"""Fixed-node diffusion Monte Carlo on the trial wave functions saved by
wavefunction.py.

ALGORITHM
---------
Importance-sampled DMC propagates the mixed distribution
f(R, tau) = Psi_T(R) Phi(R, tau), which obeys

    df/dtau = (1/2) grad^2 f - div[v_D f] - [E_L(R) - E_T] f ,
    v_D = grad ln|Psi_T| .

Each step therefore consists of

  1. drift + diffusion   R' = R + dtau v_D(R) + sqrt(dtau) eta ,
  2. fixed-node rejection: any move with sign(Psi_T(R')) != sign(Psi_T(R))
     is rejected, which confines each walker to one nodal pocket and keeps
     f >= 0 ,
  3. a Metropolis accept/reject step, which removes the leading time-step
     error of the short-time propagator,
  4. a branching weight w *= exp(-dtau [ <E_L> - E_T ]).

The drift is capped with the Umrigar form v -> v * 2/(1 + sqrt(1 + 2 v^2 dtau)),
which reduces to v for small velocities and bounds the displacement by
sqrt(2 dtau) near the nodes, where v_D diverges. The local energy entering the
branching factor is clipped to a window of half-width 2/sqrt(dtau) about the
running estimate. For Be with this trial function the window is 200-400 times
the standard deviation of E_L, so the cutoff is present but essentially never
active; it is a safeguard, not a regularisation that does work.

PROTOCOL (changed with respect to the previous version)
--------
All lengths are specified in IMAGINARY TIME, not in steps:

    tau_equil   projection time discarded before accumulation,
    tau_block   length of one accumulation block,
    tau_prod    total accumulated imaginary time.

The previous driver fixed the number of steps and varied dtau, so equilibration,
block length and statistics all changed by 4x along a time-step series, and the
residual contamination of the mixed estimator (always positive, largest at the
smallest dtau) mimicked a time-step slope. With tau-based lengths every point of
a series has the same physics except dtau.

Population. Each walker carries a continuous weight (static shapes, JAX
friendly). Every `feedback_every` steps E_T is adjusted by feedback on the total
weight and the weights are renormalised to total N (this changes no estimator).
Comb resampling is done only when the effective sample size
ESS = (sum w)^2 / sum w^2 drops below ess_min * N. Resampling on a fixed step
counter, as before, was almost a no-op (weight dispersion ~1e-3 over 5 steps)
that still injected resampling noise 1600 times per run. After resampling the
cached ln|Psi|, sign and E_L are gathered with the resampling indices instead
of being recomputed.

Errors. Blocks of length tau_block >~ 10 tau_corr (tau_corr ~ 0.1 a.u. here)
are essentially independent; stats.blocking_analysis still checks for a
plateau and says so if none is reached.

Seeds. Every run in a series gets its own seed; with a shared seed and identical
step counts, the previous series drew the SAME Gaussian increments at every
dtau, only rescaled, which made the points strongly correlated.

Everything a run produces (blocks, equilibration curve E(tau), acceptance,
crossing rate, ESS, number of resamplings) is saved to .npz, so the analysis
can be redone without rerunning.

WHAT THE RESULT MEANS
---------------------
E_FN is variational, E_FN >= E_0, with equality only if the nodes are exact.
It depends on the nodal surface of Psi_T and on nothing else, which is checked
by test_jastrow_independence(). Because of the bound, E_FN - E_exact is quoted
with a ONE-SIDED 95% upper limit when it is not resolved from zero.
"""

import math
import time

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp
import numpy as np

jax.config.update("jax_enable_x64", True)

from . import wavefunction as vmc
from .stats import blocking_analysis, weighted_mean
from . import data_path

E_EXACT_BE = vmc.E_EXACT          # single source of truth
E_EXACT_HE = -2.903724
E_EXACT_H = -0.5


# ============================================================================
# Generic engine (unchanged).  All functions take R of shape (n_elec, 3) and
# are already bound to the wave-function parameters.
# ============================================================================
def capped_drift(v, dtau):
    """Umrigar drift cap, in the numerically stable algebraic form.

    scale = (sqrt(1 + 2 v^2 dtau) - 1) / (v^2 dtau) = 2/(1 + sqrt(1 + 2 v^2 dtau))

    The second form avoids the 0/0 cancellation at small v. The cap is applied
    per electron: it leaves small drifts unchanged and bounds the displacement
    by sqrt(2 dtau) where v_D diverges, i.e. close to the nodal surface.
    """
    v2 = jnp.sum(v ** 2, axis=-1, keepdims=True)
    scale = 2.0 / (1.0 + jnp.sqrt(1.0 + 2.0 * v2 * dtau))
    return v * scale


def comb_resample(key, R, w):
    """Stratified (comb) resampling: fixed population, minimal added variance."""
    n = w.shape[0]
    wn = w / jnp.sum(w) * n
    c = jnp.cumsum(wn)
    u = jax.random.uniform(key)
    idx = jnp.searchsorted(c, u + jnp.arange(n))
    idx = jnp.clip(idx, 0, n - 1)
    return R[idx], idx


def make_dmc(logpsi_fn, sign_fn, drift_fn, eloc_fn):
    """Build the vectorised one-step propagator.

    logpsi_fn(R) -> ln|Psi_T|
    sign_fn(R)   -> +/-1
    drift_fn(R)  -> grad_R ln|Psi_T|, shape (n_elec, 3)
    eloc_fn(R)   -> E_L(R)
    """

    def one_walker(key, R, lp, sgn, el, dtau, E_T, e_ref):
        k1, k2 = jax.random.split(key)

        v = capped_drift(drift_fn(R), dtau)
        eta = jax.random.normal(k1, R.shape, dtype=R.dtype) * jnp.sqrt(dtau)
        R_new = R + dtau * v + eta

        lp_new = logpsi_fn(R_new)
        s_new = sign_fn(R_new)
        v_new = capped_drift(drift_fn(R_new), dtau)

        # log of the drift-diffusion Green functions, forward and reverse
        g_fwd = -jnp.sum((R_new - R - dtau * v) ** 2) / (2.0 * dtau)
        g_bwd = -jnp.sum((R - R_new - dtau * v_new) ** 2) / (2.0 * dtau)

        log_acc = 2.0 * (lp_new - lp) + g_bwd - g_fwd
        crossed = (s_new * sgn) < 0.0
        u = jax.random.uniform(k2, dtype=R.dtype)
        accept = jnp.logical_and(jnp.log(u) < log_acc, jnp.logical_not(crossed))

        el_new = eloc_fn(R_new)

        R_out = jnp.where(accept, R_new, R)
        lp_out = jnp.where(accept, lp_new, lp)
        s_out = jnp.where(accept, s_new, sgn)
        el_out = jnp.where(accept, el_new, el)

        # Branching factor with a symmetric cutoff on the local energy.
        ecut = 2.0 / jnp.sqrt(dtau)
        e_old_c = jnp.clip(el, e_ref - ecut, e_ref + ecut)
        e_new_c = jnp.clip(el_out, e_ref - ecut, e_ref + ecut)
        dlogw = -dtau * (0.5 * (e_old_c + e_new_c) - E_T)

        return (R_out, lp_out, s_out, el_out, dlogw,
                accept.astype(R.dtype), crossed.astype(R.dtype))

    @jax.jit
    def step(keys, R, lp, sgn, el, logw, dtau, E_T, e_ref):
        out = jax.vmap(one_walker, in_axes=(0, 0, 0, 0, 0, None, None, None))(
            keys, R, lp, sgn, el, dtau, E_T, e_ref)
        R, lp, sgn, el, dlogw, acc, cross = out
        return R, lp, sgn, el, logw + dlogw, jnp.mean(acc), jnp.mean(cross)

    @jax.jit
    def init_state(R):
        lp = jax.vmap(logpsi_fn)(R)
        sgn = jax.vmap(sign_fn)(R)
        el = jax.vmap(eloc_fn)(R)
        return lp, sgn, el

    return step, init_state


def n_steps_for(tau, dtau):
    """Number of steps covering imaginary time tau (at least one)."""
    return max(1, int(math.ceil(tau / dtau - 1e-9)))


# ============================================================================
# Driver.
# ============================================================================
def run_dmc(logpsi_fn, sign_fn, drift_fn, eloc_fn, R0,
            dtau=0.005,
            tau_equil=10.0,
            tau_block=1.0,
            tau_prod=100.0,
            tau_chunk=0.1,
            feedback_every=5,
            tau_feedback=1.0,
            ess_min=0.95,
            e_ref=None,
            seed=0,
            label="",
            verbose=True):
    """Run fixed-node DMC with all lengths in imaginary time.

    tau_chunk      resolution of the recorded equilibration curve E(tau).
    tau_feedback   strength of the population control: E_T is displaced from
                   the running energy by -ln(W/N)/tau_feedback.
    ess_min        resample when ESS < ess_min * N.
    """
    key = jax.random.PRNGKey(seed)
    step, init_state = make_dmc(logpsi_fn, sign_fn, drift_fn, eloc_fn)

    R = jnp.asarray(R0)
    n_walk = R.shape[0]
    lp, sgn, el = init_state(R)
    logw = jnp.zeros(n_walk, dtype=R.dtype)
    logN = np.log(n_walk)

    if e_ref is None:
        e_ref = float(jnp.mean(el))
    E_T = e_ref
    e_run = e_ref

    n_equil = n_steps_for(tau_equil, dtau)
    n_chunk = n_steps_for(tau_chunk, dtau)
    steps_per_block = n_steps_for(tau_block, dtau)
    n_blocks = n_steps_for(tau_prod, tau_block)

    if verbose:
        print(f"\n--- DMC {label} ---")
        print(f"  dtau = {dtau}, walkers = {n_walk}, seed = {seed}")
        print(f"  tau_equil = {tau_equil} ({n_equil} steps), "
              f"tau_block = {tau_block} ({steps_per_block} steps), "
              f"tau_prod = {tau_prod} ({n_blocks} blocks)")
        print(f"  initial <E_L> = {e_ref:.5f}")

    state = dict(R=R, lp=lp, sgn=sgn, el=el, logw=logw, E_T=E_T, key=key,
                 n_resample=0, t=0)

    def advance(n_steps, e_run):
        """Advance n_steps, accumulating the weighted mixed estimator."""
        s = state
        num = den = acc_sum = cross_sum = ess_sum = 0.0
        n_res0 = s["n_resample"]
        for _ in range(n_steps):
            s["key"], sk, rk = jax.random.split(s["key"], 3)
            keys = jax.random.split(sk, n_walk)
            (s["R"], s["lp"], s["sgn"], s["el"], s["logw"], a, c) = step(
                keys, s["R"], s["lp"], s["sgn"], s["el"], s["logw"],
                dtau, s["E_T"], e_run)
            acc_sum += float(a)
            cross_sum += float(c)

            # Weights are renormalised to total N every few steps, so exp is
            # safe and each generation enters with its own total weight.
            w = jnp.exp(s["logw"])
            num += float(jnp.sum(w * s["el"]))
            den += float(jnp.sum(w))
            s["t"] += 1

            if s["t"] % feedback_every == 0:
                ratio = float(logsumexp(s["logw"])) - logN
                s["E_T"] = e_run - ratio / tau_feedback
                s["logw"] = s["logw"] - ratio
                log_ess = (2.0 * float(logsumexp(s["logw"]))
                           - float(logsumexp(2.0 * s["logw"])))
                ess = math.exp(log_ess) / n_walk
                ess_sum += ess
                if ess < ess_min:
                    _, idx = comb_resample(rk, s["R"], jnp.exp(s["logw"]))
                    s["R"], s["lp"] = s["R"][idx], s["lp"][idx]
                    s["sgn"], s["el"] = s["sgn"][idx], s["el"][idx]
                    s["logw"] = jnp.zeros_like(s["logw"])
                    s["n_resample"] += 1
        n_fb = max(1, n_steps // feedback_every)
        return (num / den, acc_sum / n_steps, cross_sum / n_steps,
                ess_sum / n_fb, s["n_resample"] - n_res0)

    # ---- equilibration: record the projection curve E(tau) ---------------
    t0 = time.time()
    eq_tau, eq_E = [], []
    done = 0
    while done < n_equil:
        m = min(n_chunk, n_equil - done)
        e_c, a, c, _, _ = advance(m, e_run)
        done += m
        eq_tau.append(done * dtau)
        eq_E.append(e_c)
        e_run = 0.7 * e_run + 0.3 * e_c
    if verbose:
        print(f"  equilibration in {time.time()-t0:.1f}s   "
              f"E_run = {e_run:.5f}  acc = {a*100:.1f}%  cross = {c*100:.3f}%")

    # ---- accumulation ------------------------------------------------------
    t0 = time.time()
    blocks, accs, crosses, esss, nres = [], [], [], [], []
    for b in range(n_blocks):
        eb, a, c, ess, nr = advance(steps_per_block, e_run)
        blocks.append(eb)
        accs.append(a)
        crosses.append(c)
        esss.append(ess)
        nres.append(nr)
        e_run = 0.9 * e_run + 0.1 * eb
        if verbose and ((b + 1) % max(1, n_blocks // 10) == 0):
            print(f"  block {b+1:4d}/{n_blocks}  E_block = {eb:.5f}  "
                  f"E_mean = {np.mean(blocks):.5f}  acc = {a*100:.1f}%  "
                  f"cross = {c*100:.3f}%  <ESS/N> = {ess:.3f}  "
                  f"resamplings = {sum(nres)}")

    blk = blocking_analysis(blocks, min_bins=16)
    res = dict(E=blk["mean"], dE=blk["err"], dE_err=blk["err_err"],
               plateau=blk["plateau"], tau_int_blocks=blk["tau_int"],
               dtau=dtau, n_walkers=n_walk, seed=seed,
               tau_equil=tau_equil, tau_block=tau_block, tau_prod=tau_prod,
               steps_per_block=steps_per_block,
               blocks=np.array(blocks), acc_blocks=np.array(accs),
               cross_blocks=np.array(crosses), ess_blocks=np.array(esss),
               resample_blocks=np.array(nres),
               equil_tau=np.array(eq_tau), equil_E=np.array(eq_E),
               acceptance=float(np.mean(accs)),
               crossing=float(np.mean(crosses)),
               wall_time=time.time() - t0)
    if verbose:
        print(f"  accumulation in {res['wall_time']:.1f}s")
        print(f"  E_DMC = {res['E']:.6f} +/- {res['dE']:.6f}"
              f"  (blocking level {blk['level']}, "
              f"{'plateau' if blk['plateau'] else ('too few blocks to test for a plateau' if len(blk['errs']) < 2 else 'NO PLATEAU: lengthen tau_block')})")
        print(f"  acc = {res['acceptance']*100:.2f}%  "
              f"node crossings = {res['crossing']*100:.4f}%  "
              f"resamplings = {sum(nres)}")
    return res


def save_run(res, path, **extra):
    """Write every array and scalar of a run to .npz, inside DATA_DIR."""
    path = data_path(path)
    out = {k: v for k, v in res.items() if not isinstance(v, (dict, list))}
    out.update(extra)
    np.savez(path, **out)
    print(f"[saved] {path}")


def bind(wf, params=None):
    """(logpsi, sign, drift, eloc) closures for run_dmc."""
    p = wf["params"] if params is None else params
    return (lambda R: wf["log_psi"](p, R),
            lambda R: wf["sign_psi"](p, R),
            lambda R: wf["grad_log_psi"](p, R),
            lambda R: wf["local_energy"](p, R))


def report_vs_exact(E, dE, E_vmc=None, dE_vmc=None, e_exact=E_EXACT_BE):
    """Print E_FN - E_exact honestly: a value if resolved, else a one-sided
    95% upper bound (E_FN >= E_exact by the variational property)."""
    d = (E - e_exact) * 1e3
    s = dE * 1e3
    print(f"    E_FN - exact   = {d:+.3f} +/- {s:.3f} mHa  ({d/s:.1f} sigma)")
    if d < 2.0 * s:
        print(f"    -> not resolved from zero; one-sided 95% bound "
              f"< {max(d, 0.0) + 1.645 * s:.3f} mHa")
    if E_vmc is not None:
        g = (E_vmc - E) * 1e3
        sg = math.hypot(dE, dE_vmc or 0.0) * 1e3
        print(f"    E_VMC - E_FN   = {g:+.3f} +/- {sg:.3f} mHa  ({g/sg:.1f} sigma)")
        err = (E_vmc - e_exact) * 1e3
        print(f"    of the VMC error ({err:.3f} mHa), the Jastrow gap is "
              f"{g/err*100:.0f}% and the nodal part {d/err*100:.0f} "
              f"+/- {s/err*100:.0f}%")


# ============================================================================
# Validation on systems with known answers.
# ============================================================================
def _hydrogen_fns(alpha):
    """Psi_T = exp(-alpha r), one electron. No nodes, so DMC must be exact."""
    def logpsi(R):
        return -alpha * jnp.sqrt(jnp.sum(R ** 2) + 1e-14)

    def sign(R):
        return jnp.array(1.0)

    def drift(R):
        r = jnp.sqrt(jnp.sum(R ** 2) + 1e-14)
        return -alpha * R / r

    def eloc(R):
        r = jnp.sqrt(jnp.sum(R ** 2) + 1e-14)
        # E_L = (alpha - 1)/r - alpha^2/2
        return (alpha - 1.0) / r - 0.5 * alpha ** 2

    return logpsi, sign, drift, eloc


def _helium_fns(alpha):
    """Psi_T = exp(-alpha(r1+r2)), two electrons, no Jastrow, no nodes."""
    def logpsi(R):
        r = jnp.sqrt(jnp.sum(R ** 2, axis=-1) + 1e-14)
        return -alpha * jnp.sum(r)

    def sign(R):
        return jnp.array(1.0)

    def drift(R):
        r = jnp.sqrt(jnp.sum(R ** 2, axis=-1, keepdims=True) + 1e-14)
        return -alpha * R / r

    def eloc(R):
        r = jnp.sqrt(jnp.sum(R ** 2, axis=-1) + 1e-14)
        r12 = jnp.sqrt(jnp.sum((R[0] - R[1]) ** 2) + 1e-14)
        lap = jnp.sum(-2.0 * alpha / r)
        grad2 = 2.0 * alpha ** 2
        return -0.5 * (lap + grad2) - 2.0 / r[0] - 2.0 / r[1] + 1.0 / r12

    return logpsi, sign, drift, eloc


def linear_fit(x, y, s):
    """Weighted linear fit y = a + b x; returns (a, da, b, db, chi2)."""
    x, y, s = map(lambda v: np.asarray(v, float), (x, y, s))
    w = 1.0 / s ** 2
    A = np.vstack([np.ones_like(x), x]).T
    cov = np.linalg.inv(A.T @ (w[:, None] * A))
    beta = cov @ (A.T @ (w * y))
    chi2 = float(np.sum(w * (y - A @ beta) ** 2))
    return beta[0], math.sqrt(cov[0, 0]), beta[1], math.sqrt(cov[1, 1]), chi2


def validate(n_walkers=4000, alpha=1.3, dtaus=(0.020, 0.010, 0.005, 0.0025),
             tau_equil=5.0, tau_block=1.0, tau_prod=200.0, seed=1):
    """Hydrogen with a wrong exponent: nodeless, so DMC must give -0.5 at
    dtau -> 0. Same imaginary-time protocol and a distinct seed at every dtau.
    """
    print("=" * 72)
    print("  VALIDATION: hydrogen, time-step series")
    print("=" * 72)
    fns = _hydrogen_fns(alpha)
    res = []
    for i, dt in enumerate(dtaus):
        R0 = jax.random.normal(jax.random.PRNGKey(seed + 1000 + i),
                               (n_walkers, 1, 3), dtype=jnp.float64) * 0.8
        r = run_dmc(*fns, R0, dtau=dt, tau_equil=tau_equil,
                    tau_block=tau_block, tau_prod=tau_prod, seed=seed + i,
                    label=f"H, dtau = {dt}", verbose=False)
        res.append(r)
        print(f"  dtau = {dt:<8.4f}  E = {r['E']:.6f} +/- {r['dE']:.6f}   "
              f"({(r['E'] - E_EXACT_H)*1e3:+.3f} mHa)")

    a, da, b, db, chi2 = linear_fit([r["dtau"] for r in res],
                                    [r["E"] for r in res],
                                    [r["dE"] for r in res])
    print(f"\n  dtau -> 0     E = {a:.6f} +/- {da:.6f}   "
          f"({(a - E_EXACT_H)/da:+.1f} sigma from -0.5)")
    print(f"  slope         {b:+.4f} +/- {db:.4f} Eh per unit dtau   "
          f"chi2 = {chi2:.1f} / {len(res) - 2} dof")
    ok = abs(a - E_EXACT_H) < 3.0 * da
    print("\n  validation", "PASSED" if ok else "FAILED")
    return ok


def test_jastrow_independence(npz_path, dtau=0.005, tau_equil=10.0,
                              tau_block=1.0, tau_prod=60.0, seed=7):
    """E_FN must depend on the nodes alone, not on the Jastrow factor."""
    print("=" * 72)
    print("  JASTROW INDEPENDENCE OF E_FN")
    print("=" * 72)
    wf = vmc.load_wavefunction(npz_path, verbose=False)
    params = wf["params"]
    kw = dict(dtau=dtau, tau_equil=tau_equil, tau_block=tau_block,
              tau_prod=tau_prod)

    res_full = run_dmc(*bind(wf), wf["R"], seed=seed, label="full Psi_T", **kw)

    # determinant only: zero the network, push the Pade b to its bound
    p0 = {"mlp": jax.tree_util.tree_map(jnp.zeros_like, params["mlp"]),
          "orb": dict(params["orb"])}
    p0["orb"]["b_par"] = jnp.asarray(vmc.B_MAX)
    p0["orb"]["b_anti"] = jnp.asarray(vmc.B_MAX)
    if "j1" in params:                      # one-body term off as well
        p0["j1"] = jax.tree_util.tree_map(jnp.zeros_like, params["j1"])
    res_det = run_dmc(*bind(wf, p0), wf["R"], seed=seed + 1,
                      label="determinant only (same nodes)", **kw)

    diff = res_full["E"] - res_det["E"]
    sig = math.hypot(res_full["dE"], res_det["dE"])
    print(f"\n  full Psi_T        : {res_full['E']:.6f} +/- {res_full['dE']:.6f}")
    print(f"  determinant only  : {res_det['E']:.6f} +/- {res_det['dE']:.6f}")
    print(f"  difference        : {diff*1e3:+.3f} mHa  ({abs(diff)/sig:.1f} sigma)")
    return res_full, res_det


def test_population_bias(npz_path, populations=(1000, 2000, 4000),
                         dtau=0.0025, tau_equil=10.0, tau_block=1.0,
                         tau_prod=60.0, seed=17):
    """Check convergence of E_FN with walker population."""
    print("=" * 72)
    print("  POPULATION-BIAS TEST")
    print("=" * 72)
    wf = vmc.load_wavefunction(npz_path, verbose=False)
    R0 = wf["R"]
    if R0.shape[0] < max(populations):
        raise ValueError(f"{npz_path} contains only {R0.shape[0]} walkers")
    e_vmc = float(wf["meta"]["E"])
    results = []
    for i, nw in enumerate(populations):
        r = run_dmc(*bind(wf), R0[:nw], dtau=dtau, tau_equil=tau_equil,
                    tau_block=tau_block, tau_prod=tau_prod, e_ref=e_vmc,
                    seed=seed + i, label=f"population = {nw}")
        results.append(r)
        print(f"  nw = {nw:5d}  E_FN = {r['E']:.6f} +/- {r['dE']:.6f}")
    return results


# ============================================================================
# Production.
# ============================================================================
def run_production(npz_path, dtau=0.0025, tau_equil=10.0, tau_block=1.0,
                   tau_prod=150.0, n_walkers=None, seed=2026, tag=None):
    """One long run at a single small time step: the number to quote.

    At dtau = 0.0025 the node-crossing rate is ~0.002% and, if the time-step
    bias scales like the crossing rate (~dtau^1.5 here), it is ~0.01 mHa,
    below the statistical error this run reaches (~0.03 mHa for tau_prod = 150
    and 4000 walkers). Check the crossing rate printed at the end.
    """
    wf = vmc.load_wavefunction(npz_path, verbose=True)
    R0 = wf["R"] if n_walkers is None else wf["R"][:n_walkers]
    e_vmc = float(wf["meta"]["E"])
    de_vmc = float(wf["meta"]["dE"])
    tag = tag or npz_path.rsplit("/", 1)[-1].replace(".npz", "")

    r = run_dmc(*bind(wf), R0, dtau=dtau, tau_equil=tau_equil,
                tau_block=tau_block, tau_prod=tau_prod, e_ref=e_vmc,
                seed=seed, label=f"production, {tag}")
    print(f"\n  SUMMARY  {tag}")
    print(f"    E_VMC          = {e_vmc:.6f} +/- {de_vmc:.6f}")
    print(f"    E_FN           = {r['E']:.6f} +/- {r['dE']:.6f}   "
          f"(dtau = {dtau}, tau_prod = {tau_prod})")
    report_vs_exact(r["E"], r["dE"], e_vmc, de_vmc)
    save_run(r, f"dmc_prod_{tag}.npz", E_vmc=e_vmc, dE_vmc=de_vmc,
             wf_path=npz_path)
    return r


def run_timestep_series(npz_path, dtaus=(0.010, 0.005, 0.0025),
                        tau_equil=10.0, tau_block=1.0, tau_prod=40.0,
                        n_walkers=None, seed=11, tag=None, verbose=True):
    """Same imaginary-time protocol at every dtau, distinct seed per point.

    Reports the linear extrapolation AND the crossing-rate scaling; the
    recommended number to quote is still the production run at the smallest
    dtau (see run_production), with the series as a consistency check.
    """
    wf = vmc.load_wavefunction(npz_path, verbose=verbose)
    R0 = wf["R"] if n_walkers is None else wf["R"][:n_walkers]
    e_vmc = float(wf["meta"]["E"])
    de_vmc = float(wf["meta"]["dE"])
    tag = tag or npz_path.rsplit("/", 1)[-1].replace(".npz", "")

    out = []
    for i, dt in enumerate(dtaus):
        r = run_dmc(*bind(wf), R0, dtau=dt, tau_equil=tau_equil,
                    tau_block=tau_block, tau_prod=tau_prod, e_ref=e_vmc,
                    seed=seed + 101 * i, label=f"dtau = {dt}", verbose=verbose)
        save_run(r, f"dmc_ts_{tag}_dt{dt:g}.npz", E_vmc=e_vmc, wf_path=npz_path)
        out.append(r)

    x = np.array([r["dtau"] for r in out])
    y = np.array([r["E"] for r in out])
    s = np.array([r["dE"] for r in out])
    cr = np.array([r["crossing"] for r in out])

    print(f"\n  TIME-STEP SERIES  {tag}")
    for r in out:
        print(f"    dtau = {r['dtau']:<8.4f}  E = {r['E']:.6f} +/- {r['dE']:.6f}"
              f"   crossing = {r['crossing']*100:.4f}%")
    m, dm, chi2c = weighted_mean(y, s)
    print(f"    constant fit   E = {m:.6f} +/- {dm:.6f}   "
          f"chi2 = {chi2c:.2f} / {len(y)-1} dof")
    if len(out) >= 3:
        a, da, b, db, chi2 = linear_fit(x, y, s)
        print(f"    linear fit     E(0) = {a:.6f} +/- {da:.6f}   slope "
              f"{b*1e3:+.1f} +/- {db*1e3:.1f} mHa/unit dtau   "
              f"chi2 = {chi2:.2f} / {len(y)-2} dof")
    ok = cr > 0
    if ok.sum() >= 2:
        k, _ = np.polyfit(np.log(x[ok]), np.log(cr[ok]), 1)
        print(f"    crossing rate ~ dtau^{k:.2f}")
    else:
        k = np.nan

    np.savez(data_path(f"dmc_ts_{tag}_summary.npz"), dtau=x, E=y, dE=s, crossing=cr,
             acceptance=np.array([r["acceptance"] for r in out]),
             crossing_exponent=k, E_vmc=e_vmc, dE_vmc=de_vmc,
             tau_equil=tau_equil, tau_block=tau_block, tau_prod=tau_prod)
    print(f"[saved] dmc_ts_{tag}_summary.npz")
    return out


def compare_nodal_surfaces(paths, labels=None, dtau=0.005, tau_equil=10.0,
                           tau_block=1.0, tau_prod=60.0, n_walkers=None,
                           seed=31):
    """E_FN of several trial functions at ONE time step (the dtau dependence is
    not resolvable and largely cancels in differences).

    Run nodes first: if the nodal mismatch between two files is
    ~1e-4, their E_FN cannot differ measurably and this run is not needed.
    """
    labels = labels or [p.rsplit("/", 1)[-1].replace(".npz", "") for p in paths]
    rows = []
    for i, (p, lab) in enumerate(zip(paths, labels)):
        wf = vmc.load_wavefunction(p, verbose=False)
        R0 = wf["R"] if n_walkers is None else wf["R"][:n_walkers]
        e_vmc = float(wf["meta"]["E"])
        r = run_dmc(*bind(wf), R0, dtau=dtau, tau_equil=tau_equil,
                    tau_block=tau_block, tau_prod=tau_prod, e_ref=e_vmc,
                    seed=seed + 7 * i, label=lab)
        save_run(r, f"dmc_cmp_{lab.replace(' ', '_')}.npz", E_vmc=e_vmc,
                 dE_vmc=float(wf["meta"]["dE"]), wf_path=p)
        rows.append((lab, e_vmc, float(wf["meta"]["dE"]), r["E"], r["dE"]))

    print(f"\n{'='*88}\n  NODAL SURFACE COMPARISON  (dtau = {dtau})\n{'='*88}")
    print(f"{'wave function':<24}{'E_VMC':>13}{'E_FN':>13}{'err':>10}"
          f"{'gap (mHa)':>12}{'vs exact':>12}")
    for lab, ev, dev, ef, def_ in rows:
        print(f"{lab[:24]:<24}{ev:>13.6f}{ef:>13.6f}{def_:>10.6f}"
              f"{(ev - ef)*1e3:>12.3f}{(ef - E_EXACT_BE)*1e3:>12.3f}")
    if len(rows) > 1:
        e = np.array([r[3] for r in rows])
        s = np.array([r[4] for r in rows])
        m, dm, chi2 = weighted_mean(e, s)
        print(f"\n  common-value fit: E_FN = {m:.6f} +/- {dm:.6f}, "
              f"chi2 = {chi2:.2f} / {len(e)-1} dof")
    np.savez(data_path("dmc_nodal_comparison.npz"), labels=np.array(labels),
             paths=np.array(paths),
             E_vmc=np.array([r[1] for r in rows]),
             dE_vmc=np.array([r[2] for r in rows]),
             E_fn=np.array([r[3] for r in rows]),
             dE_fn=np.array([r[4] for r in rows]), dtau=dtau)
    print("[saved] dmc_nodal_comparison.npz")
    return rows


if __name__ == "__main__":
    WF = "multidet_quad_cas1.npz"

    # 1. validate the propagator (run once after any change to the engine)
    # validate()

    # 2. the number to quote: one long run at small dtau (~1.5 h at 4000 walkers)
    run_production(WF, dtau=0.0025, tau_prod=150.0)

    # 3. isolate the NODAL effect of the radial modulation: same protocol on
    #    the unmodulated model (train it with wavefunction, freeze="modulation")
    # run_production("multidet_nomod_cas1.npz", dtau=0.0025, tau_prod=150.0)

    # 4. consistency check of the time step (optional, same tau protocol)
    # run_timestep_series(WF, dtaus=(0.010, 0.005, 0.0025), tau_prod=40.0)