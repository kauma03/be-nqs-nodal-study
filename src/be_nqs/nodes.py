"""Nodal analysis of saved trial wave functions. No DMC is needed.

Run it on the .npz files written by wavefunction.run_pipeline, e.g. the
four radial parametrisations of the cas_level = 1 model. It answers, from the
files already on disk:

  1. FACTORISATION.  psi_det = Phi(R) N(R) with Phi = prod_i phi_1s(r_i) > 0,
     checked to machine precision on the saved walkers, together with the
     assumptions it needs (m_nl(r) > 0, f(r) monotone).

  2. GAUGE.  The node depends on the orbitals only through
         f(r)   = phi_2s / phi_1s      (defined up to f -> a f + b)
         h_l(r) = exp[(Z - a_l) r] m_l / m_1s      (defined up to h -> mu h)
     and on c_l, which transforms as c_l -> c_l a^2 / mu_l^2. Hence
       - a modulation common to all orbitals is a one-body factor and is
         not nodal; only the ratios q_nl = m_nl / m_1s are (so m_1s is
         nodal only through them);
       - c_p is not a property of the node, only of the chosen normalisation.
     The script prints the first-order gauge invariants c_k(nl) - c_k(1s) and
     writes the radial curves m, q, f, h for the figures.

  3. A GAUGE-INVARIANT MIXING WEIGHT.  For each channel (s^2, p^2, d^2, ...)
         w_l = || c_l T_l Phi e^J ||^2 / sum_k || c_k T_k Phi e^J ||^2 ,
     the fraction of the norm of Psi carried by that channel (cross terms
     between channels are dropped in the denominator, so the w_l sum to one).
     It is invariant under the transformations that leave Psi itself
     unchanged (f -> a f + b, h -> mu h with c_l -> c_l a^2/mu^2), so it tests
     whether the spread of c_p is a normalisation artefact. It is NOT
     invariant under moving a positive factor between determinant and Jastrow
     (e.g. a common factor on all m_nl): that leaves the node unchanged but
     changes Psi, and w_l with it. Node identity is tested by the mismatch
     (point 4), not by w_l. It is finite: it is estimated as  w_l = < (c_l T_l)^2 / sum_k (c_k T_k)^2 >  under the
     nodeless guiding density  e^{2J} Phi^2 sum_k (c_k T_k)^2 , so the
     estimator is bounded in [0, 1] and has finite variance.

     NOTE. The RMS ratio || c_p T_2 || / || T_1 || over |Psi|^2 proposed
     earlier is gauge invariant but NOT finite: T_1^2 grows like
     exp[4 (Z - a_2s) r] for a valence electron at large r while |Psi|^2
     decays only like exp[-2 (a_2s) r] times polynomials, so the integral
     diverges and a sample estimate is dominated by the outermost walker.
     The same objection applies to c_p <h^2 r^2> / Var(f). w_l replaces both.

  4. NODAL MISMATCH.  For two trial functions A, B:
         phi_AB = P_{|Psi_A|^2}[ sign Psi_A != sign Psi_B ],
     the |Psi_A|^2-weight of the region where the two nodes disagree. If it is
     ~1e-4 the two surfaces coincide wherever the electrons are, and no DMC
     run can separate them.

  5. SIGN BALANCE.  By antisymmetry exactly half of |Psi|^2 lies in each sign
     domain; the sampled fraction is an ergodicity check.

Everything is saved to node_analysis.npz, read by figures.py.
"""

import time

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from . import wavefunction as vmc
from .stats import weighted_mean
from . import data_path

Z = vmc.Z_NUC
L_OF = {"2p": 1, "3d": 2, "4f": 3}
CHANNEL_LABEL = {"2s": "s^2", "2p": "p^2", "3d": "d^2", "4f": "f^2"}


# ============================================================================
# The nodal function N(R), channel by channel.
# ============================================================================
def make_nodal_fns(cas_level):
    """Return comps(params, R) -> (log Phi, T, coef).

    N(R) = sum_k coef_k T_k(R), with T_0 = [f1 - f0][f3 - f2] (coef_0 = 1)
    and T_l = K_l(0,2) - K_l(0,3) - K_l(1,2) + K_l(1,3) for l >= 1.
    """
    shells = vmc.shell_names(cas_level)
    high = [s for s in shells if s in vmc.HIGH_SHELLS]

    def comps(params, R):
        po = params["orb"]
        r = vmc.norm_stable(R)
        t = vmc.t_cusp(r)
        m = {s: vmc.radial_mod(t, po[f"m{s}_1"], po[f"m{s}_2"]) for s in shells}

        a2s = jnp.clip(po["a2s"], *vmc.EXP_CLIP["2s"])
        c = 1.0 / (Z - a2s)
        log_phi = jnp.sum(-Z * r + jnp.log(jnp.abs(m["1s"])))

        f = (r - c) * jnp.exp((Z - a2s) * r) * m["2s"] / m["1s"]
        T = [(f[1] - f[0]) * (f[3] - f[2])]
        coef = [jnp.asarray(1.0, dtype=R.dtype)]

        dot = R @ R.T
        r2r2 = jnp.outer(r ** 2, r ** 2)
        for s in high:
            a = jnp.clip(po[f"a{s}"], *vmc.EXP_CLIP[s])
            h = jnp.exp((Z - a) * r) * m[s] / m["1s"]
            l = L_OF[s]
            if l == 1:
                ang = dot
            elif l == 2:
                ang = dot ** 2 - r2r2 / 3.0
            else:
                ang = dot ** 3 - 0.6 * r2r2 * dot
            K = jnp.outer(h, h) * ang
            T.append(K[0, 2] - K[0, 3] - K[1, 2] + K[1, 3])
            coef.append(po[vmc.MIX_KEY[s]])

        return log_phi, jnp.stack(T), jnp.stack(coef)

    return comps, ["2s"] + high


# ============================================================================
# 1. Factorisation check.
# ============================================================================
def check_factorisation(wf, comps):
    """Compare psi_det from the wave function with Phi * N on the saved walkers.

    The residual is normalised by the size of the individual terms,
    Phi * sqrt(sum_k (c_k T_k)^2), not by |psi_det|, so that walkers close to
    the node (where psi_det is a small difference of large terms) do not
    inflate it artificially.
    """
    params = wf["params"]

    def one(R):
        psi_det, _ = wf["parts"](params, R)
        log_phi, T, cf = comps(params, R)
        phi = jnp.exp(log_phi)
        N = jnp.sum(cf * T)
        scale = phi * jnp.sqrt(jnp.sum((cf * T) ** 2)) + 1e-300
        return jnp.abs(psi_det - phi * N) / scale, jnp.sign(psi_det) == jnp.sign(N)

    res, same_sign = jax.jit(jax.vmap(one))(wf["R"])
    return float(jnp.max(res)), float(jnp.mean(same_sign))


def chain_mean_and_error(x):
    """Mean and error of a (n_blocks, n_walkers, ...) array of per-walker values.

    The walkers are independent Markov chains, so each chain's time average is
    an independent sample and the error is std(chain means)/sqrt(n_walkers).
    This is exact whatever the autocorrelation along each chain, which matters
    here: a walker keeps its sign for many sweeps, so block averages over the
    same walkers are strongly correlated and blocking over a few tens of blocks
    cannot detect it.
    """
    per_chain = np.mean(np.asarray(x), axis=0)
    n = per_chain.shape[0]
    return per_chain.mean(axis=0), per_chain.std(axis=0, ddof=1) / np.sqrt(n)


# ============================================================================
# 2-3. Channel weights under the nodeless guiding density.
# ============================================================================
def channel_weights(wf, comps, n_equil=20, n_blocks=40, sweeps_per_block=10,
                    seed=0):
    """Estimate w_l for every channel; returns (w, dw) arrays."""
    params = wf["params"]
    parts = wf["parts"]

    def guide_log(p, R):
        log_phi, T, cf = comps(p, R)
        _, J = parts(p, R)
        return log_phi + J + 0.5 * jnp.log(jnp.sum((cf * T) ** 2) + 1e-300)

    @jax.jit
    def fractions(R_batch):
        def one(R):
            _, T, cf = comps(params, R)
            x = (cf * T) ** 2
            return x / (jnp.sum(x) + 1e-300)
        return jax.vmap(one)(R_batch)              # (n_walkers, n_channels)

    mcmc = vmc.make_mcmc(guide_log)
    key = jax.random.PRNGKey(seed)
    R = wf["R"]
    n = R.shape[0]
    step = wf["step"]

    for _ in range(n_equil):
        key, sk = jax.random.split(key)
        R, acc = mcmc(jax.random.split(sk, n), R, params, step, 5)
        step = vmc.adapt_step(step, float(acc))

    blocks = []
    for _ in range(n_blocks):
        key, sk = jax.random.split(key)
        R, _ = mcmc(jax.random.split(sk, n), R, params, step, sweeps_per_block)
        blocks.append(np.asarray(fractions(R)))
    return chain_mean_and_error(blocks)


# ============================================================================
# 4-5. Nodal mismatch and sign balance.
# ============================================================================
def sign_statistics(wfs, n_equil=10, n_blocks=40, sweeps_per_block=10, seed=1):
    """phi[a, b] = P_{|Psi_a|^2}(sign Psi_a != sign Psi_b), and P(Psi_a > 0).

    Returns (phi, dphi, frac_pos, dfrac_pos), each indexed by the sampling
    wave function a (and b for phi).
    """
    k = len(wfs)
    sign_fns = [jax.jit(jax.vmap(lambda R, w=w: w["sign_psi"](w["params"], R)))
                for w in wfs]
    phi = np.zeros((k, k))
    dphi = np.zeros((k, k))
    fpos = np.zeros(k)
    dfpos = np.zeros(k)

    for a, wa in enumerate(wfs):
        mcmc = vmc.make_mcmc(wa["log_psi"])
        key = jax.random.PRNGKey(seed + 100 * a)
        R = wa["R"]
        n = R.shape[0]
        step = wa["step"]
        for _ in range(n_equil):
            key, sk = jax.random.split(key)
            R, acc = mcmc(jax.random.split(sk, n), R, wa["params"], step, 5)
            step = vmc.adapt_step(step, float(acc))

        pos, mis = [], []
        for _ in range(n_blocks):
            key, sk = jax.random.split(key)
            R, _ = mcmc(jax.random.split(sk, n), R, wa["params"], step,
                        sweeps_per_block)
            sa = np.asarray(sign_fns[a](R))
            pos.append(sa > 0)
            mis.append(np.stack([sa * np.asarray(sign_fns[b](R)) < 0
                                 for b in range(k)], axis=-1))

        fpos[a], dfpos[a] = chain_mean_and_error(np.array(pos, dtype=float))
        phi[a], dphi[a] = chain_mean_and_error(np.array(mis, dtype=float))
    return phi, dphi, fpos, dfpos


# ============================================================================
# Radial curves for the figures.
# ============================================================================
def radial_curves(wfs, r_grid, r_star=2.0):
    """m_nl, q_nl, f, h and their gauge-fixed versions on a common grid.

    Gauge fixing, for the comparison between parametrisations:
        f_hat(r) = [f(r) - f(0)] / [f(r*) - f(0)]      (removes a, b)
        h_hat(r) = h(r) / h(r*)                         (removes mu)
    """
    out = {}
    for i, wf in enumerate(wfs):
        prof = vmc.radial_profiles(wf["params"]["orb"], wf["cas_level"], r_grid)
        prof_star = vmc.radial_profiles(wf["params"]["orb"], wf["cas_level"],
                                        np.array([0.0, r_star]))
        f0, fs = prof_star["f"]
        rows = {
            "m_1s": prof["m"]["1s"], "m_2s": prof["m"]["2s"],
            "m_2p": prof["m"]["2p"],
            "q_2s": prof["q"]["2s"], "q_2p": prof["q"]["2p"],
            "f": prof["f"], "h_2p": prof["h"]["2p"],
            "f_hat": (prof["f"] - f0) / (fs - f0),
            "h_hat": prof["h"]["2p"] / prof_star["h"]["2p"][1],
        }
        for name, v in rows.items():
            out.setdefault(name, []).append(v)
    return {k: np.array(v) for k, v in out.items()}


def radial_density(R, r_grid):
    """One-body radial distribution P(r) (all electrons), normalised to 1."""
    r = np.linalg.norm(np.asarray(R), axis=-1).ravel()
    edges = np.concatenate([[r_grid[0]], 0.5 * (r_grid[1:] + r_grid[:-1]),
                            [r_grid[-1]]])
    p, _ = np.histogram(r, bins=edges, density=True)
    return p


# ============================================================================
# Driver.
# ============================================================================
def analyse(paths, labels=None, out_path="node_analysis.npz",
            n_blocks=40, sweeps_per_block=10, r_max=6.0, r_star=2.0):
    labels = labels or [p.rsplit("/", 1)[-1].replace(".npz", "") for p in paths]
    t0 = time.time()

    print("=" * 78)
    print("  LOADING")
    print("=" * 78)
    wfs = []
    for p, lab in zip(paths, labels):
        print(f"\n[{lab}]  {p}")
        wf = vmc.load_wavefunction(p, verbose=True)
        wf["label"] = lab
        wf["diag"] = vmc.radial_diagnostics(wf["params"]["orb"],
                                            wf["cas_level"], R=wf["R"],
                                            verbose=True)
        wfs.append(wf)

    cas_levels = {w["cas_level"] for w in wfs}
    shells_common = vmc.shell_names(min(cas_levels))

    # ---- parameters and first-order gauge invariants -----------------------
    print("\n" + "=" * 78)
    print("  PARAMETERS AND FIRST-ORDER GAUGE INVARIANTS")
    print("=" * 78)
    print("  Multiplying every m_nl by (1 + eps t) shifts every c1 by eps, so "
          "c1(nl) - c1(1s)")
    print("  is invariant at first order; c1(1s) by itself is gauge "
          "dependent.\n")
    head = f"{'':<16}{'E_VMC':>12}{'err':>9}{'c_p':>9}{'a_2s':>8}{'a_2p':>8}"
    for s in shells_common:
        head += f"{'c1(' + s + ')':>10}"
    head += f"{'D1(2s)':>9}{'D1(2p)':>9}{'D2(2s)':>9}{'D2(2p)':>9}"
    print(head)
    for w in wfs:
        po = w["params"]["orb"]
        c1 = {s: vmc._clipped(po, f"m{s}_1", vmc.M1_MIN, vmc.M1_MAX)
              for s in shells_common}
        c2 = {s: vmc._clipped(po, f"m{s}_2", vmc.M2_MIN, vmc.M2_MAX)
              for s in shells_common}
        line = (f"{w['label'][:16]:<16}{float(w['meta']['E']):>12.5f}"
                f"{float(w['meta']['dE']):>9.5f}{float(po['cp']):>+9.4f}"
                f"{float(po['a2s']):>8.4f}{float(po['a2p']):>8.4f}")
        for s in shells_common:
            line += f"{c1[s]:>+10.4f}"
        line += (f"{c1['2s'] - c1['1s']:>+9.3f}{c1['2p'] - c1['1s']:>+9.3f}"
                 f"{c2['2s'] - c2['1s']:>+9.3f}{c2['2p'] - c2['1s']:>+9.3f}")
        print(line)

    def rel_spread(v):
        v = np.asarray(v, float)
        return (v.max() - v.min()) / abs(v.mean())

    if len(wfs) > 1:
        cols = {
            "c1(1s)": [vmc._clipped(w["params"]["orb"], "m1s_1", vmc.M1_MIN,
                                    vmc.M1_MAX) for w in wfs],
            "D1(2s)": [vmc._clipped(w["params"]["orb"], "m2s_1", vmc.M1_MIN,
                                    vmc.M1_MAX)
                       - vmc._clipped(w["params"]["orb"], "m1s_1", vmc.M1_MIN,
                                      vmc.M1_MAX) for w in wfs],
            "D1(2p)": [vmc._clipped(w["params"]["orb"], "m2p_1", vmc.M1_MIN,
                                    vmc.M1_MAX)
                       - vmc._clipped(w["params"]["orb"], "m1s_1", vmc.M1_MIN,
                                      vmc.M1_MAX) for w in wfs],
        }
        print("\n  spread across parametrisations (max - min):")
        for k, v in cols.items():
            print(f"    {k:<8} {np.max(v) - np.min(v):.3f}")

    # ---- factorisation -------------------------------------------------------
    print("\n" + "=" * 78)
    print("  FACTORISATION  psi_det = Phi * N")
    print("=" * 78)
    comps_by = {}
    fact = []
    for w in wfs:
        comps, channels = make_nodal_fns(w["cas_level"])
        comps_by[w["label"]] = (comps, channels)
        res, same = check_factorisation(w, comps)
        fact.append(res)
        print(f"  {w['label']:<16} max relative residual = {res:.2e}   "
              f"sign(psi_det) = sign(N) on {same*100:.3f}% of walkers")

    # ---- channel weights -------------------------------------------------------
    print("\n" + "=" * 78)
    print("  GAUGE-INVARIANT CHANNEL WEIGHTS  w_l")
    print("=" * 78)
    weights, dweights, chans = [], [], []
    for i, w in enumerate(wfs):
        comps, channels = comps_by[w["label"]]
        wl, dwl = channel_weights(w, comps, n_blocks=n_blocks,
                                  sweeps_per_block=sweeps_per_block, seed=i)
        weights.append(wl)
        dweights.append(dwl)
        chans.append(channels)
        txt = "  ".join(f"w({CHANNEL_LABEL[c]}) = {x:.4e} +/- {dx:.1e}"
                        for c, x, dx in zip(channels, wl, dwl))
        print(f"  {w['label']:<16} {txt}")

    # p^2 channel is common to all files
    cp = np.array([float(w["params"]["orb"]["cp"]) for w in wfs])
    wp = np.array([wl[1] for wl in weights])
    dwp = np.array([dwl[1] for dwl in dweights])
    if len(wfs) > 1 and np.all(dwp > 0):
        m, dm, chi2 = weighted_mean(wp, dwp)
        print(f"\n  c_p  : relative spread {rel_spread(cp)*100:5.1f}%")
        print(f"  w_p  : relative spread {rel_spread(wp)*100:5.1f}%   "
              f"weighted mean {m:.4e} +/- {dm:.1e}   "
              f"chi2 = {chi2:.1f} / {len(wp)-1} dof")
        if chi2 / (len(wp) - 1) < 2.0:
            print("  -> w_p agrees across parametrisations within errors while "
                  "c_p does not: the")
            print("     spread of c_p is invisible in a gauge-invariant weight, "
                  "as a gauge effect must be.")
        else:
            print("  -> w_p differs between parametrisations: the wave functions"
                  " carry genuinely")
            print("     different p^2 weight (this does not by itself mean "
                  "different nodes; see mismatch).")

    # ---- mismatch and sign balance ----------------------------------------------
    print("\n" + "=" * 78)
    print("  NODAL MISMATCH  phi[a,b] = P_|Psi_a|^2 (sign Psi_a != sign Psi_b)")
    print("=" * 78)
    phi, dphi, fpos, dfpos = sign_statistics(
        wfs, n_blocks=n_blocks, sweeps_per_block=sweeps_per_block)
    print(f"{'sampled from':<18}" + "".join(f"{w['label'][:16]:>18}"
                                            for w in wfs))
    n_w = [w["R"].shape[0] for w in wfs]

    def cell(a, b):
        if a == b:
            return f"{'-':>18}"
        if phi[a, b] == 0.0:
            # no disagreement seen on any chain: one-sided 95% bound
            return f"{'0 (<%.1e)' % (3.0 / n_w[a]):>18}"
        return f"{'%.2e+/-%.0e' % (phi[a, b], dphi[a, b]):>18}"

    for a, w in enumerate(wfs):
        print(f"{w['label'][:18]:<18}" + "".join(cell(a, b)
                                                for b in range(len(wfs))))
    if np.any(phi > 0.5):
        print("  NOTE: an entry above 0.5 means the two files use opposite "
              "global sign conventions; read it as 1 - phi.")
    print("\n  sign balance (must be 0.5 by antisymmetry):")
    for a, w in enumerate(wfs):
        z = (fpos[a] - 0.5) / max(dfpos[a], 1e-12)
        print(f"  {w['label']:<16} P(Psi > 0) = {fpos[a]:.4f} +/- "
              f"{dfpos[a]:.4f}   ({z:+.1f} sigma)")

    # ---- radial curves ----------------------------------------------------------
    r_grid = np.linspace(0.0, r_max, 601)
    curves = radial_curves(wfs, r_grid, r_star=r_star)

    # Exact gauge-invariant mixing coefficient c~_p = c_p [h(r*)/(f(r*)-f(0))]^2
    # (no expectation value, no statistical error). With f_hat and h_hat it
    # fixes the node completely.
    cp_t = []
    for w in wfs:
        ps = vmc.radial_profiles(w["params"]["orb"], w["cas_level"],
                                 np.array([0.0, r_star]))
        f0, fs = ps["f"]
        cp_t.append(float(w["params"]["orb"]["cp"])
                    * (ps["h"]["2p"][1] / (fs - f0)) ** 2)
    cp_t = np.array(cp_t)
    print("\n" + "=" * 78)
    print(f"  EXACT GAUGE-INVARIANT MIXING  c~_p = c_p [h(r*)/(f(r*)-f(0))]^2,"
          f"  r* = {r_star}")
    print("=" * 78)
    for w, c, ct in zip(wfs, cp, cp_t):
        print(f"  {w['label']:<16} c_p = {c:+.4f}   c~_p = {ct:+.5f}")
    dens = radial_density(wfs[0]["R"], r_grid)

    save = dict(
        labels=np.array(labels), paths=np.array(paths),
        E_vmc=np.array([float(w["meta"]["E"]) for w in wfs]),
        dE_vmc=np.array([float(w["meta"]["dE"]) for w in wfs]),
        cp=cp, cp_tilde=cp_t, wp=wp, dwp=dwp,
        a2s=np.array([float(w["params"]["orb"]["a2s"]) for w in wfs]),
        a2p=np.array([float(w["params"]["orb"]["a2p"]) for w in wfs]),
        fact_residual=np.array(fact),
        phi=phi, dphi=dphi, frac_pos=fpos, dfrac_pos=dfpos,
        r=r_grid, r_star=r_star, density=dens,
        radial_ok=np.array([w["diag"]["ok"] for w in wfs]),
        **{f"curve_{k}": v for k, v in curves.items()})
    for i, (wl, dwl, ch) in enumerate(zip(weights, dweights, chans)):
        save[f"w_{i}"] = wl
        save[f"dw_{i}"] = dwl
        save[f"channels_{i}"] = np.array(ch)
    out_path = data_path(out_path)
    np.savez(out_path, **save)
    print(f"\n[saved] {out_path}   ({time.time() - t0:.0f}s)")
    return save


if __name__ == "__main__":
    # Edit to match the files on disk. The label is what appears in the
    # tables and in the figure legends.
    FILES = {
        "quad A":      "multidet_quadA_cas1.npz",
        "quad B":      "multidet_quad_cas1.npz",
        "orthogonal":  "multidet_orth_cas1.npz",
        "linear":      "multidet_lin_cas1.npz",
    }
    analyse(list(FILES.values()), labels=list(FILES.keys()))