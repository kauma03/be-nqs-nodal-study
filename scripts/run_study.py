"""Run the whole cas_level = 1 study from scratch, in stages.

    python run_study.py <stage> [--quick] [--force]

Stages, in the order they must run:

    validate   hydrogen time-step series: checks the DMC propagator    (~15 min)
    vmc        trains the four trial functions from scratch             (longest)
    nodes      nodal analysis on the four .npz files                    (minutes)
    dmc        production E_FN for quadA and nomod, Jastrow-independence
               test, time-step series on quadA                          (~5 h)
    extra      production E_FN for quadB and lin, population bias       (~3 h)
    cas        d^2 channel: trains cas2 (flexible) and cas2rigid, nodal
               analysis with quadA, production E_FN for cas2             (~2 h)
    j1         explicit one-body Jastrow: trains quadJ1, nomodJ1, fix1s,
               full; nodal analysis; production E_FN (tau_prod = 100)    (~4 h)
    figures    all figures for paper 1 from whatever is on disk
    summary    one table with every number, also written to study_summary.txt

    day        = validate + vmc + nodes
    night      = dmc + figures + summary
    night2     = extra (productions) + cas + j1 + population bias
                 + figures + summary, most important first               (~8.5 h)

Every stage skips outputs that already exist (use --force to redo them), so an
interrupted run can simply be relaunched. --quick shrinks every run to a few
seconds: use it once to check that the pipeline works end to end, in a
SEPARATE folder, since quick outputs have the same file names.

The four trial functions all come from the current wavefunction module:

    quadA   quadratic modulation, seed 42
    quadB   quadratic modulation, seed 7      (replica: seed-to-seed spread)
    lin     c2 frozen at 0 (linear modulation), seed 42
    nomod   all modulation frozen (m = 1), seed 42
            -> E_FN(quadA) - E_FN(nomod) is the NODAL effect of the modulation

and, for the CAS stage (1s^2 [2s^2 + 2p^2 + 3d^2]):

    cas2       flexible orbitals, seed 42
    cas2rigid  all modulation frozen, seed 42
            -> E_FN(cas2) - E_FN(quadA) is the NODAL effect of the d^2 channel

and, for the one-body stage (OneBodyJastrow: a small network of ln(1 + t)):

    quadJ1     quadA + one-body Jastrow        does J1 take over the one-body
                                               work of m_1s? same node?
    nomodJ1    nomod + one-body Jastrow        does J1 alone recover the
                                               0.31 mHa "Jastrow side"?
    fix1s      m_1s = 1, quadratic m_2s, m_2p, one-body Jastrow
               NOT gauge-equivalent to quadJ1: with m_1s = 1 the nodal ratios
               q_nl = m_nl are bounded quadratics and cannot decay like
               1/m_1s. -> E_FN(fix1s) - E_FN(quadJ1) tests whether that decay
               is nodally relevant
    full       cas2 + one-body Jastrow         everything together
"""

import argparse
import os
import time

import numpy as np

from be_nqs import wavefunction as vmc
from be_nqs import dmc
from be_nqs import nodes as na
from be_nqs import figures as figs
from be_nqs import data_path, results_path

MOD_KEYS = vmc.modulation_keys(1)
MODELS = {                  # tag: (seed, frozen orbital parameters, cas_level)
    "quadA": (42, (), 1),
    "quadB": (7, (), 1),
    "lin":   (42, tuple(k for k in MOD_KEYS if k.endswith("_2")), 1),
    "nomod": (42, MOD_KEYS, 1),
}
CAS_MODELS = {
    "cas2":      (42, (), 2),
    "cas2rigid": (42, vmc.modulation_keys(2), 2),
}
J1_MODELS = {
    "quadJ1":  (42, (), 1),
    "nomodJ1": (42, MOD_KEYS, 1),
    "fix1s":   (42, ("m1s_1", "m1s_2"), 1),
    "full":    (42, (), 2),
}
ONE_BODY = set(J1_MODELS)
ALL_MODELS = {**MODELS, **CAS_MODELS, **J1_MODELS}
PRODUCTION_MAIN = ("quadA", "nomod")
PRODUCTION_EXTRA = ("quadB", "lin")
# DMC production seeds. The first version used 2026 + len(tag), which gave
# quadA, quadB and nomod the SAME seed (2031). quadA and nomod were run with
# it on 2026-09-30 and are kept as they are (see check_seed_correlation);
# every other run gets its own seed.
DMC_SEEDS = {"quadA": 2031, "nomod": 2031, "quadB": 2041, "lin": 2043,
             "cas2": 2051, "quadJ1": 2061, "nomodJ1": 2063, "fix1s": 2065,
             "full": 2067}

FULL = dict(
    vmc=dict(n_chains=4000, n_epochs=3500, n_sweeps=10, n_blocks=200,
             block_sweeps=150),
    nodes=dict(n_blocks=40, sweeps_per_block=10),
    prod=dict(dtau=0.0025, tau_equil=10.0, tau_block=1.0, tau_prod=150.0),
    # one-body stage: shorter production (error ~0.045 instead of ~0.037 mHa)
    # so that the whole of night2 fits in one night
    prod_short=dict(dtau=0.0025, tau_equil=10.0, tau_block=1.0,
                    tau_prod=100.0),
    jastrow=dict(dtau=0.005, tau_equil=10.0, tau_block=1.0, tau_prod=60.0),
    series=dict(dtaus=(0.010, 0.005, 0.0025), tau_equil=10.0, tau_block=1.0,
                tau_prod=40.0),
    popbias=dict(populations=(1000, 2000, 4000), dtau=0.0025, tau_equil=10.0,
                 tau_block=1.0, tau_prod=60.0),
    validate=dict(n_walkers=4000, dtaus=(0.020, 0.010, 0.005, 0.0025),
                  tau_equil=5.0, tau_block=1.0, tau_prod=200.0),
)
QUICK = dict(
    vmc=dict(n_chains=128, n_epochs=40, n_sweeps=2, n_blocks=8,
             block_sweeps=3),
    nodes=dict(n_blocks=6, sweeps_per_block=3),
    prod=dict(dtau=0.01, tau_equil=0.3, tau_block=0.25, tau_prod=2.0),
    prod_short=dict(dtau=0.01, tau_equil=0.3, tau_block=0.25, tau_prod=2.0),
    jastrow=dict(dtau=0.01, tau_equil=0.3, tau_block=0.25, tau_prod=1.0),
    series=dict(dtaus=(0.02, 0.01), tau_equil=0.3, tau_block=0.25,
                tau_prod=1.0),
    popbias=dict(populations=(64, 128), dtau=0.01, tau_equil=0.3,
                 tau_block=0.25, tau_prod=1.0),
    validate=dict(n_walkers=500, dtaus=(0.02, 0.01), tau_equil=1.0,
                  tau_block=0.5, tau_prod=10.0),
)


def wf_path(tag):
    return data_path(f"multidet_{tag}_cas{ALL_MODELS[tag][2]}.npz")


def banner(txt):
    print("\n" + "#" * 78 + f"\n#  {txt}\n" + "#" * 78, flush=True)


def stage_file(name):
    """Where a stage writes: .npz are data, everything else is a result."""
    return data_path(name) if str(name).endswith(".npz") else results_path(name)


def skip(path, force):
    path = stage_file(path)
    if os.path.exists(path) and not force:
        print(f"  [skip] {path} exists (use --force to redo)")
        return True
    return False


# ============================================================================
def stage_validate(cfg, force):
    banner("STAGE validate: DMC propagator on hydrogen")
    if skip("validation_hydrogen.txt", force):
        return
    ok = dmc.validate(**cfg["validate"])
    with open(results_path("validation_hydrogen.txt"), "w") as fh:
        fh.write("PASSED\n" if ok else "FAILED\n")
    if not ok:
        if cfg is QUICK:
            print("  (quick mode: too few walkers and too short to validate "
                  "anything; continuing)")
        else:
            raise SystemExit("validation FAILED: do not trust production "
                             "numbers")


def stage_vmc(cfg, force, models=None):
    models = MODELS if models is None else models
    banner(f"STAGE vmc: training {', '.join(models)}")
    for tag, (seed, freeze, level) in models.items():
        if skip(wf_path(tag), force):
            continue
        t0 = time.time()
        vmc.run_pipeline(cas_level=level, seed=seed, tag=f"{tag}_cas{level}",
                         freeze=freeze, one_body=(tag in ONE_BODY),
                         **cfg["vmc"])
        print(f"  [{tag}] done in {(time.time() - t0)/60:.1f} min", flush=True)


def stage_nodes(cfg, force):
    banner("STAGE nodes: nodal analysis")
    if skip("node_analysis.npz", force):
        return
    tags = list(MODELS)
    na.analyse([wf_path(t) for t in tags], labels=tags, **cfg["nodes"])


def stage_dmc(cfg, force, tags=PRODUCTION_MAIN, with_tests=True,
              prod_key="prod"):
    banner(f"STAGE dmc: production for {', '.join(tags)}")
    for tag in tags:
        if skip(f"dmc_prod_{tag}.npz", force):
            continue
        dmc.run_production(wf_path(tag), tag=tag, seed=DMC_SEEDS[tag],
                           **cfg[prod_key])
    if not with_tests:
        return

    banner("Jastrow independence of E_FN (quadA)")
    if not skip("dmc_jastrow_independence.npz", force):
        full, det = dmc.test_jastrow_independence(wf_path("quadA"),
                                                  **cfg["jastrow"])
        np.savez(data_path("dmc_jastrow_independence.npz"),
                 E_full=full["E"], dE_full=full["dE"],
                 E_det=det["E"], dE_det=det["dE"],
                 blocks_full=full["blocks"], blocks_det=det["blocks"])

    banner("Time-step series (quadA), same imaginary-time protocol")
    if not skip("dmc_ts_quadA_summary.npz", force):
        dmc.run_timestep_series(wf_path("quadA"), tag="quadA", verbose=False,
                                **cfg["series"])


def stage_extra(cfg, force):
    stage_extra_prod(cfg, force)
    stage_popbias(cfg, force)


def stage_extra_prod(cfg, force):
    stage_dmc(cfg, force, tags=PRODUCTION_EXTRA, with_tests=False)


def stage_popbias(cfg, force):
    banner("Population bias (quadA)")
    if not skip("dmc_population_bias.npz", force):
        res = dmc.test_population_bias(wf_path("quadA"), **cfg["popbias"])
        np.savez(data_path("dmc_population_bias.npz"),
                 populations=np.array(cfg["popbias"]["populations"]),
                 E=np.array([r["E"] for r in res]),
                 dE=np.array([r["dE"] for r in res]))


def stage_cas(cfg, force):
    """Does the d^2 channel move the node?

    The VMC scan found no energy gain from 3d^2 (and 4f^2). Since the
    modulation showed that half of a VMC gain can be nodal and half one-body,
    VMC is not the right judge: E_FN(cas2) - E_FN(quadA) is. The rigid model
    gives the gauge-invariant weight w_d with and without radial freedom,
    which tests whether the collapse of c_d seen before was real.
    """
    stage_vmc(cfg, force, models=CAS_MODELS)
    banner("CAS nodal analysis (quadA, cas2rigid, cas2)")
    if not skip("node_analysis_cas.npz", force):
        tags = ["quadA", "cas2rigid", "cas2"]
        na.analyse([wf_path(t) for t in tags], labels=tags,
                   out_path=data_path("node_analysis_cas.npz"), **cfg["nodes"])
    stage_dmc(cfg, force, tags=("cas2",), with_tests=False)


def stage_j1(cfg, force):
    """Explicit one-body Jastrow, with and without orbital modulation.

    The night-1 decomposition split the VMC gain of the modulation into
    0.26 mHa nodal and 0.31 mHa on the positive-factor side, attributed to
    the one-body work that m_1s does for a Jastrow without a one-body term.
    These models test that attribution from the other side.
    """
    stage_vmc(cfg, force, models=J1_MODELS)
    banner("One-body nodal analysis")
    if not skip("node_analysis_j1.npz", force):
        tags = ["quadA", "quadJ1", "fix1s", "nomod", "nomodJ1", "full"]
        tags = [t for t in tags if os.path.exists(wf_path(t))]
        na.analyse([wf_path(t) for t in tags], labels=tags,
                   out_path=data_path("node_analysis_j1.npz"), **cfg["nodes"])
    stage_dmc(cfg, force, tags=("quadJ1", "nomodJ1", "fix1s", "full"),
              with_tests=False, prod_key="prod_short")


def stage_figures(cfg, force):
    banner("STAGE figures")
    figs.main()


# ============================================================================
def fn_difference(pa, pb, la, lb):
    """E_FN(a) - E_FN(b) with its error.

    If the two runs used the same seed and the same number of blocks, the
    block series may be correlated (shared random increments); the correlation
    coefficient of the blocks is then measured and included:
        var(diff) = s_a^2 + s_b^2 - 2 rho s_a s_b .
    """
    diff = (float(pa["E"]) - float(pb["E"])) * 1e3
    sa, sb = float(pa["dE"]) * 1e3, float(pb["dE"]) * 1e3
    rho = 0.0
    same_seed = int(pa["seed"]) == int(pb["seed"])
    if same_seed and len(pa["blocks"]) == len(pb["blocks"]):
        rho = float(np.corrcoef(pa["blocks"], pb["blocks"])[0, 1])
    sd = np.sqrt(max(sa ** 2 + sb ** 2 - 2 * rho * sa * sb, 0.0))
    txt = (f"{la} - {lb:<6} = {diff:+.3f} +/- {sd:.3f} mHa  "
           f"({diff/sd:+.1f} sigma)")
    if same_seed:
        txt += (f"   [same seed: block correlation rho = {rho:+.2f}, "
                f"included; +/- {np.hypot(sa, sb):.3f} if ignored]")
    return txt


def timestep_fits(ts, prod):
    """Fit E(dtau) with the series AND the independent production point.

    The data cannot tell a linear from a quadratic time-step bias, so several
    forms are reported; the spread of their intercepts is a systematic error.
    """
    x = np.append(ts["dtau"], float(prod["dtau"]))
    y = (np.append(ts["E"], float(prod["E"])) - vmc.E_EXACT) * 1e3
    s = np.append(ts["dE"], float(prod["dE"])) * 1e3
    w = 1.0 / s ** 2
    lines = ["Time-step analysis (quadA): series + production point, "
             "E_FN - exact in mHa"]
    for xx, yy, ss, c in zip(ts["dtau"], (ts["E"] - vmc.E_EXACT) * 1e3,
                             ts["dE"] * 1e3, ts["crossing"]):
        lines.append(f"  series      dtau = {xx:<7g} {yy:+.3f} +/- {ss:.3f}"
                     f"   crossing {c*100:.4f}%")
    lines.append(f"  production  dtau = {float(prod['dtau']):<7g} "
                 f"{y[-1]:+.3f} +/- {s[-1]:.3f}")
    m = np.sum(w * y) / np.sum(w)
    lines.append(f"  constant        {m:+.3f} +/- {1/np.sqrt(w.sum()):.3f}"
                 f"   chi2 = {np.sum(w * (y - m) ** 2):.2f}/{len(y) - 1}")
    k = float(ts["crossing_exponent"])
    icpt = []
    for name, p in (("linear", 1.0), ("quadratic", 2.0),
                    (f"dtau^{k:.2f}", k)):
        if not np.isfinite(p):
            continue
        A = np.vstack([np.ones_like(x), x ** p]).T
        C = np.linalg.inv(A.T @ (w[:, None] * A))
        b = C @ A.T @ (w * y)
        chi2 = float(np.sum(w * (y - A @ b) ** 2))
        icpt.append(b[0])
        lines.append(f"  {name:<15} {b[0]:+.3f} +/- {np.sqrt(C[0, 0]):.3f}"
                     f"   chi2 = {chi2:.2f}/{len(y) - 2}   bias at "
                     f"{float(prod['dtau']):g}: {b[1] * float(prod['dtau'])**p:+.3f}")
    if icpt:
        lines.append(f"  -> dtau -> 0 intercepts span {min(icpt):+.3f} .. "
                     f"{max(icpt):+.3f} mHa (model systematic)")
    return "\n".join(lines)


# (a, b, what the comparison tests), printed when both exist
VMC_PAIRS = [
    ("nomodJ1", "nomod", "does J1 alone recover the ~0.31 mHa positive-factor gain?"),
    ("quadJ1", "quadA", "does J1 do better than m_1s as a one-body factor?"),
    ("fix1s", "quadJ1", "cost of fixing m_1s = 1 with bounded quadratics"),
]
FN_PAIRS = [
    ("quadB", "quadA", "control: same f_hat, h_hat, c~_p -> same E_FN expected"),
    ("lin", "quadA", "lin between quadA and nomod if c~_p / w_p order the nodes"),
    ("cas2", "quadA", "nodal effect of the d^2 channel"),
    ("nomodJ1", "nomod", "J1 re-optimises c_p, alpha: does the node move?"),
    ("quadJ1", "quadA", "same nodal family: same E_FN if the optimiser finds the same node"),
    ("fix1s", "quadJ1", "is the 1/m_1s decay of q_nl nodally relevant?"),
    ("full", "cas2", "does J1 change the node found with d^2?"),
    ("full", "quadJ1", "d^2 channel in presence of J1"),
]


def stage_summary(cfg, force):
    """Collect every number on disk into one table (and study_summary.txt)."""
    lines = []
    out = lines.append
    ex = vmc.E_EXACT

    out("=" * 78)
    out(f"  Be, cas_level = 1   exact = {ex:.6f}   HF limit = {vmc.E_HF:.6f}")
    out("=" * 78)

    out("\nVMC")
    out(f"{'model':<10}{'E_VMC':>13}{'err':>10}{'Var':>9}{'corr %':>9}"
        f"{'c_p':>9}{'a_2p':>8}{'c1(1s)':>8}  J1  frozen")
    for tag, (_, frz, level) in ALL_MODELS.items():
        if not os.path.exists(wf_path(tag)):
            if tag in MODELS:
                out(f"{tag:<10}  (missing)")
            continue
        d = np.load(wf_path(tag))
        corr = (float(d["E"]) - vmc.E_HF) / vmc.E_CORR * 100
        if len(frz) == len(vmc.modulation_keys(level)):
            fz = "all m"
        elif frz and all(k.endswith("_2") for k in frz):
            fz = "c2"
        else:
            fz = ",".join(frz) if frz else "-"
        j1 = "yes" if ("one_body" in d.files and bool(d["one_body"])) else " - "
        extra_txt = f"  c_d = {float(d['cd']):+.4f}" if "cd" in d.files else ""
        out(f"{tag:<10}{float(d['E']):>13.6f}{float(d['dE']):>10.6f}"
            f"{float(d['var']):>9.4f}{corr:>9.2f}{float(d['cp']):>+9.4f}"
            f"{float(d['a2p']):>8.4f}{float(d['m1s_1']):>+8.3f}  {j1}  "
            f"{fz}{extra_txt}")

    out("\n  VMC comparisons (one-body tests)")
    for a_, b_, what in VMC_PAIRS:
        if os.path.exists(wf_path(a_)) and os.path.exists(wf_path(b_)):
            da, db = np.load(wf_path(a_)), np.load(wf_path(b_))
            diff = (float(da["E"]) - float(db["E"])) * 1e3
            sd = np.hypot(float(da["dE"]), float(db["dE"])) * 1e3
            out(f"    {a_} - {b_:<8} = {diff:+.3f} +/- {sd:.3f} mHa "
                f"({diff/sd:+.1f} sigma)   {what}")

    if os.path.exists(data_path("node_analysis.npz")):
        a = np.load(data_path("node_analysis.npz"))
        labs = list(a["labels"])
        out("\nNODES")
        ct = figs.cp_tilde(dict(a))
        out(f"{'model':<8}{'c_p':>9}{'c~_p':>10}{'w_p':>12}{'err':>10}"
            f"{'P(Psi>0)':>11}")
        for i, lab in enumerate(labs):
            out(f"{lab:<8}{a['cp'][i]:>+9.4f}{ct[i]:>+10.5f}{a['wp'][i]:>12.5f}"
                f"{a['dwp'][i]:>10.5f}{a['frac_pos'][i]:>11.4f}")
        out(f"  max factorisation residual: {a['fact_residual'].max():.1e}")
        out("  nodal mismatch phi[a,b] (row = sampled from):")
        out("          " + "".join(f"{l:>11}" for l in labs))
        for i, lab in enumerate(labs):
            out(f"  {lab:<8}" + "".join(
                f"{'-':>11}" if i == j else f"{a['phi'][i, j]:>11.2e}"
                for j in range(len(labs))))

    if os.path.exists(data_path("node_analysis_j1.npz")):
        a = np.load(data_path("node_analysis_j1.npz"))
        ct = figs.cp_tilde(dict(a))
        out("\nONE-BODY NODAL ANALYSIS")
        out(f"{'model':<8}{'c_p':>9}{'c~_p':>10}{'w_p':>12}{'err':>10}")
        for i, lab in enumerate(a["labels"]):
            out(f"{lab:<8}{a['cp'][i]:>+9.4f}{ct[i]:>+10.5f}"
                f"{a['wp'][i]:>12.5f}{a['dwp'][i]:>10.5f}")

    if os.path.exists(data_path("node_analysis_cas.npz")):
        a = np.load(data_path("node_analysis_cas.npz"))
        out("\nCAS CHANNEL WEIGHTS (gauge invariant)")
        for i, lab in enumerate(a["labels"]):
            txt = "  ".join(f"w({c}) = {x:.5f}({dx*1e5:.0f})" for c, x, dx in
                            zip(a[f"channels_{i}"], a[f"w_{i}"], a[f"dw_{i}"]))
            out(f"  {lab:<10} {txt}")

    prods = {t: data_path(f"dmc_prod_{t}.npz") for t in ALL_MODELS}
    have = {t: np.load(p) for t, p in prods.items() if os.path.exists(p)}
    if have:
        out("\nFN-DMC production")
        out(f"{'model':<8}{'E_FN':>13}{'err':>10}{'-exact(mHa)':>13}"
            f"{'95% bound':>11}{'gap(mHa)':>10}{'cross %':>9}{'plateau':>9}")
        for t, p in have.items():
            d, s = (float(p["E"]) - ex) * 1e3, float(p["dE"]) * 1e3
            bound = max(d, 0.0) + 1.645 * s
            gap = (float(p["E_vmc"]) - float(p["E"])) * 1e3
            out(f"{t:<8}{float(p['E']):>13.6f}{float(p['dE']):>10.6f}"
                f"{d:>+13.3f}{bound:>11.3f}{gap:>10.3f}"
                f"{float(p['crossing'])*100:>9.4f}{str(bool(p['plateau'])):>9}")
        if "quadA" in have:
            out("\n  E_FN differences with respect to quadA (same dtau, so the "
                "time-step bias largely cancels)")
            for t, p in have.items():
                if t == "quadA":
                    continue
                out("    " + fn_difference(have["quadA"], p, "quadA", t))
        pairs = [(a_, b_, w) for a_, b_, w in FN_PAIRS
                 if a_ in have and b_ in have]
        if pairs:
            out("\n  Targeted E_FN comparisons")
            for a_, b_, what in pairs:
                out("    " + fn_difference(have[a_], have[b_], a_, b_)
                    + f"\n        -> {what}")

    ts_path = data_path("dmc_ts_quadA_summary.npz")
    if os.path.exists(ts_path) and "quadA" in have:
        out("\n" + timestep_fits(np.load(ts_path), have["quadA"]))

    if os.path.exists(data_path("dmc_jastrow_independence.npz")):
        j = np.load(data_path("dmc_jastrow_independence.npz"))
        diff = (float(j["E_full"]) - float(j["E_det"])) * 1e3
        sd = np.hypot(float(j["dE_full"]), float(j["dE_det"])) * 1e3
        out(f"\nJastrow independence: full {float(j['E_full']):.6f} +/- "
            f"{float(j['dE_full']):.6f}   det-only {float(j['E_det']):.6f} +/- "
            f"{float(j['dE_det']):.6f}   diff {diff:+.3f} +/- {sd:.3f} mHa "
            f"({diff/sd:+.1f} sigma)")

    if os.path.exists(data_path("dmc_population_bias.npz")):
        pb = np.load(data_path("dmc_population_bias.npz"))
        out("\nPopulation bias (quadA)")
        for n, e, s in zip(pb["populations"], pb["E"], pb["dE"]):
            out(f"  N = {int(n):5d}  E = {e:.6f} +/- {s:.6f}")

    txt = "\n".join(lines)
    print(txt)
    with open(results_path("study_summary.txt"), "w") as fh:
        fh.write(txt + "\n")
    print("\n[saved] study_summary.txt")


STAGES = {
    "validate": [stage_validate],
    "vmc": [stage_vmc],
    "nodes": [stage_nodes],
    "dmc": [stage_dmc],
    "extra": [stage_extra],
    "figures": [stage_figures],
    "summary": [stage_summary],
    "cas": [stage_cas],
    "day": [stage_validate, stage_vmc, stage_nodes],
    "night": [stage_dmc, stage_figures, stage_summary],
    "j1": [stage_j1],
    "popbias": [stage_popbias],
    "night2": [stage_extra_prod, stage_cas, stage_j1, stage_popbias,
               stage_figures, stage_summary],
}

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("stage", choices=list(STAGES))
    ap.add_argument("--quick", action="store_true",
                    help="tiny runs to check the pipeline (separate folder!)")
    ap.add_argument("--force", action="store_true",
                    help="redo outputs that already exist")
    args = ap.parse_args()
    cfg = QUICK if args.quick else FULL
    t0 = time.time()
    for fn in STAGES[args.stage]:
        fn(cfg, args.force)
    print(f"\n[{args.stage}] finished in {(time.time() - t0)/60:.1f} min")