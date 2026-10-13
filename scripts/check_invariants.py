"""Robustness of the E_FN vs c~_p relation to the choice of gauge fixing.

Read-only analysis of the files written by run_study.py; it does not touch
the pipeline. Run it in the folder of the study:

    python scripts/check_invariants.py

The node of the CAS(2,4) trial function is fixed by (f_hat, h_hat, c~_p),
where the scalar c~_p depends on how the gauge (f -> a f + b, h -> mu h) is
fixed. Two families of exact invariants are compared:

  point gauge    c~_p(r*) = c_p [h(r*) / (f(r*) - f(0))]^2
  matched gauge  c~_p^LS  = c_p a^2 / mu^2, where a and mu bring f - f(0)
                 and h of the model closest, in a density-weighted
                 relative least-squares sense, to those of the reference
                 model quadA

Both are exactly invariant under the gauge. The matched gauge has no free
point: its weight is the one-body radial density saved in node_analysis.npz
(model quadA), the same for every model and fixed before looking at E_FN.
The relative rms residual of the match measures how far the radial shapes
f_hat, h_hat of a model are from those of the reference: c~_p is a complete
description of the node only when that residual is small. For each choice the script fits E_FN - E_exact = a + b |c~_p| on three
model sets and prints the slope significance, chi^2 and Spearman rank
correlation, and the ranking of the models.

The cas2 and full models also carry a d^2 channel that c~_p ignores; the
"two-channel" sets exclude them.
"""
import os

import numpy as np
from scipy.stats import spearmanr

from be_nqs import data_path

E_EXACT = -14.667356
FILES = ("node_analysis.npz", "node_analysis_cas.npz", "node_analysis_j1.npz")
SETS = {
    "all 9": ["quadA", "quadB", "lin", "nomod", "cas2", "quadJ1",
              "nomodJ1", "fix1s", "full"],
    "two-channel (7)": ["quadA", "quadB", "lin", "nomod", "quadJ1",
                        "nomodJ1", "fix1s"],
    "modulated two-channel (5)": ["quadA", "quadB", "lin", "quadJ1", "fix1s"],
}
R_STARS = (0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0)


CURVE_KEYS = {"f": ("curve_f", "f"), "h": ("curve_h_2p", "h_2p", "curve_h")}


def _first(a, names):
    for n in names:
        if n in a.files:
            return a[n]
    return None


def load_models():
    models, rho, r_ref = {}, None, None
    for path in (data_path(f) for f in FILES):
        if not os.path.exists(path):
            print(f"[missing] {path}")
            continue
        a = np.load(path)
        f_all = _first(a, CURVE_KEYS["f"])
        h_all = _first(a, CURVE_KEYS["h"])
        if f_all is None or h_all is None or "r" not in a.files:
            print(f"[skip] {path}: no radial curves. Keys in the file:\n  "
                  + ", ".join(sorted(a.files)))
            continue
        r = a["r"]
        if rho is None and "density" in a.files:
            rho, r_ref = a["density"], r
        cpt = a["cp_tilde"] if "cp_tilde" in a.files else None
        r_star = float(a["r_star"]) if "r_star" in a.files else np.nan
        for i, lab in enumerate(a["labels"]):
            lab = str(lab)
            if lab in models:
                continue
            models[lab] = dict(r=r, f=f_all[i], h=h_all[i],
                               cp=float(a["cp"][i]),
                               cpt_saved=float(cpt[i]) if cpt is not None else np.nan,
                               r_star_saved=r_star, src=path)
    for lab, m in models.items():
        p = data_path(f"dmc_prod_{lab}.npz")
        if os.path.exists(p):
            d = np.load(p)
            m["y"] = (float(d["E"]) - E_EXACT) * 1e3
            m["s"] = float(d["dE"]) * 1e3
    return models, rho, r_ref


def cpt_point(m, r_star):
    f0 = m["f"][0]
    fs = np.interp(r_star, m["r"], m["f"])
    hs = np.interp(r_star, m["r"], m["h"])
    return m["cp"] * (hs / (fs - f0)) ** 2


def cpt_matched(m, ref, rho, r_ref):
    """c_p a^2 / mu^2 with a, mu from a density-weighted relative LS match
    of (f - f(0), h) to the reference model. Returns (c~_p, res_f, res_h)."""
    r = m["r"]
    rho_m = np.interp(r, r_ref, rho)
    g_ref = np.interp(r, ref["r"], ref["f"]) - ref["f"][0]
    h_ref = np.interp(r, ref["r"], ref["h"])
    g_m = m["f"] - m["f"][0]
    ok = (rho_m > 0) & (r > 0)
    wf = rho_m[ok] / g_ref[ok] ** 2
    wh = rho_m[ok] / h_ref[ok] ** 2
    a = np.sum(wf * g_m[ok] * g_ref[ok]) / np.sum(wf * g_m[ok] ** 2)
    mu = np.sum(wh * m["h"][ok] * h_ref[ok]) / np.sum(wh * m["h"][ok] ** 2)
    res_f = np.sqrt(np.sum(wf * (a * g_m[ok] - g_ref[ok]) ** 2) / np.sum(rho_m[ok]))
    res_h = np.sqrt(np.sum(wh * (mu * m["h"][ok] - h_ref[ok]) ** 2) / np.sum(rho_m[ok]))
    return m["cp"] * a ** 2 / mu ** 2, res_f, res_h


def fit(x, y, s):
    w = 1.0 / s ** 2
    A = np.vstack([np.ones_like(x), x]).T
    cov = np.linalg.inv(A.T @ (w[:, None] * A))
    beta = cov @ (A.T @ (w * y))
    chi2 = float(np.sum(w * (y - A @ beta) ** 2))
    c = np.sum(w * y) / np.sum(w)
    chi2_c = float(np.sum(w * (y - c) ** 2))
    return beta[1], np.sqrt(cov[1, 1]), chi2, chi2_c


def report(name, xs, models, tags):
    tags = [t for t in tags if t in models and "y" in models[t]]
    if len(tags) < 3:
        return
    x = np.abs(np.array([xs[t] for t in tags]))
    y = np.array([models[t]["y"] for t in tags])
    s = np.array([models[t]["s"] for t in tags])
    b, db, chi2, chi2_c = fit(x, y, s)
    rs, p = spearmanr(x, y)
    n = len(tags)
    print(f"    {name:<28} slope {b:+8.1f} +/- {db:5.1f} ({b/db:+5.1f} sigma)"
          f"   chi2 {chi2:5.1f}/{n-2}   const chi2 {chi2_c:5.1f}/{n-1}"
          f"   Spearman {rs:+.2f} (p = {p:.3f})")


def main():
    models, rho, r_ref = load_models()
    print("\nsanity: c~_p recomputed at the saved r* vs the saved value")
    for lab, m in models.items():
        rs = m["r_star_saved"] if np.isfinite(m["r_star_saved"]) else 2.0
        print(f"  {lab:<10} {cpt_point(m, rs):+.5f}   saved "
              + (f"{m['cpt_saved']:+.5f}" if np.isfinite(m["cpt_saved"])
                 else "  (none)") + f"   [{m['src']}]")
    missing = [t for t in SETS["all 9"] if t not in models]
    if missing:
        print("models without radial curves: " + ", ".join(missing))
    if rho is not None:
        cdf = np.cumsum(rho) / np.sum(rho)
        q = [r_ref[np.searchsorted(cdf, v)] for v in (0.05, 0.25, 0.5, 0.75, 0.95)]
        print("\none-body radial density quantiles 5/25/50/75/95%: "
              + ", ".join(f"{v:.2f}" for v in q) + " bohr")

    choices = [(f"r* = {rs}", {t: cpt_point(m, rs) for t, m in models.items()})
               for rs in R_STARS]
    if rho is not None and "quadA" in models:
        matched = {t: cpt_matched(m, models["quadA"], rho, r_ref)
                   for t, m in models.items()}
        print("\nmatched gauge (reference quadA): relative rms residual of the"
              " radial match")
        for t, (c, rf, rh) in matched.items():
            print(f"  {t:<10} c~_p^LS = {c:+.5f}   res(f) = {rf:.3f}"
                  f"   res(h) = {rh:.3f}")
        choices.append(("matched gauge", {t: v[0] for t, v in matched.items()}))
    choices.append(("raw c_p (reference)", {t: m["cp"] for t, m in models.items()}))

    for label, xs in choices:
        order = sorted((t for t in xs if t in SETS["all 9"]),
                       key=lambda t: -abs(xs[t]))
        print(f"\n{label}   ranking by |c~_p|: " + " > ".join(order))
        for name, tags in SETS.items():
            report(name, xs, models, tags)


if __name__ == "__main__":
    main()