"""Figures for paper 1, drawn at their final REVTeX size (column 3.375",
full width 7.0") and saved as vector PDF + PNG preview.

    python -m be_nqs.figures

reads whatever results exist and skips the figures whose input is missing:

    node_analysis.npz            -> fig_gauge_radial, fig_mixing_gauge
                                    (nodes)
    dmc_prod_<tag>.npz           -> fig_dmc_equilibration, fig_error_budget
                                    (one per model) and fig_error_budget_all
                                    (dmc.run_production)
    both                         -> fig_node_order (E_FN against the mixing
                                    measures, once >= 2 models have DMC)

fig_node_mismatch is no longer drawn by main(): with zero disagreement for
every pair it is an empty matrix; the result belongs in the text.
    dmc_ts_<tag>_summary.npz     -> fig_dmc_timestep      (dmc.run_timestep_series)

A figure whose input is missing is skipped with a message, not drawn from
stale numbers: the September series (OLD protocol, fixed step counts and a
seed shared across dtau) was superseded by dmc_ts_quadA_summary.npz and is no
longer carried here as a fallback.

Colour: parametrisations use a fixed categorical order (blue, orange, aqua,
violet; validated for colour-vision deficiency on all pairs). Each series also
has its own line style and marker, so identity never rests on colour alone.
"""

import glob
import os

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from . import wavefunction as vmc
from .stats import blocking_analysis
from . import data_path

COL_W, FULL_W = vmc.COL_W, vmc.FULL_W
E_EXACT = vmc.E_EXACT

SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"]
SERIES_LS = ["-", "--", "-.", ":"]
SERIES_MK = ["o", "s", "^", "D"]
INK = "#0b0b0b"
INK_2 = "#52514e"
BAND = "#e4e3df"

def style(i):
    k = i % len(SERIES_COLORS)
    return dict(color=SERIES_COLORS[k], ls=SERIES_LS[k])


def _markevery(n, i):
    return (int(n * (0.08 + 0.05 * i)) % n, max(1, n // 7))


def sampled_band(ax, r_lo, r_hi):
    ax.axvspan(r_lo, r_hi, color=BAND, lw=0, zorder=0)


# ============================================================================
# Nodal ingredients across parametrisations.
# ============================================================================
def fig_gauge_radial(a, fname="fig_gauge_radial.pdf"):
    """Top: the raw modulations m_nl(r). Bottom: what the node depends on.

    (a-c) m_1s, m_2s, m_2p: not individually meaningful.
    (d-e) q_nl = m_nl / m_1s: the only nodal combination of the modulations.
    (f)   gauge-fixed f_hat = (f - f(0))/(f(r*) - f(0)) and h_hat = h/h(r*),
          as ratios to the first model. Together with the exact invariant
          c~_p (see cp_tilde) they fix the node completely.
    The shaded band marks the central 90% of sampled electron radii.
    """
    r = a["r"]
    labels = list(a["labels"])
    n = len(labels)
    # central 90% of the one-body radial distribution
    cdf = np.cumsum(a["density"]) * (r[1] - r[0])
    cdf /= cdf[-1]
    r_lo, r_hi = r[np.searchsorted(cdf, 0.05)], r[np.searchsorted(cdf, 0.95)]

    fig, axs = plt.subplots(2, 3, figsize=(FULL_W, 3.9), sharex=True,
                            gridspec_kw={"hspace": 0.18, "wspace": 0.42})
    panels = [(axs[0, 0], "curve_m_1s", r"$m_{1s}(r)$"),
              (axs[0, 1], "curve_m_2s", r"$m_{2s}(r)$"),
              (axs[0, 2], "curve_m_2p", r"$m_{2p}(r)$"),
              (axs[1, 0], "curve_q_2s", r"$q_{2s}=m_{2s}/m_{1s}$"),
              (axs[1, 1], "curve_q_2p", r"$q_{2p}=m_{2p}/m_{1s}$")]
    for ax, key, ylab in panels:
        sampled_band(ax, r_lo, r_hi)
        for i in range(n):
            ax.plot(r, a[key][i], lw=0.9, marker=SERIES_MK[i % 4], ms=2.6,
                    markevery=_markevery(len(r), i), **style(i))
        ax.axhline(1.0, color=INK_2, lw=0.4, ls=":")
        ax.set_ylabel(ylab)
        ax.grid(True)

    # (f) the gauge-fixed nodal functions, as ratios to the reference model
    # (the first label). f_hat(0) = 0 by construction, so r < 0.3 is left out.
    ax = axs[1, 2]
    sampled_band(ax, r_lo, r_hi)
    f_hat, h_hat = a["curve_f_hat"], a["curve_h_hat"]
    ok = r >= 0.3
    in_band = ok & (r >= r_lo) & (r <= r_hi)
    with np.errstate(divide="ignore", invalid="ignore"):
        rf = f_hat / f_hat[0]
        rh = h_hat / h_hat[0]
    for i in range(n):
        c = SERIES_COLORS[i % 4]
        mk = dict(marker=SERIES_MK[i % 4], ms=2.6)
        ax.plot(r[ok], rf[i][ok], color=c, ls="-", lw=0.9,
                markevery=_markevery(int(ok.sum()), i), **mk)
        ax.plot(r[ok], rh[i][ok], color=c, ls="--", lw=0.9,
                markevery=_markevery(int(ok.sum()), i), mfc="white", **mk)
    vals = np.concatenate([rf[:, in_band].ravel(), rh[:, in_band].ravel()])
    vals = vals[np.isfinite(vals)]
    ax.set_yscale("log")
    lo, hi = vals.min(), vals.max()
    ax.set_ylim(lo / 1.12, hi * 1.12)
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
    ax.yaxis.set_major_locator(LogLocator(subs=(1.0, 2.0, 5.0)))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.axhline(1.0, color=INK_2, lw=0.4, ls=":")
    ax.set_ylabel(rf"ratio to {labels[0]}")
    ax.legend(handles=[Line2D([], [], color=INK, ls="-", label=r"$\hat f$"),
                       Line2D([], [], color=INK, ls="--", mfc="white",
                              label=r"$\hat h$")],
              loc="upper left")
    ax.grid(True)

    for ax in axs[1]:
        ax.set_xlabel(r"$r$ ($a_0$)")
    for k, ax in enumerate(axs.ravel()):
        ax.text(0.0, 1.02, f"({'abcdef'[k]})", transform=ax.transAxes,
                ha="left", va="bottom", fontsize=7)
    handles = [Line2D([], [], marker=SERIES_MK[i % 4], ms=3, label=labels[i],
                      **style(i)) for i in range(n)]
    handles.append(plt.Rectangle((0, 0), 1, 1, color=BAND,
                                 label="90% of electron radii"))
    fig.legend(handles=handles, loc="lower center", ncol=n + 1,
               bbox_to_anchor=(0.5, 0.93), frameon=False)
    vmc._save(fig, fname)
    plt.close(fig)


def cp_tilde(a):
    """Exact gauge-invariant mixing coefficient, one per model.

        c~_p = c_p [ h(r*) / (f(r*) - f(0)) ]^2 .

    Under f -> a f + b, h -> mu h (the transformations that leave Psi_det
    unchanged up to a constant) c_p -> c_p a^2/mu^2 and c~_p is invariant.
    With f_hat, h_hat it fixes the node: N is proportional to
        [f_hat_1 - f_hat_0][f_hat_3 - f_hat_2] + c~_p (w_hat_1 - w_hat_0).(w_hat_3 - w_hat_2).
    Unlike w_p it involves no expectation value, so it has no statistical error.
    """
    if "cp_tilde" in a:
        return np.asarray(a["cp_tilde"], float)
    r = a["r"]
    k = int(np.argmin(np.abs(r - float(a["r_star"]))))
    f, h = a["curve_f"], a["curve_h_2p"]
    return a["cp"] * (h[:, k] / (f[:, k] - f[:, 0])) ** 2


def _rows(ax, x, y, xerr=None):
    for i in range(len(x)):
        kw = dict(marker=SERIES_MK[i % 4], ms=4.5, ls="none",
                  color=SERIES_COLORS[i % 4])
        if xerr is None:
            ax.plot(x[i], y[i], **kw)
        else:
            ax.errorbar(x[i], y[i], xerr=xerr[i], capsize=1.5,
                        elinewidth=0.7, **kw)


def fig_mixing_gauge(a, fname="fig_mixing_gauge.pdf"):
    """|c_p| (raw), |c~_p| (exact invariant) and w_p (norm weight), per model.

    Magnitudes are plotted so that "more p^2 character" is to the right in
    every panel: c_p and the two invariants order the models differently.
    """
    labels = list(a["labels"])
    n = len(labels)
    y = np.arange(n)[::-1]
    cp, ct, wp, dwp = np.abs(a["cp"]), np.abs(cp_tilde(a)), a["wp"], a["dwp"]

    fig, axs = plt.subplots(1, 3, figsize=(FULL_W * 0.8, 0.75 + 0.3 * n),
                            sharey=True, gridspec_kw={"wspace": 0.1})
    _rows(axs[0], cp, y)
    _rows(axs[1], ct, y)
    _rows(axs[2], wp, y, xerr=dwp)
    axs[0].set_yticks(y)
    axs[0].set_yticklabels(labels)
    axs[0].set_xlabel("$|c_p|$\nraw coefficient")
    axs[1].set_xlabel(f"$|\\tilde c_p|$\ninvariant, $r^*={float(a['r_star']):g}$")
    axs[2].set_xlabel("$w_p$\n$p^2$ weight")
    for k, ax in enumerate(axs):
        ax.grid(True, axis="x")
        ax.set_ylim(-0.6, n - 0.4)
        ax.text(0.0, 1.02, f"({'abc'[k]})", transform=ax.transAxes,
                ha="left", va="bottom", fontsize=7)
        ax.margins(x=0.15)
    vmc._save(fig, fname)
    plt.close(fig)


NODE_FILES = ("node_analysis.npz", "node_analysis_cas.npz",
              "node_analysis_j1.npz")


def collect_mixing(files=NODE_FILES):
    """Merge the mixing measures of every model found in the nodal analyses
    (first occurrence of a label wins)."""
    rows = {}
    for fn in files:
        fn = data_path(fn)
        if not os.path.exists(fn):
            continue
        a = dict(np.load(fn))
        ct = cp_tilde(a)
        for i, lab in enumerate(a["labels"]):
            lab = str(lab)
            if lab not in rows:
                rows[lab] = dict(cp=float(a["cp"][i]), ct=float(ct[i]),
                                 wp=float(a["wp"][i]), dwp=float(a["dwp"][i]))
    return rows


def _wfit(x, y, s):
    w = 1.0 / s ** 2
    A = np.vstack([np.ones_like(x), x]).T
    C = np.linalg.inv(A.T @ (w[:, None] * A))
    b = C @ A.T @ (w * y)
    return b, C, float(np.sum(w * (y - A @ b) ** 2))


def fig_node_order(rows, prods, fname="fig_node_order.pdf"):
    """Which mixing measure orders the nodes? E_FN - exact against |c_p|,
    |c~_p| and w_p for every model with both a nodal analysis and a DMC run.
    Each point is labelled with its model; a weighted straight line and its
    chi^2 per degree of freedom are drawn in every panel, so that the panel in
    which the line fails (c_p) reads as directly as those in which it holds."""
    tags = [t for t in rows if t in prods]
    if len(tags) < 3:
        return
    e = np.array([(float(prods[t]["E"]) - E_EXACT) * 1e3 for t in tags])
    de = np.array([float(prods[t]["dE"]) * 1e3 for t in tags])
    xs = [(np.array([abs(rows[t]["cp"]) for t in tags]), None,
           r"$|c_p|$  (raw)"),
          (np.array([abs(rows[t]["ct"]) for t in tags]), None,
           r"$|\tilde c_p|$  (exact invariant)"),
          (np.array([rows[t]["wp"] for t in tags]),
           np.array([rows[t]["dwp"] for t in tags]), r"$w_p$  ($p^2$ weight)")]

    fig, axs = plt.subplots(1, 3, figsize=(FULL_W, 2.5), sharey=True,
                            gridspec_kw={"wspace": 0.08})
    for k, (ax, (x, dx, lab)) in enumerate(zip(axs, xs)):
        b, C, chi2 = _wfit(x, e, de)
        xx = np.linspace(x.min(), x.max(), 50)
        ax.plot(xx, b[0] + b[1] * xx, color=INK_2, lw=0.7, ls="--", zorder=1)
        ax.errorbar(x, e, yerr=de, xerr=dx, marker="o", ms=3.5, ls="none",
                    color=SERIES_COLORS[0], capsize=1.5, elinewidth=0.7,
                    zorder=2)
        for xi, yi, t in zip(x, e, tags):
            ax.annotate(t, (xi, yi), xytext=(3, 3), textcoords="offset points",
                        fontsize=5.5, color=INK_2)
        ax.axhline(0.0, color=INK, lw=0.5, ls=":")
        ax.text(0.03, 0.04, rf"linear fit  $\chi^2/\nu = {chi2:.1f}/{len(x) - 2}$",
                transform=ax.transAxes, ha="left", va="bottom", fontsize=6.5,
                color=INK_2)
        ax.set_xlabel(lab)
        ax.grid(True)
        ax.margins(x=0.12)
        ax.text(0.0, 1.02, f"({'abc'[k]})", transform=ax.transAxes,
                ha="left", va="bottom", fontsize=7)
    axs[0].set_ylabel(r"$E_{\rm FN}-E_{\rm exact}$ (mHa)")
    vmc._save(fig, fname)
    plt.close(fig)


def fig_node_mismatch(a, fname="fig_node_mismatch.pdf"):
    """phi[a, b]: |Psi_a|^2 weight where the two nodes disagree (log scale)."""
    labels = list(a["labels"])
    phi = np.array(a["phi"], float)
    n = len(labels)
    vals = phi[~np.eye(n, dtype=bool)]
    floor = max(vals[vals > 0].min() / 3 if np.any(vals > 0) else 1e-5, 1e-6)
    show = np.where(np.eye(n, dtype=bool), np.nan, np.maximum(phi, floor))

    fig, ax = plt.subplots(figsize=(COL_W, 0.9 + 0.42 * n))
    im = ax.imshow(np.log10(show), cmap="Blues", vmin=np.log10(floor),
                   vmax=max(np.log10(show[~np.isnan(show)]).max(),
                            np.log10(floor) + 1))
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            txt = "0" if phi[i, j] == 0 else f"{phi[i, j]:.1e}"
            dark = np.log10(show[i, j]) > 0.6 * im.get_clim()[1] + \
                0.4 * im.get_clim()[0]
            ax.text(j, i, txt, ha="center", va="center", fontsize=6,
                    color="white" if dark else INK)
    ax.set_xticks(range(n))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels)
    ax.set_ylabel(r"sampled from $|\Psi_a|^2$")
    ax.set_xlabel(r"compared with $\mathrm{sign}\,\Psi_b$")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label(r"$\log_{10}\phi_{ab}$")
    vmc._save(fig, fname)
    plt.close(fig)


# ============================================================================
# DMC.
# ============================================================================
def fig_dmc_timestep(ts, fname="fig_dmc_timestep.pdf"):
    """(a) E_FN - E_exact vs dtau; (b) node-crossing rate vs dtau (log-log)."""
    x, y, s = ts["dtau"], ts["E"], ts["dE"]
    cr = ts["crossing"]
    d = (y - E_EXACT) * 1e3
    sd = s * 1e3

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(COL_W, 3.5),
                                 gridspec_kw={"hspace": 0.45})
    w = 1.0 / sd ** 2
    m = np.sum(w * d) / np.sum(w)
    dm = 1.0 / np.sqrt(np.sum(w))
    xx = np.array([0.0, x.max() * 1.08])
    a1.fill_between(xx, m - dm, m + dm, color=BAND, lw=0,
                    label=f"constant fit {m:+.3f}({dm*1e3:.0f}) mHa")
    a1.errorbar(x, d, yerr=sd, marker="o", ms=3.5, ls="none",
                color=SERIES_COLORS[0], capsize=1.5, elinewidth=0.7,
                label=r"$E_{\rm FN}(\Delta\tau)$")
    a1.axhline(0.0, color=INK, lw=0.6, ls=":")
    a1.text(0.002 * xx[1], 0.0, "exact", va="bottom", ha="left", fontsize=6,
            color=INK_2)
    a1.set_xlim(xx)
    top = max(np.max(d + sd), m + dm)
    bot = min(np.min(d - sd), 0.0)
    a1.set_ylim(bot - 0.15 * (top - bot), top + 0.9 * (top - bot))
    a1.set_xlabel(r"$\Delta\tau$ (a.u.)")
    a1.set_ylabel(r"$E_{\rm FN}-E_{\rm exact}$ (mHa)")
    a1.legend(loc="upper right", ncol=2, fontsize=6)
    a1.grid(True)
    if "label" in ts:
        a1.set_title(ts["label"], fontsize=7, color=INK_2)

    ok = cr > 0
    a2.loglog(x[ok], cr[ok] * 100, marker="o", ms=3.5, ls="none",
              color=SERIES_COLORS[0])
    if ok.sum() >= 2:
        k, c0 = np.polyfit(np.log(x[ok]), np.log(cr[ok]), 1)
        xf = np.geomspace(x[ok].min() * 0.8, x[ok].max() * 1.25, 50)
        a2.loglog(xf, np.exp(c0) * xf ** k * 100, color=INK_2, lw=0.7,
                  ls="--", label=rf"$\propto\Delta\tau^{{{k:.2f}}}$")
        a2.legend(loc="upper left")
    from matplotlib.ticker import FixedLocator, NullLocator, FuncFormatter
    a2.xaxis.set_major_locator(FixedLocator(x))
    a2.xaxis.set_minor_locator(NullLocator())
    a2.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    from matplotlib.ticker import LogLocator
    a2.yaxis.set_major_locator(LogLocator(subs=(1.0, 2.0, 5.0)))
    a2.yaxis.set_minor_locator(NullLocator())
    a2.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    a2.set_xlabel(r"$\Delta\tau$ (a.u.)")
    a2.set_ylabel("node crossings (%)")
    a2.grid(True, which="both")
    vmc._save(fig, fname)
    plt.close(fig)


def fig_dmc_equilibration(p, fname="fig_dmc_equilibration.pdf"):
    """(a) projection curve E(tau) with an exponential fit, giving the
    effective relaxation gap; (b) blocking ladder of the production blocks."""
    tau, E = p["equil_tau"], p["equil_E"]
    E_inf = float(p["E"])
    y = (E - E_inf) * 1e3

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(COL_W, 3.6),
                                 gridspec_kw={"hspace": 0.45})
    a1.plot(tau, y, marker="o", ms=2, lw=0.6, color=SERIES_COLORS[0],
            label=r"$E(\tau)-E_{\rm FN}$")
    gap_txt = ""
    try:
        from scipy.optimize import curve_fit
        f = lambda t, A, D: A * np.exp(-D * t)
        popt, pcov = curve_fit(f, tau, y, p0=(max(y[0], 0.1), 1.0),
                               maxfev=20000)
        dD = float(np.sqrt(pcov[1, 1]))
        gap_txt = f"Delta_eff = {popt[1]:.3f} +/- {dD:.3f} Eh"
        # Draw the fit only if the gap is actually resolved (2 sigma);
        # otherwise the curve is too noisy to constrain it.
        if np.isfinite(dD) and popt[1] > 2.0 * dD:
            tt = np.linspace(0, tau.max(), 200)
            a1.plot(tt, f(tt, *popt), color=INK_2, ls="--", lw=0.8,
                    label=rf"fit, $\Delta_{{\rm eff}}={popt[1]:.2f}"
                          rf"\pm{dD:.2f}\,E_h$")
        else:
            gap_txt += "  (not resolved: fit not drawn)"
    except Exception as exc:                        # fit is a diagnostic only
        gap_txt = f"exponential fit failed: {exc}"
    a1.axhline(0.0, color=INK, lw=0.5, ls=":")
    a1.set_xlabel(r"$\tau$ (a.u.)")
    a1.set_ylabel(r"$E(\tau)-E_{\rm FN}$ (mHa)")
    a1.legend(loc="upper right")
    a1.grid(True)
    print(f"  equilibration: {gap_txt}")

    blk = blocking_analysis(p["blocks"], min_bins=16)
    lev = np.arange(len(blk["errs"]))
    a2.errorbar(lev, blk["errs"] * 1e3, yerr=blk["errs_err"] * 1e3,
                marker="o", ms=3, color=SERIES_COLORS[0], capsize=1.5,
                elinewidth=0.7, lw=0.7)
    a2.axhline(blk["err"] * 1e3, color=INK_2, ls="--", lw=0.7,
               label=f"plateau, level {blk['level']}")
    a2.set_xticks(lev)
    a2.set_xticklabels([f"{k}\n{n}" for k, n in zip(lev, blk["n"])],
                       fontsize=6)
    a2.set_xlabel("rebinning level / number of bins")
    a2.set_ylabel(r"$\sigma(E_{\rm FN})$ (mHa)")
    a2.legend(loc="lower right")
    a2.grid(True)
    vmc._save(fig, fname)
    plt.close(fig)


def _budget_numbers(p):
    v = (float(p["E_vmc"]) - E_EXACT) * 1e3
    f = (float(p["E"]) - E_EXACT) * 1e3
    return v, f, float(p["dE_vmc"]) * 1e3, float(p["dE"]) * 1e3


def fig_error_budget(p, fname="fig_error_budget.pdf"):
    """Where the VMC error goes: Jastrow gap vs nodal error, on one axis."""
    fig, ax = plt.subplots(figsize=(COL_W, 1.55))
    _draw_error_budget(ax, p)
    vmc._save(fig, fname)
    plt.close(fig)


# Top-to-bottom order of the combined figure: worst node first.
BUDGET_ORDER = ("nomod", "nomodJ1", "lin", "quadB", "quadA", "quadJ1",
                "fix1s", "cas2", "full")


def fig_error_budget_all(prods, fname="fig_error_budget_all.pdf"):
    """The single-model error budgets stacked on a SHARED energy axis, so that
    the lengths of the arrows (Jastrow gap) and the distances from zero
    (nodal error) can be compared across models by eye."""
    tags = [t for t in BUDGET_ORDER if t in prods]
    tags += sorted(t for t in prods if t not in tags)
    if len(tags) < 2:
        return
    nums = [_budget_numbers(prods[t]) for t in tags]
    lo = min(-0.1, min(f - 2 * sf for v, f, sv, sf in nums))
    hi = max(v + 2.5 * sv for v, f, sv, sf in nums)

    fig, axs = plt.subplots(len(tags), 1, sharex=True,
                            figsize=(COL_W, 0.35 + (1.15 if len(tags) <= 5
                                                    else 0.9) * len(tags)),
                            gridspec_kw={"hspace": 0.12})
    for k, (ax, t) in enumerate(zip(axs, tags)):
        _draw_error_budget(ax, prods[t], xlim=(lo, hi),
                           xlabel=(k == len(tags) - 1))
        ax.text(0.99, 0.95, t, transform=ax.transAxes, ha="right", va="top",
                fontsize=7, fontweight="bold", color=INK)
    vmc._save(fig, fname)
    plt.close(fig)


def _draw_error_budget(ax, p, xlim=None, xlabel=True):
    v, f, sv, sf = _budget_numbers(p)
    ax.errorbar([v], [1], xerr=[sv], marker="o", ms=4, color=SERIES_COLORS[1],
                capsize=2, elinewidth=0.8, ls="none")
    ax.errorbar([f], [0], xerr=[sf], marker="o", ms=4, color=SERIES_COLORS[0],
                capsize=2, elinewidth=0.8, ls="none")
    ax.axvline(0.0, color=INK, lw=0.6, ls=":")
    if f < 2 * sf:                                   # not resolved: show bound
        bound = max(f, 0.0) + 1.645 * sf
        ax.annotate("", xy=(bound, -0.32), xytext=(0.0, -0.32),
                    arrowprops=dict(arrowstyle="|-|", lw=0.6, color=INK_2,
                                    shrinkA=0, shrinkB=0, mutation_scale=2))
        ax.text(bound / 2, -0.38, f"95% bound < {bound:.2f}", ha="center",
                va="top", fontsize=6, color=INK_2)
    ax.annotate("", xy=(f, 0.5), xytext=(v, 0.5),
                arrowprops=dict(arrowstyle="<->", lw=0.6, color=INK_2))
    ax.text((v + f) / 2, 0.56, f"Jastrow gap {v - f:.2f} mHa", ha="center",
            va="bottom", fontsize=6, color=INK_2)
    ax.set_yticks([0, 1])
    ax.set_yticklabels([r"FN-DMC", r"VMC"])
    ax.set_ylim(-0.75, 1.5)
    ax.set_xlim(xlim if xlim is not None
                else (min(-0.1, f - 2 * sf), v + 2.5 * sv))
    if xlabel:
        ax.set_xlabel(r"$E-E_{\rm exact}$ (mHa)")
    ax.grid(True, axis="x")


# ============================================================================
def main():
    vmc.setup_plot_style()

    prods = {}
    for path in sorted(glob.glob(data_path("dmc_prod_*.npz"))):
        p = dict(np.load(path))
        tag = os.path.basename(path)[len("dmc_prod_"):-4]
        prods[tag] = p
        fig_dmc_equilibration(p, fname=f"fig_dmc_equilibration_{tag}.pdf")
        fig_error_budget(p, fname=f"fig_error_budget_{tag}.pdf")
    if prods:
        fig_error_budget_all(prods)
    else:
        print("no dmc_prod_*.npz: run dmc.run_production first")

    if os.path.exists(data_path("node_analysis.npz")):
        a = dict(np.load(data_path("node_analysis.npz")))
        fig_gauge_radial(a)
        fig_mixing_gauge(a)
    else:
        print("node_analysis.npz not found: run nodes.py first")

    rows = collect_mixing()
    if rows and prods:
        fig_node_order(rows, prods)

    series = sorted(glob.glob(data_path("dmc_ts_*_summary.npz")))
    for path in series:
        ts = dict(np.load(path))
        tag = os.path.basename(path)[len("dmc_ts_"):-len("_summary.npz")]
        fig_dmc_timestep(ts, fname=f"fig_dmc_timestep_{tag}.pdf")
    if not series:
        print("no dmc_ts_*_summary.npz: run dmc.run_timestep_series first")


if __name__ == "__main__":
    main()