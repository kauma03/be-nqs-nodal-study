"""Small statistics helpers shared by the VMC analysis and the DMC driver.

blocking_analysis replaces the old `max(errs)` rule. Taking the maximum over
the rebinning ladder is biased upwards (by 15-20% for a 400-block series),
because the last levels have few bins and their error estimates fluctuate by
20-30%. Here the ladder stops at `min_bins` bins, and the error bar is read at
the first level after which no significant growth occurs (a plateau test in
the spirit of Flyvbjerg & Petersen, J. Chem. Phys. 91, 461 (1989)). If no
plateau is reached the blocks are too short: the remedy is a longer block, not
a different estimator, and the function says so.
"""

import numpy as np


def blocking_analysis(x, min_bins=32, n_sigma=2.0):
    """Blocking error with an explicit plateau test.

    Returns a dict with
        mean, err        the mean and its error bar (read at the plateau)
        err_err          statistical uncertainty of err
        level            rebinning level of the plateau (0 = raw series)
        plateau          False if the error was still growing at the last level
        tau_int          integrated autocorrelation time, in units of samples
        n, errs, errs_err  the full ladder, for plotting
    """
    y = np.asarray(x, dtype=float)
    mean = float(np.mean(y))
    ns, es, des = [], [], []
    while len(y) >= min_bins:
        n = len(y)
        e = y.std(ddof=1) / np.sqrt(n)
        ns.append(n)
        es.append(e)
        des.append(e / np.sqrt(2.0 * (n - 1)))
        y = y[: n // 2 * 2].reshape(-1, 2).mean(axis=1)

    if not es:
        # Too short for any blocking: return the naive error, flagged.
        n = len(x)
        e = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
        return dict(mean=mean, err=e, err_err=np.nan, level=0, plateau=False,
                    tau_int=np.nan, n=np.array([n]), errs=np.array([e]),
                    errs_err=np.array([np.nan]))

    es, des = np.array(es), np.array(des)
    k_star = len(es) - 1
    for k in range(len(es)):
        later = range(k + 1, len(es))
        if all(es[j] <= es[k] + n_sigma * np.hypot(des[k], des[j])
               for j in later):
            k_star = k
            break
    plateau = k_star < len(es) - 1
    return dict(mean=mean, err=float(es[k_star]), err_err=float(des[k_star]),
                level=int(k_star), plateau=bool(plateau),
                tau_int=float(0.5 * (es[k_star] / es[0]) ** 2) if es[0] > 0 else np.nan,
                n=np.array(ns), errs=es, errs_err=des)


def weighted_mean(y, s):
    """Inverse-variance weighted mean, its error, and chi^2 about it."""
    y, s = np.asarray(y, float), np.asarray(s, float)
    w = 1.0 / s ** 2
    m = float(np.sum(w * y) / np.sum(w))
    return m, float(1.0 / np.sqrt(np.sum(w))), float(np.sum(w * (y - m) ** 2))
