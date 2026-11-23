"""Paired statistics over label times: bootstrap CI of the mean difference and Wilcoxon test."""
from __future__ import annotations

import numpy as np


def paired_compare(a: np.ndarray, b: np.ndarray, n_boot: int = 2000, seed: int = 0) -> dict:
    """Compare two methods (or protocols) scored on the same label times.

    Args:
        a: (n,) per-label-time scores of the first method.
        b: (n,) per-label-time scores of the second method, in the same order.
        n_boot: bootstrap resamples for the confidence interval.
        seed: seed of the bootstrap.

    Returns:
        A dict with n, mean_diff (mean of a - b), win_rate (share of label times with a > b),
        ci95 (95% percentile bootstrap interval of the mean difference) and wilcoxon_p
        (two-sided signed-rank test; NaN for fewer than 6 label times or all-zero differences).
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    out = {"n": int(d.size), "mean_diff": float(d.mean()) if d.size else float("nan"),
           "win_rate": float(np.mean(d > 0)) if d.size else float("nan")}
    rng = np.random.default_rng(seed)
    if d.size >= 2:
        bs = rng.choice(d, size=(n_boot, d.size), replace=True).mean(axis=1)
        out["ci95"] = [float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))]
    else:
        out["ci95"] = [float("nan"), float("nan")]
    try:
        from scipy.stats import wilcoxon
        out["wilcoxon_p"] = float(wilcoxon(d).pvalue) if np.any(d != 0) and d.size >= 6 else float("nan")
    except Exception:
        out["wilcoxon_p"] = float("nan")
    return out
