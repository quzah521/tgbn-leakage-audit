"""Within-period placebo stream (Section 6.3).

Inside each block of edges that share a label period and a split, the payloads
(source, destination, weight) are permuted over the fixed timestamps. Every per-period total is
preserved and only the timing within a period is destroyed, so a method that uses per-period
totals only scores the same on the real and the placebo stream; a difference measures what the
timing within a period contributes.
"""
from __future__ import annotations

import numpy as np


def within_period_shuffle(src, dst, w, ts, label_ts, split_id, rng):
    """Permute (src, dst, w) among the edges of the same (label period, split); keep ts.

    Args:
        src, dst, w, ts: (E,) edge arrays.
        label_ts: label times that delimit the periods.
        split_id: (E,) 0, 1 or 2 for the train, validation and test split.
        rng: numpy random Generator.

    Returns:
        The permuted (src, dst, w).
    """
    period = np.searchsorted(np.sort(label_ts), ts, side="right").astype(np.int64)
    grp = period * 4 + split_id.astype(np.int64)
    order = np.argsort(grp, kind="stable")
    g = grp[order]
    cuts = np.flatnonzero(np.r_[True, g[1:] != g[:-1], True])
    perm = np.arange(src.size)
    for a, b in zip(cuts[:-1], cuts[1:]):
        if b - a > 1:
            pos = order[a:b]
            perm[pos] = pos[rng.permutation(b - a)]
    return src[perm], dst[perm], w[perm]
