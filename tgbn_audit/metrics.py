"""Exact NDCG@k for TGB node affinity prediction, computed on sparse candidates.

The result equals sklearn.metrics.ndcg_score(y_true, y_score, k) with ignore_ties=False, which
the Evaluator of TGB calls. Every class that is not a candidate has score 0; all of them are
represented by one pseudo entry per row, so that ties among zero scores are averaged exactly as
scikit-learn does.
"""
from __future__ import annotations

import numpy as np


def _discount_cumsum(k: int) -> np.ndarray:
    """Dc[p] = sum of the DCG discounts of the first p positions."""
    disc = 1.0 / (np.log(np.arange(k) + 2) / np.log(2))
    return np.concatenate([[0.0], np.cumsum(disc)])


def idcg_rows(Y: np.ndarray, k: int = 10) -> np.ndarray:
    """Ideal DCG@k of each row of Y (scikit-learn ignores ties for the normalizer)."""
    Y = np.asarray(Y, dtype=np.float64)
    kk = min(k, Y.shape[1])
    if Y.shape[1] > kk:
        top = -np.partition(-Y, kk - 1, axis=1)[:, :kk]
    else:
        top = Y.copy()
    top = -np.sort(-top, axis=1)
    disc = 1.0 / (np.log(np.arange(kk) + 2) / np.log(2))
    return top @ disc


def ndcg_rows(rows: np.ndarray, scores: np.ndarray, y: np.ndarray, n_rows: int,
              n_classes: int, ytot: np.ndarray, idcg: np.ndarray, k: int = 10) -> np.ndarray:
    """Per-row NDCG@k with the tie averaging of scikit-learn.

    Args:
        rows: (nnz,) row of each candidate entry, in any order.
        scores: (nnz,) predicted score of each candidate; non-candidates score 0.
        y: (nnz,) true label value of each candidate.
        n_rows: number of rows.
        n_classes: number of classes.
        ytot: (n_rows,) sum of each full true label row (candidates and non-candidates).
        idcg: (n_rows,) ideal DCG@k of each full true label row (see idcg_rows).
        k: rank cutoff.

    Returns:
        (n_rows,) NDCG@k of each row; 0 for rows whose ideal DCG is 0.
    """
    Dc = _discount_cumsum(k)
    rows = np.asarray(rows, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64) + 0.0  # turns -0.0 into 0.0
    y = np.asarray(y, dtype=np.float64)
    cnt_c = np.bincount(rows, minlength=n_rows).astype(np.float64)
    ysum_c = np.bincount(rows, weights=y, minlength=n_rows)
    rest_cnt = n_classes - cnt_c
    rest_y = np.maximum(np.asarray(ytot, dtype=np.float64) - ysum_c, 0.0)

    # one pseudo entry per row stands for all non-candidates (score 0)
    er = np.concatenate([rows, np.arange(n_rows, dtype=np.int64)])
    es = np.concatenate([scores, np.zeros(n_rows)])
    ec = np.concatenate([np.ones(rows.size), rest_cnt])
    ey = np.concatenate([y, rest_y])
    keep = ec > 0
    er, es, ec, ey = er[keep], es[keep], ec[keep], ey[keep]
    order = np.lexsort((-es, er))
    er, es, ec, ey = er[order], es[order], ec[order], ey[order]

    # groups of tied scores within a row: each gets the mean gain over the positions it spans
    newg = np.ones(er.size, dtype=bool)
    newg[1:] = (er[1:] != er[:-1]) | (es[1:] != es[:-1])
    gid = np.cumsum(newg) - 1
    G = int(gid[-1]) + 1
    g_row = er[newg]
    g_cnt = np.bincount(gid, weights=ec, minlength=G)
    g_y = np.bincount(gid, weights=ey, minlength=G)
    cum = np.cumsum(g_cnt)
    first = np.flatnonzero(np.r_[True, g_row[1:] != g_row[:-1]])
    seg_len = np.diff(np.r_[first, G])
    row_off = np.repeat((cum - g_cnt)[first], seg_len)
    start = cum - g_cnt - row_off
    end = start + g_cnt
    s_i = np.minimum(start, k).astype(np.int64)
    e_i = np.minimum(end, k).astype(np.int64)
    contrib = (g_y / g_cnt) * (Dc[e_i] - Dc[s_i])
    dcg = np.bincount(g_row, weights=contrib, minlength=n_rows)
    safe = np.where(idcg > 0, idcg, 1.0)
    return np.where(idcg > 0, dcg / safe, 0.0)
