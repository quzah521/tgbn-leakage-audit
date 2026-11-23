"""Training-free channels and blends for node affinity prediction (Sections 4.4 and 5.1).

`ChannelState` keeps per-(node, class) statistics, updated in stream order. Channel names in
the code and their names in the paper:

    PF          persistent forecast: the most recent label (TGB heuristic)
    EMA<b>      exponential moving average of labels with factor b, initialized with the first
                label; EMA<(k-1)/k> is the MovingAverage heuristic of TGB (MA in the paper),
                EMA0.95 is "EMA (beta = 0.95)"
    CNT         edge counter: edge weight since the node's previous prediction, then reset
                (the edge "semi-labels" of NAVIS)
    CUM         total edge weight over the whole history
    PCEMA<b>    moving average of the per-period edge totals (CNT at each prediction)
    DEC<m>P     edge weight decayed exponentially with time constant m * P
                (P = median label interval; DEC0.125P is the recency counter, 3 hours on the
                daily datasets)

PF and EMA<b> read labels; all other channels read edges only. When a label may be used as an
input is decided by the evaluation stream (protocol.NodeAffinityStream), which emits a
"release" event; the driver then calls ChannelState.apply_labels.

A blend scores candidates with a non-negative linear combination of row-normalized channels.
Its weights are fit on a sample of validation label times (select_blends) and frozen for test.
Blend names in the code and in the paper:

    Fit-LabelOnly   label blend
    Fit-EdgeOnly    edge blend
    Fit-Time        hybrid blend (all channels)
    Fit-NoTime      hybrid blend without the decayed channels DEC<m>P (Section 6.3)
    Blend2, Blend3  greedy starting points of the fits above; not reported in the paper
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np

from .metrics import idcg_rows, ndcg_rows


def _row_normalize(M: np.ndarray) -> np.ndarray:
    s = M.sum(axis=1, keepdims=True)
    return np.divide(M, s, out=np.zeros_like(M), where=s > 0)


class ChannelState:
    """Per-(node, class) channel matrices, updated by edges, predictions and released labels."""

    def __init__(self, n_nodes: int, n_classes: int, betas: Sequence[float],
                 pc_betas: Sequence[float], tau_mults: Sequence[float], period: float,
                 dtype=np.float32):
        """
        Args:
            n_nodes: number of node ids.
            n_classes: number of classes (candidates of a label).
            betas: factors of the label moving averages EMA<b>.
            pc_betas: factors of the moving averages of per-period edge totals PCEMA<b>.
            tau_mults: time constants of the decayed edge weights, in units of `period`.
            period: median label interval P, in the time unit of the edges.
            dtype: dtype of the channel matrices.
        """
        self.N, self.C = int(n_nodes), int(n_classes)
        self.betas = [float(b) for b in betas]
        self.pc_betas = [float(b) for b in pc_betas]
        self.tau_mults = [float(m) for m in tau_mults]
        self.taus = [m * float(period) for m in self.tau_mults]
        z = lambda: np.zeros((self.N, self.C), dtype=dtype)
        self.pf = z()
        self.has_label = np.zeros(self.N, dtype=bool)
        self.ema = [z() for _ in self.betas]
        self.cnt = z()
        self.cum = z()
        self.has_pc = np.zeros(self.N, dtype=bool)
        self.pc = [z() for _ in self.pc_betas]
        self.dec = [z() for _ in self.taus]
        # time of the last update of each (node, class) entry of the decayed channels
        self.clock = np.zeros((self.N, self.C), dtype=np.float64) if self.taus else None
        self.now = -np.inf
        self.n_events = 0

    def channel_names(self) -> List[str]:
        """Channel names in the fixed column order used everywhere."""
        names = ["PF"] + [f"EMA{b:g}" for b in self.betas] + ["CNT", "CUM"]
        names += [f"PCEMA{b:g}" for b in self.pc_betas]
        names += [f"DEC{m:g}P" for m in self.tau_mults]
        return names

    def timing_mask(self) -> np.ndarray:
        """Boolean mask of the channels that see the timing of edges (DEC<m>P)."""
        return np.array([n.startswith("DEC") for n in self.channel_names()])

    def label_mask(self) -> np.ndarray:
        """Boolean mask of the channels that read labels (PF, EMA<b>)."""
        return np.array([n == "PF" or n.startswith("EMA") for n in self.channel_names()])

    def memory_gb(self) -> float:
        """Memory of all channel matrices, in GB."""
        n_mat = 1 + len(self.betas) + 2 + len(self.pc_betas) + len(self.taus)
        b = n_mat * self.N * self.C * self.pf.itemsize
        if self.clock is not None:
            b += self.clock.nbytes
        return b / 1e9

    def process(self, s: np.ndarray, d: np.ndarray, t: np.ndarray, w: np.ndarray) -> None:
        """Add a batch of edges (source, destination class, time, weight) to the edge channels."""
        if s.size == 0:
            return
        C = self.C
        key = s.astype(np.int64) * C + d.astype(np.int64)
        uniq, inv = np.unique(key, return_inverse=True)
        us, ud = np.divmod(uniq, C)
        w = w.astype(np.float64)
        ws = np.bincount(inv, weights=w, minlength=uniq.size)
        self.cnt[us, ud] += ws
        self.cum[us, ud] += ws
        T = float(t.max())
        if self.taus:
            # decay the old value to the batch time T, then add each edge decayed from its own time
            age_pair = T - self.clock[us, ud]
            dt_ev = T - t.astype(np.float64)
            for j, tau in enumerate(self.taus):
                add = np.bincount(inv, weights=w * np.exp(-dt_ev / tau), minlength=uniq.size)
                self.dec[j][us, ud] = self.dec[j][us, ud] * np.exp(-age_pair / tau) + add
            self.clock[us, ud] = T
        self.now = max(self.now, T)
        self.n_events += int(s.size)

    def channel_rows(self, nodes: np.ndarray) -> List[np.ndarray]:
        """Row-normalized channel matrices of `nodes` at the current time, in channel order."""
        out = [self.pf[nodes]]
        out += [M[nodes] for M in self.ema]
        out += [self.cnt[nodes], self.cum[nodes]]
        out += [M[nodes] for M in self.pc]
        if self.taus:
            # Bring every entry to a common reference time. The decay factor is shared by the
            # whole row, so it cancels in the row normalization; working in log space avoids
            # underflow for entries that have not been updated for a long time.
            ref = self.now if np.isfinite(self.now) else 0.0
            age = ref - self.clock[nodes]
            for M, tau in zip(self.dec, self.taus):
                V = M[nodes].astype(np.float64)
                with np.errstate(divide="ignore"):
                    logv = np.log(V) - age / tau
                mx = np.max(logv, axis=1, keepdims=True)
                mx[~np.isfinite(mx)] = 0.0
                out.append(np.exp(logv - mx))
        return [_row_normalize(np.asarray(M, dtype=np.float64)) for M in out]

    def after_predict(self, nodes: np.ndarray) -> None:
        """Update PCEMA<b> with the edge counter read at this prediction, then reset the counter."""
        cn = _row_normalize(self.cnt[nodes].astype(np.float64))
        first_pc = ~self.has_pc[nodes]
        for b, M in zip(self.pc_betas, self.pc):
            old = M[nodes].astype(np.float64)
            M[nodes] = np.where(first_pc[:, None], cn, b * old + (1 - b) * cn)
        self.has_pc[nodes] = True
        self.cnt[nodes] = 0.0  # as get_semilabels in NAVIS, which zeroes the rows it read

    def apply_labels(self, nodes: np.ndarray, Y: np.ndarray) -> None:
        """Update PF and EMA<b> with released labels; a node's first label initializes them, as in TGB."""
        Y = np.asarray(Y, dtype=np.float64)
        first = ~self.has_label[nodes]
        for b, M in zip(self.betas, self.ema):
            old = M[nodes].astype(np.float64)
            M[nodes] = np.where(first[:, None], Y, b * old + (1 - b) * Y)
        self.pf[nodes] = Y
        self.has_label[nodes] = True


@dataclass
class LabelBatch:
    """The labels of one label time and the channel values of their candidates, in sparse form.

    A candidate is a class with a non-zero value in at least one channel; every other class
    scores 0 under every blend.
    """

    rows: np.ndarray      # (nnz,) row (labeled node) of each candidate
    F: np.ndarray         # (nnz, n_channels) row-normalized channel values, float32
    y: np.ndarray         # (nnz,) true label value of each candidate
    ytot: np.ndarray      # (n,) sum of each full label row
    idcg: np.ndarray      # (n,) ideal DCG@k of each full label row
    n: int
    C: int

    def ndcg(self, w: np.ndarray, k: int = 10) -> float:
        """Mean NDCG@k over the rows when candidates are scored with channel weights w."""
        return float(np.mean(self.ndcg_rows(w, k))) if self.n else float("nan")

    def ndcg_rows(self, w: np.ndarray, k: int = 10) -> np.ndarray:
        """Per-row NDCG@k when candidates are scored with channel weights w."""
        sc = self.F.astype(np.float64) @ np.asarray(w, dtype=np.float64)
        return ndcg_rows(self.rows, sc, self.y, self.n, self.C, self.ytot, self.idcg, k)

    def subsample(self, max_rows: int, rng: np.random.Generator) -> "LabelBatch":
        """Keep a random subset of at most max_rows rows."""
        if self.n <= max_rows:
            return self
        keep_rows = np.sort(rng.choice(self.n, size=max_rows, replace=False))
        remap = -np.ones(self.n, dtype=np.int64)
        remap[keep_rows] = np.arange(max_rows)
        m = remap[self.rows] >= 0
        return LabelBatch(remap[self.rows[m]], self.F[m], self.y[m], self.ytot[keep_rows],
                          self.idcg[keep_rows], max_rows, self.C)


def build_label_batch(state: ChannelState, nodes: np.ndarray, Y: np.ndarray, k: int = 10,
                      chunk: int = 1024) -> LabelBatch:
    """Read the channels of `nodes` and pair them with their labels Y (one label time)."""
    Y = np.asarray(Y, dtype=np.float64)
    n, C = Y.shape
    rows_l, F_l, y_l = [], [], []
    for a in range(0, n, chunk):
        b = min(n, a + chunk)
        chans = state.channel_rows(nodes[a:b])
        mask = np.zeros((b - a, C), dtype=bool)
        for M in chans:
            mask |= M > 0
        r, c = np.nonzero(mask)
        rows_l.append(r + a)
        F_l.append(np.stack([M[r, c] for M in chans], axis=1).astype(np.float32))
        y_l.append(Y[a:b][r, c])
    d = len(state.channel_names())
    rows = np.concatenate(rows_l) if rows_l else np.zeros(0, np.int64)
    F = np.concatenate(F_l) if F_l else np.zeros((0, d), np.float32)
    y = np.concatenate(y_l) if y_l else np.zeros(0)
    return LabelBatch(rows, F, y, Y.sum(axis=1), idcg_rows(Y, k), n, C)


def concat_batches(batches: List[LabelBatch]):
    """Concatenate label batches; returns (combined LabelBatch, label-time index of each row)."""
    off = 0
    rows, F, y, ytot, idcg, bid = [], [], [], [], [], []
    for i, b in enumerate(batches):
        rows.append(b.rows + off)
        F.append(b.F)
        y.append(b.y)
        ytot.append(b.ytot)
        idcg.append(b.idcg)
        bid.append(np.full(b.n, i))
        off += b.n
    C = batches[0].C
    big = LabelBatch(np.concatenate(rows), np.concatenate(F), np.concatenate(y),
                     np.concatenate(ytot), np.concatenate(idcg), off, C)
    return big, np.concatenate(bid)


class ValStore:
    """Reservoir sample of validation label times (each subsampled to max_rows nodes).

    The blend weights are fit on this sample (Section 5.1: up to 60 label times, 400 nodes each).
    """

    def __init__(self, max_label_times: int, max_rows: int, seed: int = 0):
        self.M, self.max_rows = int(max_label_times), int(max_rows)
        self.rng = np.random.default_rng(seed)
        self.items: List[LabelBatch] = []
        self.seen = 0
        self._big = None

    def add(self, lb: LabelBatch) -> None:
        """Offer one label time to the reservoir."""
        if lb.n == 0:
            return
        self.seen += 1
        sub = lb.subsample(self.max_rows, self.rng)
        if len(self.items) < self.M:
            self.items.append(sub)
        else:
            j = self.rng.integers(0, self.seen)
            if j < self.M:
                self.items[j] = sub
        self._big = None

    def score(self, w: np.ndarray, k: int = 10) -> float:
        """NDCG@k of channel weights w: mean over label times of the per-time mean, as in TGB."""
        if self._big is None:
            self._big = concat_batches(self.items)
        big, bid = self._big
        g = big.ndcg_rows(w, k)
        per_lt = np.bincount(bid, weights=g) / np.bincount(bid)
        return float(per_lt.mean())


def onehot(d: int, j: int) -> np.ndarray:
    """Weight vector of length d that selects channel j alone."""
    e = np.zeros(d)
    e[j] = 1.0
    return e


def select_blends(store: ValStore, names: List[str], lambdas=(0.1, 0.2, 0.35, 0.5),
                  mus=(0.05, 0.1, 0.2, 0.35, 0.5), k: int = 10, log=print) -> Dict[str, dict]:
    """Fit all blends on the validation sample.

    Blend2 and Blend3 are greedy starting points: the best label channel plus one edge channel
    without timing (weight in `lambdas`), then plus one decayed channel (weight in `mus`).
    The reported blends are refined from them by coord_ascent.

    Args:
        store: validation sample.
        names: channel names (ChannelState.channel_names()).
        lambdas: candidate weights of the edge channel in Blend2.
        mus: candidate weights of the decayed channel in Blend3.
        k: rank cutoff of NDCG.
        log: function that receives progress messages.

    Returns:
        {blend name: {"w": weights, "val_sample": NDCG@k on the sample, "desc": readable weights}}
        for Blend2, Blend3, Fit-NoTime, Fit-Time, Fit-LabelOnly and Fit-EdgeOnly.
    """
    d = len(names)
    idx = {n: i for i, n in enumerate(names)}
    bases = [n for n in names if n == "PF" or n.startswith("EMA")]
    partners = [None, "CNT", "CUM"] + [n for n in names if n.startswith("PCEMA")]
    decs = [n for n in names if n.startswith("DEC")]
    timing = np.array([n.startswith("DEC") for n in names])

    # Blend2: best label channel plus one edge channel without timing
    best2 = (-1.0, None, "")
    for b in bases:
        for p in partners:
            lams = [0.0] if p is None else lambdas
            for lam in lams:
                w = (1 - lam) * onehot(d, idx[b])
                if p is not None:
                    w = w + lam * onehot(d, idx[p])
                s = store.score(w, k)
                if s > best2[0]:
                    best2 = (s, w, f"{b}" + ("" if p is None else f"+{lam:g}*{p}"))
    log(f"  Blend2 (no timing) best on val-sample: {best2[2]}  ndcg={best2[0]:.4f}")

    # Blend3: Blend2 plus one decayed channel (mu = 0 keeps Blend2)
    best3 = (best2[0], best2[1], best2[2] + " (+no DEC)")
    for dn in decs:
        for mu in mus:
            w = (1 - mu) * best2[1] + mu * onehot(d, idx[dn])
            s = store.score(w, k)
            if s > best3[0]:
                best3 = (s, w, best2[2] + f" +{mu:g}*{dn}")
    log(f"  Blend3 (+timing) best on val-sample: {best3[2]}  ndcg={best3[0]:.4f}")

    # label and edge blends start from their best single channel
    is_label = np.array([n == "PF" or n.startswith("EMA") for n in names])
    singles = {j: store.score(onehot(d, j), k) for j in range(d)}
    jl = max(np.flatnonzero(is_label), key=lambda j: singles[j])
    je = max(np.flatnonzero(~is_label), key=lambda j: singles[j])
    fit_l, s_l = coord_ascent(store, onehot(d, jl), np.flatnonzero(is_label), k)
    log(f"  Fit-LabelOnly val-sample ndcg={s_l:.4f}")
    fit_e, s_e = coord_ascent(store, onehot(d, je), np.flatnonzero(~is_label), k)
    log(f"  Fit-EdgeOnly  val-sample ndcg={s_e:.4f}")

    # hybrid blends start from the greedy blends
    fit_nt, s_nt = coord_ascent(store, best2[1], np.flatnonzero(~timing), k)
    log(f"  Fit-NoTime val-sample ndcg={s_nt:.4f}")
    w0 = best3[1] if s_nt <= best3[0] else fit_nt
    fit_t, s_t = coord_ascent(store, w0, np.arange(d), k)
    log(f"  Fit-Time   val-sample ndcg={s_t:.4f}")
    return {
        "Blend2": {"w": best2[1], "val_sample": best2[0], "desc": best2[2]},
        "Blend3": {"w": best3[1], "val_sample": best3[0], "desc": best3[2]},
        "Fit-NoTime": {"w": fit_nt, "val_sample": s_nt, "desc": _desc(fit_nt, names)},
        "Fit-Time": {"w": fit_t, "val_sample": s_t, "desc": _desc(fit_t, names)},
        "Fit-LabelOnly": {"w": fit_l, "val_sample": s_l, "desc": _desc(fit_l, names)},
        "Fit-EdgeOnly": {"w": fit_e, "val_sample": s_e, "desc": _desc(fit_e, names)},
    }


def _desc(w: np.ndarray, names: List[str]) -> str:
    return " + ".join(f"{w[j]:.3f}*{names[j]}" for j in np.flatnonzero(w > 1e-9))


def coord_ascent(store: ValStore, w0: np.ndarray, allowed: np.ndarray, k: int = 10,
                 sweeps: int = 3):
    """Coordinate search for non-negative weights that sum to 1, maximizing NDCG on the sample.

    Args:
        store: validation sample.
        w0: starting weights (entries outside `allowed` are set to 0).
        allowed: indices of the channels the blend may use.
        k: rank cutoff of NDCG.
        sweeps: maximum number of passes over the allowed channels.

    Returns:
        (weights, NDCG@k on the sample).
    """
    w = np.asarray(w0, dtype=np.float64).copy()
    w[~np.isin(np.arange(w.size), allowed)] = 0.0
    if w.sum() <= 0:
        w[allowed[0]] = 1.0
    w /= w.sum()
    best = store.score(w, k)
    for _ in range(sweeps):
        improved = False
        for j in allowed:
            cands = {0.0, w[j] * 0.5, w[j] * 2.0, w[j] + 0.05, w[j] + 0.2, max(w[j] - 0.05, 0.0)}
            cand_best = (best, None)
            for v in cands:
                w2 = w.copy()
                w2[j] = v
                if w2.sum() <= 0:
                    continue
                w2 /= w2.sum()
                s = store.score(w2, k)
                if s > cand_best[0] + 1e-7:
                    cand_best = (s, w2)
            if cand_best[1] is not None:
                best, w = cand_best
                improved = True
        if not improved:
            break
    return w, best
