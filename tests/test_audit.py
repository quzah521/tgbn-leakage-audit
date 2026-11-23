"""Tests of the evaluation stream and the heuristics on synthetic streams (no TGB, no torch).

    [1] the sparse NDCG@10 equals sklearn.metrics.ndcg_score, which the TGB Evaluator calls
    [2] stream invariants: the strict cutoff never passes an edge at or after t_q before the
        prediction at t_q, and a delayed label is never released before t_q + delay
    [3] same-timestamp edges (as on tgbn-trade): the edge counter is perfect under the TGB loop
        and not under the strict cutoff
    [4] early label release (as on the daily datasets): labels whose content lies after their
        stamp favor label-based heuristics until they are released at their availability time
    [5] batch-boundary edges: under the TGB loop, edges after t_q in the triggering batch help an
        edge-based heuristic when labels describe [t_q, t_q + P)
    [6] placebo (Section 6.3): planted timing within periods makes Fit-Time beat Fit-NoTime and
        the placebo removes the gain; without planted timing there is no gain
    [7] the EMA channel with b = (k-1)/k equals the MovingAverage(window=k) heuristic of TGB

Usage:
    python tests/test_audit.py      # or: pytest tests
"""
import os
import sys
import tempfile

import numpy as np
from sklearn.metrics import ndcg_score

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from tgbn_audit import NodeAffinityStream, idcg_rows, ndcg_rows  # noqa: E402
from tgbn_audit import pipeline  # noqa: E402


class FakeLabels:
    """Label source with the pointer semantics of NodePropPredDataset.find_next_labels_batch."""

    def __init__(self, label_dict):
        self.label_dict = label_dict
        self.label_ts = np.sort(np.array(list(label_dict.keys())))
        self.idx = 0

    def get_label_time(self):
        return self.label_ts[-1] if self.idx >= self.label_ts.size else self.label_ts[self.idx]

    def get_node_label(self, cur_t):
        if self.idx >= self.label_ts.size:
            return None
        ts = self.label_ts[self.idx]
        if cur_t >= ts:
            self.idx += 1
            dct = self.label_dict[ts]
            nodes = np.array(list(dct.keys()))
            Y = np.stack([dct[k] for k in dct]).astype(np.float32)
            return np.full(nodes.size, ts, dtype=int), nodes.astype(np.int64), Y
        return None

    def reset(self):
        self.idx = 0


def synth(timing: bool, seed=0, U=200, C=40, P=1000, n_periods=80, rate=20.0, p_fav=0.6,
          label_offset=0):
    """Synthetic stream in which users switch their favorite class.

    With timing=True, users switch at random moments within periods; with timing=False, only at
    period starts. The label at L_j is the normalized count vector of the user's edges in
    [L_j + label_offset*P, L_j + (label_offset+1)*P).
    """
    rng = np.random.default_rng(seed)
    T_end = n_periods * P
    src, dst, ts = [], [], []
    for u in range(U):
        bg = rng.dirichlet(np.full(C, 0.3))
        n_sw = rng.poisson(T_end / (2 * P))
        sw = np.sort(rng.uniform(0, T_end, n_sw))
        if not timing:
            sw = np.unique(np.floor(sw / P) * P)
        favs = rng.integers(0, C, sw.size + 1)
        n_ev = rng.poisson(rate * n_periods)
        t = np.sort(rng.integers(0, T_end, n_ev))
        f = favs[np.searchsorted(sw, t, side="right")]
        cls = np.where(rng.random(n_ev) < p_fav, f, rng.choice(C, n_ev, p=bg))
        src.append(np.full(n_ev, C + u))
        dst.append(cls)
        ts.append(t)
    src, dst, ts = map(np.concatenate, (src, dst, ts))
    o = np.argsort(ts, kind="stable")
    src, dst, ts = src[o].astype(np.int64), dst[o].astype(np.int64), ts[o].astype(np.int64)
    label_dict = {}
    for j in range(1, n_periods - label_offset):
        L = j * P
        a = L + label_offset * P
        i0, i1 = np.searchsorted(ts, a), np.searchsorted(ts, a + P)
        dct = {}
        for u in np.unique(src[i0:i1]):
            m = src[i0:i1] == u
            v = np.bincount(dst[i0:i1][m], minlength=C).astype(float)
            dct[int(u)] = v / v.sum()
        label_dict[L] = dct
    q1, q2 = np.quantile(ts, [0.70, 0.85])
    masks = (ts <= q1, (ts > q1) & (ts <= q2), ts > q2)
    return src, dst, ts, np.ones(src.size), masks, FakeLabels(label_dict), C


def synth_same_timestamp(seed=0, U=60, C=30, n_years=30, per_year=200):
    """Stream like tgbn-trade: all edges of year Y carry the stamp Y and label(Y) sums them."""
    rng = np.random.default_rng(seed)
    src, dst, ts = [], [], []
    pref = rng.dirichlet(np.full(C, 0.5), size=U)
    for y in range(n_years):
        pref = 0.7 * pref + 0.3 * rng.dirichlet(np.full(C, 0.5), size=U)
        for u in range(U):
            k = rng.poisson(per_year / U * 3) + 1
            src.append(np.full(k, C + u))
            dst.append(rng.choice(C, k, p=pref[u]))
            ts.append(np.full(k, y))
    src, dst, ts = (np.concatenate(x).astype(np.int64) for x in (src, dst, ts))
    label_dict = {}
    for y in range(n_years):
        m = ts == y
        dct = {}
        for u in np.unique(src[m]):
            v = np.bincount(dst[m][src[m] == u], minlength=C).astype(float)
            dct[int(u)] = v / v.sum()
        label_dict[y] = dct
    yrs = ts
    masks = (yrs < 0.7 * n_years, (yrs >= 0.7 * n_years) & (yrs < 0.85 * n_years), yrs >= 0.85 * n_years)
    return src, dst, ts, np.ones(src.size), masks, FakeLabels(label_dict), C


def run(data, cutoff, out, extra=()):
    """Run the heuristic pipeline on synthetic data."""
    args = ["--name", "synth", "--cutoff", cutoff, "--weight", "one", "--out_dir", out,
            "--fit_max_label_times", "30"] + list(extra)
    return pipeline.main(args, data=data)


def test_ndcg():
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(300):
        n, C = rng.integers(1, 40), rng.integers(11, 80)
        Y = (rng.random((n, C)) < rng.uniform(0.02, 0.5)) * rng.integers(0, 4, (n, C)).astype(float)
        S = np.zeros((n, C))
        m = rng.random((n, C)) < rng.uniform(0, 1)
        S[m] = np.round(rng.normal(size=m.sum()), rng.integers(0, 3))
        r, c = np.nonzero(m | (S != 0))
        got = ndcg_rows(r, S[r, c], Y[r, c], n, C, Y.sum(1), idcg_rows(Y, 10), 10).mean()
        worst = max(worst, abs(got - ndcg_score(Y, S, k=10)))
    print(f"[1] sparse NDCG vs sklearn: max abs diff {worst:.2e}")
    assert worst < 1e-9


def test_stream_invariants():
    src, dst, ts, w, masks, labels, C = synth(True)
    splits = [np.flatnonzero(m) for m in masks]
    delay = 2500
    for cutoff in ["strict", "protocol"]:
        stream = NodeAffinityStream(ts, labels, 200, cutoff, label_delay=delay)
        stream.reset()
        max_seen, n_pred, n_rel, worst_rel = -np.inf, 0, 0, np.inf
        late_edges, released_since = 0, []
        for idx, ph in zip(splits, ["train", "val", "test"]):
            for ev in stream.run_split(idx, ph):
                if ev[0] == "edges":
                    max_seen = max(max_seen, ts[ev[1]].max())
                elif ev[0] == "predict":
                    n_pred += 1
                    late_edges += int(max_seen >= ev[1])
                    # every label released before this prediction must be available by now
                    for t_lab in released_since:
                        worst_rel = min(worst_rel, ev[1] - (t_lab + delay))
                    released_since = []
                else:
                    n_rel += 1
                    released_since.append(ev[1])
        print(f"[2] {cutoff:8s}: {n_pred} predictions, {late_edges} with an edge at/after t_q "
              f"already ingested; labels released {n_rel}, min(t_release - (t_q + delay)) = {worst_rel}")
        if cutoff == "strict":
            assert late_edges == 0
        assert worst_rel >= 0


def test_same_timestamp(out):
    data = synth_same_timestamp()
    res = {c: run(data, c, out, ["--runs", "real"])[c] for c in ["protocol", "strict"]}
    cnt = {c: res[c]["test_mean"]["real"]["CNT"] for c in res}
    print(f"[3] same-timestamp target leakage: CNT test NDCG protocol={cnt['protocol']:.4f}, "
          f"strict={cnt['strict']:.4f}")
    assert cnt["protocol"] > 0.999 and cnt["strict"] < 0.99


def test_lookahead(out):
    data = synth(True, label_offset=3)
    js = run(data, "strict", out, ["--runs", "real", "--label_delays", "0,4"])
    d0 = js["strict_delay0"]["comparisons"]["real: Fit-LabelOnly vs Fit-EdgeOnly"]
    d4 = js["strict_delay4"]["comparisons"]["real: Fit-LabelOnly vs Fit-EdgeOnly"]
    print(f"[4] label look-ahead: LabelOnly - EdgeOnly at delay 0: {d0['mean_diff']:+.4f}, "
          f"at the availability delay (4P): {d4['mean_diff']:+.4f}")
    assert d0["mean_diff"] > 0.02 and d4["mean_diff"] < d0["mean_diff"] - 0.02


def test_batch_boundary(out):
    data = synth(True, label_offset=0, rate=5.0)
    res = {c: run(data, c, out, ["--runs", "real"])[c] for c in ["protocol", "strict"]}
    e = {c: res[c]["test_mean"]["real"]["Fit-EdgeOnly"] for c in res}
    leak = {c: res[c]["diagnostics"]["real"]["leak_mean_edges"] for c in res}
    print(f"[5] batch boundary: Fit-EdgeOnly protocol={e['protocol']:.4f}, strict={e['strict']:.4f}, "
          f"gain from leaked edges {e['protocol'] - e['strict']:+.4f} "
          f"(edges at/after t_q ingested per prediction: protocol {leak['protocol']:.1f}, strict {leak['strict']:.1f})")
    assert leak["strict"] == 0
    assert e["protocol"] > e["strict"]


def test_timing_placebo(out):
    c = run(synth(True), "strict", out)["strict"]["comparisons"]
    k_gain, k_did = "real: Fit-Time vs Fit-NoTime", "timing gain: real minus placebo (difference-in-differences)"
    print(f"[6] planted timing: Fit-Time - Fit-NoTime {pipeline._fmt_cmp(c[k_gain])}")
    print(f"    difference-in-differences vs placebo {pipeline._fmt_cmp(c[k_did])}")
    assert c[k_gain]["ci95"][0] > 0 and c[k_did]["ci95"][0] > 0
    c = run(synth(False), "strict", out)["strict"]["comparisons"]
    print(f"    null stream: difference-in-differences {pipeline._fmt_cmp(c[k_did])}")


class _TGBMovingAverage:
    """Update and query logic of MovingAverage (modules/heuristics.py in the TGB repository)."""

    def __init__(self, num_class, window=7):
        self.dict, self.num_class, self.window = {}, num_class, window

    def update_dict(self, node_id, label):
        if node_id in self.dict:
            self.dict[node_id] = (self.dict[node_id] * (self.window - 1) + label) / self.window
        else:
            self.dict[node_id] = label

    def query_dict(self, node_id):
        return self.dict.get(node_id, np.zeros(self.num_class))


def test_tgb_moving_average():
    """EMA channel with b = (k-1)/k reproduces TGB's MovingAverage(window=k) (the first label initializes it)."""
    from tgbn_audit.heuristics import ChannelState
    rng = np.random.default_rng(3)
    N, C, worst = 30, 12, 0.0
    for k in (6, 7):
        b = pipeline.tgb_ma_beta("synth") if k == 7 else (k - 1) / k
        st = ChannelState(N, C, [b], [], [], 1.0)
        ref = _TGBMovingAverage(C, window=k)
        for _ in range(40):
            nodes = np.sort(rng.choice(N, size=rng.integers(1, N), replace=False))
            Y = rng.random((nodes.size, C)) * (rng.random((nodes.size, C)) < 0.3)
            Y = (Y / np.maximum(Y.sum(1, keepdims=True), 1e-12)).astype(np.float32)
            got = st.ema[0][nodes]  # value used for prediction, before the update
            exp = np.stack([ref.query_dict(int(u)) for u in nodes])
            worst = max(worst, float(np.abs(got - exp).max()))
            st.apply_labels(nodes, Y)
            for u, y in zip(nodes, Y):
                ref.update_dict(int(u), y)
    print(f"[7] EMA b=(k-1)/k vs TGB MovingAverage(window=k), k in (6, 7): max abs diff {worst:.2e}")
    assert worst < 1e-5
    assert pipeline.resolve_betas("0.5,0.7,tgb,0.95", "tgbn-genre")[2] == 5 / 6
    assert pipeline.ma_key("tgbn-genre") == "EMA0.833333"
    assert all(pipeline.ma_key(n) == "EMA0.857143" for n in ("tgbn-trade", "tgbn-reddit", "tgbn-token"))


if __name__ == "__main__":
    out = tempfile.mkdtemp()
    test_ndcg()
    test_stream_invariants()
    test_same_timestamp(out)
    test_lookahead(out)
    test_batch_boundary(out)
    test_timing_placebo(out)
    test_tgb_moving_average()
    print(f"\nall tests passed (outputs under {out})")
