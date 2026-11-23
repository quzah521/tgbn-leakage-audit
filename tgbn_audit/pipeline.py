"""Run the training-free heuristics under a chosen protocol (Sections 4.4, 5 and 6).

One run makes a train pass, a validation pass (blend weights are fit on a validation sample) and
a test pass under a given edge cutoff and label delay, on the real stream and optionally on the
within-period placebo. It writes, per (dataset, cutoff, edge weight, label delay), a directory

    <out_dir>/<name>_<cutoff>_<weight>[_delay<d>]/
        summary.md               readable report
        summary.json             read by scripts/make_tables.py
        per_label_time_test.csv  test NDCG@10 per label time and configuration (paired tests)
        log.txt

and, when several delays are given, <name>_<weight>_delay_sweep.md. Run it through
scripts/run_heuristics.py.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import time
from collections import defaultdict
from fractions import Fraction
from typing import Dict, List

import numpy as np

from .data import AVAILABILITY_DELAY, TGB_MA_WINDOW, load_tgb, median_label_interval
from .heuristics import ChannelState, ValStore, build_label_batch, onehot, select_blends
from .placebo import within_period_shuffle
from .protocol import NodeAffinityStream
from .stats import paired_compare


def drive(stream: NodeAffinityStream, idx, phase, state: ChannelState, src, dst, ts, w,
          on_predict=None) -> None:
    """Feed the events of one split to a ChannelState.

    Args:
        stream: evaluation stream.
        idx: edge indices of the split.
        phase: "train", "val" or "test".
        state: channel state to update.
        src, dst, ts, w: (E,) edge arrays.
        on_predict: optional callback (label time, nodes, Y), called before the state is
            updated for the prediction, i.e. with the channels as they are at prediction time.
    """
    for ev in stream.run_split(idx, phase):
        kind = ev[0]
        if kind == "edges":
            i = ev[1]
            state.process(src[i], dst[i], ts[i], w[i])
        elif kind == "release":
            state.apply_labels(ev[2], ev[3])
        else:  # "predict"
            _, lt, nodes, Y = ev
            if on_predict is not None:
                on_predict(lt, nodes, Y)
            state.after_predict(nodes)


def run_pipeline(src, dst, ts, w, splits, labels, n_nodes, C, period, args, cutoff, tag,
                 log, label_delay=0.0) -> dict:
    """Train pass, validation pass with blend fitting, and test pass on one stream.

    Args:
        src, dst, ts, w: (E,) edge arrays (dst are class ids).
        splits: edge indices of the train, validation and test split.
        labels: label source (data.TGBLabels or a test double).
        n_nodes: number of node ids.
        C: number of classes.
        period: median label interval P.
        args: parsed command-line arguments (channel settings, sample sizes, seed).
        cutoff: "protocol" or "strict".
        tag: name of the run in the log ("real" or "placebo").
        log: logging function.
        label_delay: release delay in units of P.

    Returns:
        Per-label-time validation and test scores of every channel and blend, the selected
        blend weights, and the edge-side diagnostics of the stream.
    """
    state = ChannelState(n_nodes, C, args.betas, args.pc_betas, args.tau_mults, period,
                         dtype=np.float32)
    names = state.channel_names()
    d = len(names)
    log(f"[{tag}] channels: {names}  state memory ~{state.memory_gb():.2f} GB")
    stream = NodeAffinityStream(ts, labels, args.batch_size, cutoff, label_delay * period)
    stream.reset()
    singles = {n: onehot(d, j) for j, n in enumerate(names)}

    t0 = time.time()
    drive(stream, splits[0], "train", state, src, dst, ts, w)
    log(f"[{tag}] train pass done: {stream.n_label_times['train']} label times, "
        f"{time.time() - t0:.0f}s")

    store = ValStore(args.fit_max_label_times, args.fit_max_rows, seed=args.seed)
    val = defaultdict(list)
    val_lt: List[int] = []
    nnz_per_row: List[float] = []

    def on_val(lt, nodes, Y):
        lb = build_label_batch(state, nodes, Y, k=10)
        val_lt.append(lt)
        nnz_per_row.append(lb.rows.size / max(lb.n, 1))
        for n, wv in singles.items():
            val[n].append(lb.ndcg(wv))
        store.add(lb)

    t0 = time.time()
    drive(stream, splits[1], "val", state, src, dst, ts, w, on_val)
    log(f"[{tag}] val pass done: {len(val_lt)} label times, "
        f"{np.mean(nnz_per_row) if nnz_per_row else 0:.1f} candidate classes/node, "
        f"{time.time() - t0:.0f}s")

    t0 = time.time()
    sel = select_blends(store, names, k=10, log=log)
    log(f"[{tag}] selection done ({len(store.items)} val label times in sample), "
        f"{time.time() - t0:.0f}s")
    configs = dict(singles)
    for k_, v in sel.items():
        configs[k_] = v["w"]

    test = defaultdict(list)
    test_lt: List[int] = []

    def on_test(lt, nodes, Y):
        lb = build_label_batch(state, nodes, Y, k=10)
        test_lt.append(lt)
        for n, wv in configs.items():
            test[n].append(lb.ndcg(wv))

    t0 = time.time()
    drive(stream, splits[2], "test", state, src, dst, ts, w, on_test)
    log(f"[{tag}] test pass done: {len(test_lt)} label times, {time.time() - t0:.0f}s")
    labels.reset()

    leak = np.asarray(stream.leak, dtype=float)
    return {
        "names": names,
        "timing_mask": state.timing_mask().tolist(),
        "val_lt": val_lt,
        "val": {k_: np.asarray(v) for k_, v in val.items()},
        "test_lt": test_lt,
        "test": {k_: np.asarray(v) for k_, v in test.items()},
        "selected": {k_: {"w": v["w"].tolist(), "val_sample": v["val_sample"],
                          "desc": v["desc"]} for k_, v in sel.items()},
        "leak_mean_edges": float(leak.mean()) if leak.size else 0.0,
        "max_now_minus_lt": float(np.max(stream.now_minus_lt)) if stream.now_minus_lt else float("nan"),
        "n_label_times": dict(stream.n_label_times),
    }


# ---- reporting -------------------------------------------------------------------------


def _fmt_cmp(c: dict) -> str:
    return (f"diff={c['mean_diff']:+.4f}  95%CI=[{c['ci95'][0]:+.4f},{c['ci95'][1]:+.4f}]  "
            f"win={c['win_rate']:.2f}  Wilcoxon p={c['wilcoxon_p']:.3g}  (n={c['n']})")


def _with_ma(d: dict, name: str) -> dict:
    """Add the alias "MA" for the channel that equals the MovingAverage heuristic of TGB."""
    k = ma_key(name)
    if k in d:
        d["MA"] = d[k]
    return d


def report(res: Dict[str, dict], args, cutoff, meta, out_dir, log) -> dict:
    """Write summary.md, summary.json and per_label_time_test.csv of one run; return the JSON."""
    real = res.get("real")
    plac = res.get("placebo")
    names = real["names"]
    mean = lambda a: float(np.mean(a)) if len(a) else float("nan")
    lines = [f"# heuristic audit: {args.name}  cutoff={cutoff}  label_delay={meta.get('label_delay', 0):g}*P"
             f"  edge weight={args.weight}", ""]
    lines += [f"- events {meta['n_events']:,}, classes {meta['C']}, nodes {meta['n_nodes']:,}",
              f"- median label interval P = {meta['period']:.6g} (time units of the dataset)",
              f"- edge list time order: {meta['n_backward_steps']:,} backward steps, "
              f"largest jump back {meta['max_backward_jump']}",
              f"- label times evaluated: val {len(real['val_lt'])}, test {len(real['test_lt'])}",
              f"- edges processed after L_i but before predicting L_i (mean per label time): "
              f"{real['leak_mean_edges']:.1f}",
              f"- max over label times of (latest processed edge time - L_i): "
              f"{real['max_now_minus_lt']:.6g}  (negative = no edge at/after L_i was used)",
              f"- TGB MovingAverage(window={TGB_MA_WINDOW.get(args.name, 7)}) = channel "
              f"{ma_key(args.name)}", ""]

    lines += ["## Validation (full val, single channels)", "", "| channel | NDCG@10 |", "|---|---|"]
    for n in names:
        lines.append(f"| {n} | {mean(real['val'][n]):.4f} |")
    lines += ["", "## Selected blends (chosen on a val sample)", ""]
    for run_name, r in res.items():
        for k_, v in r["selected"].items():
            lines.append(f"- [{run_name}] **{k_}**: {v['desc']}  (val-sample {v['val_sample']:.4f})")
    lines += ["", "## Test NDCG@10 (mean over label times)", ""]
    hdr = "| config | real |" + (" placebo | real-placebo |" if plac else "")
    lines += [hdr, "|---|---|" + ("---|---|" if plac else "")]
    order = ["Fit-Time", "Fit-NoTime", "Blend3", "Blend2", "Fit-LabelOnly", "Fit-EdgeOnly"] + names
    for n in order:
        r = mean(real["test"][n])
        if plac:
            p = mean(plac["test"][n])
            lines.append(f"| {n} | {r:.4f} | {p:.4f} | {r - p:+.4f} |")
        else:
            lines.append(f"| {n} | {r:.4f} |")

    cmp = {}
    tm = np.array(real["timing_mask"])
    nt_names = [n for n, f in zip(names, tm) if not f]
    t_names = [n for n, f in zip(names, tm) if f]
    best_nt = max(nt_names, key=lambda n: mean(real["val"][n]))
    best_t = max(t_names, key=lambda n: mean(real["val"][n])) if t_names else None
    cmp["real: Fit-LabelOnly vs Fit-EdgeOnly"] = paired_compare(real["test"]["Fit-LabelOnly"], real["test"]["Fit-EdgeOnly"])
    cmp["real: Fit-Time vs Fit-NoTime"] = paired_compare(real["test"]["Fit-Time"], real["test"]["Fit-NoTime"])
    cmp["real: Blend3 vs Blend2"] = paired_compare(real["test"]["Blend3"], real["test"]["Blend2"])
    if best_t:
        cmp[f"real: best DEC single ({best_t}) vs best no-timing single ({best_nt})"] = \
            paired_compare(real["test"][best_t], real["test"][best_nt])
    if plac:
        same_lt = real["test_lt"] == plac["test_lt"]
        cmp["placebo: Fit-Time vs Fit-NoTime"] = paired_compare(plac["test"]["Fit-Time"], plac["test"]["Fit-NoTime"])
        if same_lt:
            cmp["Fit-Time: real vs placebo"] = paired_compare(real["test"]["Fit-Time"], plac["test"]["Fit-Time"])
            cmp["Fit-NoTime: real vs placebo (sanity, should be ~0)"] = paired_compare(
                real["test"]["Fit-NoTime"], plac["test"]["Fit-NoTime"])
            gain_r = np.asarray(real["test"]["Fit-Time"]) - np.asarray(real["test"]["Fit-NoTime"])
            gain_p = np.asarray(plac["test"]["Fit-Time"]) - np.asarray(plac["test"]["Fit-NoTime"])
            cmp["timing gain: real minus placebo (difference-in-differences)"] = paired_compare(gain_r, gain_p)
    lines += ["", "## Paired comparisons over test label times", ""]
    for k_, c in cmp.items():
        lines.append(f"- {k_}: {_fmt_cmp(c)}")

    text = "\n".join(lines)
    with open(os.path.join(out_dir, "summary.md"), "w", encoding="utf-8") as f:
        f.write(text + "\n")
    log("\n" + text)

    with open(os.path.join(out_dir, "per_label_time_test.csv"), "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["run", "config", "label_t", "ndcg"])
        for run_name, r in res.items():
            for n, arr in r["test"].items():
                for lt, v in zip(r["test_lt"], arr):
                    wr.writerow([run_name, n, lt, f"{v:.6f}"])
    js = {"args": vars(args), "cutoff": cutoff, "meta": meta, "comparisons": cmp,
          "ma_key": ma_key(args.name),  # the MovingAverage channel of TGB, also stored as "MA"
          "test_mean": {rn: _with_ma({n: mean(a) for n, a in r["test"].items()}, args.name)
                        for rn, r in res.items()},
          "val_mean": {rn: _with_ma({n: mean(a) for n, a in r["val"].items()}, args.name)
                       for rn, r in res.items()},
          "selected": {rn: r["selected"] for rn, r in res.items()},
          "n_label_times": {rn: r["n_label_times"] for rn, r in res.items()},
          "diagnostics": {rn: {"leak_mean_edges": r["leak_mean_edges"],
                               "max_now_minus_lt": r["max_now_minus_lt"]} for rn, r in res.items()}}
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(js, f, indent=2, default=float)
    return js


# ---- command line ----------------------------------------------------------------------


def parse_floats(s: str) -> List[float]:
    """Parse a comma-separated list of numbers; fractions such as "6/7" are allowed."""
    return [float(Fraction(x.strip())) for x in s.split(",") if x.strip()]


def tgb_ma_beta(name: str) -> float:
    """Factor b = (k - 1) / k of the MovingAverage heuristic of TGB on this dataset.

    Datasets that TGB does not configure (e.g. the synthetic test data) use its default k = 7.
    """
    k = TGB_MA_WINDOW.get(name, 7)
    return (k - 1) / k


def resolve_betas(spec: str, name: str) -> List[float]:
    """Parse --betas; the token "tgb" stands for the MovingAverage factor of TGB on this dataset."""
    out = []
    for x in spec.split(","):
        x = x.strip()
        if not x:
            continue
        out += [tgb_ma_beta(name)] if x == "tgb" else parse_floats(x)
    return sorted(set(out), key=out.index)


def ma_key(name: str) -> str:
    """Name of the channel that equals the MovingAverage heuristic of TGB on this dataset."""
    return f"EMA{tgb_ma_beta(name):g}"


def build_argparser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", default="tgbn-genre",
                   help="tgbn-trade | tgbn-genre | tgbn-reddit | tgbn-token")
    p.add_argument("--root", default="datasets", help="dataset folder passed to py-tgb (as in NAVIS)")
    p.add_argument("--cutoff", default="strict",
                   help="strict | protocol | strict,protocol (protocol = TGB reference loop)")
    p.add_argument("--runs", default="real,placebo",
                   help="real and/or placebo (within-period shuffle, Section 6.3)")
    p.add_argument("--batch_size", type=int, default=200)
    p.add_argument("--weight", default="abs", choices=["abs", "one"],
                   help="edge weight: abs = |edge feature| (as NAVIS), one = count edges")
    p.add_argument("--betas", default="0.5,0.7,tgb,0.95",
                   help="factors of the label moving averages; 'tgb' = (k-1)/k of TGB's "
                        "MovingAverage(window=k) on this dataset (data.TGB_MA_WINDOW)")
    p.add_argument("--pc_betas", type=parse_floats, default=parse_floats("0.5,6/7"),
                   help="factors of the moving averages of per-period edge totals")
    p.add_argument("--tau_mults", type=parse_floats, default=parse_floats("0.125,0.5,2,8"),
                   help="time constants of the decayed edge weights, in units of P")
    p.add_argument("--label_delays", type=str, default="0",
                   help="comma list of label release delays in units of P, e.g. 0,6; "
                        "'avail' = the availability delay of the dataset (data.AVAILABILITY_DELAY)")
    p.add_argument("--fit_max_label_times", type=int, default=60,
                   help="validation label times sampled to fit the blends")
    p.add_argument("--fit_max_rows", type=int, default=400,
                   help="labeled nodes sampled per validation label time")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out_dir", default=os.path.join("results", "heuristics"))
    return p


def main(argv=None, data=None):
    """Command-line entry point.

    Args:
        argv: command-line arguments (default: sys.argv).
        data: optional (src, dst, ts, w, masks, labels, n_classes) to run on instead of a TGB
            dataset; used by the synthetic tests.

    Returns:
        {"<cutoff>" or "<cutoff>_delay<d>": summary JSON of that run}.
    """
    args = build_argparser().parse_args(argv)
    args.betas = resolve_betas(args.betas, args.name)
    args.label_delays = [AVAILABILITY_DELAY[args.name] if x.strip() == "avail" else parse_floats(x)[0]
                         for x in args.label_delays.split(",") if x.strip()]
    if data is None:
        src, dst, ts, w, _feat, masks, labels, C = load_tgb(args.name, args.root, args.weight)
    else:
        src, dst, ts, w, masks, labels, C = data
    n_nodes = int(max(src.max(), dst.max())) + 1
    try:  # labeled nodes may have no edges
        n_nodes = max(n_nodes, 1 + max(max(dd.keys()) for dd in labels.label_dict.values()))
    except Exception:
        pass
    assert dst.max() < C, "destination ids must be class ids (true for all tgbn datasets)"
    lt_sorted = np.unique(np.asarray(labels.label_ts))
    period = median_label_interval(lt_sorted)
    splits = [np.flatnonzero(m) for m in masks]
    split_id = np.zeros(src.size, dtype=np.int64)
    split_id[masks[1]] = 1
    split_id[masks[2]] = 2

    dts = np.diff(ts)
    n_inv = int(np.sum(dts < 0))
    max_back = int(-dts.min()) if n_inv else 0
    print(f"time order: {n_inv:,} backward steps in the edge list (largest jump back {max_back})",
          flush=True)
    meta = {"n_events": int(src.size), "C": C, "n_nodes": n_nodes, "period": period,
            "n_backward_steps": n_inv, "max_backward_jump": max_back}
    all_js = {}
    sweep = []
    jobs = [(c.strip(), dl) for c in args.cutoff.split(",") if c.strip() for dl in args.label_delays]
    for cutoff, dl in jobs:
        meta = dict(meta, label_delay=dl)
        suffix = "" if dl == 0 else f"_delay{dl:g}"
        out_dir = os.path.join(args.out_dir, f"{args.name}_{cutoff}_{args.weight}{suffix}")
        os.makedirs(out_dir, exist_ok=True)
        logf = open(os.path.join(out_dir, "log.txt"), "w", encoding="utf-8")

        def log(msg):
            print(msg, flush=True)
            logf.write(msg + "\n")
            logf.flush()

        log(f"dataset {args.name}: {src.size:,} events, C={C}, n_nodes={n_nodes:,}, "
            f"label times={lt_sorted.size}, P={period:g}, cutoff={cutoff}, "
            f"label_delay={dl:g}*P")
        res = {}
        for run in [r.strip() for r in args.runs.split(",") if r.strip()]:
            if run == "real":
                s_, d_, w_ = src, dst, w
            elif run == "placebo":
                s_, d_, w_ = within_period_shuffle(src, dst, w, ts, labels.label_ts, split_id,
                                                   np.random.default_rng(args.seed + 1))
            else:
                raise ValueError(f"unknown run {run!r} (expected real or placebo)")
            res[run] = run_pipeline(s_, d_, ts, w_, splits, labels, n_nodes, C, period, args,
                                    cutoff, run, log, label_delay=dl)
        js = report(res, args, cutoff, meta, out_dir, log)
        all_js[cutoff if len(args.label_delays) == 1 else f"{cutoff}_delay{dl:g}"] = js
        sweep.append((cutoff, dl, js))
        logf.close()
    if len(args.label_delays) > 1:
        write_sweep(sweep, args, period)
    return all_js


def write_sweep(sweep, args, period):
    """Write the label-delay sweep table <name>_<weight>_delay_sweep.md."""
    keys = ["PF", "MA", "EMA0.95", "Fit-LabelOnly", "Fit-EdgeOnly", "Fit-NoTime", "Fit-Time"]
    lines = [f"# label-delay sweep: {args.name}  (delay in units of P = {period:g})", "",
             "Test NDCG@10 (real run). A label revealed at L is usable only from L + delay.", "",
             "| cutoff | delay | " + " | ".join(keys) + " | LabelOnly - EdgeOnly (95% CI) |",
             "|---|---|" + "---|" * len(keys) + "---|"]
    for cutoff, dl, js in sweep:
        tm = js["test_mean"]["real"]
        c = js["comparisons"]["real: Fit-LabelOnly vs Fit-EdgeOnly"]
        lines.append(f"| {cutoff} | {dl:g} | " + " | ".join(f"{tm.get(k, float('nan')):.4f}" for k in keys)
                     + f" | {c['mean_diff']:+.4f} [{c['ci95'][0]:+.4f},{c['ci95'][1]:+.4f}] |")
    text = "\n".join(lines)
    path = os.path.join(args.out_dir, f"{args.name}_{args.weight}_delay_sweep.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    print("\n" + text, flush=True)


if __name__ == "__main__":
    main()
