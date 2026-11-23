"""Paired comparison of the original and the leakage-free protocol for the heuristics (Section 5.2).

For each dataset, the per_label_time_test.csv files of the two runs written by
scripts/run_heuristics.py are paired by test label time. For each method we report the mean
under each protocol, the relative change, the mean paired difference (leakage-free minus
original) with a 95% bootstrap CI (2,000 resamples), and the Wilcoxon signed-rank p-value,
which is not computed for fewer than 6 label times (tgbn-trade).

Usage:
    python scripts/compare_protocols.py                  # reads results/heuristics/
    python scripts/compare_protocols.py --results <dir> --out <file.md>

Runs compared (directory names under --results), as in Section 5.1:
  tgbn-trade   original = tgbn-trade_protocol_abs      leakage-free = tgbn-trade_strict_abs
  tgbn-genre   original = tgbn-genre_strict_abs        leakage-free = tgbn-genre_strict_abs_delay7
  tgbn-reddit  original = tgbn-reddit_strict_one       leakage-free = tgbn-reddit_strict_one_delay6
  tgbn-token   original = tgbn-token_strict_abs        leakage-free = tgbn-token_strict_abs_delay6
"""
import argparse
import csv
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from tgbn_audit.data import TGB_MA_WINDOW  # noqa: E402
from tgbn_audit.stats import paired_compare  # noqa: E402

PAIRS = [  # dataset, original run, leakage-free run
    ("tgbn-trade", "tgbn-trade_protocol_abs", "tgbn-trade_strict_abs"),
    ("tgbn-genre", "tgbn-genre_strict_abs", "tgbn-genre_strict_abs_delay7"),
    ("tgbn-reddit", "tgbn-reddit_strict_one", "tgbn-reddit_strict_one_delay6"),
    ("tgbn-token", "tgbn-token_strict_abs", "tgbn-token_strict_abs_delay6"),
]
METHODS = [  # label in the table, config name in per_label_time_test.csv ("MA" is resolved per dataset)
    ("PF", "PF"),
    ("MA", "MA"),
    ("EMA (0.95)", "EMA0.95"),
    ("Label blend", "Fit-LabelOnly"),
    ("Edge blend", "Fit-EdgeOnly"),
    ("Hybrid blend", "Fit-Time"),
]


def load(path):
    """Read per_label_time_test.csv; returns {config: {label time: NDCG@10}} of the real run."""
    out = defaultdict(dict)
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            if r["run"] == "real":
                out[r["config"]][int(float(r["label_t"]))] = float(r["ndcg"])
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", default=os.path.join("results", "heuristics"))
    p.add_argument("--out", default=None, help="markdown file (default: <results>/../protocol_comparison.md)")
    p.add_argument("--names", default=None, help="comma list of datasets (default: all four)")
    args = p.parse_args()
    keep = None if args.names is None else {n.strip() for n in args.names.split(",")}

    lines = ["# Original vs leakage-free protocol (heuristics, test NDCG@10)", "",
             "diff = leakage-free minus original, paired by test label time; "
             "CI = 95% bootstrap (2,000 resamples)", "",
             "| dataset | method | original | leakage-free | rel. change | diff [95% CI] | Wilcoxon p | n |",
             "|---|---|---|---|---|---|---|---|"]
    missing = []
    for name, orig, lf in PAIRS:
        if keep is not None and name not in keep:
            continue
        paths = [os.path.join(args.results, d, "per_label_time_test.csv") for d in (orig, lf)]
        if not all(os.path.exists(x) for x in paths):
            missing += [x for x in paths if not os.path.exists(x)]
            continue
        A, B = load(paths[0]), load(paths[1])
        k = TGB_MA_WINDOW[name]
        for label, cfg in METHODS:
            if cfg == "MA":
                cfg = f"EMA{(k - 1) / k:g}"
                label = f"MA (k={k})"
            if cfg not in A or cfg not in B:
                lines.append(f"| {name} | {label} | — | — | — | config {cfg} not found | — | — |")
                continue
            lts = sorted(set(A[cfg]) & set(B[cfg]))
            a = np.array([A[cfg][t] for t in lts])
            b = np.array([B[cfg][t] for t in lts])
            c = paired_compare(b, a)
            ma, mb = a.mean(), b.mean()
            rel = (mb - ma) / ma * 100 if ma else float("nan")
            pv = c["wilcoxon_p"]
            lines.append(f"| {name} | {label} | {ma:.4f} | {mb:.4f} | {rel:+.1f}% | "
                         f"{c['mean_diff']:+.4f} [{c['ci95'][0]:+.4f}, {c['ci95'][1]:+.4f}] | "
                         f"{'—' if not np.isfinite(pv) else f'{pv:.2g}'} | {c['n']} |")
    if missing:
        lines += ["", "Missing (run scripts/run_heuristics.py first; per-label-time scores are not part of",
                  "reference_results/):"] + [f"- {m}" for m in missing]
    text = "\n".join(lines)
    out = args.out or os.path.join(os.path.dirname(os.path.abspath(args.results)), "protocol_comparison.md")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
