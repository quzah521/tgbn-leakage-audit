"""Reproduce every number, table and figure of the paper (Linux, macOS and Windows).

Usage:
    python reproduce.py test                         # synthetic checks, about two minutes, no data
    python reproduce.py probe                        # label windows (Table 3, Figure 2)
    python reproduce.py loop                         # loop diagnostics (Table 3, S1)
    python reproduce.py heuristics                   # heuristics, both protocols (Tables 4, 5; S2-S6)
    python reproduce.py navis --navis_dir NAVIS/NAVIS              # NAVIS, seeds 1-3 (GPU)
    python reproduce.py navis --navis_dir NAVIS/NAVIS --skip_done  # only the missing NAVIS runs
    python reproduce.py tables                       # results/tables.md, results/tables.tex
    python reproduce.py figure                       # Figures 2-4 in results/figures/
    python reproduce.py all                          # test, probe, loop, heuristics, tables, figure

Options: --only tgbn-genre,tgbn-reddit restricts a step to some datasets; --dry_run prints the
commands without running them (with --skip_done, the navis step then lists the missing runs).
py-tgb downloads each dataset on first use; tgbn-token needs well over 10 GB of disk and a lot of
memory. Every step writes to results/; reference_results/ holds our own outputs.
"""
import argparse
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
DATASETS = ["tgbn-trade", "tgbn-genre", "tgbn-reddit", "tgbn-token"]
RES = os.path.join(ROOT, "results")

# Heuristic runs: (dataset, arguments of scripts/run_heuristics.py). Forms: S same-timestamp
# edges, B batch-boundary edges; LF = leakage-free protocol.
HEURISTICS = [
    ("tgbn-trade", "--cutoff strict,protocol --runs real"),                       # S, Table 4, S2
    ("tgbn-genre", "--cutoff strict,protocol --runs real,placebo"),               # B, original, placebo
    ("tgbn-genre", "--cutoff strict --runs real,placebo --label_delays avail"),   # leakage-free, placebo
    ("tgbn-reddit", "--cutoff strict,protocol --runs real --weight one"),         # B, original
    ("tgbn-reddit", "--cutoff strict --runs real,placebo --weight one --label_delays avail"),  # LF, placebo
    ("tgbn-reddit", "--cutoff strict --runs real,placebo --weight abs"),          # weighting, placebo
    ("tgbn-reddit", "--cutoff strict --runs real --weight abs --label_delays 1,3,avail"),  # delay sweep
    ("tgbn-genre", "--cutoff strict --runs real --label_delays 5,6"),             # delay check (Section 5.5)
    ("tgbn-token", "--cutoff strict,protocol --runs real"),                       # B, original
    ("tgbn-token", "--cutoff strict --runs real,placebo --label_delays avail"),   # leakage-free, placebo
]

# NAVIS runs per seed: (dataset, arguments of navis_audit.py, rerun of a published number?).
# On the two large datasets, training stops after 10 epochs without improvement on validation.
# The seed-2 runs on tgbn-reddit ran all 50 epochs. In edge mode, validation improved in every
# epoch up to the best one (19 with absolute, 18 with unit weights), so early stopping selects the
# same epoch (per-epoch scores: reference_results/navis/navis_audit_results_epochs.jsonl).
NAVIS = [
    ("tgbn-trade", "", False),                                    # label mode, original
    ("tgbn-trade", "--global_state prev", False),                 # label mode, leakage-free (d = 0)
    ("tgbn-trade", "--multi --cutoff protocol", False),           # edge mode, original
    ("tgbn-trade", "--multi --cutoff strict", False),             # edge mode, leakage-free
    ("tgbn-genre", "", False),
    ("tgbn-genre", "--global_state prev", False),                 # global-state ablation
    ("tgbn-genre", "--global_state prev --label_delay 7", False),
    ("tgbn-genre", "--multi --cutoff protocol", False),
    ("tgbn-genre", "--multi --cutoff strict", False),
    ("tgbn-reddit", "--patience 10", True),                       # paper uses the published 0.569
    ("tgbn-reddit", "--multi --cutoff protocol --reg_device cpu --patience 10", True),  # published 0.552
    ("tgbn-reddit", "--global_state prev --label_delay 6 --patience 10", False),
    ("tgbn-reddit", "--multi --cutoff strict --reg_device cpu --patience 10", False),
    ("tgbn-reddit", "--multi --cutoff strict --edge_weight one --reg_device cpu --patience 10", False),
    ("tgbn-token", "--patience 10", False),                       # reproduces the published 0.513
    ("tgbn-token", "--multi --cutoff protocol --reg_device cpu --patience 10", True),  # published 0.444
    ("tgbn-token", "--global_state prev --label_delay 6 --patience 10", False),
    ("tgbn-token", "--multi --cutoff strict --reg_device cpu --patience 10", False),
]


def run(cmd, cwd=ROOT, dry=False):
    """Print a command and run it (unless dry); stop on the first failure."""
    print("$", " ".join(cmd), flush=True)
    if not dry:
        subprocess.run(cmd, cwd=cwd, check=True)


def wanted(name, only):
    """True if the dataset is selected by --only (all datasets by default)."""
    return only is None or name in only


def step_test(a):
    run([PY, os.path.join("tests", "test_probe.py")], dry=a.dry_run)
    run([PY, os.path.join("tests", "test_audit.py")], dry=a.dry_run)


def step_probe(a):
    names = [n for n in DATASETS if wanted(n, a.only)]
    args = ["--all"] if len(names) == len(DATASETS) else ["--name", names[0]] if len(names) == 1 else None
    if args is None:  # several but not all: one process each
        for n in names:
            run([PY, "-m", "label_probe", "tgb", "--name", n, "--out_dir", os.path.join(RES, "probe")], dry=a.dry_run)
        return
    run([PY, "-m", "label_probe", "tgb", *args, "--out_dir", os.path.join(RES, "probe")], dry=a.dry_run)


def step_loop(a):
    for n in DATASETS:  # one process per dataset keeps the memory of the large datasets apart
        if wanted(n, a.only):
            run([PY, os.path.join("scripts", "audit_loop.py"), "--names", n,
                 "--out", os.path.join(RES, "loop", "loop_diagnostics.json")], dry=a.dry_run)


def step_heuristics(a):
    for n, args in HEURISTICS:
        if wanted(n, a.only):
            run([PY, os.path.join("scripts", "run_heuristics.py"), "--name", n, *args.split(),
                 "--out_dir", os.path.join(RES, "heuristics")], dry=a.dry_run)


def step_navis(a):
    if not a.navis_dir:
        sys.exit("navis needs --navis_dir <NAVIS checkout>/NAVIS (see navis/README.md)")
    nd = os.path.abspath(a.navis_dir)
    run([PY, os.path.join("navis", "apply_patch.py"), nd], dry=a.dry_run)
    out = os.path.join(RES, "navis", "navis_audit_results.jsonl")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    names = [n for n in DATASETS if wanted(n, a.only)]
    # py-tgb 2.3.0 returns the raw tgbn-token amounts on the first load after a download and their
    # logarithm on every later load; load each dataset once so that every run sees the cached form
    for n in names:
        run([PY, "-c", "from tgb.nodeproppred.dataset import NodePropPredDataset as D; "
                       f"D(name='{n}', root='datasets', preprocess=True)"], cwd=nd, dry=a.dry_run)
    done = navis_done(out) if a.skip_done else set()
    todo = []
    for s in [x.strip() for x in a.seeds.split(",") if x.strip()]:
        for n, args, published in NAVIS:
            if n in names and (a.rerun_published or not published):
                if navis_key(n, args, s) in done:
                    continue
                pat = [] if "--patience" in args.split() else ["--patience", str(a.patience)]
                todo.append([PY, "navis_audit.py", "--name", n, "--seed", s, *pat, "--out", out, *args.split()])
    if a.skip_done:
        print(f"{len(todo)} NAVIS runs missing from {out}", flush=True)
    for cmd in todo:
        run(cmd, cwd=nd, dry=a.dry_run)


def navis_key(name, args, seed):
    """Key of one navis_audit.py run, matching the fields of its result line."""
    t = args.split()
    val = lambda flag, default: t[t.index(flag) + 1] if flag in t else default
    return (name, "--multi" in t, val("--global_state", "leaky"), val("--cutoff", "protocol"),
            float(val("--label_delay", 0)), val("--edge_weight", "abs"), int(seed))


def navis_done(path):
    """Configurations that already have a result line in the NAVIS results file."""
    done = set()
    if os.path.exists(path):
        import json
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    done.add((r["name"], bool(r["multi"]), r["global_state"], r["cutoff"],
                              float(r.get("label_delay", 0)), r.get("edge_weight") or "abs", int(r["seed"])))
    return done


def step_tables(a):
    run([PY, os.path.join("scripts", "make_tables.py"), "--results", RES,
         "--latex", os.path.join(RES, "tables.tex")], dry=a.dry_run)
    run([PY, os.path.join("scripts", "compare_protocols.py"), "--results", os.path.join(RES, "heuristics")],
        dry=a.dry_run)


def step_figure(a):
    run([PY, os.path.join("scripts", "plot_probe.py"), "--results", os.path.join(RES, "probe"),
         "--out", os.path.join(RES, "figures", "probe.pdf"), "--png", os.path.join(RES, "figures", "probe.png")],
        dry=a.dry_run)
    run([PY, os.path.join("scripts", "plot_inputs.py"), "--results", RES,
         "--out", os.path.join(RES, "figures", "inputs.pdf"), "--png", os.path.join(RES, "figures", "inputs.png")],
        dry=a.dry_run)
    run([PY, os.path.join("scripts", "plot_memory.py"), "--results", RES,
         "--out", os.path.join(RES, "figures", "memory.pdf"), "--png", os.path.join(RES, "figures", "memory.png")],
        dry=a.dry_run)


STEPS = {"test": step_test, "probe": step_probe, "loop": step_loop, "heuristics": step_heuristics,
         "navis": step_navis, "tables": step_tables, "figure": step_figure}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=list(STEPS) + ["all"])
    ap.add_argument("--only", default=None, help="comma list of datasets (default: all four)")
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--navis_dir", default=None, help="the NAVIS/ folder of a NAVIS checkout (commit 2b503ef)")
    ap.add_argument("--seeds", default="1,2,3", help="NAVIS seeds (the paper reports the mean over 1, 2, 3)")
    ap.add_argument("--patience", type=int, default=0,
                    help="NAVIS early stopping on val (0 = off); the tgbn-reddit and tgbn-token runs set 10 themselves")
    ap.add_argument("--skip_done", action="store_true",
                    help="NAVIS: skip configurations and seeds that already have a line in results/navis/")
    ap.add_argument("--rerun_published", action="store_true",
                    help="NAVIS: also rerun the original-protocol runs for which the paper uses published numbers")
    a = ap.parse_args()
    a.only = None if a.only is None else {n.strip() for n in a.only.split(",") if n.strip()}
    steps = ["test", "probe", "loop", "heuristics", "tables", "figure"] if a.step == "all" else [a.step]
    for s in steps:
        print(f"\n== {s} ==", flush=True)
        STEPS[s](a)


if __name__ == "__main__":
    main()
