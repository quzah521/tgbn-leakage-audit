# Reference results

Our outputs, in the layout that `scripts/make_tables.py` reads:

```
probe/        python -m label_probe tgb --all (py-tgb 2.3.0, seed 0): <name>.json, <name>.md, overview.md
loop/         python scripts/audit_loop.py: loop_diagnostics.json
heuristics/   python scripts/run_heuristics.py: <name>_<cutoff>_<weight>[_delay<d>]/summary.json
navis/        navis_audit_results.jsonl (one line per NAVIS run), navis_audit_results_epochs.jsonl
              (per-epoch scores of the seed-2 edge-mode runs on tgbn-reddit), and
              navis_paper_reported.json (scores published by NAVIS, used where marked with †)
logs/         console logs of the heuristic and probe runs, and of the MovingAverage scripts of TGB
figures/      Figures 2-4 of the paper (scripts/plot_probe.py, plot_inputs.py, plot_memory.py)
tables.md     python scripts/make_tables.py --results reference_results
tables.tex    LaTeX rows of Tables 4 and 5 and of S1 and S3 (same command with --latex)
```

* The heuristic summaries were converted from the console logs in `logs/` and therefore keep
  four decimals; `tables.md` rounds half up. They contain no per-label-time scores, so the paired
  tests of `compare_protocols.py` need fresh runs (`python reproduce.py heuristics`).
* The tgbn-genre summaries use the MovingAverage of TGB with k = 6 (β = 5/6); our MA there
  matches the TGB script (`logs/tgb_moving_average_scripts.log`) and the leaderboard.
* The probe chooses the window with the highest cosine among those with coverage ≥ 0.99. On
  tgbn-genre, whose label day straddles two calendar days, this gives [t+4, t+7) and an
  availability delay of 7 days; the paper reports the look-ahead there as 5-6 days for this
  reason, whereas `tables.md` shows a*·Δ = 6 days.
* NAVIS: seeds 1, 2 and 3 for every configuration of Table 4; the global-state ablation on
  tgbn-genre (footnote in Section 3.3) has seed 2 only. The seed-2 edge-mode line on tgbn-reddit
  with absolute weights was reconstructed from
  its per-epoch scores (best validation epoch) and is marked with a `note` field.
