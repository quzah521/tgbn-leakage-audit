# Tables

## Table 3: label windows and loop diagnostics (test split)

Window and delay from the label-window probe; look-ahead under the TGB loop = a*·Δ (a* = last offset of the window); n̄_=, n̄_>: mean edges stamped at / after t_q processed before the prediction; max δ: largest edge horizon. Forms: S same-timestamp, E early label release, B batch boundary.

| dataset | label window | availability delay d | look-ahead | n̄_= | n̄_> | max δ | forms |
|---|---|---|---|---|---|---|---|
| tgbn-trade | [t+0, t+1) yr | 1 yr | none | 19,125.7 | 0.0 | 0 yr | S |
| tgbn-genre | [t+4, t+7) d | 7 d | 6 d | 1.6 | 99.8 | 2.3 h | E, B |
| tgbn-reddit | [t+5, t+6) d | 6 d | 5 d | 0.7 | 90.9 | 6.1 min | E, B |
| tgbn-token | [t+5, t+6) d | 6 d | 5 d | 3.5 | 108.2 | 2.8 min | E, B |

## S1: loop diagnostics on all splits (Section 4.3)

TGB loop; last column: strict cutoff (edges at/after t_q, max δ), which must be 0 / negative.

| dataset | split | query times | n̄_= | n̄_> | max n_> | leaky | max δ | strict |
|---|---|---|---|---|---|---|---|---|
| tgbn-trade | train | 22 | 14,089.3 | 0.0 | 0 | 100.0% | 0 yr | 0 / -1 yr |
|  | val | 4 | 18,056.2 | 0.0 | 0 | 100.0% | 0 yr | 0 / -1 yr |
|  | test | 3 | 19,125.7 | 0.0 | 0 | 100.0% | 0 yr | 0 / -1 yr |
| tgbn-genre | train | 1250 | 1.2 | 100.0 | 268 | 100.0% | 18.0 h | 0 / -55 s |
|  | val | 163 | 1.7 | 104.6 | 199 | 100.0% | 26.1 min | 0 / -55 s |
|  | test | 166 | 1.6 | 99.8 | 200 | 99.4% | 2.3 h | 0 / -55 s |
| tgbn-reddit | train | 929 | 0.3 | 101.1 | 200 | 100.0% | 7.9 h | 0 / -1 s |
|  | val | 83 | 0.6 | 102.9 | 199 | 100.0% | 17.2 min | 0 / -1 s |
|  | test | 77 | 0.7 | 90.9 | 199 | 98.7% | 6.1 min | 0 / -1 s |
| tgbn-token | train | 733 | 0.5 | 106.1 | 400 | 100.0% | 2.2 d | 0 / -1 s |
|  | val | 29 | 3.2 | 89.0 | 181 | 100.0% | 1.7 min | 0 / -2 s |
|  | test | 22 | 3.5 | 108.2 | 200 | 95.5% | 2.8 min | 0 / -1 s |

## S2: information revealed by each form, test NDCG@10 (Section 4.4)

TGB: reference loop (E: labels released immediately); Removed: only this form removed (E: delayed release; S, B: strict cutoff). Gain = TGB − Removed.

| form | dataset | method | TGB | Removed | Gain |
|---|---|---|---|---|---|
| S | tgbn-trade | Edge counter | 1.000 | 0.855 | +0.145 |
| E | tgbn-genre | PF | 0.357 | 0.311 | +0.046 |
| E | tgbn-reddit | PF | 0.369 | 0.308 | +0.061 |
| E | tgbn-token | PF | 0.430 | 0.362 | +0.068 |
| B | tgbn-genre | Recency counter | 0.404 | 0.404 | +0.0002 |
| B | tgbn-reddit | Recency counter | 0.450 | 0.450 | +0.0000 |
| B | tgbn-token | Recency counter | 0.411 | 0.411 | +0.0000 |

## Table 4: test NDCG@10, original vs leakage-free protocol

Heuristics: Section 5.1 setup (daily datasets: strict cutoff in both protocols). NAVIS: original = reference loop with the original global state; leakage-free = delayed release with the past global state (label mode) or strict cutoff (edge mode). † = value reported in the NAVIS paper; ± = std over seeds.

| method | trade Orig. | trade LF | trade Δ% | genre Orig. | genre LF | genre Δ% | reddit Orig. | reddit LF | reddit Δ% | token Orig. | token LF | token Δ% |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| *Label-based* |  |  |  |  |  |  |  |  |  |  |  |  |
| PF | 0.855 | 0.855 | 0.0 | 0.357 | 0.311 | -12.9 | 0.369 | 0.308 | -16.5 | 0.430 | 0.362 | -15.7 |
| MA | 0.823 | 0.823 | 0.0 | 0.509 | 0.476 | -6.5 | 0.559 | 0.531 | -5.0 | 0.508 | 0.451 | -11.3 |
| EMA (β=0.95) | 0.725 | 0.725 | 0.0 | 0.530 | 0.511 | -3.6 | 0.572 | 0.555 | -2.9 | 0.486 | 0.440 | -9.4 |
| Label blend | 0.858 | 0.858 | 0.0 | 0.532 | 0.511 | -4.0 | 0.576 | 0.555 | -3.7 | 0.513 | 0.451 | -12.0 |
| NAVIS (label mode) | 0.863 ± 0.0002 (n=3) | 0.863 ± 0.0001 (n=3) | 0.0 | 0.527 ± 0.0003 (n=3) | 0.501 ± 0.0001 (n=3) | -5.0 | 0.569† | 0.549 ± 0.0001 (n=3) | -3.5 | 0.513 ± 0.0001 (n=3) | 0.439 ± 0.0001 (n=3) | -14.4 |
| *Edge-based* |  |  |  |  |  |  |  |  |  |  |  |  |
| NAVIS (edge mode) | 0.998 ± 0.0000 (n=3) | 0.863 ± 0.0002 (n=3) | -13.5 | 0.520 ± 0.0004 (n=3) | 0.520 ± 0.0001 (n=3) | 0.0 | 0.552† | 0.552 ± 0.0001 (n=3) | 0.0 | 0.444† | 0.444 ± 0.0001 (n=3) | 0.0 |
| Edge blend | 1.000 | 0.863 | -13.7 | 0.527 | 0.527 | 0.0 | 0.554 | 0.554 | 0.0 | 0.447 | 0.447 | 0.0 |
| *Hybrid* |  |  |  |  |  |  |  |  |  |  |  |  |
| Hybrid blend | 1.000 | 0.863 | -13.7 | 0.542 | 0.530 | -2.2 | 0.580 | 0.561 | -3.3 | 0.513 | 0.454 | -11.5 |

## Table 5: ranks, original → leakage-free (Section 5.4)

Scores that agree to three decimals share a rank; Kendall's τ-b between the two rankings.

| method | trade | genre | reddit | token |
|---|---|---|---|---|
| PF | 6 → 6 | 8 → 8 | 8 → 8 | 8 → 8 |
| MA | 7 → 7 | 7 → 7 | 5 → 7 | 4 → 2 |
| EMA (β=0.95) | 8 → 8 | 3 → 4 | 3 → 2 | 5 → 6 |
| Label blend | 5 → 5 | 2 → 4 | 2 → 2 | 1 → 2 |
| NAVIS (label mode) | 4 → 1 | 4 → 6 | 4 → 6 | 1 → 7 |
| NAVIS (edge mode) | 3 → 1 | 6 → 3 | 7 → 5 | 7 → 5 |
| Edge blend | 1 → 1 | 4 → 2 | 6 → 4 | 6 → 4 |
| Hybrid blend | 1 → 1 | 1 → 1 | 1 → 1 | 1 → 1 |
| Kendall's τ | 0.90 | 0.59 | 0.69 | 0.46 |

## S3: label-release delay sweep, tgbn-reddit (Section 5.5)

Test NDCG@10 when a label is used only from its stamp + delay (strict cutoff, |score| weights); share = (0 d − 3 d) / (0 d − 6 d).

| method | 0 d | 1 d | 3 d | 6 d | share at 3 d |
|---|---|---|---|---|---|
| PF | 0.369 | 0.369 | 0.314 | 0.308 | 90% |
| MA | 0.559 | 0.559 | 0.538 | 0.531 | 77% |
| EMA (β=0.95) | 0.572 | 0.572 | 0.560 | 0.555 | 70% |
| Label blend | 0.576 | 0.576 | 0.560 | 0.555 | 76% |
| Hybrid blend | 0.579 | 0.579 | 0.563 | 0.558 | 77% |

tgbn-genre around its availability delay (text of Section 5.5):

| delay | PF | MA | EMA (β=0.95) | Label blend | Hybrid blend | edge blend | label − edge [95% CI] |
|---|---|---|---|---|---|---|---|
| 0 d | 0.357 | 0.509 | 0.530 | 0.532 | 0.542 | 0.527 | +0.0049 [+0.0028, +0.0070] |
| 5 d | 0.320 | 0.483 | 0.516 | 0.516 | 0.531 | 0.527 | -0.0117 [-0.0138, -0.0093] |
| 6 d | 0.316 | 0.479 | 0.513 | 0.513 | 0.530 | 0.527 | -0.0147 [-0.0166, -0.0127] |
| 7 d | 0.311 | 0.476 | 0.511 | 0.511 | 0.530 | 0.527 | -0.0166 [-0.0188, -0.0141] |

## S4: label-based minus edge-based, Figure 3 (Section 5.3)

Heuristics: label blend − edge blend, paired over test label times (95% bootstrap CI); NAVIS: label mode − edge mode.

| dataset | heuristics, original | heuristics, leakage-free | NAVIS, original | NAVIS, leakage-free |
|---|---|---|---|---|
| tgbn-trade | -0.1425 [-0.1665, -0.1115] | -0.0056 [-0.0091, -0.0033] | -0.135 | 0.000 |
| tgbn-genre | +0.0049 [+0.0028, +0.0070] | -0.0166 [-0.0188, -0.0141] | +0.007 | -0.020 |
| tgbn-reddit | +0.0226 [+0.0219, +0.0233] | +0.0014 [+0.0008, +0.0019] | +0.017 | -0.003 |
| tgbn-token | +0.0657 [+0.0610, +0.0707] | +0.0043 [+0.0036, +0.0050] | +0.069 | -0.005 |

## S5: edge weighting, tgbn-reddit, leakage-free (Section 5.5)

| edge weight | edge blend | NAVIS edge mode |
|---|---|---|
| |score| | 0.538 | 0.552 ± 0.0001 (n=3) |
| unit (edge count) | 0.554 | 0.557 ± 0.0001 (n=3) |

## S6: within-period placebo (Section 6.3)

Gain of the decayed edge weights (Fit-Time − Fit-NoTime) on the real and on the within-period-shuffled stream, and their difference (difference-in-differences).

| dataset | cutoff | weight | label delay | gain, real | gain, placebo | real − placebo [95% CI] | Wilcoxon p |
|---|---|---|---|---|---|---|---|
| tgbn-genre | strict | abs | 7 d | +0.0023 [+0.0015, +0.0030] | +0.0017 [+0.0008, +0.0026] | +0.0005 [-0.0000, +0.0012] | 0.19 |
| tgbn-reddit | strict | one | 6 d | +0.0010 [+0.0008, +0.0012] | +0.0011 [+0.0009, +0.0012] | -0.0000 [-0.0001, +0.0000] | 0.4 |
| tgbn-token | strict | abs | 6 d | +0.0013 [+0.0010, +0.0016] | +0.0012 [+0.0010, +0.0014] | +0.0001 [-0.0001, +0.0003] | 0.39 |
| tgbn-genre | strict | abs | 0 d | +0.0010 [+0.0004, +0.0017] | +0.0009 [+0.0003, +0.0015] | +0.0002 [-0.0003, +0.0006] | 0.96 |
| tgbn-genre | protocol | abs | 0 d | +0.0009 [+0.0004, +0.0015] | +0.0008 [+0.0000, +0.0014] | +0.0002 [-0.0006, +0.0011] | 0.73 |
| tgbn-reddit | strict | abs | 0 d | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | — |

## S7: all NAVIS runs

| dataset | mode | global state | cutoff | label delay (P) | edge weight | seeds | test (mean ± std) | best epoch | regularizer on |
|---|---|---|---|---|---|---|---|---|---|
| tgbn-genre | edge | - | protocol | 0 | abs | 1,2,3 | 0.520 ± 0.0004 (n=3) | 17,16,17 | cuda |
| tgbn-genre | edge | - | strict | 0 | abs | 1,2,3 | 0.520 ± 0.0001 (n=3) | 17,17,18 | cuda |
| tgbn-genre | label | leaky | - | 0 | - | 1,2,3 | 0.527 ± 0.0003 (n=3) | 13,12,13 | cuda |
| tgbn-genre | label | prev | - | 0 | - | 2 | 0.527 | 14 | cuda |
| tgbn-genre | label | prev | - | 7 | - | 1,2,3 | 0.501 ± 0.0001 (n=3) | 23,23,23 | cuda |
| tgbn-reddit | edge | - | strict | 0 | abs | 1,2,3 | 0.552 ± 0.0001 (n=3) | 19,19,19 | cpu |
| tgbn-reddit | edge | - | strict | 0 | one | 1,2,3 | 0.557 ± 0.0001 (n=3) | 18,18,18 | cpu |
| tgbn-reddit | label | prev | - | 6 | - | 1,2,3 | 0.549 ± 0.0001 (n=3) | 18,18,18 | cuda |
| tgbn-token | edge | - | strict | 0 | abs | 1,2,3 | 0.444 ± 0.0001 (n=3) | 22,22,23 | cpu |
| tgbn-token | label | leaky | - | 0 | - | 1,2,3 | 0.513 ± 0.0001 (n=3) | 16,16,15 | cuda |
| tgbn-token | label | prev | - | 6 | - | 1,2,3 | 0.439 ± 0.0001 (n=3) | 24,23,23 | cuda |
| tgbn-trade | edge | - | protocol | 0 | abs | 1,2,3 | 0.998 ± 0.0000 (n=3) | 50,50,50 | cuda |
| tgbn-trade | edge | - | strict | 0 | abs | 1,2,3 | 0.863 ± 0.0002 (n=3) | 42,42,33 | cuda |
| tgbn-trade | label | leaky | - | 0 | - | 1,2,3 | 0.863 ± 0.0002 (n=3) | 37,38,48 | cuda |
| tgbn-trade | label | prev | - | 0 | - | 1,2,3 | 0.863 ± 0.0001 (n=3) | 37,38,44 | cuda |

