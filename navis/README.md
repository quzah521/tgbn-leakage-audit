# NAVIS under the original and the leakage-free protocol

NAVIS (Feldman et al., ICLR 2026) is the state of the art on TGB node affinity prediction.
`navis_audit.patch` turns its released training script, `NAVIS/navis.py` at commit `2b503ef`
of <https://github.com/orfeld415/NAVIS>, into `navis_audit.py`. The patch

* adds the two switches of the leakage-free protocol (Section 3.3 of the paper): the strict edge
  cutoff for the edge mode and delayed label release for the label mode;
* adds a replacement for the global state of the label mode, which in the original reads the
  labels of other nodes at the same query time (footnote in Section 3.3);
* makes the dataset, seed, number of epochs, early stopping, learning rate and hinge margin
  selectable (the original fixes tgbn-trade, seed 2, 50 epochs, lr 1e-4 and margin 1e-3), and
  appends one JSON line per finished run (and per epoch) to a results file;
* imports `LambdaNDCGLoss2` from the vendored `ltr_loss.py` when pytorchltr is not installed.

With the default flags, `navis_audit.py` trains exactly like the original script.

## Setup

```bash
git clone https://github.com/orfeld415/NAVIS
git -C NAVIS checkout 2b503ef
python navis/apply_patch.py NAVIS/NAVIS      # writes NAVIS/NAVIS/navis_audit.py and ltr_loss.py
```

`apply_patch.py` needs neither git nor `patch` and stops if `navis.py` is not the expected
version. `python reproduce.py navis --navis_dir NAVIS/NAVIS` applies the patch and then runs
every configuration of the paper (see the main README for the requirements).

## Switches

| Flag | Values | Meaning |
|---|---|---|
| `--name` | tgbn-trade (default), tgbn-genre, tgbn-reddit, tgbn-token | dataset |
| `--multi` | off / on | off: label mode (inputs are past labels); on: edge mode (inputs are edge "semi-labels") |
| `--cutoff` | `protocol` (default), `strict` | edge mode: with `strict`, only edges with t < t_q are processed before predicting t_q; the rest of the batch is held back |
| `--label_delay D` | 0 (default) | label mode: a label stamped L becomes an input from L + D·P on (P = median label interval); genre 7, reddit 6, token 6 |
| `--global_state` | `leaky` (default), `prev`, `none` | label mode: `leaky` is the original (the current label of node i−1); `prev` is the mean of the most recently released labels |
| `--edge_weight` | `abs` (default), `one` | edge mode: \|edge feature\| (original) or edge counts |
| `--reg_device` | `cuda` (default), `cpu` | device of the hinge regularizer (see below) |
| `--seed`, `--epochs`, `--patience` | 2, 50, 0 | seed, maximum epochs, early stopping on validation (0 = off) |
| `--lr`, `--delta` | 1e-4, 1e-3 | learning rate and margin of the hinge regularizer (the values of the original script) |
| `--out` | `navis_audit_results.jsonl` | results file; per-epoch lines go to `<out>_epochs.jsonl` |

Leakage-free configurations: label mode `--global_state prev --label_delay <d>`; edge mode
`--multi --cutoff strict`. As in the original script, the test score is taken at the epoch with
the best validation score.

## Learning rate without leakage

NAVIS reports no hyperparameter search; its script fixes lr = 1e-4 and the margin 1e-3. To check
that our leakage-free comparison does not depend on these values, `reproduce.py navis_tune`
searches the learning rate for each leakage-free configuration, with training, model selection
and testing all under the leakage-free protocol:

```bash
python reproduce.py navis_tune --navis_dir NAVIS/NAVIS                   # lr in {3e-5, 1e-4, 3e-4, 1e-3}, seed 2
python reproduce.py navis_tune --navis_dir NAVIS/NAVIS --stage confirm   # seeds 1 and 3 of each selected lr
python reproduce.py tables                                              # S8 in results/tables.md
```

The rate with the best validation score on seed 2 is selected (ties go to 1e-4, whose runs come
from the main results). `--tune_lrs` and `--tune_deltas` change the grid; `--only` restricts it
to some datasets. The runs go to `results/navis/navis_tune.jsonl`, apart from the main results.

## Reproduction of the published numbers

With both switches off, the patched script reproduces the published test NDCG@10 to within 0.002
for the label mode on tgbn-trade and tgbn-token and for both modes on tgbn-genre. On tgbn-trade,
the edge mode reaches 0.998 under the reference loop, whereas the NAVIS paper reports 0.863, the
value we obtain with the strict cutoff (Section 5.2 of our paper).

## `--reg_device cpu` on the large datasets

In edge mode on tgbn-reddit, the `HindgeRegularization` of NAVIS occasionally triggered a CUDA
device-side assert (`scatter gather kernel index out of bounds`) with PyTorch 2.10 and CUDA 12.8,
at a different step in each run, although the inputs of a failing step were finite and the same
computation succeeded on the CPU. We therefore compute this regularizer on the CPU in edge mode
on tgbn-reddit and tgbn-token. The formula is unchanged and the gradients flow back to the model
on the GPU; the `reg_device` field of each result line records where it was computed.
