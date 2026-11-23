# Third-party code

| component | where | license | how it is used |
|---|---|---|---|
| TGB (`py-tgb`) | installed with pip | MIT, Copyright (c) 2023 TGB Team | datasets, label pointer, evaluator semantics; not redistributed |
| pytorchltr `LambdaNDCGLoss2` | `navis/ltr_loss.py` | MIT, Copyright (c) 2020 Rolf Jagerman (full text: `navis/LICENSE_pytorchltr.md`) | vendored, with the one-line exponent clamp that NAVIS's instructions ask for |
| NAVIS | `navis/navis_audit.patch` | **no license file in the NAVIS repository** | only the patch is distributed; it is applied to the user's own NAVIS checkout (commit 2b503ef) by `navis/apply_patch.py`. `regularization.py` is taken from that checkout |

The TGB `MovingAverage` heuristic is reimplemented in `tgbn_audit/heuristics.py` (an EMA channel);
`tests/test_audit.py` contains a ten-line reference copy of its update rule to check equivalence.
