"""
Replay the loss step that crashed in navis_audit.py --debug_checks, in seconds instead of hours.

Usage (from the NAVIS/NAVIS directory, next to regularization.py and ltr_loss.py):
    python debug_repro.py debug_failing_batch.pt

It prints:
  1. versions and a short description of the saved batch (shapes, ties, all-zero rows, binary path or not);
  2. whether the same computation passes on CPU;
  3. on GPU: the index ranges returned by every topk / argsort inside the loss, before anything is gathered;
  4. finally the real GPU loss with CUDA_LAUNCH_BLOCKING=1, so a crash points at the exact line.
Used to diagnose the CUDA assert described in navis/README.md.
"""
import os
import sys

os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")  # must be set before CUDA is initialised

import torch  # noqa: E402

from ltr_loss import LambdaNDCGLoss2, rank_by_score  # noqa: E402  (ships with navis_audit)
from regularization import HindgeRegularization  # noqa: E402

K = 20
DELTA = 0.001
path = sys.argv[1] if len(sys.argv) > 1 else "debug_failing_batch.pt"
try:
    dbg = torch.load(path, map_location="cpu", weights_only=False)
except TypeError:  # older torch without weights_only
    dbg = torch.load(path, map_location="cpu")
print(f"[repro] torch={torch.__version__} cuda={torch.version.cuda} "
      f"(run used torch={dbg.get('torch')} cuda={dbg.get('cuda')})")
print(f"[repro] crashed stage={dbg.get('stage')} epoch={dbg.get('epoch')} label_t={dbg.get('label_t')}")

P, Y = dbg["preds_full"].float(), dbg["labels_full"].float()


def describe(name, x):
    fin = torch.isfinite(x)
    print(f"  {name}: shape={tuple(x.shape)} dtype={x.dtype} finite={bool(fin.all())} "
          f"min={x[fin].min().item() if fin.any() else 'nan'} max={x[fin].max().item() if fin.any() else 'nan'}")


print("[repro] full-width inputs")
describe("preds", P)
describe("labels", Y)
rows_zero_label = int((Y.sum(1) == 0).sum())
rows_const_pred = int((P.max(1).values == P.min(1).values).sum())
print(f"  rows with all-zero labels={rows_zero_label}, rows whose preds are all equal={rows_const_pred}, "
      f"nonzero labels per row: min={int((Y > 0).sum(1).min())} max={int((Y > 0).sum(1).max())}")


def chain(P, Y, dev):
    P, Y = P.to(dev), Y.to(dev)
    _, pi = torch.topk(P, K)
    _, li = torch.topk(Y, K)
    ti = torch.unique(torch.cat([pi, li], dim=1), dim=1)
    return torch.gather(P, 1, ti), torch.gather(Y, 1, ti), ti


P2, Y2, ti = chain(P, Y, "cpu")
U = P2.shape[1]
binary = bool(torch.all((Y2 == 0) | (Y2 == 1)))
n = torch.count_nonzero(Y2, dim=1).clamp(max=K)
print(f"[repro] after gather: U={U} columns, binary_path={binary}, n: min={int(n.min())} max={int(n.max())}, "
      f"rows with n=0: {int((n == 0).sum())}")

crit, reg = LambdaNDCGLoss2(), HindgeRegularization(top_k=K, delta=DELTA)
try:
    with torch.no_grad():
        a = crit(P2, Y2, n).mean()
        b = reg(P2, Y2)
    print(f"[repro] CPU: OK (lambda={a.item():.4f}, reg={b.item():.4f})")
except Exception as e:  # noqa: BLE001
    print(f"[repro] CPU: FAILED -> {type(e).__name__}: {e}")
    raise

if not torch.cuda.is_available():
    print("[repro] no GPU here; stop.")
    sys.exit(0)

dev = "cuda"
# GPU, step by step: check every index tensor before it is used
if dbg.get("stage") in ("loss", "regularization", "backward") and "preds" in dbg:
    Pg, Yg = dbg["preds"].to(dev), dbg["labels"].to(dev)
    print("[repro] GPU uses the gathered tensors saved at crash time")
else:
    Pg, Yg, _ = chain(P, Y, dev)
U = Pg.shape[1]


def check(name, idx, upper):
    lo, hi = int(idx.min()), int(idx.max())
    flag = "OK" if 0 <= lo and hi < upper else "OUT OF RANGE"
    print(f"  {name}: min={lo} max={hi} allowed=[0,{upper - 1}]  {flag}")


print("[repro] GPU index ranges")
ng = torch.count_nonzero(Yg, dim=1).clamp(max=K)
check("reg pred_top (topk scores)", torch.topk(Pg, K, dim=1).indices, U)
if bool(torch.all((Yg == 0) | (Yg == 1))):
    ms = torch.where(Yg == 1, Pg, torch.full_like(Pg, -float("inf")))
else:
    ms = torch.where(Yg > 0, Yg, torch.full_like(Yg, -float("inf")))
check("reg true_top (topk masked)", torch.topk(ms, K, dim=1).indices, U)
check("lambda ranking (argsort scores)", rank_by_score(Pg, ng), U)
check("lambda max_dcg ranking (argsort labels)", rank_by_score(Yg.double(), ng), U)

print("[repro] GPU: running the real loss 50 times with CUDA_LAUNCH_BLOCKING=1 ...")
for i in range(50):
    Pi = Pg.clone().requires_grad_(True)
    loss = crit(Pi, Yg, ng).mean() + reg(Pi, Yg)
    loss.backward()
    torch.cuda.synchronize()
print("[repro] GPU: OK on all 50 tries (the crash did not reproduce from these inputs)")
