"""Point-in-time audit of TGB node affinity prediction (tgbn-*), in pure NumPy.

Main entry points:
    NodeAffinityStream: the TGB evaluation loop with the two switches of the leakage-free
        protocol (strict edge cutoff, delayed label release); Algorithm 1 of the paper.
    TGBLabels, load_tgb: loading the TGB datasets through py-tgb.
    AVAILABILITY_DELAY: availability delay of each dataset, recovered by the label-window probe.
"""
from .data import AVAILABILITY_DELAY, TGB_MA_WINDOW, TGBLabels, load_tgb, median_label_interval
from .metrics import idcg_rows, ndcg_rows
from .protocol import NodeAffinityStream

__version__ = "1.0.0"
__all__ = ["AVAILABILITY_DELAY", "TGB_MA_WINDOW", "TGBLabels", "load_tgb", "median_label_interval",
           "idcg_rows", "ndcg_rows", "NodeAffinityStream"]
