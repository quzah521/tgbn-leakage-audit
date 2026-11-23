"""Run the training-free heuristics under the TGB loop and the leakage-free protocol.

Usage (py-tgb downloads the datasets on first use; outputs go to results/heuristics/):
    python scripts/run_heuristics.py --name tgbn-trade  --cutoff strict,protocol --runs real
    python scripts/run_heuristics.py --name tgbn-genre  --cutoff strict --runs real --label_delays 0,avail
    python scripts/run_heuristics.py --name tgbn-reddit --cutoff strict --runs real --weight one \
        --label_delays 0,avail

Main options:
    --cutoff        protocol = TGB reference loop; strict = only edges with t < t_q before predicting t_q
    --label_delays  a label stamped t_q becomes an input at t_q + delay * P; 'avail' = the
                    availability delay of the dataset (genre 7 days, reddit and token 6 days)
    --runs          real and/or placebo (within-period shuffle, Section 6.3)
    --weight        abs = |edge feature| (as NAVIS), one = count edges
Run with -h for all options.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from tgbn_audit.pipeline import main  # noqa: E402

if __name__ == "__main__":
    main()
