"""Utility functions for SiGNN.

Includes random seed setting, argument printing, and adjacency matrix
manipulation helpers.
"""

import numpy as np
import torch
from texttable import Texttable


def set_seed(seed):
    """Set random seed for reproducibility.

    Parameters
    ----------
    seed : int
        Random seed value.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def tab_printer(args):
    """Print arguments in a tabular format.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments.

    Note
    ----
    Requires the ``texttable`` package. Install via ``pip install texttable``.
    """
    args = vars(args)
    keys = sorted(args.keys())
    t = Texttable()
    t.add_rows(
        [["Parameter", "Value"]]
        + [[k.replace("_", " "), args[k]] for k in keys]
    )
    print(t.draw())
