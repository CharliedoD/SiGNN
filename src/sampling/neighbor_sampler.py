"""Neighborhood sampling utilities.

This module provides graph neighborhood samplers used during
mini-batch training, including a standard sampler (via C++ extension)
and a random walk sampler (via ``torch_cluster``).
"""

import numpy as np
import scipy.sparse as sp
import torch

from sample_neighbor import sample_neighbor_cpu

try:
    import torch_cluster
except ImportError:
    torch_cluster = None


class Sampler:
    """Standard neighborhood sampler using C++ extension.

    Parameters
    ----------
    adj_matrix : scipy.sparse.csr_matrix
        Adjacency matrix in CSR format.
    """

    def __init__(self, adj_matrix: sp.csr_matrix):
        self.rowptr = torch.LongTensor(adj_matrix.indptr)
        self.col = torch.LongTensor(adj_matrix.indices)

    def __call__(self, nodes, size, replace=True):
        nodes = nodes.to("cpu")
        nbr = sample_neighbor_cpu(self.rowptr, self.col, nodes, size, replace)
        return nbr


class RandomWalkSampler:
    """Random walk based neighborhood sampler.

    Uses ``torch_cluster.random_walk`` for sampling.

    Parameters
    ----------
    adj_matrix : scipy.sparse.csr_matrix
        Adjacency matrix in CSR format.
    p : float
        Return parameter for random walk.
    q : float
        In-out parameter for random walk.
    """

    def __init__(self, adj_matrix: sp.csr_matrix, p: float = 1.0, q: float = 1.0):
        self.rowptr = torch.LongTensor(adj_matrix.indptr)
        self.col = torch.LongTensor(adj_matrix.indices)
        self.p = p
        self.q = q
        assert torch_cluster, "Please install 'torch_cluster' first."

    def __call__(self, nodes, size, replace=True):
        nbr = torch.ops.torch_cluster.random_walk(
            self.rowptr, self.col, nodes, size, self.p, self.q
        )[0][:, 1:]
        return nbr


def eliminate_selfloops(adj_matrix):
    """Remove self-loops from an adjacency matrix.

    Parameters
    ----------
    adj_matrix : scipy.sparse matrix or numpy.ndarray
        Input adjacency matrix.

    Returns
    -------
    scipy.sparse matrix or numpy.ndarray
        Adjacency matrix without self-loops.
    """
    if sp.issparse(adj_matrix):
        adj_matrix = adj_matrix - sp.diags(adj_matrix.diagonal(), format="csr")
        adj_matrix.eliminate_zeros()
    else:
        adj_matrix = adj_matrix - np.diag(adj_matrix)
    return adj_matrix


def add_selfloops(adj_matrix: sp.csr_matrix):
    """Add self-loops to an adjacency matrix.

    Parameters
    ----------
    adj_matrix : scipy.sparse.csr_matrix
        Input adjacency matrix.

    Returns
    -------
    scipy.sparse.csr_matrix
        Adjacency matrix with self-loops.
    """
    adj_matrix = eliminate_selfloops(adj_matrix)
    return adj_matrix + sp.eye(adj_matrix.shape[0], dtype=adj_matrix.dtype, format="csr")
