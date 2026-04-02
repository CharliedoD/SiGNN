"""Base dataset class and common utilities for dynamic graph datasets.

Provides the ``Dataset`` base class with shared functionality for loading
features, splitting nodes/edges, and iterating over temporal snapshots.
"""

import math
from collections import namedtuple
from typing import Optional

import numpy as np
import scipy.sparse as sp
import torch
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

Data = namedtuple("Data", ["x", "edge_index"])


def standard_normalization(arr):
    """Apply standard normalization across nodes for each time step.

    Parameters
    ----------
    arr : numpy.ndarray
        Feature array of shape ``(n_steps, n_node, n_dim)``.

    Returns
    -------
    numpy.ndarray
        Normalized feature array of the same shape.
    """
    n_steps, n_node, n_dim = arr.shape
    arr_norm = preprocessing.scale(
        np.reshape(arr, [n_steps, n_node * n_dim]), axis=1
    )
    arr_norm = np.reshape(arr_norm, [n_steps, n_node, n_dim])
    return arr_norm


def edges_to_adj(edges, num_nodes, undirected=True):
    """Convert edge arrays to a sparse adjacency matrix.

    Parameters
    ----------
    edges : tuple[numpy.ndarray, numpy.ndarray]
        Tuple of ``(row, col)`` index arrays.
    num_nodes : int
        Total number of nodes.
    undirected : bool
        If True, make the adjacency matrix symmetric.

    Returns
    -------
    scipy.sparse.csr_matrix
        Sparse adjacency matrix.
    """
    row, col = edges
    data = np.ones(len(row))
    adj = sp.csr_matrix((data, (row, col)), shape=(num_nodes, num_nodes))
    if undirected:
        adj = adj.maximum(adj.T)
    adj[adj > 1] = 1
    return adj


def merge_edges(edges, step=1):
    """Merge consecutive edge snapshots.

    Parameters
    ----------
    edges : list[numpy.ndarray]
        List of edge arrays for each time step.
    step : int
        Number of consecutive snapshots to merge.

    Returns
    -------
    list[numpy.ndarray]
        Merged edge snapshots.
    """
    if step == 1:
        return edges
    i = 0
    length = len(edges)
    out = []
    while i < length:
        e = edges[i : i + step]
        if len(e):
            out.append(np.hstack(e))
        i += step
    print(f"Edges merged from {len(edges)} timestamps to {len(out)} timestamps")
    return out


class Dataset:
    """Base class for dynamic graph datasets.

    Parameters
    ----------
    name : str or None
        Dataset name (used to locate data files).
    root : str
        Root directory containing dataset folders.
    """

    def __init__(self, name=None, root="./data"):
        self.name = name
        self.root = root
        self.x = None
        self.y = None
        self.num_features = None
        self.adj = []
        self.adj_evolve = []
        self.edges = []
        self.edges_evolve = []

    def _read_feature(self):
        """Read node features from a ``.npy`` file."""
        import os.path as osp
        filename = osp.join(self.root, self.name, f"{self.name}.npy")
        if osp.exists(filename):
            return np.load(filename)
        return None

    def split_nodes(
        self,
        train_size: float = 0.4,
        val_size: float = 0.0,
        test_size: float = 0.6,
        random_state: Optional[int] = None,
    ):
        """Split nodes into train, validation, and test sets.

        Parameters
        ----------
        train_size : float
            Fraction of nodes for training.
        val_size : float
            Fraction of nodes for validation.
        test_size : float
            Fraction of nodes for testing.
        random_state : int or None
            Random seed for reproducible splits.
        """
        val_size = 0.0 if val_size is None else val_size
        assert train_size + val_size + test_size <= 1.0

        y = self.y
        train_nodes, test_nodes = train_test_split(
            torch.arange(y.size(0)),
            train_size=train_size + val_size,
            test_size=test_size,
            random_state=random_state,
            stratify=y,
        )

        if val_size:
            train_nodes, val_nodes = train_test_split(
                train_nodes,
                train_size=train_size / (train_size + val_size),
                random_state=random_state,
                stratify=y[train_nodes],
            )
        else:
            val_nodes = None

        self.train_nodes = train_nodes
        self.val_nodes = val_nodes
        self.test_nodes = test_nodes

    def split_edges(
        self,
        train_stamp: float = 0.7,
        train_size: float = None,
        val_size: float = 0.1,
        test_size: float = 0.2,
        random_state: int = None,
    ):
        """Split edges temporally into train, validation, and test sets.

        Parameters
        ----------
        train_stamp : float
            Fraction (or absolute count) of timestamps for training.
        train_size : float or None
            If set, subsample this fraction from training edges.
        val_size : float
            Fraction of total edges for validation.
        test_size : float
            Fraction of total edges for testing.
        random_state : int or None
            Random seed.
        """
        if random_state is not None:
            torch.manual_seed(random_state)

        num_edges = self.edges[-1].size(-1)
        train_stamp = (
            train_stamp
            if train_stamp >= 1
            else math.ceil(len(self) * train_stamp)
        )

        train_edges = torch.LongTensor(np.hstack(self.edges_evolve[:train_stamp]))
        if train_size is not None:
            assert 0 < train_size < 1
            num_train = math.floor(train_size * num_edges)
            perm = torch.randperm(train_edges.size(1))[:num_train]
            train_edges = train_edges[:, perm]

        num_val = math.floor(val_size * num_edges)
        num_test = math.floor(test_size * num_edges)
        testing_edges = torch.LongTensor(np.hstack(self.edges_evolve[train_stamp:]))
        perm = torch.randperm(testing_edges.size(1))

        assert num_val + num_test <= testing_edges.size(1)

        self.train_stamp = train_stamp
        self.train_edges = train_edges
        self.val_edges = testing_edges[:, perm[:num_val]]
        self.test_edges = testing_edges[:, perm[num_val : num_val + num_test]]

    def __getitem__(self, time_index: int):
        x = self.x[time_index]
        edge_index = self.edges[time_index]
        snapshot = Data(x=x, edge_index=edge_index)
        return snapshot

    def __next__(self):
        if self.t < len(self):
            snapshot = self.__getitem__(self.t)
            self.t = self.t + 1
            return snapshot
        else:
            self.t = 0
            raise StopIteration

    def __iter__(self):
        self.t = 0
        return self

    def __len__(self):
        return len(self.adj)

    def __repr__(self):
        return self.name
