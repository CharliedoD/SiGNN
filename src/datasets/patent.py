"""Patent dynamic graph dataset."""

import os.path as osp
from collections import defaultdict

import numpy as np
import torch
from tqdm import tqdm

from src.datasets.base import Dataset, edges_to_adj, merge_edges, standard_normalization


class Patent(Dataset):
    """US Patent citation dynamic graph dataset.

    Parameters
    ----------
    root : str
        Root directory containing the ``patent/`` data folder.
    normalize : bool
        Whether to apply standard normalization to features.
    """

    def __init__(self, root="./data", normalize=True):
        super().__init__(name="patent", root=root)
        edges_evolve = self._read_graph()
        y = self._read_label()
        edges_evolve = merge_edges(edges_evolve, step=2)
        x = self._read_feature()

        if x is not None:
            if normalize:
                x = standard_normalization(x)
            self.num_features = x.shape[-1]
            self.x = torch.FloatTensor(x)

        self.num_nodes = y.size
        self.num_features = x.shape[-1]
        self.num_classes = y.max() + 1

        edges = [edges_evolve[0]]
        for e_now in edges_evolve[1:]:
            e_last = edges[-1]
            edges.append(np.hstack([e_last, e_now]))

        self.adj = [edges_to_adj(edge, num_nodes=self.num_nodes) for edge in edges]
        self.adj_evolve = [
            edges_to_adj(edge, num_nodes=self.num_nodes) for edge in edges_evolve
        ]
        self.edges = [torch.LongTensor(edge) for edge in edges]
        self.edges_evolve = edges_evolve

        self.x = torch.FloatTensor(x)
        self.y = torch.LongTensor(y)

    def _read_graph(self):
        filename = osp.join(self.root, self.name, f"{self.name}_edges.json")
        time_edges = defaultdict(list)
        with open(filename) as f:
            for line in tqdm(f, desc="Loading patent edges"):
                src, dst, date, _, _ = eval(line)
                date = date // 1e4
                time_edges[date].append((src, dst))

        edges = []
        for time in sorted(time_edges):
            edges.append(np.transpose(time_edges[time]))
        return edges

    def _read_label(self):
        filename = osp.join(self.root, self.name, f"{self.name}_nodes.json")
        labels = []
        with open(filename) as f:
            for line in tqdm(f, desc="Loading patent nodes"):
                node, _, date, label = eval(line)
                date = date // 1e4
                labels.append(label - 1)
        labels = np.array(labels)
        return labels
