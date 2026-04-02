"""DBLP dynamic graph dataset."""

import os.path as osp
from collections import defaultdict

import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder

from signn.datasets.base import Dataset, edges_to_adj, standard_normalization


class DBLP(Dataset):
    """DBLP co-authorship dynamic graph dataset.

    Parameters
    ----------
    root : str
        Root directory containing the ``dblp/`` data folder.
    normalize : bool
        Whether to apply standard normalization to features.
    """

    def __init__(self, root="./data", normalize=True):
        super().__init__(name="dblp", root=root)
        edges_evolve, self.num_nodes = self._read_graph()
        x = self._read_feature()
        y = self._read_label()

        if x is not None:
            if normalize:
                x = standard_normalization(x)
            self.num_features = x.shape[-1]
            self.x = torch.FloatTensor(x)

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

        self.y = torch.LongTensor(y)

    def _read_graph(self):
        filename = osp.join(self.root, self.name, f"{self.name}.txt")
        d = defaultdict(list)
        num_nodes = 0
        with open(filename) as f:
            for line in f:
                src, dst, t = line.strip().split()
                src, dst = int(src), int(dst)
                d[t].append((src, dst))
                num_nodes = max(num_nodes, src)
                num_nodes = max(num_nodes, dst)
        num_nodes += 1

        edges = []
        for time in sorted(d):
            row, col = zip(*d[time])
            edge_now = np.vstack([row, col])
            edges.append(edge_now)
        return edges, num_nodes

    def _read_label(self):
        filename = osp.join(self.root, self.name, "node2label.txt")
        nodes = []
        labels = []
        with open(filename) as f:
            for line in f:
                node, label = line.strip().split()
                nodes.append(int(node))
                labels.append(label)

        nodes = np.array(nodes)
        labels = LabelEncoder().fit_transform(labels)
        assert np.allclose(nodes, np.arange(nodes.size))
        return labels
