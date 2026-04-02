"""Tmall dynamic graph dataset."""

import os.path as osp
from collections import defaultdict

import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from signn.datasets.base import Dataset, edges_to_adj, merge_edges, standard_normalization


class Tmall(Dataset):
    """Tmall user-item interaction dynamic graph dataset.

    Parameters
    ----------
    root : str
        Root directory containing the ``tmall/`` data folder.
    normalize : bool
        Whether to apply standard normalization to features.
    """

    def __init__(self, root="./data", normalize=True):
        super().__init__(name="tmall", root=root)
        edges_evolve, self.num_nodes = self._read_graph()
        x = self._read_feature()

        y, labeled_nodes = self._read_label()

        # Reindexing: place labeled nodes first
        others = set(range(self.num_nodes)) - set(labeled_nodes.tolist())
        new_index = np.hstack([labeled_nodes, list(others)])
        whole_nodes = np.arange(self.num_nodes)
        mapping_dict = dict(zip(new_index, whole_nodes))
        mapping = np.vectorize(mapping_dict.get)(whole_nodes)
        edges_evolve = [mapping[e] for e in edges_evolve]

        edges_evolve = merge_edges(edges_evolve, step=10)

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

        self.mapping = mapping
        self.y = torch.LongTensor(y)

    def _read_graph(self):
        filename = osp.join(self.root, self.name, f"{self.name}.txt")
        d = defaultdict(list)
        num_nodes = 0
        with open(filename) as f:
            for line in tqdm(f, desc="Loading edges"):
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
            for line in tqdm(f, desc="Loading nodes"):
                node, label = line.strip().split()
                nodes.append(int(node))
                labels.append(label)

        labeled_nodes = np.array(nodes)
        labels = LabelEncoder().fit_transform(labels)
        return labels, labeled_nodes
