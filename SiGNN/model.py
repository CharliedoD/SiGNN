"""SiGNN: Spike-induced Graph Neural Network for Dynamic Graph Representation Learning.

This module defines the main ``SiGNN`` model that combines multi-granularity
temporal aggregation with spiking neural networks for dynamic node classification.
"""

import torch
import torch.nn as nn

from signn.layers import TALayer
from signn.neuron import reset_net
from signn.sampling import Sampler, RandomWalkSampler, add_selfloops


class SiGNN(nn.Module):
    """SiGNN model for dynamic graph node classification.

    Parameters
    ----------
    dataset : Dataset
        A dynamic graph dataset instance (provides adjacency matrices, features, etc.).
    in_features : int
        Input feature dimension.
    out_features : int
        Number of output classes.
    hids : list[int]
        Hidden dimensions for each aggregation hop.
    alpha : float
        Surrogate gradient smoothing factor.
    p : float
        Proportion of neighbors sampled from cumulative graph vs. evolving graph.
    dropout : float
        Dropout probability.
    bias : bool
        Whether to use bias in linear layers.
    aggr : str
        Aggregation method (``'mean'`` or ``'sum'``).
    sampler : str
        Sampler type: ``'sage'`` for standard sampling, ``'rw'`` for random walk.
    surrogate : str
        Surrogate gradient function name.
    sizes : list[int]
        Neighborhood sample sizes for each hop.
    concat : bool
        Whether to concatenate self and neighbor representations.
    act : str
        Neuron type (reserved, currently uses BLIF).
    nchannels : int
        Number of temporal aggregation channels.
    invth : float
        Initial voltage threshold for spiking neurons.
    device : torch.device
        Device to place tensors on.
    """

    def __init__(self, dataset, in_features, out_features, hids=(128, 64),
                 alpha=1.0, p=0.5, dropout=0.7, bias=True, aggr="mean",
                 sampler="sage", surrogate="triangle", sizes=(5, 2),
                 concat=False, act="BLIF", nchannels=3, invth=1,
                 device=torch.device("cpu")):
        super().__init__()

        self.dataset = dataset
        self.device = device

        if sampler == "rw":
            self.sampler = [
                RandomWalkSampler(add_selfloops(adj_matrix))
                for adj_matrix in dataset.adj
            ]
            self.sampler_t = [
                RandomWalkSampler(add_selfloops(adj_matrix))
                for adj_matrix in dataset.adj_evolve
            ]
        elif sampler == "sage":
            self.sampler = [
                Sampler(add_selfloops(adj_matrix))
                for adj_matrix in dataset.adj
            ]
            self.sampler_t = [
                Sampler(add_selfloops(adj_matrix))
                for adj_matrix in dataset.adj_evolve
            ]
        else:
            raise ValueError(f"Unknown sampler type: {sampler}")

        ta_layers = nn.ModuleList()
        for _ in range(nchannels):
            ta_layers.append(
                TALayer(
                    in_features,
                    hids=hids,
                    sizes=sizes,
                    v_threshold=invth,
                    alpha=alpha,
                    surrogate=surrogate,
                    concat=concat,
                    bias=bias,
                    aggr=aggr,
                    dropout=dropout,
                )
            )

        num_steps = len(dataset)

        self.ta_layers = ta_layers
        self.sizes = sizes
        self.p = p
        self.mtg_agg = nn.Linear(hids[-1], out_features)
        self.pooling_1 = nn.Conv1d(
            groups=hids[-1],
            in_channels=hids[-1],
            out_channels=hids[-1],
            kernel_size=num_steps,
        )
        self.pooling_2 = nn.Conv1d(
            groups=hids[-1],
            in_channels=hids[-1],
            out_channels=hids[-1],
            kernel_size=(num_steps // 2 + num_steps % 2),
        )
        self.pooling_3 = nn.Conv1d(
            groups=hids[-1],
            in_channels=hids[-1],
            out_channels=hids[-1],
            kernel_size=(num_steps // 3 + num_steps % 3),
        )

    def encode(self, nodes):
        """Encode nodes into embeddings using multi-granularity temporal aggregation.

        Parameters
        ----------
        nodes : torch.Tensor
            Node indices to encode.

        Returns
        -------
        torch.Tensor
            Node embeddings of shape ``(num_nodes, out_features)``.
        """
        dataset = self.dataset
        device = self.device
        embeddings1 = []
        embeddings2 = []
        embeddings3 = []
        sizes = self.sizes

        for time_step in range(len(dataset)):
            snapshot = dataset[time_step]
            sampler = self.sampler[time_step]
            sampler_t = self.sampler_t[time_step]
            x = snapshot.x
            h = [x[nodes].to(device)]
            num_nodes = [nodes.size(0)]
            nbr = nodes

            for size in sizes:
                size_1 = max(int(size * self.p), 1)
                size_2 = size - size_1
                if size_2 > 0:
                    nbr_1 = sampler(nbr, size_1).view(nbr.size(0), size_1)
                    nbr_2 = sampler_t(nbr, size_2).view(nbr.size(0), size_2)
                    nbr = torch.cat([nbr_1, nbr_2], dim=1).flatten()
                else:
                    nbr = sampler(nbr, size_1).view(-1)
                num_nodes.append(nbr.size(0))
                h.append(x[nbr].to(device))

            if time_step % 1 == 0:
                o1 = self.ta_layers[0](h, num_nodes)
                embeddings1.append(o1)

            if time_step % 2 == 0:
                o2 = self.ta_layers[1](h, num_nodes)
                embeddings2.append(o2)

            if (time_step + 1) % 3 == 0 and len(dataset) == 27:
                o3 = self.ta_layers[2](h, num_nodes)
                embeddings3.append(o3)
            elif time_step % 3 == 0 and len(dataset) != 27:
                o3 = self.ta_layers[2](h, num_nodes)
                embeddings3.append(o3)

        emb1 = torch.stack(embeddings1)
        emb1 = emb1.permute(1, 2, 0)
        emb1 = self.pooling_1(emb1).squeeze(dim=2)

        emb2 = torch.stack(embeddings2)
        emb2 = emb2.permute(1, 2, 0)
        emb2 = self.pooling_2(emb2).squeeze(dim=2)

        emb3 = torch.stack(embeddings3)
        emb3 = emb3.permute(1, 2, 0)
        emb3 = self.pooling_3(emb3).squeeze(dim=2)

        embeddings = torch.stack([emb1, emb2, emb3], dim=0)
        embeddings = torch.mean(embeddings, dim=0)
        embeddings = self.mtg_agg(embeddings)

        reset_net(self)
        return embeddings

    def forward(self, nodes):
        """Forward pass: encode nodes and return class logits.

        Parameters
        ----------
        nodes : torch.Tensor
            Node indices.

        Returns
        -------
        torch.Tensor
            Class logits of shape ``(num_nodes, out_features)``.
        """
        return self.encode(nodes)
