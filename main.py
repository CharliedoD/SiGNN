"""SiGNN: Spike-induced Graph Neural Network for Dynamic Graph Representation Learning.

This is the main entry point for training and evaluating the SiGNN model
on dynamic graph node classification tasks.
"""

import argparse
import time
import torch
import torch.nn as nn
from sklearn import metrics
from torch.utils.data import DataLoader
from tqdm import tqdm
from signn.datasets import DBLP, Tmall, Patent
from signn.model import SiGNN
from signn.utils import set_seed, tab_printer


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="SiGNN: Spike-induced Graph Neural Network")
    parser.add_argument("--dataset", type=str, default="DBLP", help="Dataset name: DBLP, Tmall, or Patent.")
    parser.add_argument("--sizes", type=int, nargs="+", default=[5, 2], help="Neighborhood sampling size for each layer.")
    parser.add_argument("--hids", type=int, nargs="+", default=[128, 64], help="Hidden units for each layer.")
    parser.add_argument("--aggr", type=str, default="mean", help="Aggregate function: 'mean' or 'sum'.")
    parser.add_argument("--sampler", type=str, default="sage", help="Neighborhood sampler: 'sage' or 'rw'.")
    parser.add_argument("--surrogate", type=str, default="arctan", help="Surrogate gradient: 'sigmoid', 'triangle', 'arctan', 'mg', 'super'.")
    parser.add_argument("--neuron", type=str, default="BLIF", help="Spiking neuron type: IF, LIF, or BLIF.")
    parser.add_argument("--batch_size", type=int, default=1024, help="Batch size for training.")
    parser.add_argument("--lr", type=float, default=5e-3, help="Learning rate.")
    parser.add_argument("--train_size", type=float, default=0.4, help="Fraction of nodes for training.")
    parser.add_argument("--alpha", type=float, default=1.0, help="Smooth factor for surrogate learning.")
    parser.add_argument("--p", type=float, default=0.5, help="Fraction of sampled neighborhoods from cumulative graph.")
    parser.add_argument("--dropout", type=float, default=0.6, help="Dropout probability.")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs.")
    parser.add_argument("--concat", action="store_true", help="Concatenate self and neighbor representations.")
    parser.add_argument("--seed", type=int, default=2024, help="Random seed.")
    parser.add_argument("--nchannels", type=int, default=3, help="Number of temporal aggregation channels.")
    parser.add_argument("--cuda", type=str, default="cuda:0", help="CUDA device.")
    parser.add_argument("--invth", type=float, default=1.0, help="Initial voltage threshold for spiking neurons.")

    try:
        args = parser.parse_args()
        args.test_size = 1 - args.train_size
        args.train_size = args.train_size - 0.05
        args.val_size = 0.05
        args.split_seed = 42
        tab_printer(args)
    except SystemExit:
        parser.print_help()
        exit(0)

    return args


def load_dataset(name):
    """Load a dataset by name.

    Parameters
    ----------
    name : str
        One of ``'dblp'``, ``'tmall'``, or ``'patent'``.

    Returns
    -------
    Dataset
        The loaded dataset instance.
    """
    name_lower = name.lower()
    if name_lower == "dblp":
        return DBLP()
    elif name_lower == "tmall":
        return Tmall()
    elif name_lower == "patent":
        return Patent()
    else:
        raise ValueError(
            f"{name} is invalid. Only datasets (DBLP, Tmall, Patent) are available."
        )


def train_epoch(model, train_loader, optimizer, loss_fn, y):
    """Run one training epoch.

    Parameters
    ----------
    model : SiGNN
        The model to train.
    train_loader : DataLoader
        Training data loader.
    optimizer : torch.optim.Optimizer
        Optimizer.
    loss_fn : nn.Module
        Loss function.
    y : torch.Tensor
        Label tensor on device.
    """
    model.train()
    for nodes in tqdm(train_loader, desc="Training"):
        optimizer.zero_grad()
        loss_fn(model(nodes), y[nodes]).backward()
        optimizer.step()


@torch.no_grad()
def evaluate(model, loader, y):
    """Evaluate model on a data loader.

    Parameters
    ----------
    model : SiGNN
        The model to evaluate.
    loader : DataLoader
        Evaluation data loader.
    y : torch.Tensor
        Label tensor on device.

    Returns
    -------
    tuple[float, float]
        Macro-F1 and Micro-F1 scores.
    """
    model.eval()
    logits = []
    labels = []
    for nodes in loader:
        logits.append(model(nodes))
        labels.append(y[nodes])
    logits = torch.cat(logits, dim=0).cpu()
    labels = torch.cat(labels, dim=0).cpu()
    preds = logits.argmax(1)
    macro_f1 = metrics.f1_score(labels, preds, average="macro")
    micro_f1 = metrics.f1_score(labels, preds, average="micro")
    return macro_f1, micro_f1


def main():
    """Main training and evaluation loop."""
    args = parse_args()
    assert len(args.hids) == len(args.sizes), \
        "Number of hidden dimensions must equal number of sampling sizes!"

    # Load dataset
    data = load_dataset(args.dataset)

    # Split nodes
    data.split_nodes(
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        random_state=args.split_seed,
    )

    set_seed(args.seed)

    device = torch.device(args.cuda)
    y = data.y.to(device)

    # Data loaders
    train_loader = DataLoader(
        data.train_nodes.tolist(),
        pin_memory=False,
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        data.test_nodes.tolist() if data.val_nodes is None else data.val_nodes.tolist(),
        pin_memory=False,
        batch_size=args.batch_size,
        shuffle=False,
    )
    test_loader = DataLoader(
        data.test_nodes.tolist(),
        pin_memory=False,
        batch_size=args.batch_size,
        shuffle=False,
    )

    # Build model
    model = SiGNN(
        dataset=data,
        in_features=data.num_features,
        out_features=data.num_classes,
        alpha=args.alpha,
        dropout=args.dropout,
        sampler=args.sampler,
        p=args.p,
        aggr=args.aggr,
        concat=args.concat,
        sizes=args.sizes,
        surrogate=args.surrogate,
        hids=args.hids,
        act=args.neuron,
        bias=True,
        nchannels=args.nchannels,
        invth=args.invth,
        device=device,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    # Training loop
    best_val_metric = 0
    best_test_metric = [0, 0]
    start = time.time()

    for epoch in range(1, args.epochs + 1):
        train_epoch(model, train_loader, optimizer, loss_fn, y)
        val_metric = evaluate(model, val_loader, y)
        test_metric = evaluate(model, test_loader, y)

        if val_metric[1] >= best_val_metric:
            best_val_metric = val_metric[1]
            best_test_metric = test_metric

        elapsed = time.time() - start
        print(
            f"Epoch: {epoch:03d}, "
            f"Val: {val_metric[1]:.4f}, "
            f"Test: {test_metric[1]:.4f}, "
            f"Best: Macro-{best_test_metric[0]:.4f}, "
            f"Micro-{best_test_metric[1]:.4f}, "
            f"Time elapsed {elapsed:.2f}s"
        )


if __name__ == "__main__":
    main()
