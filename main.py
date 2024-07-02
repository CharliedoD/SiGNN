import argparse
import time

import torch
import torch.nn as nn
from sklearn import metrics
from torch.utils.data import DataLoader
from tqdm import tqdm
from SiGNN import dataset, neuron
from SiGNN.layers import TAlayer
from SiGNN.utils import (RandomWalkSampler, Sampler, add_selfloops,
                            set_seed, tab_printer)



class SiGNN(nn.Module):
    def __init__(self, in_features, out_features, hids=[128, 64], alpha=1.0, p=0.5,
                 dropout=0.7, bias=True, aggr='mean', sampler='sage',
                 surrogate='triangle', sizes=[5, 2], concat=False, act='BLIF', nchannels=3, invth=1):

        super().__init__()

        if sampler == 'rw':
            self.sampler = [RandomWalkSampler(
                add_selfloops(adj_matrix)) for adj_matrix in data.adj]
            self.sampler_t = [RandomWalkSampler(add_selfloops(
                adj_matrix)) for adj_matrix in data.adj_evolve]
        elif sampler == 'sage':
            self.sampler = [Sampler(add_selfloops(adj_matrix))
                            for adj_matrix in data.adj]
            self.sampler_t = [Sampler(add_selfloops(adj_matrix))
                              for adj_matrix in data.adj_evolve]
        else:
            raise ValueError(sampler)

        TA_layers = nn.ModuleList()
        for i in range(nchannels):
            TA_layers.append(TAlayer(in_features, hids=hids, sizes=sizes, v_threshold=invth, 
                                     alpha=alpha, surrogate=surrogate, concat=concat, bias=bias, aggr=aggr, dropout=dropout))


        num_steps = len(data)

        self.TA_layers = TA_layers
        self.sizes = sizes
        self.p = p
        self.MTGagg = nn.Linear(hids[-1], out_features)
        self.pooling_1 = nn.Conv1d(groups=hids[-1], in_channels=hids[-1], out_channels=hids[-1], kernel_size=num_steps)
        self.pooling_2 = nn.Conv1d(groups=hids[-1], in_channels=hids[-1], out_channels=hids[-1], kernel_size=(num_steps//2 + num_steps%2))
        self.pooling_3 = nn.Conv1d(groups=hids[-1], in_channels=hids[-1], out_channels=hids[-1], kernel_size=(num_steps//3 + num_steps%3))

        
    def encode(self, nodes):
        embeddings1 = []
        embeddings2 = []
        embeddings3 = []
        sizes = self.sizes
        for time_step in range(len(data)):
            snapshot = data[time_step]
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
                o1 = self.TA_layers[0](h, num_nodes)
                embeddings1.append(o1)
            
            if time_step % 2 == 0:
                o2 = self.TA_layers[1](h, num_nodes)
                embeddings2.append(o2)

            if (time_step + 1) % 3 == 0 and len(data)==27:  
                o3 = self.TA_layers[2](h, num_nodes)           
                embeddings3.append(o3)
            elif time_step % 3 == 0 and len(data)!=27:
                o3 = self.TA_layers[2](h, num_nodes)            
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
        embeddings = self.MTGagg(embeddings)
        
        neuron.reset_net(self)
        return embeddings

    def forward(self, nodes):
        embeddings = self.encode(nodes)
        return embeddings
        
        
        
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", nargs="?", default="DBLP", help="Datasets (DBLP, Tmall, Patent). (default: DBLP)")
parser.add_argument('--sizes', type=int, nargs='+', default=[5, 2], help='Neighborhood sampling size for each layer. (default: [5, 2])')
parser.add_argument('--hids', type=int, nargs='+', default=[128, 64], help='Hidden units for each layer. (default: [128, 48])')
parser.add_argument("--aggr", nargs="?", default="mean", help="Aggregate function ('mean', 'sum'). (default: 'mean')")
parser.add_argument("--sampler", nargs="?", default="sage", help="Neighborhood Sampler")
parser.add_argument("--surrogate", nargs="?", default="arctan", help="Surrogate function ('sigmoid', 'triangle', 'arctan', 'mg', 'super'). (default: 'sigmoid')")
parser.add_argument("--neuron", nargs="?", default="BLIF", help="Spiking neuron used for training. (IF, LIF, BLIF). (default: LIF")
parser.add_argument('--batch_size', type=int, default=1024, help='Batch size for training. (default: 1024)')
parser.add_argument('--lr', type=float, default=5e-3,  help='Learning rate for training. (default: 5e-3)')
parser.add_argument('--train_size', type=float, default=0.4, help='Ratio of nodes for training. (default: 0.4)')
parser.add_argument('--alpha', type=float, default=1.0, help='Smooth factor for surrogate learning. (default: 1.0)')
parser.add_argument('--p', type=float, default=0.5, help='Percentage of sampled neighborhoods for g_t. (default: 0.5)')
parser.add_argument('--dropout', type=float, default=0.6, help='Dropout probability. (default: 0.6)')
parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs. (default: 100)')
parser.add_argument('--concat', action='store_true', help='Whether to concat node representation and neighborhood representations. (default: False)')
parser.add_argument('--seed', type=int, default=2024, help='Random seed for model. (default: 2024)')
parser.add_argument('--nchannels', type=int, default=3, help='num of LIF models')
parser.add_argument('--cuda', type=str, default='cuda:0', help='which card')
parser.add_argument('--invth', type=float, default=1.0, help='Hidden units for each layer. (default: [128, 16])')


try:
    args = parser.parse_args()
    args.test_size = 1 - args.train_size
    args.train_size = args.train_size - 0.05
    args.val_size = 0.05
    args.split_seed = 42
    tab_printer(args)
except:
    parser.print_help()
    exit(0)

assert len(args.hids) == len(args.sizes), "must be equal!"

if args.dataset.lower() == "dblp":
    data = dataset.DBLP()
elif args.dataset.lower() == "tmall":
    data = dataset.Tmall()
elif args.dataset.lower() == "patent":
    data = dataset.Patent()
else:
    raise ValueError(
        f"{args.dataset} is invalid. Only datasets (dblp, tmall, patent) are available.")

# train:val:test
data.split_nodes(train_size=args.train_size, val_size=args.val_size,
                 test_size=args.test_size, random_state=args.split_seed)

set_seed(args.seed)

device = torch.device(args.cuda)
y = data.y.to(device)

train_loader = DataLoader(data.train_nodes.tolist(), pin_memory=False, batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(data.test_nodes.tolist() if data.val_nodes is None else data.val_nodes.tolist(),
                        pin_memory=False, batch_size=args.batch_size, shuffle=False)
test_loader = DataLoader(data.test_nodes.tolist(), pin_memory=False, batch_size=args.batch_size, shuffle=False)

model = SiGNN(data.num_features, data.num_classes, alpha=args.alpha,
                 dropout=args.dropout, sampler=args.sampler, p=args.p,
                 aggr=args.aggr, concat=args.concat, sizes=args.sizes, surrogate=args.surrogate,
                 hids=args.hids, act=args.neuron, bias=True, nchannels=args.nchannels, invth=args.invth).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
loss_fn = nn.CrossEntropyLoss()


def train():
    model.train()
    for nodes in tqdm(train_loader, desc='Training'):
        optimizer.zero_grad()
        loss_fn(model(nodes), y[nodes]).backward()
        optimizer.step()


@torch.no_grad()
def test(loader):
    model.eval()
    logits = []
    labels = []
    for nodes in loader:
        logits.append(model(nodes))
        labels.append(y[nodes])
    logits = torch.cat(logits, dim=0).cpu()
    labels = torch.cat(labels, dim=0).cpu()
    logits = logits.argmax(1)
    metric_macro = metrics.f1_score(labels, logits, average='macro')
    metric_micro = metrics.f1_score(labels, logits, average='micro')
    return metric_macro, metric_micro


best_val_metric = test_metric = 0
best_test_metric = [0, 0]
start = time.time()
for epoch in range(1, args.epochs + 1):
    train()
    val_metric, test_metric = test(val_loader), test(test_loader)
    if val_metric[1] >= best_val_metric:
        best_val_metric = val_metric[1]
        best_test_metric = test_metric
    end = time.time()
    print(
        f'Epoch: {epoch:03d}, Val: {val_metric[1]:.4f}, Test: {test_metric[1]:.4f}, Best: Macro-{best_test_metric[0]:.4f}, Micro-{best_test_metric[1]:.4f}, Time elapsed {end - start:.2f}s')
