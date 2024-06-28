import torch
import torch.nn as nn
import torch.nn.functional as F
from SiGNN import neuron



class TAlayer(nn.Module):
    def __init__(self, in_features,
                 hids=[128, 64],
                 sizes=[5, 2],
                 v_threshold=1.0, 
                 alpha=1.0,
                 surrogate='triangle',
                 concat=False,
                 bias=False,
                 aggr='mean',
                 dropout=0.5):

        super().__init__()

        self.Agg_1 = Aggregator(in_features, hids[0], concat=concat, bias=bias, aggr=aggr)
        self.Agg_2 = Aggregator(hids[0], hids[1], concat=concat, bias=bias, aggr=aggr)
        
        self.snn_1 = neuron.BLIF(hid=hids[0], v_threshold=v_threshold, alpha=alpha, surrogate=surrogate)
        self.snn_2 = neuron.BLIF(hid=hids[1], v_threshold=v_threshold, alpha=alpha, surrogate=surrogate)
        
        self.dropout = nn.Dropout(dropout)
        self.sizes = sizes


    def forward(self, h, num_nodes):
    
        for i in range(len(self.sizes)):
            self_x = h[:-1]
            neigh_x = []
            for j, n_x in enumerate(h[1:]):
                neigh_x.append(n_x.view(-1, self.sizes[j], h[0].size(-1)))
            if i != len(self.sizes) - 1:
                out_t, out_x = self.Agg_1(self_x, neigh_x)
                out_t = self.snn_1(out_t)
                out_s = torch.mul(out_x, out_t)
                out_s = self.dropout(out_s)
                h = torch.split(out_s, num_nodes[:-(i + 1)])
            else:
                out_t, out_x = self.Agg_2(self_x, neigh_x)
                out_t = self.snn_2(out_t)
                out = torch.mul(out_x, out_t)
                out = self.dropout(out) 

        return out
        
               
class Aggregator(nn.Module):
    def __init__(self, in_features, out_features,
                 aggr='mean',
                 concat=False,
                 bias=False):

        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.concat = concat

        self.aggr = aggr
        self.aggregator = {'mean': torch.mean, 'sum': torch.sum}[aggr]

        self.lin_l = nn.Linear(in_features, out_features, bias=bias)
        self.lin_r = nn.Linear(in_features, out_features, bias=bias)
        
        self.lin_l_t = nn.Linear(in_features, out_features, bias=bias)
        self.lin_r_t = nn.Linear(in_features, out_features, bias=bias)
        
        self.act = nn.Sigmoid()
        self.relu = nn.ReLU()



    def forward(self, x, neigh):

        if not isinstance(neigh, torch.Tensor):
            neigh_h = []
            for i in range(len(neigh)):
                h = self.aggregator(neigh[i], dim=1)
                neigh_h.append(h)
            neigh = torch.cat(neigh_h, dim=0)
             
        if not isinstance(x, torch.Tensor):
            x = torch.cat(x, dim=0)
            
        
        self_x = self.lin_l(x)
        self_s = self.lin_l_t(x)
        neigh_x = self.lin_r(neigh)
        neigh_s = self.lin_r_t(neigh)
        
        out_t =  self_s + neigh_s
        out_x =  self.act(self_x + neigh_x)

        return out_t, out_x

    def __repr__(self):
        return f"{self.__class__.__name__}({self.in_features}, {self.out_features}, aggr={self.aggr})"
        



