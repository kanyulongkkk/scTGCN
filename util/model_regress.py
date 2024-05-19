

import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, GATv2Conv
from torch.nn import Linear


class Net_encoder(nn.Module):
    def __init__(self, input_size):
        super(Net_encoder, self).__init__()
        self.input_size = input_size
        self.k = 64
        self.f = 64

        self.encoder = nn.Sequential(
            nn.Linear(self.input_size, 64),
            nn.ReLU(),
            nn.LayerNorm(64)
        )

    def forward(self, data):
        data = data.float().view(-1, self.input_size)
        embedding = self.encoder(data)
        
        return embedding



class Net_cell(nn.Module):
    def __init__(self, num_of_class):
        super(Net_cell, self).__init__()
        self.cell = nn.Sequential(
            nn.Linear(64, num_of_class)
        )

        self.cell = GNN(num_of_class)
    def forward(self, embedding, edge_index):
        cell_prediction = self.cell(embedding,edge_index)

        return cell_prediction

class GNN(torch.nn.Module):
    def __init__(self, num_of_class) :
        super(GNN, self).__init__()
        self.gcn = GCNConv(64,32)
        self.out = Linear(32, num_of_class)
    
    def forward(self, x ,edge_index):
        print(len(edge_index))
        h = self.gcn(x, edge_index)
        embedding = torch.relu(h)
        z = self.out(embedding)
        return z
 
