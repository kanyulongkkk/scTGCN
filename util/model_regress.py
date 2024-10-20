import torch
import torch.nn as nn
from torch_geometric.nn import GATv2Conv
from torch.nn import Linear

class Net_encoder(nn.Module):
    def __init__(self, input_size):
        super(Net_encoder, self).__init__()
        self.input_size = input_size
        self.encoder = nn.Sequential(
            nn.Linear(self.input_size, 64),
            nn.ReLU()
        )

    def forward(self, data):
        data = data.float().view(-1, self.input_size)
        embedding = self.encoder(data)
        return embedding

class GNN(nn.Module):
    def __init__(self, num_of_class):
        super(GNN, self).__init__()
        self.num_heads = 16  # Fixed number of attention heads
        self.gat = GATv2Conv(64, 1024, heads=self.num_heads)
        self.out = Linear(1024 * self.num_heads, num_of_class)  # Adjust output layer dimensions

    def forward(self, x, edge_index):
        h = self.gat(x, edge_index)
        embedding = torch.relu(h)
        z = self.out(embedding)
        return z

class Net_cell(nn.Module):
    def __init__(self, num_of_class):
        super(Net_cell, self).__init__()
        self.cell = GNN(num_of_class)

    def forward(self, embedding, edge_index):
        cell_prediction = self.cell(embedding, edge_index)
        return cell_prediction
 
