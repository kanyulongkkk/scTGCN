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

        self.init_conv = GATv2Conv(64, 8,heads=2)
        self.num_heads = 16
        self.head_convs = nn.ModuleList([GATv2Conv(16, 1024) for _ in range(self.num_heads)])
        self.gat = GATv2Conv(1024*self.num_heads, 1024, heads=self.num_heads)  # Uncommented this line
        self.out = Linear(1024 * self.num_heads, num_of_class)  

    def forward(self, x, edge_index):
        # Initial convolution to extract features
        x = self.init_conv(x, edge_index)

        # Applying head convolutions
        head_outputs = [head_conv(x, edge_index) for head_conv in self.head_convs]

        # Concatenating the outputs from all head convolutions
        h = torch.cat(head_outputs, dim=-1)  # Concatenate along the feature dimension

        # Final GAT layer
        gat_output = self.gat(h, edge_index)

        # Final output layer
        z = self.out(gat_output)
        return z
class Net_cell(nn.Module):
    def __init__(self, num_of_class):
        super(Net_cell, self).__init__()
        self.cell = GNN(num_of_class)

    def forward(self, embedding, edge_index):
        cell_prediction = self.cell(embedding, edge_index)
        return cell_prediction
 
