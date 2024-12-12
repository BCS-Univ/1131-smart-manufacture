import torch.nn as nn
from torch_geometric.nn import GCNConv
import torch

class DetectionGCN(nn.Module):
  def __init__(self, input_dim, hidden_dim, output_dim, layer_num=20):
    super(DetectionGCN, self).__init__()
    self.layers = nn.ModuleList()

    self.layers.append(GCNConv(input_dim, hidden_dim))

    for _ in range(layer_num - 2):
      self.layers.append(GCNConv(hidden_dim, hidden_dim))

    self.layers.append(GCNConv(hidden_dim, output_dim))

  def forward(self, x, edge_index):
    for layer in self.layers[:-1]:
      x = torch.relu(layer(x, edge_index))
    x = self.layers[-1](x, edge_index)
    return x