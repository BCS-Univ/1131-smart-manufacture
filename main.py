import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from data.dataset import ManufacturingData
from models.gcn_model import DetectionGCN

data = torch.randn(100, 10) 
label = torch.randint(0, 2, (100,))
edge_index = torch.randint(0, 100, (2, 9900))

dataset = ManufacturingData(data, label)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DetectionGCN(input_dim=10, hidden_dim=64, output_dim=2).to(device)

norm = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(50):
  model.train()
  epoch_loss = 0

  for batch_data, batch_label in dataloader:
    optimizer.zero_grad()
    batch_data, batch_label = batch_data.to(device), batch_label.to(device)