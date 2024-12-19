import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from data.dataset import ManufacturingData
from models.gcn_model import DetectionGCN
import numpy as np

import time


def jagged_sleep(base_time=0.2, shape=2.0):
    pareto_sample = np.random.pareto(shape) + 1
    sleep_time = base_time * pareto_sample
    time.sleep(sleep_time)

data = torch.randn(128, 10) 
label = torch.randint(0, 2, (128,))
edge_index = torch.randint(0, 32, (2, 200))

dataset = ManufacturingData(data, label)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, drop_last=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DetectionGCN(input_dim=10, hidden_dim=64, output_dim=2).to(device)

norm = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(50):
  model.train()
  epoch_loss = 0

  for i, (batch_data, batch_labels) in enumerate(dataloader):
      # print(f"Batch Data: {batch_data}")
      optimizer.zero_grad()
      batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)
      output = model(batch_data, edge_index.to(device))

      loss = norm(output, batch_labels)

      loss.backward()
      optimizer.step()
      epoch_loss += loss.item()
      print(f"Batch{i + 1}Loss: {loss}")
      jagged_sleep()


  print(f"Epoch {epoch + 1}, Loss: {epoch_loss / len(dataloader)}")

model.eval()
with torch.no_grad():
    for batch_data, _ in dataloader:
        batch_data = batch_data.to(device)
        output = model(batch_data, edge_index.to(device))
        predictions = torch.argmax(output, dim=1)
        print(output)
        print(predictions)
        print("--------------------------")
