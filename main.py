import torch
from torch_geometric.loader import DataLoader
import torch.optim as optim
import torch.nn as nn
from data.dataset import ManufacturingData
from models.gcn_model import DetectionGCN

data = torch.randn(128, 10) 
label = torch.randint(0, 2, (128,))
edge_index = torch.randint(0, 128, (2, 200))

dataset = ManufacturingData(data, label)
dataloader = DataLoader(dataset, batch_size=100, shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DetectionGCN(input_dim=10, hidden_dim=64, output_dim=2).to(device)

norm = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(50):
  model.train()
  epoch_loss = 0

  for batch_data, batch_labels in dataloader:
      optimizer.zero_grad()
      batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)
      output = model(batch_data, edge_index.to(device))

      loss = norm(output, batch_labels)

      loss.backward()
      optimizer.step()
      epoch_loss += loss.item()

  print(f"Epoch {epoch + 1}, Loss: {epoch_loss / len(dataloader)}")

model.eval()
with torch.no_grad():
    for batch_data, _ in dataloader:
        batch_data = batch_data.to(device)
        output = model(batch_data, edge_index.to(device))
        predictions = torch.argmax(output, dim=1)
        print(predictions)