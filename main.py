import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from torch.cuda.amp import autocast
from data.dataset import ManufacturingData
from models.gcn_model import DetectionGCN
import numpy as np

data = torch.randn(128, 10) 
label = torch.randint(0, 2, (128,))
edge_index = torch.randint(0, 32, (2, 200))

dataset = ManufacturingData(data, label)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, drop_last=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DetectionGCN(input_dim=10, hidden_dim=64, output_dim=2).to(device)

norm = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

def cyclical_lr(step_size, min_lr=3e-4, max_lr=3e-3):
    def compute_lr(current_step):
        cycle = np.floor(1 + current_step/(2 * step_size))
        x = abs(current_step/step_size - 2 * cycle + 1)
        lr = min_lr + (max_lr - min_lr) * max(0, 1 - x)
        return lr
    return compute_lr

# Calculate step_size as 2-10 times the number of iterations in an epoch
step_size = len(dataloader) * 4  # 4 epochs per cycle
clr = cyclical_lr(step_size)
current_step = 0

for epoch in range(50):
    model.train()
    epoch_loss = 0

    for i, (batch_data, batch_labels) in enumerate(dataloader):
        # Update learning rate
        lr = clr(current_step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        optimizer.zero_grad()
        batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)

        with torch.amp.autocast('cuda'):
            output = model(batch_data, edge_index.to(device))
            loss = norm(output, batch_labels)

        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        print(f"Batch {i + 1} Loss: {loss:.4f}, LR: {lr:.6f}")
        current_step += 1

    print(f"Epoch {epoch + 1}, Loss: {epoch_loss / len(dataloader):.4f}")

model.eval()
with torch.no_grad():
    for batch_data, _ in dataloader:
        batch_data = batch_data.to(device)
        output = model(batch_data, edge_index.to(device))
        predictions = torch.argmax(output, dim=1)
        print(output)
