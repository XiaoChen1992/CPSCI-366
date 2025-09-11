# fashion_mnist_mlp.py
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Reproducibility
torch.manual_seed(666)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data
transform = transforms.Compose([
    transforms.ToTensor(),                      # [0,1]
    transforms.Normalize((0.5,), (0.5,))        # mean/std for single channel
])

train_ds = datasets.FashionMNIST(root="./data", train=True, download=True, transform=transform)
test_ds  = datasets.FashionMNIST(root="./data", train=False, download=True, transform=transform)

train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
test_loader  = DataLoader(test_ds,  batch_size=256, shuffle=False)

# Model: 28*28 -> 256 -> 128 -> 10
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.net(x)

import torch
import torch.nn as nn
import torch.nn.functional as F

# class MLP(nn.Module):
#     # 28*28 -> 256 -> 128 -> 10 with ReLU
#     def __init__(self):
#         super().__init__()
#         self.fc1 = nn.Linear(28*28, 256)
#         self.fc2 = nn.Linear(256, 128)
#         self.fc3 = nn.Linear(128, 10)

#     def forward(self, x):
#         x = torch.flatten(x, 1)       # same as nn.Flatten()
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)               # logits
#         return x

model = MLP().to(device)

# Loss & Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Train
def train_one_epoch(loader):
    model.train()
    total_loss = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
    return total_loss / len(loader.dataset)

# Evaluate
@torch.no_grad()
def evaluate(loader):
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return correct / total

# Run
epochs = 5
for epoch in range(1, epochs + 1):
    train_loss = train_one_epoch(train_loader)
    test_acc = evaluate(test_loader)
    print(f"Epoch {epoch:02d} | train_loss: {train_loss:.4f} | test_acc: {test_acc*100:.2f}%")
