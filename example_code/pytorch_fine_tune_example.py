import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models

# 1. Data preprocessing
transform = transforms.Compose([
    transforms.Resize(224),  # Resize images to 224x224 for ResNet
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True)

testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False)

# 2. Load pretrained model
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

# 3. Freeze all layers
for param in model.parameters():
    param.requires_grad = False

# 4. Replace the final layer
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 10)  # CIFAR-10 has 10 classes

# 5. Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

# 6. Training loop (1 epoch demo)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

for epoch in range(1):
    model.train()
    running_loss = 0.0
    for images, labels in trainloader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch [{epoch+1}] Loss: {running_loss/len(trainloader):.4f}")

# 7. Simple test evaluation
model.eval()
correct, total = 0, 0
with torch.no_grad():
    for images, labels in testloader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (preds == labels).sum().item()
print(f"Test Accuracy: {100 * correct / total:.2f}%")

# PyTorch Lightning Fine-tuning Example
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, datasets, transforms
import pytorch_lightning as pl
from torch.utils.data import DataLoader

# 1. Define LightningModule
class LitResNet(pl.LightningModule):
    def __init__(self, num_classes=10, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        # Freeze all pretrained layers
        for param in self.model.parameters():
            param.requires_grad = False
        # Replace final layer
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        acc = (y_hat.argmax(dim=1) == y).float().mean()
        self.log_dict({"train_loss": loss, "train_acc": acc}, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        acc = (y_hat.argmax(dim=1) == y).float().mean()
        self.log_dict({"val_loss": loss, "val_acc": acc}, prog_bar=True)

    def configure_optimizers(self):
        return optim.Adam(self.model.fc.parameters(), lr=self.hparams.lr)

# 2. Data preparation
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

train_ds = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
val_ds = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=32)

# 3. Training
trainer = pl.Trainer(max_epochs=1, accelerator="auto")
model = LitResNet()
trainer.fit(model, train_loader, val_loader)

