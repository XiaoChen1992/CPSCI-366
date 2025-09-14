# Assignment 1: Deep Learning for Tabular Data — Mobile Price Classification

## Overview

In this assignment, you will build a **Multi-Layer Perceptron (MLP)** to predict the price range of mobile phones based on their hardware specifications. You will work with a structured dataset containing various phone features such as RAM, battery power, screen size, and more.

Your goal is to **train and evaluate a deep learning model using only the `train.csv` data**, and to explore preprocessing techniques, model training strategies, and performance evaluation using suitable metrics.

---

## Dataset

- Source: [Mobile Price Classification on Kaggle](https://www.kaggle.com/datasets/iabhishekofficial/mobile-price-classification)
- Files:
  - `train.csv`: 2000 samples with 20 features + `price_range` label (0–3)
  - `test.csv`: 1000 samples, same features, no labels (used only for final ranking)
- Target variable: `price_range` (0: low cost, 1: medium cost, 2: high cost, 3: very high cost)

---

## Rules
- **Team work, 2 students in one team.**
- **DO NOT use `test.csv` for training, validation, feature selection, or scaling**.
- Only use `train.csv` during development.
- We will evaluate your model using `test.csv` and rank performance separately.
- You may consult Kaggle notebooks and blogs to explore ideas, but you **must cite all external resources**.
- **Extra credits**: The team achieves the lowest test loss will get addition 1 credit for Assignment 1.

---

## Tasks

### 1. Understand the Dataset
- Read the feature descriptions on Kaggle.
- Explore relationships between features and the target variable.
- Perform basic EDA (distribution plots, correlations, etc.).

### 2. Preprocess the Data
- Normalize or standardize features(if you think they can benefit your model).
- You may engineer new features or apply dimensionality reduction.
- Use only `train.csv` for all preprocessing steps.

### 3. Build an MLP Model
- Implement a feedforward neural network using PyTorch.
- Use at least 2 hidden layers and ReLU activation.
- Output layer must have 4 units and use softmax (via `CrossEntropyLoss`).
- Use dropout, batch norm, and weight initialization as needed.

### 4. Train and Validate
- Split the training data into train/validation sets (e.g., 80/20).
- Use early stopping, learning rate scheduling, and other best practices.
- Track training and validation accuracy and loss.

### 5. Evaluation
- Report classification accuracy.
- Optionally include confusion matrix, F1-score, etc.
- **Draw a learning curve**: accuracy (and loss) vs. epoch.

### 6. Final Prediction
- Include a function that:
  ```python
  def predict(model, X_test_loader, device):
      # Evaluate model on test data
      # Return accuracy and other classification metrics
      # TODO: Implement comprehensive evaluation
  ```
- Apply the same preprocessing pipeline to test data
- Report accuracy, precision, recall, F1-score

### 7. What to Submit
7.1. Jupyter notebook (.ipynb):

* Data loading and preprocessing

* MLP model code

* Training and validation loop

* Evaluation metrics and learning curves

* Final prediction function

7.2. Brief report (.pdf) summarizing (up to two pages, I will not grade your report based on the number of pages):

* Your model architecture and training setup

* Key findings from evaluation

* Any feature engineering

* External resources used (with links or citations)

7.3. Trained model weights (.pt)

### Grading Criteria (100 pts)
## 🏁 Grading Criteria (100 pts)

| Component                         | Points |
|----------------------------------|--------|
| Preprocessing & Feature Handling | 15     |
| MLP Model Implementation         | 20     |
| Training Strategy & Validation   | 20     |
| Learning Curve & Metrics         | 15     |
| Code Clarity & Reproducibility   | 10     |
| Report & Citations               | 10     |
| Generalization to Test Set       | 10     |

```python
"""
NOTE: This is example code to show you how to orgnize the project. This code does not contain feature propressing.
"""
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix # TODO: learn how to use these two functions
import matplotlib.pyplot as plt

# 1. Load data
def load_data(path):
    df = pd.read_csv(path)
    return df

def feature_processing(df):
    # TODO: Feature engineerig.
    return new_df

# 2. Dataset class
class MobilePriceDataset(Dataset):
    """
    You can directly use this function, we will discuss more about this function in Assignment 2.
    """
    def __init__(self, X, y):
        self.X = torch.tensor(X.values, dtype=torch.float32)
        self.y = torch.tensor(y.values, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# 3. MLP model
class MLPClassifier(nn.Module):
   # TODO: Here I built a very simple MLP, you need to build your own model.

    def __init__(self, input_dim, hidden_dims=[64, 32], output_dim=4):
        super(MLPClassifier, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], output_dim) # Do not apply softmax here!!!
        )

    def forward(self, x):
        return self.model(x)

# 4. Placeholder for main training pipeline
def main():
    # Load training data
    df = load_data("train.csv")
    df = feature_processing(df)

    # Split features and labels
    X = df.drop("price_range", axis=1)
    y = df["price_range"]

    # Preprocessing (e.g., scaling) — add later
    
    scaler = StandardScaler()  # TODO: You need to read sklearn document. 
    # https://scikit-learn.org/stable/data_transforms.html
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

    # Train/val split
    X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Prepare datasets and loaders
    train_dataset = MobilePriceDataset(X_train, y_train)
    val_dataset = MobilePriceDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    # Define model
    model = MLPClassifier(input_dim=X.shape[1])

    # Move model to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # For Mac M-series chips, you can also try:
    # device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    model = model.to(device)

    # Here is a basic traning loop
    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5)
    
    # Training parameters
    num_epochs = 100 # You can change this number

    # The following three lines are related to early stop
    early_stop_patience = 15 #TODO: Apply early stop for your model
    best_val_loss = float('inf')
    patience_counter = 0
    #########################################################
    
    # Lists to store metrics for learning curves
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    
    print(f"Training on device: {device}")
    print(f"Model has {sum(p.numel() for p in model.parameters())} parameters")
    
    # Training loop
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device) # move data to gpu
            
            # Zero gradients before every batch
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(data)
            loss = criterion(outputs, target)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Statistics
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += target.size(0)
            train_correct += (predicted == target).sum().item()
        
        # Calculate training metrics
        avg_train_loss = train_loss / len(train_loader)
        train_accuracy = 100 * train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                outputs = model(data)
                loss = criterion(outputs, target)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1) # covert proability to 0/1 label
                val_total += target.size(0)
                val_correct += (predicted == target).sum().item()
        
        # Calculate validation metrics
        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = 100 * val_correct / val_total
        
        # Store metrics for learning curves
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)
        
        # TODO: Learning rate scheduling
        
        
        # Print progress
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}]')
            print(f'Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.2f}%')
            print(f'Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.2f}%')
            print(f'LR: {optimizer.param_groups[0]["lr"]:.6f}')
            print('-' * 50)

        # TODO: Early stopping

        # TODO: save best model
        # uncomment the following line after you apply early stopping
        # model.save(model.state_dict(), 'you_path/best_model.pt')
    
    plot_learning_curves(train_losses, val_losses, train_accuracies, val_accuracies)
    
    return model, scaler


def plot_learning_curves(train_losses, val_losses, train_accuracies, val_accuracies):
    # TODO: Use this function to plot learning curve
    """Plot learning curves for loss and accuracy"""
    import matplotlib.pyplot as plt
    
    epochs = range(1, len(train_losses) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot losses
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    ax1.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # Plot accuracies
    ax2.plot(epochs, train_accuracies, 'b-', label='Training Accuracy', linewidth=2)
    ax2.plot(epochs, val_accuracies, 'r-', label='Validation Accuracy', linewidth=2)
    ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.legend(fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    model, scaler = main()

    # load the bset model
    model.load_state_dict(torch.load('your_path/best_model.pt'))
    
    def predict(model, X_test_loader, device):
        """
        Evaluate model on test data and return comprehensive metrics
        """
        model.eval()
        correct = 0
        total = 0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in X_test_loader:
                data, target = data.to(device), target.to(device)
                outputs = model(data)
                _, predicted = torch.max(outputs.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
                
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
        
        accuracy = 100 * correct / total
        
        # TODO: Add more evaluation metrics
        # Hint: Use classification_report, confusion_matrix from sklearn.metrics
        # Print or return precision, recall, F1-score for each class
        
        return accuracy
```