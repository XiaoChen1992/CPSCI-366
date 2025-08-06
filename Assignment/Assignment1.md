# 🧠 Assignment 1: Deep Learning for Tabular Data — Mobile Price Classification

## 📌 Overview

In this assignment, you will build a **Multi-Layer Perceptron (MLP)** to predict the price range of mobile phones based on their hardware specifications. You will work with a structured dataset containing various phone features such as RAM, battery power, screen size, and more.

Your goal is to **train and evaluate a deep learning model using only the `train.csv` data**, and to explore preprocessing techniques, model training strategies, and performance evaluation using suitable metrics.

---

## 📂 Dataset

- Source: [Mobile Price Classification on Kaggle](https://www.kaggle.com/datasets/iabhishekofficial/mobile-price-classification)
- Files:
  - `train.csv`: 2000 samples with 20 features + `price_range` label (0–3)
  - `test.csv`: 1000 samples, same features, no labels (used only for final ranking)
- Target variable: `price_range` (0: low cost, 1: medium cost, 2: high cost, 3: very high cost)

---

## Rules

- **DO NOT use `test.csv` for training, validation, feature selection, or scaling**.
- Only use `train.csv` during development.
- We will evaluate your model using `test.csv` and rank performance separately.
- You may consult Kaggle notebooks and blogs to explore ideas, but you **must cite all external resources**.

---

## Tasks

### 1. Understand the Dataset
- Read the feature descriptions on Kaggle.
- Explore relationships between features and the target variable.
- Perform basic EDA (distribution plots, correlations, etc.).

### 2. Preprocess the Data
- Normalize or standardize features.
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
- **Draw a learning curve**: accuracy (or loss) vs. epoch.

### 6. Final Prediction
- Include a function:
  ```python
  def predict(model, X_test):
      # Applied the exacty same data preprocessing pipeline for raw X_test to get X_test
      # returns numpy array of predicted labels
      return model.predict(X_test)
   ```

### 7. What to Submit
7.1. Jupyter notebook (.ipynb):

* Data loading and preprocessing

* MLP model code

* Training and validation loop

* Evaluation metrics and learning curves

* Final prediction function

7.2. Brief report (.pdf) summarizing:

* Your model architecture and training setup

* Key findings from evaluation

* Any feature engineering

* External resources used (with links or citations)

* Trained model weights (.pt)

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

# 1. Load data
def load_data(path):
    df = pd.read_csv(path)
    return df
# TODO: Procress df here.

# 2. Dataset class
class MobilePriceDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X.values, dtype=torch.float32)
        self.y = torch.tensor(y.values, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# 3. MLP model
class MLPClassifier(nn.Module):
   # TODO: Here I built a very simple MLP, you may want to build your own model.

    def __init__(self, input_dim, hidden_dims=[64, 32], output_dim=4):
        super(MLPClassifier, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], output_dim)
        )

    def forward(self, x):
        return self.model(x)

# 4. Placeholder for main training pipeline
def main():
    # Load training data
    df = load_data("train.csv")

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
    model = model.to(device)

    # TODO: Add training loop, validation, evaluation, learning curve, etc.

if __name__ == "__main__":
    main()
    # TODO: save model's weights (.pt)
```