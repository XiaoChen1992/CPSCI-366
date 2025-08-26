import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch
import torch.nn as nn
import torch.optim as optim

# Load data
data = pd.read_csv("train.csv")

# Drop ID
data.drop(columns=["Id"], inplace=True)

# Split features and target
X = data.drop(columns=["SalePrice"])
y = data["SalePrice"]

# Separate numerical and categorical features
num_features = X.select_dtypes(include=[np.number]).columns.tolist()
cat_features = X.select_dtypes(include=["object"]).columns.tolist()

# Define preprocessing pipeline
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse=False))
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, num_features),
    ("cat", categorical_transformer, cat_features)
])

# Define simple MLP model
class MLP(nn.Module):
    def __init__(self, input_dim):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        return self.model(x)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# K-Fold CV
kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold = 0
scores = []

for train_idx, val_idx in kf.split(X):
    print(f"\n🔁 Fold {fold + 1}")
    
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    # Fit transformer on train, transform both train and val
    X_train_proc = preprocessor.fit_transform(X_train)
    X_val_proc = preprocessor.transform(X_val)
    
    # Convert to tensors
    X_train_tensor = torch.tensor(X_train_proc, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values.reshape(-1, 1), dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val_proc, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val.values.reshape(-1, 1), dtype=torch.float32)
    
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    # Model
    model = MLP(X_train_proc.shape[1]).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    # Training loop
    for epoch in range(50):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
    
    # Evaluation
    model.eval()
    with torch.no_grad():
        val_preds = model(X_val_tensor.to(device)).cpu().numpy().squeeze()
        val_rmse = np.sqrt(mean_squared_error(y_val, val_preds))
        print(f"Fold {fold + 1} RMSE: {val_rmse:.2f}")
        scores.append(val_rmse)
        fold += 1

print("Cross-validated RMSE scores:", scores)
print("Average RMSE:", np.mean(scores))
