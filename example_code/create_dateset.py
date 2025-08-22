import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

# Step 1: Generate fake data
np.random.seed(42)
num_samples = 1000

# Numerical features
numerical_data = np.random.rand(num_samples, 2) * 100  # Scale features by 100

# Categorical features (0, 1, 2)
categorical_data = np.random.randint(0, 3, size=(num_samples, 1))

# Convert categorical data to PyTorch tensor
categorical_data_tensor = torch.tensor(categorical_data, dtype=torch.long)

# One-hot encode categorical data using PyTorch
categorical_data_encoded = torch.nn.functional.one_hot(categorical_data_tensor.squeeze(), num_classes=3)

# Convert numerical data to PyTorch tensor
numerical_data_tensor = torch.tensor(numerical_data, dtype=torch.float32)

# Regression labels
labels = np.random.rand(num_samples) * 100  # Range from 0 to 100
labels_tensor = torch.tensor(labels, dtype=torch.float32)

# Combine numerical and categorical data
# Ensure categorical data is of type float32 to match numerical data
data = torch.cat((numerical_data_tensor, categorical_data_encoded.type(torch.float32)), dim=1)

# Step 2: Create a PyTorch dataset
class RegressionDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
        
    def __len__(self):
        """
        This function is required by PyTorch to determine the length of the dataset.
        """
        return len(self.labels)
        
    def __getitem__(self, idx):
        """
        This funcion is required by PyTorch to retrieve a sample from the dataset.
        """
        feature = self.features[idx]
        label = self.labels[idx]
        return feature, label

# Prepare dataset
dataset = RegressionDataset(data, labels_tensor)

# Example DataLoader (can be used in training loop)
data_loader = DataLoader(dataset, batch_size=10, shuffle=True)

# Testing the DataLoader
for features, labels in data_loader:
    print("Batch features:", features)
    print("Batch labels:", labels)
    break  # Only show the first batch
