"""
@Description: This script demonstrates how to create and load different types of tensors using PyTorch.
@Author: Shawn Chen
@Date: 2025-07-29
"""
# Documentation: https://pytorch.org/docs/stable/tensors.html

import torch
import pandas as pd

# Create a tensor with only one value
tensor_single = torch.tensor(5)
print("Single Value Tensor:")
print(tensor_single)

# Create a 1D tensor (vector)
tensor_a = torch.tensor([1, 2, 3])
print("Tensor A:")
print(tensor_a)

# Create a 2D tensor (matrix)
tensor_b = torch.tensor([[1, 2, 3], [4, 5, 6]])
print("Tensor B:")
print(tensor_b)

# Create a 3D tensor (tensor)
tensor_c = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
print("Tensor C:")
print(tensor_c)

# Read a csv file and create a tensor from it
# Load IRIS data from a CSV file
iris_data = pd.read_csv('https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv', header=None)
print("Iris Data:")
print(iris_data)

# Separate numeric features (first 4 columns) from string labels (last column)
iris_features = iris_data.iloc[:, :-1]  # All columns except the last one
iris_labels = iris_data.iloc[:, -1]     # Only the last column (species names)

# Convert only the numeric features to a tensor
iris_tensor = torch.tensor(iris_features.values, dtype=torch.float32)
print("Iris Features Tensor:")
print(iris_tensor)
print("Shape:", iris_tensor.shape)

# Show the labels separately
print("Iris Labels:")
print(iris_labels.unique())

# Load an image and create a tensor from it
from PIL import Image
from torchvision import transforms

# read Hamilton College Logo
image_path = './example_data/h_logo.jpg'
image = Image.open(image_path)

# Define a transform to convert the image to a tensor
transform = transforms.ToTensor()

# Apply the transform to the image
image_tensor = transform(image)
print("Image Tensor:")
print(image_tensor)
print("Shape:", image_tensor.shape)

