import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import time

# Load and preprocess the data
data = pd.read_csv('assets/Housing.csv')
X = data.drop('price', axis=1)
y = data['price']

# Convert categorical variables to numeric
X = pd.get_dummies(X, columns=['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea', 'furnishingstatus'])

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
y_scaled = scaler.fit_transform(y.values.reshape(-1, 1))

# Split the data
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_train = torch.FloatTensor(X_train)
y_train = torch.FloatTensor(y_train)
X_val = torch.FloatTensor(X_val)
y_val = torch.FloatTensor(y_val)

#send tensors to gpu
X_train = X_train.cuda()
y_train = y_train.cuda()
X_val = X_val.cuda()
y_val = y_val.cuda()

# Define the neural network
class HousingNN(nn.Module):
    def __init__(self, input_size):
        super(HousingNN, self).__init__()
        self.layer1 = nn.Linear(input_size, 8)  # Hidden layer with 8 nodes
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(8, 1)  # Output layer
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x

# Initialize the model
input_size = X_train.shape[1]
model = HousingNN(input_size)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

#send model to gpu
model = model.cuda()

# Training loop
num_epochs = 100
start_time = time.time()

for epoch in range(num_epochs):
    # Forward pass
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    
    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

training_time = time.time() - start_time

# Evaluation
model.eval()
with torch.no_grad():
    # Training set predictions
    train_outputs = model(X_train)
    train_loss = criterion(train_outputs, y_train)
    
    # Validation set predictions
    val_outputs = model(X_val)
    val_loss = criterion(val_outputs, y_val)
    
    # Calculate metrics for training set
    train_pred = train_outputs.cpu().numpy()
    train_true = y_train.cpu().numpy()
    train_r2 = 1 - np.sum((train_true - train_pred) ** 2) / np.sum((train_true - train_true.mean()) ** 2)
    train_mae = np.mean(np.abs(train_true - train_pred))
    train_rmse = np.sqrt(np.mean((train_true - train_pred) ** 2))
    
    # Calculate metrics for validation set
    val_pred = val_outputs.cpu().numpy()
    val_true = y_val.cpu().numpy()
    val_r2 = 1 - np.sum((val_true - val_pred) ** 2) / np.sum((val_true - val_true.mean()) ** 2)
    val_mae = np.mean(np.abs(val_true - val_pred))
    val_rmse = np.sqrt(np.mean((val_true - val_pred) ** 2))

print(f'\nTraining Time: {training_time:.2f} seconds')
print('\nTraining Set Metrics:')
print(f'MSE Loss: {train_loss.item():.4f}')
print(f'R-squared: {train_r2:.4f}')
print(f'MAE: {train_mae:.4f}')
print(f'RMSE: {train_rmse:.4f}')

print('\nValidation Set Metrics:')
print(f'MSE Loss: {val_loss.item():.4f}')
print(f'R-squared: {val_r2:.4f}')
print(f'MAE: {val_mae:.4f}')
print(f'RMSE: {val_rmse:.4f}')
