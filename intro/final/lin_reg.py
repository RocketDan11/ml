import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from tqdm import tqdm

# Read the dataset
df = pd.read_csv('assets/healthcare-dataset-stroke-data.csv')

# Data preprocessing
# Drop id column and handle missing values
df = df.drop('id', axis=1)
df['bmi'].fillna(df['bmi'].mean(), inplace=True)

# Convert categorical variables to numeric using one-hot encoding
categorical_columns = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
df = pd.get_dummies(df, columns=categorical_columns)

# Separate features (X) and target variable (y)
X = df.drop('stroke', axis=1)
y = df['stroke']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize lists to store losses
train_losses = []
val_losses = []

# Create K-fold cross validator
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

# Train model with k-fold cross validation
for fold, (train_idx, val_idx) in enumerate(tqdm(kf.split(X_train_scaled), total=n_splits, desc="Training Folds")):
    # Split data into training and validation for this fold
    X_train_fold = X_train_scaled[train_idx]
    y_train_fold = y_train.iloc[train_idx]
    X_val_fold = X_train_scaled[val_idx]
    y_val_fold = y_train.iloc[val_idx]
    
    # Train model
    model = LinearRegression()
    model.fit(X_train_fold, y_train_fold)
    
    # Calculate losses
    train_pred = model.predict(X_train_fold)
    val_pred = model.predict(X_val_fold)
    
    train_loss = mean_squared_error(y_train_fold, train_pred)
    val_loss = mean_squared_error(y_val_fold, val_pred)
    
    train_losses.append(train_loss)
    val_losses.append(val_loss)

# Final predictions on test set
final_model = LinearRegression()
final_model.fit(X_train_scaled, y_train)
y_pred = final_model.predict(X_test_scaled)

# Calculate final metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(range(1, n_splits + 1), train_losses, label='Training Loss', marker='o')
plt.plot(range(1, n_splits + 1), val_losses, label='Validation Loss', marker='o')
plt.xlabel('Fold')
plt.ylabel('Mean Squared Error')
plt.title('Training and Validation Loss Across Folds')
plt.legend()
plt.grid(True)
plt.savefig('loss_plot.png')
plt.close()

# Print results
print("\nModel Performance:")
print(f"Final Test MSE: {mse:.4f}")
print(f"Final R-squared Score: {r2:.4f}")
print(f"\nAverage Training Loss: {np.mean(train_losses):.4f}")
print(f"Average Validation Loss: {np.mean(val_losses):.4f}")

# Feature importance
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': final_model.coef_
})
print("\nTop 10 Most Important Features:")
print(feature_importance.sort_values(by='Coefficient', key=abs, ascending=False).head(10))

