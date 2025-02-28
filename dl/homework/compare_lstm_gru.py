import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import time
from torch.utils.data import DataLoader

# Import the Shakespeare dataloader
from shakespear_loader import char_to_int, int_to_char, train_loader, test_loader

# Define the RNN models (LSTM and GRU)
class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, rnn_type='lstm'):
        super(RNNModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn_type = rnn_type.lower()
        
        # Choose RNN type
        if self.rnn_type == 'lstm':
            self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        elif self.rnn_type == 'gru':
            self.rnn = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        else:
            raise ValueError("rnn_type must be 'lstm' or 'gru'")
            
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x, hidden=None):
        # Initial hidden state
        batch_size = x.size(0)
        if hidden is None:
            if self.rnn_type == 'lstm':
                h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
                c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
                hidden = (h0, c0)
            else:  # GRU
                hidden = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        
        # Forward pass through RNN
        out, hidden = self.rnn(x, hidden)
        
        # Decode the hidden state
        out = self.fc(out)
        return out, hidden

# Training function
def train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs, device):
    """Train the model and evaluate on test set."""
    train_losses = []
    test_losses = []
    test_accuracies = []
    
    # For timing
    start_time = time.time()
    
    for epoch in range(1, num_epochs + 1):
        # Training
        model.train()
        train_loss = 0
        for batch_idx, (batch_x, batch_y) in enumerate(train_loader):
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            # Print shapes for debugging (only first batch of first epoch)
            if epoch == 1 and batch_idx == 0:
                print(f"batch_x shape: {batch_x.shape}")
                print(f"batch_y shape: {batch_y.shape}")
            
            # Convert inputs to one-hot encoding
            batch_x_one_hot = torch.nn.functional.one_hot(batch_x, num_classes=len(char_to_int)).float()
            
            # Print one-hot shape for debugging (only first batch of first epoch)
            if epoch == 1 and batch_idx == 0:
                print(f"batch_x_one_hot shape: {batch_x_one_hot.shape}")
            
            # Forward pass
            output, _ = model(batch_x_one_hot)
            
            # Print output shape for debugging (only first batch of first epoch)
            if epoch == 1 and batch_idx == 0:
                print(f"output shape: {output.shape}")
                print(f"output[:, -1, :] shape: {output[:, -1, :].shape}")
                print(f"batch_y shape: {batch_y.shape}")
            
            # We only need to predict the next character (which is our target)
            loss = criterion(output[:, -1, :], batch_y)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Average training loss
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Evaluation
        model.eval()
        test_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_idx, (batch_x, batch_y) in enumerate(test_loader):
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                
                # Convert inputs to one-hot encoding
                batch_x_one_hot = torch.nn.functional.one_hot(batch_x, num_classes=len(char_to_int)).float()
                
                # Forward pass
                output, _ = model(batch_x_one_hot)
                
                # We only need to predict the next character (which is our target)
                loss = criterion(output[:, -1, :], batch_y)
                
                test_loss += loss.item()
                
                # Calculate accuracy
                _, predicted = torch.max(output[:, -1, :], 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
        
        # Average test loss and accuracy
        test_loss /= len(test_loader)
        test_losses.append(test_loss)
        
        accuracy = 100 * correct / total
        test_accuracies.append(accuracy)
        
        # Print progress
        if epoch % 5 == 0:
            print(f"Epoch [{epoch}/{num_epochs}], Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Test Accuracy: {accuracy:.2f}%")
    
    training_time = time.time() - start_time
    
    return train_losses, test_losses, test_accuracies, training_time

# Generate text function
def generate_text(model, seed_text, predict_len=100, temperature=0.8):
    """Generate text using the trained model."""
    model.eval()
    device = next(model.parameters()).device
    
    # Process seed text
    context = [char_to_int[c] for c in seed_text]
    generated_text = seed_text
    
    with torch.no_grad():
        for _ in range(predict_len):
            # Prepare input
            x = torch.tensor(context[-20:]).unsqueeze(0).to(device)  # Use last 20 chars
            x_one_hot = torch.nn.functional.one_hot(x, num_classes=len(char_to_int)).float()
            
            # Get prediction
            output, _ = model(x_one_hot)
            output = output[:, -1, :] / temperature  # Get last character prediction
            probs = torch.nn.functional.softmax(output, dim=-1)
            
            # Sample from the predicted probability distribution
            next_char_idx = torch.multinomial(probs, 1).item()
            next_char = int_to_char[next_char_idx]
            
            # Update context and generated text
            generated_text += next_char
            context.append(next_char_idx)
    
    return generated_text

# Get model size
def get_model_size(model):
    """Calculate model size in MB"""
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_all_mb = (param_size + buffer_size) / 1024**2
    return size_all_mb

# Function to train and evaluate a model
def train_and_evaluate(rnn_type):
    """Train and evaluate a model with the specified RNN type"""
    # Hyperparameters
    hidden_size = 128
    num_layers = 2
    num_epochs = 30
    learning_rate = 0.001
    
    # Get vocabulary size
    vocab_size = len(char_to_int)
    
    # Initialize model and move to device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RNNModel(
        input_size=vocab_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        output_size=vocab_size,
        rnn_type=rnn_type
    ).to(device)
    
    # Print device information
    print(f"Training {rnn_type.upper()} on: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Train the model
    train_losses, test_losses, test_accuracies, training_time = train_model(
        model, train_loader, test_loader,
        criterion, optimizer, num_epochs, device
    )
    
    # Calculate model size
    model_size = get_model_size(model)
    
    # Report metrics
    print("\n" + "="*50)
    print(f"{rnn_type.upper()} TRAINING RESULTS")
    print("="*50)
    print(f"Final Training Loss: {train_losses[-1]:.4f}")
    print(f"Final Test Loss: {test_losses[-1]:.4f}")
    print(f"Final Test Accuracy: {test_accuracies[-1]:.2f}%")
    print(f"Training Time: {training_time:.2f} seconds")
    print(f"Model Size: {model_size:.2f} MB")
    print("="*50)
    
    return model, train_losses, test_losses, test_accuracies, training_time, model_size

# Function to compare models
def compare_models(lstm_results, gru_results):
    """Compare the performance of LSTM and GRU models"""
    # Unpack results
    lstm_model, lstm_train_losses, lstm_test_losses, lstm_accuracies, lstm_time, lstm_size = lstm_results
    gru_model, gru_train_losses, gru_test_losses, gru_accuracies, gru_time, gru_size = gru_results
    
    # Plot training and test losses
    plt.figure(figsize=(15, 10))
    
    # Plot training losses
    plt.subplot(2, 2, 1)
    plt.plot(lstm_train_losses, label='LSTM Training Loss', color='blue', linewidth=2)
    plt.plot(gru_train_losses, label='GRU Training Loss', color='red', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training Losses', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=10)
    
    # Plot test losses
    plt.subplot(2, 2, 2)
    plt.plot(lstm_test_losses, label='LSTM Test Loss', color='blue', linewidth=2)
    plt.plot(gru_test_losses, label='GRU Test Loss', color='red', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Test Losses', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=10)
    
    # Plot test accuracies
    plt.subplot(2, 2, 3)
    plt.plot(lstm_accuracies, label='LSTM Accuracy', color='blue', linewidth=2)
    plt.plot(gru_accuracies, label='GRU Accuracy', color='red', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title('Test Accuracies', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=10)
    
    # Bar chart for training time and model size
    plt.subplot(2, 2, 4)
    metrics = ['Training Time (s)', 'Model Size (MB)']
    lstm_values = [lstm_time, lstm_size]
    gru_values = [gru_time, gru_size]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    plt.bar(x - width/2, lstm_values, width, label='LSTM', color='blue')
    plt.bar(x + width/2, gru_values, width, label='GRU', color='red')
    
    plt.xlabel('Metric', fontsize=12)
    plt.ylabel('Value', fontsize=12)
    plt.title('Performance Metrics', fontsize=14)
    plt.xticks(x, metrics)
    plt.grid(True, linestyle='--', alpha=0.7, axis='y')
    plt.legend(fontsize=10)
    
    # Add text summary
    plt.figtext(0.5, 0.01, 
                f"LSTM - Final Loss: {lstm_test_losses[-1]:.4f}, Accuracy: {lstm_accuracies[-1]:.2f}%, Time: {lstm_time:.2f}s, Size: {lstm_size:.2f}MB\n"
                f"GRU - Final Loss: {gru_test_losses[-1]:.4f}, Accuracy: {gru_accuracies[-1]:.2f}%, Time: {gru_time:.2f}s, Size: {gru_size:.2f}MB",
                ha="center", fontsize=12, bbox={"facecolor":"orange", "alpha":0.2, "pad":5})
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig('lstm_vs_gru_comparison.png')
    plt.show()
    
    # Generate text with both models
    seed_text = "The quick brown fox jumps over the lazy dog"
    
    print("\nGenerating text with LSTM model:")
    lstm_generated = generate_text(lstm_model, seed_text)
    print(lstm_generated)
    
    print("\nGenerating text with GRU model:")
    gru_generated = generate_text(gru_model, seed_text)
    print(gru_generated)
    
    # Print comparison summary
    print("\n" + "="*50)
    print("MODEL COMPARISON SUMMARY")
    print("="*50)
    print(f"LSTM Final Test Loss: {lstm_test_losses[-1]:.4f}")
    print(f"GRU Final Test Loss: {gru_test_losses[-1]:.4f}")
    print(f"LSTM Final Test Accuracy: {lstm_accuracies[-1]:.2f}%")
    print(f"GRU Final Test Accuracy: {gru_accuracies[-1]:.2f}%")
    print(f"LSTM Training Time: {lstm_time:.2f} seconds")
    print(f"GRU Training Time: {gru_time:.2f} seconds")
    print(f"LSTM Model Size: {lstm_size:.2f} MB")
    print(f"GRU Model Size: {gru_size:.2f} MB")
    
    # Determine which model performed better
    if lstm_accuracies[-1] > gru_accuracies[-1]:
        accuracy_winner = "LSTM"
    elif lstm_accuracies[-1] < gru_accuracies[-1]:
        accuracy_winner = "GRU"
    else:
        accuracy_winner = "Tie"
        
    if lstm_test_losses[-1] < gru_test_losses[-1]:
        loss_winner = "LSTM"
    elif lstm_test_losses[-1] > gru_test_losses[-1]:
        loss_winner = "GRU"
    else:
        loss_winner = "Tie"
        
    if lstm_time < gru_time:
        time_winner = "LSTM"
    elif lstm_time > gru_time:
        time_winner = "GRU"
    else:
        time_winner = "Tie"
        
    if lstm_size < gru_size:
        size_winner = "LSTM"
    elif lstm_size > gru_size:
        size_winner = "GRU"
    else:
        size_winner = "Tie"
    
    print("\nWINNERS BY CATEGORY:")
    print(f"Best Accuracy: {accuracy_winner}")
    print(f"Best Loss: {loss_winner}")
    print(f"Fastest Training: {time_winner}")
    print(f"Smallest Model: {size_winner}")
    print("="*50)

if __name__ == "__main__":
    # Train LSTM model
    print("Training LSTM model...")
    lstm_results = train_and_evaluate(rnn_type='lstm')
    
    # Train GRU model
    print("\nTraining GRU model...")
    gru_results = train_and_evaluate(rnn_type='gru')
    
    # Compare the models
    compare_models(lstm_results, gru_results)
