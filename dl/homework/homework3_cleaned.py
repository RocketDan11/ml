import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

# TextDataset class for handling text data
class TextDataset(Dataset):
    """Dataset for character-level text generation."""
    def __init__(self, text, sequence_length):
        self.text = text
        self.sequence_length = sequence_length
        
        # Create character to index mapping
        chars = sorted(list(set(text)))
        self.char_to_idx = {ch: i for i, ch in enumerate(chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(chars)}
        
        # Create sequences
        self.sequences = []
        for i in range(len(text) - sequence_length):
            # Input sequence and target (next character)
            seq_in = text[i:i+sequence_length]
            seq_out = text[i+1:i+sequence_length+1]
            
            # Convert to indices
            seq_in_idx = [self.char_to_idx[c] for c in seq_in]
            seq_out_idx = [self.char_to_idx[c] for c in seq_out]
            
            self.sequences.append((seq_in_idx, seq_out_idx))
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq_in, seq_out = self.sequences[idx]
        
        # Convert to one-hot encoding
        x = torch.zeros(self.sequence_length, len(self.char_to_idx))
        for i, char_idx in enumerate(seq_in):
            x[i, char_idx] = 1.0
        
        # Target is just the indices
        y = torch.tensor(seq_out)
        
        return x, y

# Base RNN model class
class BaseRNN(nn.Module):
    """Base class for RNN models"""
    def __init__(self, input_size, hidden_size, num_layers, output_size, rnn_type='rnn'):
        super(BaseRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Choose RNN type
        if rnn_type.lower() == 'lstm':
            self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
            self.rnn_type = 'lstm'
        else:
            self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
            self.rnn_type = 'rnn'
            
        self.fc = nn.Linear(hidden_size, output_size)
        self.rnn.input_size = input_size  # Store for later use in text generation
    
    def forward(self, x, hidden=None):
        # Initial hidden state
        if hidden is None:
            if self.rnn_type == 'lstm':
                h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
                c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
                hidden = (h0, c0)
            else:
                hidden = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Forward pass through RNN
        out, hidden = self.rnn(x, hidden)
        
        # Decode the hidden state of the last time step
        out = self.fc(out)
        return out, hidden

# Training function
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    """Train the model and evaluate on validation set."""
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    # For timing
    start_time = torch.cuda.Event(enable_timing=True)
    end_time = torch.cuda.Event(enable_timing=True)
    
    start_time.record()
    
    for epoch in range(1, num_epochs + 1):
        # Training
        model.train()
        train_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            # Forward pass
            output, _ = model(batch_x)
            loss = criterion(output.view(-1, output.size(-1)), batch_y.view(-1))
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Average training loss
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                
                # Forward pass
                output, _ = model(batch_x)
                loss = criterion(output.view(-1, output.size(-1)), batch_y.view(-1))
                
                val_loss += loss.item()
                
                # Calculate accuracy
                _, predicted = torch.max(output, 2)
                total += batch_y.size(0) * batch_y.size(1)
                correct += (predicted == batch_y).sum().item()
        
        # Average validation loss and accuracy
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        accuracy = 100 * correct / total
        val_accuracies.append(accuracy)
        
        # Print progress
        if epoch % 10 == 0:
            print(f"Epoch [{epoch}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {accuracy:.2f}%")
    
    end_time.record()
    torch.cuda.synchronize()
    training_time = start_time.elapsed_time(end_time) / 1000  # Convert to seconds
    
    return train_losses, val_losses, val_accuracies, training_time

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

def generate_text(model, char_to_idx, idx_to_char, seed_text, predict_len=100, temperature=0.8):
    """Generate text using the trained model.
    
    Args:
        model: Trained RNN model
        char_to_idx: Character to index mapping
        idx_to_char: Index to character mapping
        seed_text: Initial text to start prediction from
        predict_len: Number of characters to generate
        temperature: Controls randomness (lower = more conservative)
    
    Returns:
        Generated text string
    """
    model.eval()
    device = next(model.parameters()).device
    
    # Process seed text
    context = seed_text[-model.rnn.input_size:].ljust(model.rnn.input_size)
    generated_text = seed_text
    
    with torch.no_grad():
        for _ in range(predict_len):
            # Prepare input
            x = torch.tensor([char_to_idx[c] for c in context])
            x = torch.nn.functional.one_hot(x, num_classes=len(char_to_idx)).float()
            x = x.unsqueeze(0).to(device)  # Add batch dimension
            
            # Get prediction
            output, _ = model(x)
            output = output[:, -1, :] / temperature  # Get last character prediction
            probs = torch.nn.functional.softmax(output, dim=-1)
            
            # Sample from the predicted probability distribution
            next_char_idx = torch.multinomial(probs, 1).item()
            next_char = idx_to_char[next_char_idx]
            
            # Update context and generated text
            generated_text += next_char
            context = context[1:] + next_char
    
    return generated_text

def load_model(model_path, hidden_size=128, num_layers=2, rnn_type='rnn', device='cpu'):
    """Load a trained model and its character mappings.
    
    Args:
        model_path: Path to the saved model file
        hidden_size: Hidden size used in the model
        num_layers: Number of layers used in the model
        rnn_type: Type of RNN ('rnn' or 'lstm')
        device: Device to load the model on ('cpu' or 'cuda')
    
    Returns:
        model: Loaded model
        char_to_idx: Character to index mapping
        idx_to_char: Index to character mapping
    """
    checkpoint = torch.load(model_path, map_location=device)
    char_to_idx = checkpoint['char_to_idx']
    idx_to_char = checkpoint['idx_to_char']
    
    # Initialize model with correct dimensions
    vocab_size = len(char_to_idx)
    model = BaseRNN(
        input_size=vocab_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        output_size=vocab_size,
        rnn_type=rnn_type
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    return model, char_to_idx, idx_to_char

def train_and_evaluate(rnn_type='rnn'):
    """Train and evaluate a model with the specified RNN type"""
    # Read input text
    with open('input-text.txt', 'r') as f:
        text = f.read()
    
    # Hyperparameters
    sequence_length = 50
    hidden_size = 128
    num_layers = 2
    batch_size = 32
    num_epochs = 100
    learning_rate = 0.001
    
    # Create dataset
    dataset = TextDataset(text, sequence_length)
    vocab_size = len(dataset.char_to_idx)
    
    # Split into train and validation sets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Initialize model and move to device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BaseRNN(
        input_size=vocab_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        output_size=vocab_size,
        rnn_type=rnn_type
    ).to(device)
    
    # Print device information
    print(f"Training on: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Train the model
    train_losses, val_losses, val_accuracies, training_time = train_model(
        model, train_loader, val_loader,
        criterion, optimizer, num_epochs, device
    )
    
    # Calculate model size
    model_size = get_model_size(model)
    
    # Report metrics
    print("\n" + "="*50)
    print("TRAINING RESULTS")
    print("="*50)
    print(f"Final Training Loss: {train_losses[-1]:.4f}")
    print(f"Final Validation Loss: {val_losses[-1]:.4f}")
    print(f"Final Validation Accuracy: {val_accuracies[-1]:.2f}%")
    print(f"Training Time: {training_time:.2f} seconds")
    print(f"Model Size: {model_size:.2f} MB")
    print(f"Device: {device}")
    print("="*50)
    
    # Plot training and validation losses
    plt.figure(figsize=(12, 10))
    
    plt.subplot(2, 1, 1)
    plt.plot(train_losses, label='Training Loss', color='blue', linewidth=2)
    plt.plot(val_losses, label='Validation Loss', color='red', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training and Validation Losses', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=10)
    
    plt.subplot(2, 1, 2)
    plt.plot(val_accuracies, label='Validation Accuracy', color='green', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title('Validation Accuracy', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=10)
    
    # Add text annotations for final values
    plt.figtext(0.5, 0.01, 
                f"Final Training Loss: {train_losses[-1]:.4f} | "
                f"Final Validation Loss: {val_losses[-1]:.4f} | "
                f"Final Accuracy: {val_accuracies[-1]:.2f}%",
                ha="center", fontsize=12, bbox={"facecolor":"orange", "alpha":0.2, "pad":5})
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to make room for the text
    
    # Display the plot in the notebook instead of saving it
    plt.show()
    
    # Save the trained model
    model_filename = f"{rnn_type}_model.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'char_to_idx': dataset.char_to_idx,
        'idx_to_char': dataset.idx_to_char,
        'training_metrics': {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'val_accuracies': val_accuracies,
            'training_time': training_time,
            'model_size': model_size
        }
    }, model_filename)
    
    return model, dataset.char_to_idx, dataset.idx_to_char

if __name__ == "__main__":
    # Train RNN model
    print("Training RNN model...")
    rnn_model, rnn_char_to_idx, rnn_idx_to_char = train_and_evaluate(rnn_type='rnn')
    
    # Train LSTM model
    print("\nTraining LSTM model...")
    lstm_model, lstm_char_to_idx, lstm_idx_to_char = train_and_evaluate(rnn_type='lstm')
    
    # Example of generating text with both models
    seed_text = "The quick brown fox jumps over the lazy dog"
    
    print("\nGenerating text with RNN model:")
    rnn_generated = generate_text(rnn_model, rnn_char_to_idx, rnn_idx_to_char, seed_text)
    print(rnn_generated)
    
    print("\nGenerating text with LSTM model:")
    lstm_generated = generate_text(lstm_model, lstm_char_to_idx, lstm_idx_to_char, seed_text)
    print(lstm_generated)
