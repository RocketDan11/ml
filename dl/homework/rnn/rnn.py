import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

#important to note hidden state is NOT the size of the sequence length
#but the size of the hidden state passed to the next layer

class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        """
        Initialize the RNN model.
        
        Args:
            input_size (int): Size of input features
            hidden_size (int): Number of features in the hidden state
            num_layers (int): Number of recurrent layers
            output_size (int): Size of output features
        """
        super(SimpleRNN, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # RNN layer
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True  # Expect input shape: (batch, seq_len, input_size)
        )
        
        # Fully connected layer to map RNN output to desired output size
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        """
        Forward pass of the model.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_length, input_size)
            
        Returns:
            output (torch.Tensor): Output tensor of shape (batch_size, seq_length, output_size)
            hidden (torch.Tensor): Final hidden state
        """
        # Initialize hidden state
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        
        # Forward propagate RNN
        out, hidden = self.rnn(x, h0)
        
        # Pass through fully connected layer
        # Reshape output to (batch_size * seq_length, hidden_size)
        out = out.contiguous().view(-1, self.hidden_size)
        out = self.fc(out)
        
        # Reshape back to (batch_size, seq_length, output_size)
        out = out.view(batch_size, -1, out.size(-1))
        
        return out, hidden

# Example usage
class TextDataset(Dataset):
    def __init__(self, text, sequence_length):
        self.text = text
        self.sequence_length = sequence_length
        self.char_to_idx = {char: idx for idx, char in enumerate(sorted(set(text)))}
        self.idx_to_char = {idx: char for char, idx in self.char_to_idx.items()}
        self.data_size = len(text) - sequence_length
        
    def __len__(self):
        return self.data_size
    
    def __getitem__(self, idx):
        # Get sequence and target
        sequence = self.text[idx:idx + self.sequence_length]
        target = self.text[idx + 1:idx + self.sequence_length + 1]
        
        # Convert to indices
        x = torch.tensor([self.char_to_idx[char] for char in sequence])
        y = torch.tensor([self.char_to_idx[char] for char in target])
        
        # One-hot encode
        x = torch.nn.functional.one_hot(x, num_classes=len(self.char_to_idx)).float()
        
        return x, y

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    train_losses = []
    val_losses = []
    val_accuracies = []
    start_time = torch.cuda.Event(enable_timing=True)
    end_time = torch.cuda.Event(enable_timing=True)
    
    start_time.record()
    
    for epoch in range(num_epochs):
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
                output, _ = model(batch_x)
                loss = criterion(output.view(-1, output.size(-1)), batch_y.view(-1))
                val_loss += loss.item()
                
                # Calculate accuracy
                predictions = output.view(-1, output.size(-1)).argmax(dim=1)
                targets = batch_y.view(-1)
                correct += (predictions == targets).sum().item()
                total += targets.size(0)
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        val_accuracy = 100 * correct / total
        val_accuracies.append(val_accuracy)
        
        # Print progress every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], '
                  f'Train Loss: {train_loss:.4f}, '
                  f'Val Loss: {val_loss:.4f}, '
                  f'Val Accuracy: {val_accuracy:.2f}%')
    
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

if __name__ == "__main__":
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
    model = SimpleRNN(
        input_size=vocab_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        output_size=vocab_size
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
    }, 'rnn_model.pth')

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

def load_model(model_path, device='cpu'):
    """Load a trained model and its character mappings.
    
    Args:
        model_path: Path to the saved model file
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
    model = SimpleRNN(
        input_size=vocab_size,
        hidden_size=128,  # Use the same parameters as during training
        num_layers=2,
        output_size=vocab_size
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    return model, char_to_idx, idx_to_char
