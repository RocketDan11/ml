import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
import math
import matplotlib.pyplot as plt

print(torch.cuda.is_available())  
print(torch.version.cuda)  
print(torch.cuda.get_device_name(0))
# Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

# Transformer Model
class TextTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers, dim_feedforward):
        super().__init__()
        self.embedding = nn.Linear(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.decoder = nn.Linear(d_model, vocab_size)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        src = self.embedding(src)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        output = self.decoder(output)
        return output

# Sequence Dataset
class TextDataset(Dataset):
    def __init__(self, data, seq_len, vocab_size):
        self.data = data
        self.seq_len = seq_len
        self.vocab_size = vocab_size

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        # Get sequence and target
        sequence = self.data[idx:idx + self.seq_len]
        target = self.data[idx + 1:idx + self.seq_len + 1]
        
        # Convert to one-hot encoded tensors
        x = torch.zeros(self.seq_len, self.vocab_size)
        x[range(self.seq_len), sequence] = 1
        
        # Target should be indices for CrossEntropyLoss
        return x, torch.LongTensor(target)

# Load and process text data
def load_text_data(filepath):
    with open(filepath, 'r') as f:
        text = f.read()
    
    # Get unique characters
    chars = sorted(list(set(text)))
    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    idx_to_char = {i: ch for i, ch in enumerate(chars)}
    
    # Convert text to indices
    data = [char_to_idx[ch] for ch in text]
    return np.array(data), char_to_idx, idx_to_char

def train_model(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        # Reshape output and target for CrossEntropyLoss
        output = output.view(-1, output.size(-1))
        target = target.view(-1)
        loss = criterion(output, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        total_loss += loss.item()
        
        # Calculate accuracy
        pred = output.argmax(dim=1)
        correct += (pred == target).sum().item()
        total += target.size(0)
    
    return total_loss / len(train_loader), 100. * correct / total

def validate_model(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            # Reshape output and target for CrossEntropyLoss
            output = output.view(-1, output.size(-1))
            target = target.view(-1)
            loss = criterion(output, target)
            total_loss += loss.item()
            
            # Calculate accuracy
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)
    
    return total_loss / len(val_loader), 100. * correct / total

def main():
    # Load and process text data
    data, char_to_idx, idx_to_char = load_text_data('sequence.txt')
    vocab_size = len(char_to_idx)
    
    # Hyperparameters
    d_model = 128
    nhead = 4
    num_layers = 3
    dim_feedforward = 512
    batch_size = 64
    n_epochs = 50
    seq_len = 20
    learning_rate = 0.001
    
    # Create datasets
    train_size = int(0.8 * len(data))
    train_data = data[:train_size]
    val_data = data[train_size:]
    
    train_dataset = TextDataset(train_data, seq_len, vocab_size)
    val_dataset = TextDataset(val_data, seq_len, vocab_size)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Initialize model and training components
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TextTransformer(vocab_size, d_model, nhead, num_layers, dim_feedforward).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        # Training loop
    best_val_loss = float('inf')
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    # Training loop
    best_val_loss = float('inf')
    for epoch in range(n_epochs):
        train_loss, train_acc = train_model(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate_model(model, val_loader, criterion, device)
            
         # Store metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')
        
        print(f'Epoch: {epoch+1}/{n_epochs}')
        print(f'Training Loss: {train_loss:.6f}')
        print(f'Validation Loss: {val_loss:.6f}')
        print('-' * 50)

       # Plot loss curves
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Curves')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    main()
