import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
import math
import time
import matplotlib.pyplot as plt
import ast

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Load Purepecha data from file
def load_purepecha_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
        # Extract the data list from the file
        data_str = content.strip()
        if data_str.startswith('data = '):
            data_str = data_str[7:]
        data = ast.literal_eval(data_str)
        
        # Combine all Purepecha translations into a single text
        purepecha_text = ' '.join([item[1] for item in data])
        return purepecha_text

# Load the Purepecha data
purepecha_text = load_purepecha_data('assets/purepecha_data.txt')
print(f"Loaded {len(purepecha_text)} characters of Purepecha text")

# Create character-level vocabulary
chars = sorted(list(set(purepecha_text)))
vocab_size = len(chars)
print("Unique characters:", vocab_size)
char2idx = {ch: i for i, ch in enumerate(chars)}
idx2char = {i: ch for i, ch in enumerate(chars)}

# Dataset that produces input sequences and target character (the next character)
class CharDataset(Dataset):
    def __init__(self, text, seq_length=20):
        self.text = text
        self.seq_length = seq_length
        self.data = [char2idx[ch] for ch in text]

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, idx):
        # Input sequence of length seq_length and target: the next character
        x = torch.tensor(self.data[idx:idx+self.seq_length], dtype=torch.long)
        y = torch.tensor(self.data[idx+self.seq_length], dtype=torch.long)
        return x, y

# Choose sequence length: try 10, 20, or 30 (here, we use 20)
seq_length = 20
dataset = CharDataset(purepecha_text, seq_length=seq_length)
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=batch_size)

# Model 1: Plain RNN-based model (using LSTM)
class RNNModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers=1):
        super(RNNModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, x):
        # x: (batch, seq_length)
        embedded = self.embedding(x)              # (batch, seq_length, embed_dim)
        output, (hn, cn) = self.lstm(embedded)      # output: (batch, seq_length, hidden_dim)
        # Use the final hidden state for prediction
        out = self.fc(hn[-1])                      # (batch, vocab_size)
        return out

# Model 2: RNN with cross attention.
# After processing the sequence with LSTM, we attend over all time steps using the final hidden state as query.
class RNNWithAttention(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers=1):
        super(RNNWithAttention, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True)
        # Attention parameters
        self.attn_linear = nn.Linear(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim * 2, vocab_size)
    
    def forward(self, x):
        # x: (batch, seq_length)
        embedded = self.embedding(x)             # (batch, seq_length, embed_dim)
        outputs, (hn, cn) = self.lstm(embedded)    # outputs: (batch, seq_length, hidden_dim)
        # Use the final hidden state as the query for attention
        query = hn[-1].unsqueeze(1)               # (batch, 1, hidden_dim)
        # Project outputs to get keys
        keys = self.attn_linear(outputs)          # (batch, seq_length, hidden_dim)
        # Compute dot-product attention scores (scaled)
        scores = torch.bmm(query, keys.transpose(1, 2)) / math.sqrt(keys.size(-1))  # (batch, 1, seq_length)
        attn_weights = torch.softmax(scores, dim=-1)  # (batch, 1, seq_length)
        # Compute context vector as weighted sum of outputs
        context = torch.bmm(attn_weights, outputs)  # (batch, 1, hidden_dim)
        context = context.squeeze(1)                # (batch, hidden_dim)
        query = query.squeeze(1)                    # (batch, hidden_dim)
        combined = torch.cat((query, context), dim=1)  # (batch, 2*hidden_dim)
        out = self.fc(combined)                     # (batch, vocab_size)
        return out

# Hyperparameters
embed_dim = 128
hidden_dim = 256
num_layers = 1
learning_rate = 0.003
num_epochs = 20

# Instantiate models
model_plain = RNNModel(vocab_size, embed_dim, hidden_dim, num_layers).to(device)
model_attn = RNNWithAttention(vocab_size, embed_dim, hidden_dim, num_layers).to(device)

criterion = nn.CrossEntropyLoss()
optimizer_plain = optim.Adam(model_plain.parameters(), lr=learning_rate)
optimizer_attn = optim.Adam(model_attn.parameters(), lr=learning_rate)

# Functions to train and evaluate a model
def train_model(model, optimizer, loader):
    model.train()
    total_loss = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)          # logits shape: (batch, vocab_size)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
    return total_loss / len(loader.dataset)

def evaluate_model(model, loader):
    model.eval()
    total_loss = 0
    correct = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            total_loss += loss.item() * x.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
    avg_loss = total_loss / len(loader.dataset)
    accuracy = correct / len(loader.dataset)
    return avg_loss, accuracy

# Containers for metrics
plain_train_losses = []
plain_val_losses = []
plain_val_accs = []

attn_train_losses = []
attn_val_losses = []
attn_val_accs = []

# Training loop for the plain RNN model
print("Training Plain RNN Model (without attention)")
start_time = time.time()
for epoch in range(num_epochs):
    train_loss = train_model(model_plain, optimizer_plain, train_loader)
    val_loss, val_acc = evaluate_model(model_plain, val_loader)
    plain_train_losses.append(train_loss)
    plain_val_losses.append(val_loss)
    plain_val_accs.append(val_acc)
    print(f"Plain RNN Epoch {epoch+1}/{num_epochs}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
plain_time = time.time() - start_time

# Training loop for the RNN with attention model
print("\nTraining RNN Model with Cross Attention")
start_time = time.time()
for epoch in range(num_epochs):
    train_loss = train_model(model_attn, optimizer_attn, train_loader)
    val_loss, val_acc = evaluate_model(model_attn, val_loader)
    attn_train_losses.append(train_loss)
    attn_val_losses.append(val_loss)
    attn_val_accs.append(val_acc)
    print(f"Attention RNN Epoch {epoch+1}/{num_epochs}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
attn_time = time.time() - start_time

print("\nTiming Summary:")
print(f"Plain RNN training time: {plain_time:.2f} seconds")
print(f"RNN with Attention training time: {attn_time:.2f} seconds")

# Visualization of training loss and validation accuracy
epochs = range(1, num_epochs+1)
plt.figure(figsize=(12, 5))

# Plot training loss
plt.subplot(1, 2, 1)
plt.plot(epochs, plain_train_losses, label='Plain RNN Train Loss')
plt.plot(epochs, attn_train_losses, label='Attention RNN Train Loss')
plt.xlabel('Epochs')
plt.ylabel('Training Loss')
plt.title('Training Loss Comparison')
plt.legend()
plt.grid(True)

# Plot validation accuracy
plt.subplot(1, 2, 2)
plt.plot(epochs, plain_val_accs, label='Plain RNN Val Accuracy')
plt.plot(epochs, attn_val_accs, label='Attention RNN Val Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Validation Accuracy')
plt.title('Validation Accuracy Comparison')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Function to generate text using the trained model
def generate_text(model, seed_text, length=100, temperature=0.8):
    model.eval()
    with torch.no_grad():
        # Convert seed text to indices
        seed_indices = [char2idx[ch] for ch in seed_text]
        current_indices = seed_indices.copy()
        
        generated_text = seed_text
        
        for _ in range(length):
            # Convert current sequence to tensor
            x = torch.tensor([current_indices[-seq_length:]], dtype=torch.long).to(device)
            
            # Get model prediction
            logits = model(x)
            
            # Apply temperature
            logits = logits / temperature
            
            # Sample from the distribution
            probs = torch.softmax(logits, dim=1)
            next_idx = torch.multinomial(probs, 1).item()
            
            # Add to generated text
            generated_text += idx2char[next_idx]
            current_indices.append(next_idx)
            
            # Remove oldest character if sequence is too long
            if len(current_indices) > seq_length:
                current_indices.pop(0)
    
    return generated_text

# Generate text using both models
print("\nText Generation Examples:")
seed_text = "Ná k'eri"
print(f"Seed text: {seed_text}")

print("\nPlain RNN Generated Text:")
generated_plain = generate_text(model_plain, seed_text, length=50)
print(generated_plain)

print("\nAttention RNN Generated Text:")
generated_attn = generate_text(model_attn, seed_text, length=50)
print(generated_attn)
