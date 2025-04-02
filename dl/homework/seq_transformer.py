import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import math
from sklearn.model_selection import train_test_split
import torchinfo

# Input text data
text = '''Next character prediction is a fundamental task in the field of natural language processing (NLP) that involves predicting the next character in a sequence of text based on the characters that precede it. This task is essential for various applications, including text auto-completion, spell checking, and even in the development of sophisticated AI models capable of generating human-like text. At its core, next character prediction relies on statistical models or deep learning algorithms to analyze a given sequence of text and predict which character is most likely to follow. These predictions are based on patterns and relationships learned from large datasets of text during the training phase of the model. One of the most popular approaches to next character prediction involves the use of Recurrent Neural Networks (RNNs), and more specifically, a variant called Long Short-Term Memory (LSTM) networks. RNNs are particularly well-suited for sequential data like text, as they can maintain information in 'memory' about previous characters to inform the prediction of the next character. LSTM networks enhance this capability by being able to remember long-term dependencies, making them even more effective for next character prediction tasks. Training a model for next character prediction involves feeding it large amounts of text data, allowing it to learn the probability of each character's appearance following a sequence of characters. During this training process, the model adjusts its parameters to minimize the difference between its predictions and the actual outcomes, thus improving its predictive accuracy over time. Once trained, the model can be used to predict the next character in a given piece of text by considering the sequence of characters that precede it. This can enhance user experience in text editing software, improve efficiency in coding environments with auto-completion features, and enable more natural interactions with AI-based chatbots and virtual assistants. In summary, next character prediction plays a crucial role in enhancing the capabilities of various NLP applications, making text-based interactions more efficient, accurate, and human-like. Through the use of advanced machine learning models like RNNs and LSTMs, next character prediction continues to evolve, opening new possibilities for the future of text-based technology.'''

# Positional Encoding as described in "Attention is All You Need"
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return x

# Multi-Head Attention as described in "Attention is All You Need"
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Linear projections for Q, K, V
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
    def split_heads(self, x):
        # x shape: (batch_size, seq_len, d_model)
        batch_size, seq_len = x.size(0), x.size(1)
        # Reshape to (batch_size, seq_len, num_heads, d_k)
        x = x.view(batch_size, seq_len, self.num_heads, self.d_k)
        # Transpose to (batch_size, num_heads, seq_len, d_k)
        return x.transpose(1, 2)
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # Linear projections
        q = self.W_q(query)
        k = self.W_k(key)
        v = self.W_v(value)
        
        # Split heads
        q = self.split_heads(q)
        k = self.split_heads(k)
        v = self.split_heads(v)
        
        # Scaled dot-product attention
        # q, k, v shapes: (batch_size, num_heads, seq_len, d_k)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = torch.softmax(scores, dim=-1)
        attention_output = torch.matmul(attention_weights, v)
        
        # Reshape back
        # attention_output shape: (batch_size, num_heads, seq_len, d_k)
        attention_output = attention_output.transpose(1, 2)
        # attention_output shape: (batch_size, seq_len, num_heads, d_k)
        attention_output = attention_output.contiguous().view(batch_size, -1, self.d_model)
        
        # Final linear projection
        output = self.W_o(attention_output)
        
        return output, attention_weights

# Feed Forward Network
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        return self.linear2(self.relu(self.linear1(x)))

# Transformer Encoder Layer as described in "Attention is All You Need"
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # Self attention with residual connection and layer normalization
        attn_output, _ = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed forward with residual connection and layer normalization
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x

# Complete Transformer for Character-level Prediction
class CharTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers, dropout=0.1):
        super(CharTransformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model)
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        self.final_layer = nn.Linear(d_model, vocab_size)
        
    def generate_mask(self, seq_len):
        # Create mask for self-attention (upper triangular, causal mask)
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        mask = mask.unsqueeze(0)  # Add batch dimension
        return mask
        
    def forward(self, x):
        seq_len = x.size(1)
        mask = self.generate_mask(seq_len)
        
        # Embedding and positional encoding
        x = self.embedding(x)
        x = self.positional_encoding(x)
        x = self.dropout(x)
        
        # Encoder layers
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x, mask)
        
        # Final linear layer for prediction
        output = self.final_layer(x)
        
        return output

# Preparing the dataset for sequence prediction
max_length = 30  # Maximum length of input sequences
sequences = [text[i:i + max_length] for i in range(len(text) - max_length)]
labels = [text[i + max_length] for i in range(len(text) - max_length)]

# Creating character vocabulary
chars = sorted(list(set(text)))
char_to_ix = {ch: i for i, ch in enumerate(chars)}
ix_to_char = {i: ch for i, ch in enumerate(chars)}

# Convert sequences and labels to tensors
X = torch.tensor([[char_to_ix[ch] for ch in seq] for seq in sequences], dtype=torch.long)
y = torch.tensor([char_to_ix[label] for label in labels], dtype=torch.long)

# Splitting the dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Hyperparameters as described in "Attention is All You Need" (adapted for character-level tasks)
d_model = 128       # Embedding size
num_heads = 4       # Number of attention heads
d_ff = 512          # Feed-forward dimension
num_layers = 6      # Number of encoder layers
dropout = 0.1       # Dropout rate
learning_rate = 0.0001
epochs = 50
batch_size = 32

# Model, loss, and optimizer
model = CharTransformer(len(chars), d_model, num_heads, d_ff, num_layers, dropout)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.98), eps=1e-9)

# Learning rate scheduler as described in the paper
class WarmupScheduler:
    def __init__(self, optimizer, d_model, warmup_steps=4000):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.step_num = 0
        
    def step(self):
        self.step_num += 1
        lr = self.d_model ** (-0.5) * min(self.step_num ** (-0.5), self.step_num * self.warmup_steps ** (-1.5))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

scheduler = WarmupScheduler(optimizer, d_model, warmup_steps=4000)

# Display model summary
print(f"Character vocabulary size: {len(chars)}")
print(f"Training set size: {len(X_train)}")
print(f"Validation set size: {len(X_val)}")
summary = torchinfo.summary(model, input_data=X_train[:1])
print(summary)

# Training the model
total_start_time = time.time()
for epoch in range(epochs):
    start_time = time.time()
    model.train()
    
    # Initialize batch metrics
    epoch_loss = 0
    
    # Process data in batches
    num_batches = (len(X_train) + batch_size - 1) // batch_size
    for batch in range(num_batches):
        # Get batch data
        start_idx = batch * batch_size
        end_idx = min((batch + 1) * batch_size, len(X_train))
        batch_X = X_train[start_idx:end_idx]
        batch_y = y_train[start_idx:end_idx]
        
        # Forward pass
        optimizer.zero_grad()
        output = model(batch_X)
        output = output[:, -1, :]  # Get predictions for the last position
        loss = criterion(output, batch_y)
        
        # Backward pass and optimization
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
        optimizer.step()
        scheduler.step()
        
        epoch_loss += loss.item() * (end_idx - start_idx)
    
    # Calculate average loss for the epoch
    epoch_loss /= len(X_train)
    
    # Validation
    model.eval()
    with torch.no_grad():
        val_output = model(X_val)
        val_output = val_output[:, -1, :]  # Get predictions for the last position
        val_loss = criterion(val_output, y_val)
        _, predicted = torch.max(val_output, 1)
        val_accuracy = (predicted == y_val).float().mean()
    
    if (epoch+1) % 5 == 0:
        end_time = time.time()
        execution_time = end_time - start_time
        print(f'Epoch {epoch+1}, Loss: {epoch_loss:.4f}, Validation Loss: {val_loss.item():.4f}, '
              f'Validation Accuracy: {val_accuracy.item():.4f}, Execution Time: {execution_time:.2f} seconds')
    
    # Generate a sample prediction every 10 epochs
    if (epoch+1) % 10 == 0:
        model.eval()
        # Generate sample text
        sample_idx = np.random.randint(0, len(X_val))
        sample_input = X_val[sample_idx:sample_idx+1]
        input_text = ''.join([ix_to_char[idx.item()] for idx in sample_input[0]])
        
        with torch.no_grad():
            output = model(sample_input)
            output = output[0, -1]
            _, predicted_idx = torch.max(output, 0)
        
        next_char = ix_to_char[predicted_idx.item()]
        print(f"Sample Input: '{input_text}'")
        print(f"Predicted Next Character: '{next_char}'")
        print(f"Actual Next Character: '{ix_to_char[y_val[sample_idx].item()]}'")

total_end_time = time.time()
total_execution_time = total_end_time - total_start_time
print(f'Total Execution Time: {total_execution_time:.2f} seconds')

# Save the trained model
torch.save(model.state_dict(), 'char_transformer_model.pth')

# Function to generate text using the trained model
def generate_text(model, start_text, length=100):
    model.eval()
    
    # Convert start_text to indices
    input_indices = [char_to_ix[ch] for ch in start_text]
    
    # Generate text character by character
    generated_text = start_text
    
    for _ in range(length):
        # Prepare input sequence (limited to max_length)
        if len(input_indices) > max_length:
            input_indices = input_indices[-max_length:]
        
        x = torch.tensor([input_indices], dtype=torch.long)
        
        with torch.no_grad():
            output = model(x)
            output = output[0, -1]  # Get the prediction for the last character
            
            # Sample from the output distribution
            # (using temperature=1.0, which is just softmax)
            probs = torch.softmax(output, dim=0)
            predicted_idx = torch.multinomial(probs, 1)[0].item()
            
            # Add predicted character to the generated text
            generated_text += ix_to_char[predicted_idx]
            input_indices.append(predicted_idx)
    
    return generated_text

# Generate sample text after training
print("\nGenerating sample text:")
seed_text = text[:10]  # Use the first 10 characters as seed
generated_text = generate_text(model, seed_text, length=200)
print(f"Seed: '{seed_text}'")
print(f"Generated text: '{generated_text}'")
