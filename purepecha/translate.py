import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


# Load the dataset
with open("assets/data.txt") as f:
    data_text = f.read()

# Process the data format from the file
# First cleanup - handle the data = [ part and extract just the content
content = data_text.replace("data = [", "").replace("]", "")

# Fix missing commas between entries (lines that start with a parenthesis but don't have a comma before them)
content = content.replace(")(\n", "),(\n")

# Split the entries
entries = []
entry_buffer = ""
in_entry = False

for char in content:
    if char == '(' and not in_entry:
        in_entry = True
        entry_buffer = char
    elif char == ')' and in_entry:
        entry_buffer += char
        entries.append(entry_buffer)
        entry_buffer = ""
        in_entry = False
    elif in_entry:
        entry_buffer += char

# Parse each entry
english_to_purepecha = []
for entry in entries:
    # Remove outer parentheses
    if entry.startswith('(') and entry.endswith(')'):
        inner = entry[1:-1]
        # Split by first comma to handle cases where there might be commas in the text
        parts = inner.split(',', 1)
        if len(parts) == 2:
            eng = parts[0].strip().strip('"').strip()
            pur = parts[1].strip().strip('"').strip()
            english_to_purepecha.append((eng, pur))

print(f"Loaded {len(english_to_purepecha)} English-Purepecha translation pairs")

# Special tokens for the start and end of sequences
SOS_token = 0  # Start Of Sequence Token
EOS_token = 1  # End Of Sequence Token

# Preparing the word to index mapping and vice versa
word_to_index = {"SOS": SOS_token, "EOS": EOS_token}
for pair in english_to_purepecha:
    for word in pair[0].split() + pair[1].split():
        if word not in word_to_index:
            word_to_index[word] = len(word_to_index)

index_to_word = {i: word for word, i in word_to_index.items()}

class TranslationDataset(Dataset):
    """Custom Dataset class for handling translation pairs."""
    def __init__(self, dataset, word_to_index):
        self.dataset = dataset
        self.word_to_index = word_to_index

    def __len__(self):
        # Returns the total number of translation pairs in the dataset
        return len(self.dataset)

    def __getitem__(self, idx):
        # Retrieves a translation pair by index, converts words to indices,
        # and adds the EOS token at the end of each sentence.
        input_sentence, target_sentence = self.dataset[idx]
        input_indices = [self.word_to_index[word] for word in input_sentence.split()] + [EOS_token]
        target_indices = [self.word_to_index[word] for word in target_sentence.split()] + [EOS_token]
        return torch.tensor(input_indices, dtype=torch.long), torch.tensor(target_indices, dtype=torch.long)

# Creating a DataLoader to batch and shuffle the dataset
translation_dataset = TranslationDataset(english_to_purepecha, word_to_index)
dataloader = DataLoader(translation_dataset, batch_size=1, shuffle=True)

class Encoder(nn.Module):
    """The Encoder part of the seq2seq model with attention."""
    def __init__(self, input_size, hidden_size):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)  # Embedding layer
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)  # GRU layer

    def forward(self, input, hidden):
        # Forward pass for the encoder
        embedded = self.embedding(input).view(1, 1, -1)
        output, hidden = self.gru(embedded, hidden)
        return output, hidden

    def initHidden(self):
        # Initializes hidden state
        return torch.zeros(1, 1, self.hidden_size, device=device)

class MultiHeadAttention(nn.Module):
    """Multi-Head Attention mechanism from 'Attention Is All You Need' paper."""
    def __init__(self, hidden_size, num_heads=4, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        
        # Ensure hidden_size is divisible by num_heads
        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        # Linear projections for Query, Key, and Value
        self.query_proj = nn.Linear(hidden_size, hidden_size)
        self.key_proj = nn.Linear(hidden_size, hidden_size)
        self.value_proj = nn.Linear(hidden_size, hidden_size)
        
        # Output projection
        self.output_proj = nn.Linear(hidden_size, hidden_size)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
        # Scaling factor for dot product attention
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)
        
    def forward(self, query, key, value, mask=None):
        # query, key, value shapes: [batch_size, seq_len, hidden_size]
        # For decoder self-attention: query = decoder hidden state, key/value = encoder outputs
        
        batch_size = query.shape[0]
        
        # Linear projections and split into multiple heads
        # [batch_size, seq_len, hidden_size]
        Q = self.query_proj(query)
        K = self.key_proj(key)
        V = self.value_proj(value)
        
        # Reshape for multi-head attention
        # [batch_size, seq_len, num_heads, head_dim]
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim)
        
        # Transpose to [batch_size, num_heads, seq_len, head_dim]
        Q = Q.permute(0, 2, 1, 3)
        K = K.permute(0, 2, 1, 3)
        V = V.permute(0, 2, 1, 3)
        
        # Calculate scaled dot-product attention
        # [batch_size, num_heads, query_len, key_len]
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale
        
        # Apply mask if provided (for padding or causal attention)
        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)
        
        # Apply softmax to get attention weights
        # [batch_size, num_heads, query_len, key_len]
        attention = torch.softmax(energy, dim=-1)
        
        # Apply dropout
        attention = self.dropout(attention)
        
        # Weighted sum of values
        # [batch_size, num_heads, query_len, head_dim]
        x = torch.matmul(attention, V)
        
        # Transpose and reshape back
        # [batch_size, query_len, hidden_size]
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(batch_size, -1, self.hidden_size)
        
        # Final linear projection
        output = self.output_proj(x)
        
        # Return both the output and the attention weights for visualization
        return output, attention


class Attention(nn.Module):
    """Wrapper around Multi-Head Attention for compatibility with the existing code."""
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        
        # Use the Multi-Head Attention mechanism
        self.multihead_attn = MultiHeadAttention(hidden_size, num_heads=4)
        
    def forward(self, hidden, encoder_outputs):
        # Adapt the interface to work with the existing seq2seq model
        # hidden shape: [1, 1, hidden_size]
        # encoder_outputs shape: [seq_len, 1, hidden_size]
        
        # Reshape inputs to match MultiHeadAttention expectations
        # [1, seq_len, hidden_size]
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        
        # Use hidden state as query, encoder outputs as key and value
        # Expand hidden to match batch dimension of encoder_outputs
        # [1, 1, hidden_size]
        query = hidden.permute(1, 0, 2)
        
        # Compute multi-head attention
        # output shape: [1, 1, hidden_size]
        # attn_weights shape: [1, num_heads, 1, seq_len]
        _, attn_weights = self.multihead_attn(query, encoder_outputs, encoder_outputs)
        
        # Average attention weights across heads for visualization
        # [1, seq_len]
        attn_weights = attn_weights.mean(dim=1).squeeze(0)
        
        return attn_weights

class AttentionDecoder(nn.Module):
    """The Decoder part of the seq2seq model with Transformer-style attention."""
    def __init__(self, hidden_size, output_size, dropout=0.1):
        super(AttentionDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Embedding layer
        self.embedding = nn.Embedding(output_size, hidden_size)
        
        # Multi-head attention mechanism from Transformer
        self.attention = Attention(hidden_size)
        
        # Add layer normalization for stability (from Transformer)
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.layer_norm2 = nn.LayerNorm(hidden_size)
        
        # Feed-forward network (from Transformer)
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size)
        )
        
        # GRU for maintaining sequence state (hybrid approach)
        self.gru = nn.GRU(hidden_size * 2, hidden_size, batch_first=True)
        
        # Output layer
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, encoder_outputs):
        # input shape: [1]
        # hidden shape: [1, 1, hidden_size]
        # encoder_outputs shape: [seq_len, 1, hidden_size]
        
        # Get embedding of input
        # [1, 1, hidden_size]
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)
        
        # Calculate attention weights using the Transformer-style attention
        # [1, seq_len]
        attn_weights = self.attention(hidden, encoder_outputs)
        
        # Calculate context vector using attention weights
        # [1, 1, hidden_size]
        context = torch.bmm(attn_weights.unsqueeze(0), encoder_outputs.permute(1, 0, 2))
        
        # Apply residual connection and layer normalization (Transformer style)
        # First combine the context with the embedded input
        # [1, 1, hidden_size * 2]
        combined = torch.cat((embedded, context), dim=2)
        
        # Pass through GRU (maintaining the RNN component for sequence modeling)
        # output shape: [1, 1, hidden_size]
        # hidden shape: [1, 1, hidden_size]
        gru_output, hidden = self.gru(combined, hidden)
        
        # Apply layer normalization
        normalized_output = self.layer_norm1(gru_output)
        
        # Feed-forward network with residual connection (Transformer style)
        ff_output = normalized_output + self.feed_forward(normalized_output)
        ff_output = self.layer_norm2(ff_output)
        
        # Final output projection
        # [1, output_size]
        output = self.softmax(self.out(ff_output.squeeze(0)))
        
        # Return output, hidden state, and attention weights for visualization
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

# Assuming all words in the dataset + 'SOS' and 'EOS' tokens are included in word_to_index
input_size = len(word_to_index)
hidden_size = 256  # Adjust according to your preference
output_size = len(word_to_index)

encoder = Encoder(input_size=input_size, hidden_size=hidden_size).to(device)
decoder = AttentionDecoder(hidden_size=hidden_size, output_size=output_size).to(device)

# Set the learning rate for optimization
learning_rate = 0.008

# Initializing optimizers for both encoder and decoder with Adam optimizer
encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)

def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion):
    # Initialize encoder hidden state
    encoder_hidden = encoder.initHidden()

    # Clear gradients for optimizers
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    # Calculate the length of input and target tensors
    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    # Initialize loss
    loss = 0
    
    # Store encoder outputs for attention
    encoder_outputs = torch.zeros(input_length, 1, encoder.hidden_size, device=device)

    # Encoding each word in the input
    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_tensor[ei].unsqueeze(0), encoder_hidden)
        encoder_outputs[ei] = encoder_output

    # Decoder's first input is the SOS token
    decoder_input = torch.tensor([[SOS_token]], device=device)

    # Decoder starts with the encoder's last hidden state
    decoder_hidden = encoder_hidden
    
    # Store attention weights for visualization
    attention_weights = torch.zeros(target_length, input_length)

    # Decoding loop
    for di in range(target_length):
        decoder_output, decoder_hidden, attn_weights = decoder(decoder_input, decoder_hidden, encoder_outputs)
        
        # Store attention weights
        attention_weights[di] = attn_weights.squeeze()
        
        # Choose top1 word from decoder's output
        topv, topi = decoder_output.topk(1)
        decoder_input = topi.squeeze().detach()  # Detach from history as input

        # Calculate loss
        loss += criterion(decoder_output, target_tensor[di].unsqueeze(0))
        if decoder_input.item() == EOS_token:  # Stop if EOS token is generated
            break

    # Backpropagation
    loss.backward()

    # Update encoder and decoder parameters
    encoder_optimizer.step()
    decoder_optimizer.step()

    # Return average loss and attention weights
    return loss.item() / target_length, attention_weights

# Negative Log Likelihood Loss function for calculating loss
criterion = nn.NLLLoss()

# Set number of epochs for training
# Training parameters
n_iters = 50
learning_rate = 0.005

# Initialize the encoder and decoder with appropriate parameters
encoder = Encoder(input_size, hidden_size).to(device)
decoder = AttentionDecoder(hidden_size, output_size).to(device)

# Initialize optimizers
encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)

# Function to train and validate the model
def train_and_validate(encoder, decoder, encoder_optimizer, decoder_optimizer, dataloader, criterion, n_epochs=100):
    """Train the model for specified number of epochs with validation at each epoch."""
    # Lists to store training and validation metrics
    train_losses = []
    validation_bleu_scores = []
    validation_losses = []
    best_bleu_score = 0.0
    best_encoder_wts = None
    best_decoder_wts = None
    
    # Training loop
    for epoch in range(1, n_epochs + 1):
        # Track progress
        total_loss = 0
        
        # Set models to training mode
        encoder.train()
        decoder.train()
        
        # Iterate through the dataset
        for i, (input_tensor, target_tensor) in enumerate(dataloader):
            # Move tensors to device
            input_tensor = input_tensor[0].to(device)  # Remove batch dimension
            target_tensor = target_tensor[0].to(device)  # Remove batch dimension
            
            # Train on single example
            loss, attention_weights = train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)
            total_loss += loss
            
        # Calculate average training loss for this epoch
        avg_loss = total_loss / len(dataloader)
        train_losses.append(avg_loss)
        
        # Validation phase
        encoder.eval()
        decoder.eval()
        val_loss, bleu_score = evaluate(encoder, decoder, dataloader, criterion)
        validation_losses.append(val_loss)
        validation_bleu_scores.append(bleu_score)
        
        # Save best model based on BLEU score
        if bleu_score > best_bleu_score:
            best_bleu_score = bleu_score
            best_encoder_wts = encoder.state_dict().copy()
            best_decoder_wts = decoder.state_dict().copy()
        
        # Print progress
        print(f'Epoch {epoch}/{n_epochs} | Train Loss: {avg_loss:.4f} | Val Loss: {val_loss:.4f} | BLEU: {bleu_score:.4f}')
        
    # Load best model weights
    encoder.load_state_dict(best_encoder_wts)
    decoder.load_state_dict(best_decoder_wts)
    
    return train_losses, validation_losses, validation_bleu_scores

# Define the evaluation function
def evaluate(encoder, decoder, dataloader, criterion, max_examples=None):
    """Evaluate the model on the dataset and compute BLEU score."""
    total_loss = 0
    all_bleu_scores = []
    count = 0
    
    with torch.no_grad():
        for i, (input_tensor, target_tensor) in enumerate(dataloader):
            if max_examples is not None and count >= max_examples:
                break
            
            # Move tensors to device
            input_tensor = input_tensor[0].to(device)  # Remove batch dimension
            target_tensor = target_tensor[0].to(device)  # Remove batch dimension
            
            # Initialize encoder hidden
            encoder_hidden = encoder.initHidden()
            
            # Get input and target lengths
            input_length = input_tensor.size(0)
            target_length = target_tensor.size(0)
            
            # Initialize encoder outputs tensor
            encoder_outputs = torch.zeros(input_length, 1, encoder.hidden_size, device=device)
            
            # Forward pass through encoder
            for ei in range(input_length):
                encoder_output, encoder_hidden = encoder(input_tensor[ei].unsqueeze(0), encoder_hidden)
                encoder_outputs[ei] = encoder_output
            
            # Prepare decoder input
            decoder_input = torch.tensor([[SOS_token]], device=device)
            decoder_hidden = encoder_hidden
            
            # Variables for loss calculation
            loss = 0
            
            # Decoded outputs for BLEU calculation
            decoded_words = []
            
            # Forward pass through decoder with teacher forcing
            use_teacher_forcing = False  # No teacher forcing during evaluation
            
            # Decode each token in the target
            for di in range(target_length):
                decoder_output, decoder_hidden, _ = decoder(
                    decoder_input, decoder_hidden, encoder_outputs
                )
                
                # Calculate loss
                loss += criterion(decoder_output, target_tensor[di].unsqueeze(0))
                
                # Get the most likely next word
                topv, topi = decoder_output.topk(1)
                if topi.item() == EOS_token:
                    decoded_words.append('<EOS>')
                    break
                else:
                    decoded_words.append(index_to_word[topi.item()])
                
                # Next input
                decoder_input = topi.detach()
            
            # Average loss over sequence length
            loss = loss.item() / target_length
            total_loss += loss
            
            # Convert target to list of words for BLEU calculation
            target_words = [index_to_word[idx.item()] for idx in target_tensor if idx.item() != EOS_token]
            
            # Calculate BLEU score for this sample
            # Simple BLEU implementation (precision of n-grams)
            from collections import Counter
            
            # Take only up to EOS or end of the sequence
            pred_tokens = [w for w in decoded_words if w != '<EOS>']
            ref_tokens = target_words
            
            # Calculate precision of unigrams (1-grams)
            if len(pred_tokens) == 0:
                bleu_score = 0.0
            else:
                # Count matches
                pred_counter = Counter(pred_tokens)
                ref_counter = Counter(ref_tokens)
                
                # Calculate matches
                matches = sum((pred_counter & ref_counter).values())
                
                # Calculate precision
                precision = matches / max(len(pred_tokens), 1)
                
                # Apply brevity penalty
                bp = min(1.0, len(pred_tokens) / max(len(ref_tokens), 1)) if len(ref_tokens) > 0 else 0.0
                
                # Final BLEU score
                bleu_score = bp * precision
            
            all_bleu_scores.append(bleu_score)
            count += 1
    
    # Calculate average loss and BLEU score
    avg_loss = total_loss / (count if max_examples is not None else len(dataloader))
    avg_bleu = sum(all_bleu_scores) / len(all_bleu_scores) if all_bleu_scores else 0.0
    
    return avg_loss, avg_bleu

# Start training
train_losses, validation_losses, validation_bleu_scores = train_and_validate(encoder, decoder, encoder_optimizer, decoder_optimizer, dataloader, criterion, n_epochs=n_iters)

# Define function to generate attention visualization data
def generate_attention_visualization(encoder, decoder, dataloader, n_examples=3):
    """Generate attention visualization data for a few examples."""
    attention_data = []
    count = 0
    
    with torch.no_grad():
        for i, (input_tensor, target_tensor) in enumerate(dataloader):
            if count >= n_examples:
                break
                
            # Move tensors to device
            input_tensor = input_tensor[0].to(device)  # Remove batch dimension
            target_tensor = target_tensor[0].to(device)  # Remove batch dimension
            
            # Initialize encoder hidden
            encoder_hidden = encoder.initHidden()
            
            # Get input and target lengths
            input_length = input_tensor.size(0)
            target_length = target_tensor.size(0)
            
            # Initialize encoder outputs tensor
            encoder_outputs = torch.zeros(input_length, 1, encoder.hidden_size, device=device)
            
            # Forward pass through encoder
            for ei in range(input_length):
                encoder_output, encoder_hidden = encoder(input_tensor[ei].unsqueeze(0), encoder_hidden)
                encoder_outputs[ei] = encoder_output
            
            # Prepare decoder input
            decoder_input = torch.tensor([[SOS_token]], device=device)
            decoder_hidden = encoder_hidden
            
            # Initialize attention weights for visualization
            all_attention_weights = torch.zeros(target_length, input_length)
            
            # Forward pass through decoder with teacher forcing
            decoded_words = []
            for di in range(target_length):
                decoder_output, decoder_hidden, attention_weights = decoder(
                    decoder_input, decoder_hidden, encoder_outputs
                )
                
                # Store attention weights for visualization
                all_attention_weights[di] = attention_weights.squeeze().detach().cpu()
                
                # Get the most likely next word
                topv, topi = decoder_output.topk(1)
                if topi.item() == EOS_token:
                    decoded_words.append('<EOS>')
                    break
                else:
                    decoded_words.append(index_to_word[topi.item()])
                
                # Next input is the chosen word
                decoder_input = topi.detach()
            
            # Convert tensors to lists for input and target sentences
            input_words = [index_to_word[idx.item()] for idx in input_tensor]
            target_words = [index_to_word[idx.item()] for idx in target_tensor]
            
            # Store data for visualization
            example_data = {
                'input_sentence': ' '.join(input_words),
                'target_sentence': ' '.join(target_words),
                'decoded_sentence': ' '.join(decoded_words),
                'attention_weights': all_attention_weights
            }
            
            attention_data.append(example_data)
            count += 1
    
    return attention_data

# Generate attention visualization data
attention_data = generate_attention_visualization(encoder, decoder, dataloader, n_examples=3)

def evaluate_and_show_examples(encoder, decoder, dataloader, criterion, n_examples=10):
    # Switch model to evaluation mode
    encoder.eval()
    decoder.eval()

    total_loss = 0
    correct_predictions = 0
    all_attention_weights = []

    # No gradient calculation
    with torch.no_grad():
        for i, (input_tensor, target_tensor) in enumerate(dataloader):
            # Move tensors to the correct device
            input_tensor = input_tensor[0].to(device)
            target_tensor = target_tensor[0].to(device)

            encoder_hidden = encoder.initHidden()

            input_length = input_tensor.size(0)
            target_length = target_tensor.size(0)

            loss = 0
            
            # Store encoder outputs for attention
            encoder_outputs = torch.zeros(input_length, 1, encoder.hidden_size, device=device)

            # Encoding step
            for ei in range(input_length):
                encoder_output, encoder_hidden = encoder(input_tensor[ei].unsqueeze(0), encoder_hidden)
                encoder_outputs[ei] = encoder_output

            # Decoding step
            decoder_input = torch.tensor([[SOS_token]], device=device)
            decoder_hidden = encoder_hidden

            predicted_indices = []
            attention_weights = torch.zeros(target_length, input_length)

            for di in range(target_length):
                decoder_output, decoder_hidden, attn_weights = decoder(decoder_input, decoder_hidden, encoder_outputs)
                attention_weights[di] = attn_weights.squeeze()
                topv, topi = decoder_output.topk(1)
                predicted_indices.append(topi.item())
                decoder_input = topi.squeeze().detach()

                loss += criterion(decoder_output, target_tensor[di].unsqueeze(0))
                if decoder_input.item() == EOS_token:
                    break

            # Calculate and print loss and accuracy for the evaluation
            total_loss += loss.item() / target_length
            if predicted_indices == target_tensor.tolist():
                correct_predictions += 1

            # Optionally, print some examples and plot attention
            if i < n_examples:
                predicted_sentence = ' '.join([index_to_word[index] for index in predicted_indices if index not in (SOS_token, EOS_token)])
                target_sentence = ' '.join([index_to_word[index.item()] for index in target_tensor if index.item() not in (SOS_token, EOS_token)])
                input_sentence = ' '.join([index_to_word[index.item()] for index in input_tensor if index.item() not in (SOS_token, EOS_token)])

                print(f'Input: {input_sentence}, Target: {target_sentence}, Predicted: {predicted_sentence}')
                
                # Store attention weights for visualization
                all_attention_weights.append({
                    'input': input_sentence,
                    'output': predicted_sentence,
                    'attention': attention_weights[:len(predicted_indices), :input_length].cpu().numpy()
                })

        # Print overall evaluation results
        average_loss = total_loss / len(dataloader)
        accuracy = correct_predictions / len(dataloader)
        print(f'Evaluation Loss: {average_loss}, Accuracy: {accuracy}')
        
        # Plot attention matrices for the first few examples
        if n_examples > 0 and len(all_attention_weights) > 0:
            plot_attention_matrices(all_attention_weights[:min(n_examples, len(all_attention_weights))])

def plot_attention_matrices(attention_data, n_examples=3):
    """Plot Transformer-style attention matrices for better visualization of word alignments."""
    n = min(n_examples, len(attention_data))
    fig, axes = plt.subplots(n, 1, figsize=(12, 5*n))
    if n == 1:
        axes = [axes]
    
    # Create a custom colormap for better visualization
    # Using a perceptually uniform colormap for better interpretation
    colors = [(0.0, 0.0, 0.7),  # Deep blue for low values
              (0.9, 0.9, 0.9),  # Light gray for middle values
              (0.7, 0.0, 0.0)]  # Deep red for high values
    cmap_name = 'transformer_attention_cmap'
    cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=100)
    
    for i, data in enumerate(attention_data[:n]):
        ax = axes[i]
        attention = data['attention']
        input_sentence = data['input'].split()
        output_sentence = data['output'].split()
        
        # Display the attention matrix with the custom colormap
        im = ax.imshow(attention, cmap=cm, aspect='auto', interpolation='nearest')
        
        # Add grid to better visualize the alignment between words
        ax.set_xticks(np.arange(-.5, len(input_sentence), 1), minor=True)
        ax.set_yticks(np.arange(-.5, len(output_sentence), 1), minor=True)
        ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
        
        # Set labels with better formatting
        ax.set_xticks(range(len(input_sentence)))
        ax.set_yticks(range(len(output_sentence)))
        ax.set_xticklabels(input_sentence, rotation=45, ha='right', fontsize=10, fontweight='bold')
        ax.set_yticklabels(output_sentence, fontsize=10, fontweight='bold')
        
        # Add labels and title with better formatting
        ax.set_xlabel('Input Sentence (Source)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Output Sentence (Target)', fontsize=12, fontweight='bold')
        ax.set_title(f'Transformer Attention Matrix {i+1}', fontsize=14, fontweight='bold')
        
        # Add a colorbar with better formatting
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label('Attention Weight', fontsize=10, fontweight='bold')
        
        # Add text annotations for better readability
        # Only annotate cells with significant attention to avoid clutter
        for y in range(len(output_sentence)):
            for x in range(len(input_sentence)):
                # Adjust threshold as needed - higher threshold for Transformer attention
                # as it tends to be more distributed
                if attention[y, x] > 0.15:  
                    # Choose text color based on background for better contrast
                    text_color = 'white' if attention[y, x] > 0.4 else 'black'
                    ax.text(x, y, f'{attention[y, x]:.2f}', 
                             ha='center', va='center', color=text_color, 
                             fontsize=8, fontweight='bold')
        
        # Highlight the maximum attention value for each output word
        for y in range(len(output_sentence)):
            # Find top-2 attention positions for each output word (Transformer often distributes attention)
            attention_row = attention[y]
            top_indices = torch.topk(torch.tensor(attention_row), min(2, len(input_sentence))).indices.numpy()
            
            # Highlight primary attention
            rect = plt.Rectangle((top_indices[0]-0.5, y-0.5), 1, 1, fill=False, 
                                edgecolor='yellow', linewidth=2)
            ax.add_patch(rect)
            
            # Highlight secondary attention if available
            if len(top_indices) > 1:
                rect2 = plt.Rectangle((top_indices[1]-0.5, y-0.5), 1, 1, fill=False, 
                                    edgecolor='orange', linewidth=1, linestyle='--')
                ax.add_patch(rect2)
        
        # Add a title explaining the visualization
        if i == 0:
            plt.figtext(0.5, 0.01, 
                      "Transformer-style attention shows how each output word (row) attends to input words (columns).\n" +
                      "Yellow boxes highlight primary attention, orange dashed boxes show secondary attention.", 
                      ha="center", fontsize=10, bbox={"facecolor":"lightgray", "alpha":0.5, "pad":5})
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)  # Make room for the explanation text
    plt.savefig('transformer_attention_matrices.png')
    plt.show()

# Plot the attention matrices for visualization
plot_attention_matrices(attention_data, n_examples=3)

# Visualize training and validation metrics
plt.figure(figsize=(12, 5))

# Plot losses
plt.subplot(1, 2, 1)
plt.plot(range(1, n_iters + 1), train_losses, label='Training Loss')
plt.plot(range(1, n_iters + 1), validation_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True)

# Plot BLEU scores
plt.subplot(1, 2, 2)
plt.plot(range(1, n_iters + 1), validation_bleu_scores, label='BLEU Score', color='green')
plt.xlabel('Epochs')
plt.ylabel('BLEU Score')
plt.title('Translation BLEU Score')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('english2purepecha_training_metrics.png')
plt.show()

# Print final validation metrics
final_loss = validation_losses[-1] if validation_losses else 0
final_bleu = validation_bleu_scores[-1] if validation_bleu_scores else 0
print(f'\nFinal Validation Results:')
print(f'Loss: {final_loss:.4f}')
print(f'BLEU Score: {final_bleu:.4f}')

# Save the trained model
torch.save({
    'encoder_state_dict': encoder.state_dict(),
    'decoder_state_dict': decoder.state_dict(),
    'encoder_optimizer': encoder_optimizer.state_dict(),
    'decoder_optimizer': decoder_optimizer.state_dict(),
    'word_to_index': word_to_index,
    'index_to_word': index_to_word
}, 'english_to_purepecha_model.pt')

print("Model saved to 'english_to_purepecha_model.pt'")

# Perform evaluation with examples
print(f'\nExample translations:')
evaluate_and_show_examples(encoder, decoder, dataloader, criterion, n_examples=5)
