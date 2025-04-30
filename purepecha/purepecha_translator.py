import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import torch.cuda as cuda
import torch.backends.cudnn as cudnn

# Custom Multi-Head Self-Attention module
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = attn.softmax(dim=-1)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_dim, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attention = MultiHeadSelfAttention(embed_dim, num_heads)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(
            in_features=embed_dim,
            hidden_features=mlp_dim,
            out_features=embed_dim
        )
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class PurepechaTranslator(nn.Module):
    def __init__(self, input_vocab_size, output_vocab_size, embed_dim=256, num_heads=8, num_layers=6, mlp_dim=512):
        super().__init__()
        self.input_embedding = nn.Embedding(input_vocab_size, embed_dim)
        self.output_embedding = nn.Embedding(output_vocab_size, embed_dim)
        
        self.encoder_layers = nn.ModuleList([
            TransformerEncoder(embed_dim, num_heads, mlp_dim)
            for _ in range(num_layers)
        ])
        
        self.decoder_layers = nn.ModuleList([
            TransformerEncoder(embed_dim, num_heads, mlp_dim)
            for _ in range(num_layers)
        ])
        
        self.final_layer = nn.Linear(embed_dim, output_vocab_size)
        
    def forward(self, src, tgt):
        # src: [batch_size, seq_len]
        # tgt: [batch_size, seq_len]
        
        src_emb = self.input_embedding(src)
        tgt_emb = self.output_embedding(tgt)
        
        # Encoder
        enc_out = src_emb
        for layer in self.encoder_layers:
            enc_out = layer(enc_out)
            
        # Decoder
        dec_out = tgt_emb
        for layer in self.decoder_layers:
            dec_out = layer(dec_out)
            
        # Final projection
        output = self.final_layer(dec_out)
        return output

class PurepechaDataset(Dataset):
    def __init__(self, src_texts, tgt_texts, src_vocab, tgt_vocab, max_length=50):
        self.src_texts = src_texts
        self.tgt_texts = tgt_texts
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.max_length = max_length
        
    def __len__(self):
        return len(self.src_texts)
    
    def __getitem__(self, idx):
        src_text = self.src_texts[idx]
        tgt_text = self.tgt_texts[idx]
        
        # Convert text to indices
        src_indices = [self.src_vocab.get(char, self.src_vocab['<unk>']) for char in src_text]
        tgt_indices = [self.tgt_vocab.get(char, self.tgt_vocab['<unk>']) for char in tgt_text]
        
        # Pad sequences
        src_indices = src_indices[:self.max_length] + [self.src_vocab['<pad>']] * (self.max_length - len(src_indices))
        tgt_indices = tgt_indices[:self.max_length] + [self.tgt_vocab['<pad>']] * (self.max_length - len(tgt_indices))
        
        return torch.tensor(src_indices), torch.tensor(tgt_indices)

def create_vocab(texts):
    vocab = {'<pad>': 0, '<unk>': 1, '<sos>': 2, '<eos>': 3}
    for text in texts:
        for char in text:
            if char not in vocab:
                vocab[char] = len(vocab)
    return vocab

def load_data(file_path):
    df = pd.read_csv(file_path, sep='\t', header=None, names=['english', 'purepecha'])
    return df['english'].tolist(), df['purepecha'].tolist()

def print_torch_info():
    print("\nPyTorch Information:")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {cuda.is_available()}")
    if cuda.is_available():
        print(f"Current CUDA device: {cuda.current_device()}")
        print(f"Device name: {cuda.get_device_name(0)}")
        print(f"CUDA device count: {cuda.device_count()}")
        print(f"CUDA device properties: {cuda.get_device_properties(0)}")
    print()

def calculate_accuracy(predictions, targets, pad_idx=0):
    # Calculate accuracy ignoring padding tokens
    mask = targets != pad_idx
    correct = (predictions == targets) * mask
    return correct.sum().item() / mask.sum().item()

def setup_gpu():
    if not cuda.is_available():
        print("Warning: CUDA is not available. Training will be performed on CPU.")
        return torch.device('cpu')
    
    # Set CUDA device
    device = torch.device('cuda')
    
    # Enable cuDNN benchmarking for faster training
    cudnn.benchmark = True
    
    # Set deterministic mode for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(42)
    if cuda.is_available():
        torch.cuda.manual_seed_all(42)
    
    # Clear GPU memory
    if cuda.is_available():
        torch.cuda.empty_cache()
    
    return device

def train_model(model, train_loader, val_loader, num_epochs=10, learning_rate=0.001):
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5)
    
    # Setup GPU
    device = setup_gpu()
    print(f"Using device: {device}")
    
    # Move model to GPU
    model = model.to(device)
    
    # Enable gradient scaler for mixed precision training
    scaler = torch.cuda.amp.GradScaler()
    
    best_val_loss = float('inf')
    best_model_state = None
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        train_acc = 0
        train_batches = 0
        
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        for src, tgt in train_pbar:
            src, tgt = src.to(device), tgt.to(device)
            
            optimizer.zero_grad()
            
            # Use automatic mixed precision
            with torch.cuda.amp.autocast():
                output = model(src, tgt[:, :-1])
                loss = criterion(output.view(-1, output.size(-1)), tgt[:, 1:].reshape(-1))
            
            # Scale gradients and update weights
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            # Calculate accuracy
            predictions = output.argmax(dim=-1)
            acc = calculate_accuracy(predictions, tgt[:, 1:])
            
            train_loss += loss.item()
            train_acc += acc
            train_batches += 1
            
            # Update progress bar
            train_pbar.set_postfix({
                'loss': f'{train_loss/train_batches:.4f}',
                'acc': f'{train_acc/train_batches:.4f}',
                'gpu_mem': f'{cuda.memory_allocated()/1024**2:.1f}MB'
            })
        
        avg_train_loss = train_loss / len(train_loader)
        avg_train_acc = train_acc / len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_acc = 0
        val_batches = 0
        
        val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]')
        with torch.no_grad():
            for src, tgt in val_pbar:
                src, tgt = src.to(device), tgt.to(device)
                
                with torch.cuda.amp.autocast():
                    output = model(src, tgt[:, :-1])
                    loss = criterion(output.view(-1, output.size(-1)), tgt[:, 1:].reshape(-1))
                
                predictions = output.argmax(dim=-1)
                acc = calculate_accuracy(predictions, tgt[:, 1:])
                
                val_loss += loss.item()
                val_acc += acc
                val_batches += 1
                
                val_pbar.set_postfix({
                    'loss': f'{val_loss/val_batches:.4f}',
                    'acc': f'{val_acc/val_batches:.4f}',
                    'gpu_mem': f'{cuda.memory_allocated()/1024**2:.1f}MB'
                })
        
        avg_val_loss = val_loss / len(val_loader)
        avg_val_acc = val_acc / len(val_loader)
        
        # Update learning rate
        scheduler.step(avg_val_loss)
        
        # Print epoch summary
        print(f'\nEpoch {epoch+1}/{num_epochs} Summary:')
        print(f'Training Loss: {avg_train_loss:.4f} | Training Accuracy: {avg_train_acc:.4f}')
        print(f'Validation Loss: {avg_val_loss:.4f} | Validation Accuracy: {avg_val_acc:.4f}')
        print(f'Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')
        print(f'GPU Memory Usage: {cuda.memory_allocated()/1024**2:.1f}MB')
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict()
            print('New best model saved!')
    
    # Load best model
    model.load_state_dict(best_model_state)
    return model

def main():
    print_torch_info()
    
    # Load data
    src_texts, tgt_texts = load_data('assets/purepecha_data.tsv')
    print(f"Loaded {len(src_texts)} translation pairs")
    
    # Create vocabularies
    src_vocab = create_vocab(src_texts)
    tgt_vocab = create_vocab(tgt_texts)
    print(f"Source vocabulary size: {len(src_vocab)}")
    print(f"Target vocabulary size: {len(tgt_vocab)}")
    
    # Split data
    train_src, val_src, train_tgt, val_tgt = train_test_split(
        src_texts, tgt_texts, test_size=0.2, random_state=42
    )
    print(f"Training samples: {len(train_src)}")
    print(f"Validation samples: {len(val_src)}")
    
    # Create datasets
    train_dataset = PurepechaDataset(train_src, train_tgt, src_vocab, tgt_vocab)
    val_dataset = PurepechaDataset(val_src, val_tgt, src_vocab, tgt_vocab)
    
    # Create dataloaders with num_workers for faster data loading
    num_workers = min(4, cuda.device_count() * 4) if cuda.is_available() else 0
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=32, num_workers=num_workers, pin_memory=True)
    
    # Initialize model
    model = PurepechaTranslator(
        input_vocab_size=len(src_vocab),
        output_vocab_size=len(tgt_vocab)
    )
    
    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel Summary:")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Train model
    model = train_model(model, train_loader, val_loader)

if __name__ == '__main__':
    main() 