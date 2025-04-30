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
from typing import Literal
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize

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

class RNNTranslator(nn.Module):
    def __init__(self, input_vocab_size, output_vocab_size, 
                 embed_dim=256, hidden_dim=512, num_layers=2, 
                 rnn_type: Literal['lstm', 'gru'] = 'lstm',
                 dropout=0.3):
        super().__init__()
        self.rnn_type = rnn_type
        
        # Embedding layers with dropout
        self.input_embedding = nn.Embedding(input_vocab_size, embed_dim)
        self.output_embedding = nn.Embedding(output_vocab_size, embed_dim)
        self.embedding_dropout = nn.Dropout(dropout)
        
        # Encoder RNN
        if rnn_type == 'lstm':
            self.encoder = nn.LSTM(
                input_size=embed_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0,
                batch_first=True
            )
        else:  # gru
            self.encoder = nn.GRU(
                input_size=embed_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0,
                batch_first=True
            )
            
        # Decoder RNN
        if rnn_type == 'lstm':
            self.decoder = nn.LSTM(
                input_size=embed_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0,
                batch_first=True
            )
        else:  # gru
            self.decoder = nn.GRU(
                input_size=embed_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0,
                batch_first=True
            )
            
        # Attention mechanism with dropout
        self.attention = nn.Linear(hidden_dim * 2, hidden_dim)
        self.attention_combine = nn.Linear(hidden_dim * 2, hidden_dim)
        self.attention_dropout = nn.Dropout(dropout)
        
        # Output layer with dropout
        self.final_layer = nn.Linear(hidden_dim, output_vocab_size)
        self.output_dropout = nn.Dropout(dropout)
        
    def forward(self, src, tgt):
        # src: [batch_size, seq_len]
        # tgt: [batch_size, seq_len]
        
        # Get embeddings with dropout
        src_emb = self.embedding_dropout(self.input_embedding(src))
        tgt_emb = self.embedding_dropout(self.output_embedding(tgt))
        
        # Encoder
        if self.rnn_type == 'lstm':
            enc_output, (enc_hidden, enc_cell) = self.encoder(src_emb)
        else:
            enc_output, enc_hidden = self.encoder(src_emb)
        
        # Initialize decoder hidden state with encoder's last hidden state
        if self.rnn_type == 'lstm':
            dec_hidden = (enc_hidden, enc_cell)
        else:
            dec_hidden = enc_hidden
            
        # Decoder
        dec_output = []
        for t in range(tgt_emb.size(1)):
            # Get current input
            dec_input = tgt_emb[:, t:t+1]
            
            # Run decoder step
            if self.rnn_type == 'lstm':
                dec_out, dec_hidden = self.decoder(dec_input, dec_hidden)
            else:
                dec_out, dec_hidden = self.decoder(dec_input, dec_hidden)
                
            # Apply attention with dropout
            attn_weights = torch.bmm(dec_out, enc_output.transpose(1, 2))
            attn_weights = torch.softmax(attn_weights, dim=2)
            context = torch.bmm(attn_weights, enc_output)
            
            # Combine attention with decoder output
            combined = torch.cat((dec_out, context), dim=2)
            combined = self.attention_combine(combined)
            combined = self.attention_dropout(combined)
            
            dec_output.append(combined)
            
        # Stack decoder outputs
        dec_output = torch.cat(dec_output, dim=1)
        
        # Final projection with dropout
        output = self.final_layer(self.output_dropout(dec_output))
        return output

def train_and_evaluate(model, train_loader, val_loader, num_epochs=10, learning_rate=0.001, tgt_vocab=None):
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5)
    
    device = setup_gpu()
    print(f"Using device: {device}")
    model = model.to(device)
    
    best_val_loss = float('inf')
    best_model_state = None
    best_bleu_score = 0.0
    
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
            output = model(src, tgt[:, :-1])
            loss = criterion(output.view(-1, output.size(-1)), tgt[:, 1:].reshape(-1))
            
            loss.backward()
            optimizer.step()
            
            predictions = output.argmax(dim=-1)
            acc = calculate_accuracy(predictions, tgt[:, 1:])
            
            train_loss += loss.item()
            train_acc += acc
            train_batches += 1
            
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
        all_predictions = []
        all_targets = []
        
        val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]')
        with torch.no_grad():
            for src, tgt in val_pbar:
                src, tgt = src.to(device), tgt.to(device)
                
                output = model(src, tgt[:, :-1])
                loss = criterion(output.view(-1, output.size(-1)), tgt[:, 1:].reshape(-1))
                
                predictions = output.argmax(dim=-1)
                acc = calculate_accuracy(predictions, tgt[:, 1:])
                
                val_loss += loss.item()
                val_acc += acc
                val_batches += 1
                
                # Store predictions and targets for BLEU score
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(tgt[:, 1:].cpu().numpy())
                
                val_pbar.set_postfix({
                    'loss': f'{val_loss/val_batches:.4f}',
                    'acc': f'{val_acc/val_batches:.4f}',
                    'gpu_mem': f'{cuda.memory_allocated()/1024**2:.1f}MB'
                })
        
        avg_val_loss = val_loss / len(val_loader)
        avg_val_acc = val_acc / len(val_loader)
        
        # Calculate BLEU score
        bleu_score = calculate_bleu_score(all_predictions, all_targets, tgt_vocab)
        
        # Update learning rate
        scheduler.step(avg_val_loss)
        
        # Print epoch summary
        print(f'\nEpoch {epoch+1}/{num_epochs} Summary:')
        print(f'Training Loss: {avg_train_loss:.4f} | Training Accuracy: {avg_train_acc:.4f}')
        print(f'Validation Loss: {avg_val_loss:.4f} | Validation Accuracy: {avg_val_acc:.4f}')
        print(f'BLEU Score: {bleu_score:.4f}')
        print(f'Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')
        
        # Save best model based on BLEU score
        if bleu_score > best_bleu_score:
            best_bleu_score = bleu_score
            best_model_state = model.state_dict()
            print('New best model saved!')
    
    # Load best model
    model.load_state_dict(best_model_state)
    return model

def calculate_bleu_score(predictions, targets, tgt_vocab, max_n=4):
    """
    Calculate BLEU score for translation predictions.
    
    Args:
        predictions: List of predicted token indices
        targets: List of target token indices
        tgt_vocab: Target vocabulary dictionary
        max_n: Maximum n-gram order for BLEU score calculation
    
    Returns:
        Average BLEU score across all predictions
    """
    # Create reverse vocabulary mapping
    idx_to_word = {idx: word for word, idx in tgt_vocab.items()}
    
    # Convert indices to words
    pred_texts = []
    ref_texts = []
    
    for pred, ref in zip(predictions, targets):
        # Convert indices to words, ignoring padding tokens
        pred_words = [idx_to_word[idx] for idx in pred if idx not in (tgt_vocab['<pad>'], tgt_vocab['<sos>'], tgt_vocab['<eos>'])]
        ref_words = [idx_to_word[idx] for idx in ref if idx not in (tgt_vocab['<pad>'], tgt_vocab['<sos>'], tgt_vocab['<eos>'])]
        
        if pred_words and ref_words:  # Only add non-empty sequences
            pred_texts.append(pred_words)
            ref_texts.append([ref_words])  # Wrap in list for sentence_bleu
    
    # Calculate BLEU score
    smoothie = SmoothingFunction().method1
    bleu_scores = []
    
    for pred, ref in zip(pred_texts, ref_texts):
        try:
            score = sentence_bleu(ref, pred, smoothing_function=smoothie)
            bleu_scores.append(score)
        except:
            # Skip if there's an error (e.g., empty sequences)
            continue
    
    return np.mean(bleu_scores) if bleu_scores else 0.0

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
    
    # Create datasets - using the same data for both training and validation
    train_dataset = PurepechaDataset(src_texts, tgt_texts, src_vocab, tgt_vocab)
    val_dataset = PurepechaDataset(src_texts, tgt_texts, src_vocab, tgt_vocab)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)  # No shuffle for validation
    
    # Train and evaluate LSTM model
    print("\nTraining LSTM model...")
    lstm_model = RNNTranslator(
        input_vocab_size=len(src_vocab),
        output_vocab_size=len(tgt_vocab),
        rnn_type='lstm'
    )
    lstm_model = train_and_evaluate(lstm_model, train_loader, val_loader, tgt_vocab=tgt_vocab)
    
    # Train and evaluate GRU model
    print("\nTraining GRU model...")
    gru_model = RNNTranslator(
        input_vocab_size=len(src_vocab),
        output_vocab_size=len(tgt_vocab),
        rnn_type='gru'
    )
    gru_model = train_and_evaluate(gru_model, train_loader, val_loader, tgt_vocab=tgt_vocab)

if __name__ == '__main__':
    main()
