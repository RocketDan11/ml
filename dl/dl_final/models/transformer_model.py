"""
Transformer model for sequence-to-sequence translation.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary

class PositionalEncoding(nn.Module):
    """
    Positional encoding for the transformer model.
    """
    def __init__(self, d_model, max_seq_length=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # Apply sine to even indices
        pe[:, 0::2] = torch.sin(position * div_term)
        # Apply cosine to odd indices
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add batch dimension
        pe = pe.unsqueeze(0)
        
        # Register buffer (persistent state of the module)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        """
        Forward pass of the positional encoding.
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            
        Returns:
            x: Input tensor with positional encoding [batch_size, seq_len, d_model]
        """
        # Add positional encoding to the input
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class TransformerEncoder(nn.Module):
    """
    Encoder for the transformer model.
    """
    def __init__(
        self,
        input_dim,
        embedding_dim,
        hidden_dim,
        num_layers,
        num_heads,
        ff_dim,
        dropout,
        max_seq_length=5000
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout = dropout
        
        # Token embedding layer
        self.token_embedding = nn.Embedding(input_dim, embedding_dim)
        
        # Scale embeddings by sqrt(hidden_dim)
        self.scale = math.sqrt(embedding_dim)
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(embedding_dim, max_seq_length, dropout)
        
        # Encoder layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True
        )
        
        # Encoder
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Projections if embedding_dim != hidden_dim
        self.embedding_to_hidden = nn.Linear(embedding_dim, hidden_dim) if embedding_dim != hidden_dim else None
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        """
        Forward pass of the encoder.
        
        Args:
            src: Source sequence [batch_size, src_len]
            src_mask: Mask for self-attention [src_len, src_len]
            src_key_padding_mask: Mask for padded tokens [batch_size, src_len]
            
        Returns:
            encoder_output: Encoded sequence [batch_size, src_len, hidden_dim]
        """
        # src: [batch_size, src_len]
        
        # Embed tokens and apply positional encoding
        src_embedded = self.token_embedding(src) * self.scale
        src_embedded = self.positional_encoding(src_embedded)
        # src_embedded: [batch_size, src_len, embedding_dim]
        
        # Project to hidden_dim if needed
        if self.embedding_to_hidden is not None:
            src_embedded = self.embedding_to_hidden(src_embedded)
        # src_embedded: [batch_size, src_len, hidden_dim]
        
        # Apply transformer encoder
        encoder_output = self.transformer_encoder(
            src_embedded, mask=src_mask, src_key_padding_mask=src_key_padding_mask
        )
        # encoder_output: [batch_size, src_len, hidden_dim]
        
        return encoder_output

class TransformerDecoder(nn.Module):
    """
    Decoder for the transformer model.
    """
    def __init__(
        self,
        output_dim,
        embedding_dim,
        hidden_dim,
        num_layers,
        num_heads,
        ff_dim,
        dropout,
        max_seq_length=5000
    ):
        super().__init__()
        
        self.output_dim = output_dim
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout = dropout
        
        # Token embedding layer
        self.token_embedding = nn.Embedding(output_dim, embedding_dim)
        
        # Scale embeddings by sqrt(hidden_dim)
        self.scale = math.sqrt(embedding_dim)
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(embedding_dim, max_seq_length, dropout)
        
        # Decoder layer
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True
        )
        
        # Decoder
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers)
        
        # Projections if embedding_dim != hidden_dim
        self.embedding_to_hidden = nn.Linear(embedding_dim, hidden_dim) if embedding_dim != hidden_dim else None
        
        # Output projection
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, trg, encoder_output, trg_mask=None, memory_mask=None, 
                trg_key_padding_mask=None, memory_key_padding_mask=None):
        """
        Forward pass of the decoder.
        
        Args:
            trg: Target sequence [batch_size, trg_len]
            encoder_output: Encoded sequence from encoder [batch_size, src_len, hidden_dim]
            trg_mask: Mask for self-attention in decoder [trg_len, trg_len]
            memory_mask: Mask for attention over encoder output [trg_len, src_len]
            trg_key_padding_mask: Mask for padded tokens in target [batch_size, trg_len]
            memory_key_padding_mask: Mask for padded tokens in source [batch_size, src_len]
            
        Returns:
            output: Decoded output [batch_size, trg_len, output_dim]
        """
        # trg: [batch_size, trg_len]
        # encoder_output: [batch_size, src_len, hidden_dim]
        
        # Embed tokens and apply positional encoding
        trg_embedded = self.token_embedding(trg) * self.scale
        trg_embedded = self.positional_encoding(trg_embedded)
        # trg_embedded: [batch_size, trg_len, embedding_dim]
        
        # Project to hidden_dim if needed
        if self.embedding_to_hidden is not None:
            trg_embedded = self.embedding_to_hidden(trg_embedded)
        # trg_embedded: [batch_size, trg_len, hidden_dim]
        
        # Apply transformer decoder
        decoder_output = self.transformer_decoder(
            trg_embedded, encoder_output,
            tgt_mask=trg_mask,
            memory_mask=memory_mask,
            tgt_key_padding_mask=trg_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask
        )
        # decoder_output: [batch_size, trg_len, hidden_dim]
        
        # Apply output layer
        output = self.fc_out(decoder_output)
        # output: [batch_size, trg_len, output_dim]
        
        return output

class TransformerSeq2Seq(nn.Module):
    """
    Transformer sequence-to-sequence model.
    """
    def __init__(
        self,
        encoder_vocab_size,
        decoder_vocab_size,
        encoder_embedding_dim,
        decoder_embedding_dim,
        hidden_dim,
        ff_dim,
        num_layers,
        num_heads,
        dropout,
        device=torch.device('cpu'),
        max_seq_length=5000
    ):
        super().__init__()
        
        self.encoder = TransformerEncoder(
            encoder_vocab_size,
            encoder_embedding_dim,
            hidden_dim,
            num_layers,
            num_heads,
            ff_dim,
            dropout,
            max_seq_length
        )
        
        self.decoder = TransformerDecoder(
            decoder_vocab_size,
            decoder_embedding_dim,
            hidden_dim,
            num_layers,
            num_heads,
            ff_dim,
            dropout,
            max_seq_length
        )
        
        self.device = device
        
        # Initialize parameters with Glorot / fan_avg
        self._reset_parameters()
        
        # Create SOS and EOS token indices
        self.sos_idx = 1
        self.eos_idx = 2
        
    def _reset_parameters(self):
        """
        Initialize parameters with Glorot / fan_avg.
        """
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        
    def create_masks(self, src, trg, src_pad_idx=0, trg_pad_idx=0):
        """
        Create masks for transformer.
        
        Args:
            src: Source sequence [batch_size, src_len]
            trg: Target sequence [batch_size, trg_len]
            src_pad_idx: Index of <PAD> token in source vocabulary
            trg_pad_idx: Index of <PAD> token in target vocabulary
            
        Returns:
            src_mask: Mask for encoder self-attention [src_len, src_len]
            trg_mask: Mask for decoder self-attention [trg_len, trg_len]
            src_padding_mask: Mask for padded tokens in source [batch_size, src_len]
            trg_padding_mask: Mask for padded tokens in target [batch_size, trg_len]
        """
        # Source padding mask: 1 for padding, 0 for token
        src_padding_mask = (src == src_pad_idx)
        
        # Target padding mask: 1 for padding, 0 for token
        if trg is not None:
            trg_padding_mask = (trg == trg_pad_idx)
        else:
            trg_padding_mask = None
        
        # Source mask: None as encoder self-attention is fully visible
        src_mask = None
        
        # Target mask: Mask for decoder self-attention
        if trg is not None:
            trg_len = trg.shape[1]
            
            # Create square matrix of size (trg_len, trg_len)
            trg_mask = torch.triu(torch.ones(trg_len, trg_len), diagonal=1).bool().to(self.device)
        else:
            trg_mask = None
            
        return src_mask, trg_mask, src_padding_mask, trg_padding_mask
    
    def forward(self, src, src_lengths, trg, teacher_forcing_ratio=1.0):
        """
        Forward pass of the transformer model.
        
        Args:
            src: Source sequence [batch_size, src_len]
            src_lengths: Lengths of source sequences [batch_size]
            trg: Target sequence [batch_size, trg_len]
            teacher_forcing_ratio: Not used in transformer, kept for API consistency
            
        Returns:
            output: Sequence of predictions [batch_size, trg_len-1, output_dim]
        """
        # src: [batch_size, src_len]
        # trg: [batch_size, trg_len]
        
        # Create masks
        src_mask, trg_mask, src_padding_mask, trg_padding_mask = self.create_masks(src, trg[:, :-1])
        
        # Encode the source sequence
        encoder_output = self.encoder(src, src_mask, src_padding_mask)
        # encoder_output: [batch_size, src_len, hidden_dim]
        
        # Decode the target sequence
        output = self.decoder(
            trg[:, :-1], encoder_output,
            trg_mask, None,
            trg_padding_mask, src_padding_mask
        )
        # output: [batch_size, trg_len-1, output_dim]
        
        return output
    
    def translate(self, src, src_lengths, max_length=50):
        """
        Translate a source sequence.
        
        Args:
            src: Source sequence [1, src_len]
            src_lengths: Length of source sequence [1]
            max_length: Maximum length of translation
            
        Returns:
            translation: Sequence of token indices [max_length]
        """
        self.eval()
        
        with torch.no_grad():
            # Encode the source sequence
            src_mask, _, src_padding_mask, _ = self.create_masks(src, None)
            encoder_output = self.encoder(src, src_mask, src_padding_mask)
            
            # Start with <SOS> token
            curr_tokens = torch.tensor([[self.sos_idx]], device=self.device)
            
            # Store translation
            translation = [self.sos_idx]
            
            # Decode one token at a time
            for _ in range(max_length - 1):
                # Create masks
                _, trg_mask, _, trg_padding_mask = self.create_masks(src, curr_tokens)
                
                # Decode next token
                output = self.decoder(
                    curr_tokens, encoder_output,
                    trg_mask, None,
                    trg_padding_mask, src_padding_mask
                )
                
                # Get predicted token
                pred_token = output[:, -1, :].argmax(dim=1).item()
                translation.append(pred_token)
                
                # Stop if <EOS> token is predicted
                if pred_token == self.eos_idx:
                    break
                
                # Add predicted token to current tokens
                curr_tokens = torch.cat([
                    curr_tokens, torch.tensor([[pred_token]], device=self.device)
                ], dim=1)
        
        return translation
    
    def translate_batch(self, src, src_lengths, max_length=50):
        """
        Translate a batch of source sequences.
        
        Args:
            src: Source sequences [batch_size, src_len]
            src_lengths: Lengths of source sequences [batch_size]
            max_length: Maximum length of translations
            
        Returns:
            translations: List of translations (lists of token indices)
        """
        self.eval()
        batch_size = src.shape[0]
        
        with torch.no_grad():
            # Encode the source sequences
            src_mask, _, src_padding_mask, _ = self.create_masks(src, None)
            encoder_output = self.encoder(src, src_mask, src_padding_mask)
            
            # Start with <SOS> token
            curr_tokens = torch.full((batch_size, 1), self.sos_idx, device=self.device)
            
            # Store translations
            translations = [[] for _ in range(batch_size)]
            active_idxs = list(range(batch_size))
            
            # Decode one token at a time
            for _ in range(max_length - 1):
                # Create masks
                _, trg_mask, _, trg_padding_mask = self.create_masks(src, curr_tokens)
                
                # Decode next token
                output = self.decoder(
                    curr_tokens, encoder_output,
                    trg_mask, None,
                    trg_padding_mask, src_padding_mask
                )
                
                # Get predicted tokens
                pred_tokens = output[:, -1, :].argmax(dim=1)
                
                # Update translations and active indices
                new_active_idxs = []
                for i, idx in enumerate(active_idxs):
                    token = pred_tokens[i].item()
                    translations[idx].append(token)
                    
                    # Keep only active sequences (not ended with <EOS>)
                    if token != self.eos_idx:
                        new_active_idxs.append(idx)
                
                # Stop if all sequences have ended
                if not new_active_idxs:
                    break
                
                # Update active indices
                active_idxs = new_active_idxs
                
                # If all sequences have ended, break
                if not active_idxs:
                    break
                
                # Filter active sequences and update current tokens
                active_preds = pred_tokens[active_idxs].unsqueeze(1)
                
                # Add predicted tokens to current tokens
                curr_tokens = torch.cat([
                    curr_tokens, torch.zeros((batch_size, 1), dtype=torch.long, device=self.device)
                ], dim=1)
                curr_tokens[active_idxs, -1] = active_preds.squeeze(1)
        
        return translations
    
    def model_summary(self, src_shape, trg_shape):
        """
        Print model summary.
        
        Args:
            src_shape: Shape of source tensor [batch_size, src_len]
            trg_shape: Shape of target tensor [batch_size, trg_len]
        """
        return summary(
            self,
            input_data=[
                torch.zeros(src_shape, dtype=torch.long, device=self.device),
                torch.tensor([src_shape[1]] * src_shape[0], device=self.device),
                torch.zeros(trg_shape, dtype=torch.long, device=self.device)
            ],
            col_names=["input_size", "output_size", "num_params", "trainable"],
            col_width=20,
            row_settings=["var_names"]
        )