"""
LSTM/GRU RNN model for sequence-to-sequence translation.
"""
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary

class Encoder(nn.Module):
    """
    Encoder for sequence-to-sequence model.
    Uses bidirectional LSTM/GRU to encode source sequence.
    """
    def __init__(
        self,
        input_dim,
        embedding_dim,
        hidden_dim,
        num_layers,
        dropout,
        rnn_type='lstm'
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.rnn_type = rnn_type.lower()
        
        # Embedding layer
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        
        # RNN layer
        if self.rnn_type == 'lstm':
            self.rnn = nn.LSTM(
                embedding_dim,
                hidden_dim,
                num_layers=num_layers,
                bidirectional=True,
                dropout=dropout if num_layers > 1 else 0,
                batch_first=True
            )
        elif self.rnn_type == 'gru':
            self.rnn = nn.GRU(
                embedding_dim,
                hidden_dim,
                num_layers=num_layers,
                bidirectional=True,
                dropout=dropout if num_layers > 1 else 0,
                batch_first=True
            )
        else:
            raise ValueError(f"Unsupported RNN type: {rnn_type}. Use 'lstm' or 'gru'.")
        
        # Fully connected layer to combine bidirectional states
        self.fc = nn.Linear(hidden_dim * 2, hidden_dim)
        
    def forward(self, src, src_lengths):
        """
        Forward pass of the encoder.
        
        Args:
            src: Source sequence [batch_size, src_len]
            src_lengths: Lengths of source sequences [batch_size]
            
        Returns:
            outputs: Sequence of hidden states [batch_size, src_len, hidden_dim * 2]
            hidden: Final hidden state [num_layers, batch_size, hidden_dim]
        """
        # src: [batch_size, src_len]
        
        # Embed the source sequence
        embedded = self.embedding(src)
        # embedded: [batch_size, src_len, embedding_dim]
        
        # Pack padded sequence
        packed_embedded = nn.utils.rnn.pack_padded_sequence(
            embedded, src_lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        
        # Pass through RNN
        if self.rnn_type == 'lstm':
            packed_outputs, (hidden, cell) = self.rnn(packed_embedded)
            # hidden: [num_layers * 2, batch_size, hidden_dim]
            # cell: [num_layers * 2, batch_size, hidden_dim]
        else:  # GRU
            packed_outputs, hidden = self.rnn(packed_embedded)
            # hidden: [num_layers * 2, batch_size, hidden_dim]
        
        # Unpack outputs
        outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs, batch_first=True)
        # outputs: [batch_size, src_len, hidden_dim * 2]
        
        # Process hidden state for decoder
        # Separate forward and backward states
        hidden_forward = hidden[0:self.num_layers]
        hidden_backward = hidden[self.num_layers:]
        
        # Combine forward and backward states - concatenate along last dimension
        hidden = torch.cat([hidden_forward, hidden_backward], dim=2)
        # hidden: [num_layers, batch_size, hidden_dim * 2]
        
        # Apply linear layer - reshape to apply to each layer/batch
        batch_size = hidden.size(1)
        hidden_combined = hidden.view(-1, self.hidden_dim * 2)
        hidden_combined = self.fc(hidden_combined)
        hidden = hidden_combined.view(self.num_layers, batch_size, self.hidden_dim)
        # hidden: [num_layers, batch_size, hidden_dim]
        
        # For LSTM, we need to do the same for cell state
        if self.rnn_type == 'lstm':
            cell_forward = cell[0:self.num_layers]
            cell_backward = cell[self.num_layers:]
            
            cell = torch.cat([cell_forward, cell_backward], dim=2)
            # cell: [num_layers, batch_size, hidden_dim * 2]
            
            cell_combined = cell.view(-1, self.hidden_dim * 2)
            cell_combined = self.fc(cell_combined)
            cell = cell_combined.view(self.num_layers, batch_size, self.hidden_dim)
            # cell: [num_layers, batch_size, hidden_dim]
            
            return outputs, (hidden, cell)
        
        return outputs, hidden

class Decoder(nn.Module):
    """
    Decoder for sequence-to-sequence model.
    Uses LSTM/GRU to decode output sequence.
    """
    def __init__(
        self,
        output_dim,
        embedding_dim,
        hidden_dim,
        num_layers,
        dropout,
        rnn_type='lstm'
    ):
        super().__init__()
        
        self.output_dim = output_dim
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.rnn_type = rnn_type.lower()
        
        # Embedding layer
        self.embedding = nn.Embedding(output_dim, embedding_dim)
        
        # RNN layer
        if self.rnn_type == 'lstm':
            self.rnn = nn.LSTM(
                embedding_dim,
                hidden_dim,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0,
                batch_first=True
            )
        elif self.rnn_type == 'gru':
            self.rnn = nn.GRU(
                embedding_dim,
                hidden_dim,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0,
                batch_first=True
            )
        else:
            raise ValueError(f"Unsupported RNN type: {rnn_type}. Use 'lstm' or 'gru'.")
        
        # Output layer
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, trg, hidden, cell=None):
        """
        Forward pass of the decoder for a single time step.
        
        Args:
            trg: Target token indices [batch_size, 1]
            hidden: Hidden state from encoder or previous decoder step
                   [num_layers, batch_size, hidden_dim]
            cell: Cell state for LSTM [num_layers, batch_size, hidden_dim]
            
        Returns:
            output: Prediction for target tokens [batch_size, output_dim]
            hidden: New hidden state [num_layers, batch_size, hidden_dim]
            cell: New cell state [num_layers, batch_size, hidden_dim]
        """
        # trg: [batch_size, 1]
        
        # Embed the target token
        embedded = self.embedding(trg)
        # embedded: [batch_size, 1, embedding_dim]
        
        # Apply dropout to the embedding
        embedded = self.dropout(embedded)
        
        # Pass through RNN
        if self.rnn_type == 'lstm':
            output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
            # output: [batch_size, 1, hidden_dim]
            # hidden: [num_layers, batch_size, hidden_dim]
            # cell: [num_layers, batch_size, hidden_dim]
        else:  # GRU
            output, hidden = self.rnn(embedded, hidden)
            # output: [batch_size, 1, hidden_dim]
            # hidden: [num_layers, batch_size, hidden_dim]
        
        # Predict target token
        prediction = self.fc_out(output.squeeze(1))
        # prediction: [batch_size, output_dim]
        
        if self.rnn_type == 'lstm':
            return prediction, hidden, cell
        
        return prediction, hidden

class RNNSeq2Seq(nn.Module):
    """
    Sequence-to-sequence model using LSTM/GRU.
    """
    def __init__(
        self,
        encoder_vocab_size,
        decoder_vocab_size,
        encoder_embedding_dim,
        decoder_embedding_dim,
        hidden_dim,
        num_layers,
        encoder_dropout,
        decoder_dropout,
        rnn_type='lstm',
        device=torch.device('cpu')
    ):
        super().__init__()
        
        self.encoder = Encoder(
            encoder_vocab_size,
            encoder_embedding_dim,
            hidden_dim,
            num_layers,
            encoder_dropout,
            rnn_type
        )
        
        self.decoder = Decoder(
            decoder_vocab_size,
            decoder_embedding_dim,
            hidden_dim,
            num_layers,
            decoder_dropout,
            rnn_type
        )
        
        self.device = device
        self.rnn_type = rnn_type.lower()
        
        # Initialize parameters with Glorot / fan_avg
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
        
    def forward(self, src, src_lengths, trg, use_teacher_forcing=None):
        """
        Forward pass of the sequence-to-sequence model.
        
        Args:
            src: Source sequence [batch_size, src_len]
            src_lengths: Lengths of source sequences [batch_size]
            trg: Target sequence [batch_size, trg_len]
            use_teacher_forcing: Whether to use teacher forcing (if None, uses random)
            
        Returns:
            outputs: Sequence of predictions [batch_size, trg_len-1, output_dim]
        """
        # src: [batch_size, src_len]
        # trg: [batch_size, trg_len]
        
        batch_size = src.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.output_dim
        
        # Tensor to store decoder outputs
        outputs = torch.zeros(batch_size, trg_len - 1, trg_vocab_size).to(self.device)
        
        # Encode the source sequence
        if self.rnn_type == 'lstm':
            encoder_outputs, (hidden, cell) = self.encoder(src, src_lengths)
        else:  # GRU
            encoder_outputs, hidden = self.encoder(src, src_lengths)
            cell = None
        
        # First input to the decoder is the <SOS> token
        input = trg[:, 0:1]
        
        # Teacher forcing is applied to the whole batch
        if use_teacher_forcing is None:
            use_teacher_forcing = random.random() < 0.5
        
        # Decode one token at a time
        for t in range(1, trg_len):
            # Pass through decoder
            if self.rnn_type == 'lstm':
                output, hidden, cell = self.decoder(input, hidden, cell)
            else:  # GRU
                output, hidden = self.decoder(input, hidden)
            
            # Store output
            outputs[:, t-1] = output
            
            # Teacher forcing: use actual target as next input
            if use_teacher_forcing:
                input = trg[:, t:t+1]
            # No teacher forcing: use predicted token as next input
            else:
                top1 = output.argmax(1)
                input = top1.unsqueeze(1)
        
        return outputs
    
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
            if self.rnn_type == 'lstm':
                encoder_outputs, (hidden, cell) = self.encoder(src, src_lengths)
            else:  # GRU
                encoder_outputs, hidden = self.encoder(src, src_lengths)
                cell = None
            
            # Start with <SOS> token
            input = torch.tensor([[self.decoder.embedding.weight.size(0) - 2]], device=self.device)
            
            # Store translation
            translation = [input.item()]
            
            # Decode one token at a time
            for t in range(1, max_length):
                # Pass through decoder
                if self.rnn_type == 'lstm':
                    output, hidden, cell = self.decoder(input, hidden, cell)
                else:  # GRU
                    output, hidden = self.decoder(input, hidden)
                
                # Get predicted token
                pred_token = output.argmax(1).item()
                translation.append(pred_token)
                
                # Stop if <EOS> token is predicted
                if pred_token == 2:  # <EOS> token
                    break
                
                # Use predicted token as next input
                input = output.argmax(1).unsqueeze(1)
        
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
            if self.rnn_type == 'lstm':
                encoder_outputs, (hidden, cell) = self.encoder(src, src_lengths)
            else:  # GRU
                encoder_outputs, hidden = self.encoder(src, src_lengths)
                cell = None
            
            # Start with <SOS> token
            input = torch.tensor([[1]] * batch_size, device=self.device)
            
            # Store translations
            translations = [[] for _ in range(batch_size)]
            active_idxs = list(range(batch_size))
            
            # Decode one token at a time
            for t in range(1, max_length):
                # Pass through decoder
                if self.rnn_type == 'lstm':
                    output, hidden, cell = self.decoder(input, hidden, cell)
                else:  # GRU
                    output, hidden = self.decoder(input, hidden)
                
                # Get predicted tokens
                pred_tokens = output.argmax(1)
                
                # Update translations and active indices
                new_active_idxs = []
                for i, idx in enumerate(active_idxs):
                    token = pred_tokens[i].item()
                    translations[idx].append(token)
                    
                    # Keep only active sequences (not ended with <EOS>)
                    if token != 2:  # <EOS> token
                        new_active_idxs.append(idx)
                
                # Stop if all sequences have ended
                if not new_active_idxs:
                    break
                
                # Update active indices and input
                active_idxs = new_active_idxs
                input = pred_tokens[active_idxs].unsqueeze(1)
                
                # Update hidden and cell states
                active_hidden = hidden[:, active_idxs, :]
                if self.rnn_type == 'lstm':
                    active_cell = cell[:, active_idxs, :]
                
                # If all sequences have ended, break
                if input.shape[0] == 0:
                    break
        
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