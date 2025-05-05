"""
RNN with attention model for sequence-to-sequence translation.
"""
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary

class Encoder(nn.Module):
    """
    Encoder for sequence-to-sequence model with attention.
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
        
        # Linear layers to combine bidirectional states
        self.fc_hidden = nn.Linear(hidden_dim * 2, hidden_dim)
        if self.rnn_type == 'lstm':
            self.fc_cell = nn.Linear(hidden_dim * 2, hidden_dim)
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
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
        embedded = self.dropout(self.embedding(src))
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
        # Separate and combine forward and backward states
        hidden_forward = hidden[0:self.num_layers]
        hidden_backward = hidden[self.num_layers:]
        
        hidden = torch.cat([
            hidden_forward,
            hidden_backward
        ], dim=2)
        # hidden: [num_layers, batch_size, hidden_dim * 2]
        
        # Apply linear layer to combine bidirectional state
        hidden = torch.tanh(self.fc_hidden(hidden))
        # hidden: [num_layers, batch_size, hidden_dim]
        
        # For LSTM, we need to do the same for cell state
        if self.rnn_type == 'lstm':
            cell_forward = cell[0:self.num_layers]
            cell_backward = cell[self.num_layers:]
            
            cell = torch.cat([
                cell_forward,
                cell_backward
            ], dim=2)
            # cell: [num_layers, batch_size, hidden_dim * 2]
            
            cell = torch.tanh(self.fc_cell(cell))
            # cell: [num_layers, batch_size, hidden_dim]
            
            return outputs, (hidden, cell)
        
        return outputs, hidden

class Attention(nn.Module):
    """
    Attention mechanism for sequence-to-sequence model.
    """
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super().__init__()
        
        self.encoder_attn = nn.Linear(encoder_dim, attention_dim)
        self.decoder_attn = nn.Linear(decoder_dim, attention_dim)
        self.attn = nn.Linear(attention_dim, 1)
        
    def forward(self, encoder_outputs, decoder_hidden, mask=None):
        """
        Forward pass of the attention mechanism.
        
        Args:
            encoder_outputs: Sequence of encoder hidden states [batch_size, src_len, encoder_dim]
            decoder_hidden: Decoder hidden state [batch_size, decoder_dim]
            mask: Source padding mask [batch_size, src_len]
            
        Returns:
            attention: Attention weights [batch_size, src_len]
            context: Context vector [batch_size, encoder_dim]
        """
        # encoder_outputs: [batch_size, src_len, encoder_dim]
        # decoder_hidden: [batch_size, decoder_dim]
        
        src_len = encoder_outputs.shape[1]
        
        # Reshape decoder hidden state
        decoder_hidden = decoder_hidden.unsqueeze(1).repeat(1, src_len, 1)
        # decoder_hidden: [batch_size, src_len, decoder_dim]
        
        # Calculate attention scores
        encoder_transformed = self.encoder_attn(encoder_outputs)
        # encoder_transformed: [batch_size, src_len, attention_dim]
        
        decoder_transformed = self.decoder_attn(decoder_hidden)
        # decoder_transformed: [batch_size, src_len, attention_dim]
        
        energy = torch.tanh(encoder_transformed + decoder_transformed)
        # energy: [batch_size, src_len, attention_dim]
        
        attention = self.attn(energy).squeeze(2)
        # attention: [batch_size, src_len]
        
        # Apply mask to give -inf to padding tokens
        if mask is not None:
            attention = attention.masked_fill(mask == 0, -1e10)
        
        # Apply softmax to get attention weights
        attention = F.softmax(attention, dim=1)
        # attention: [batch_size, src_len]
        
        # Calculate context vector
        context = torch.bmm(attention.unsqueeze(1), encoder_outputs).squeeze(1)
        # context: [batch_size, encoder_dim]
        
        return attention, context

class AttentionDecoder(nn.Module):
    """
    Decoder with attention for sequence-to-sequence model.
    """
    def __init__(
        self,
        output_dim,
        embedding_dim,
        hidden_dim,
        encoder_dim,
        attention_dim,
        num_layers,
        dropout,
        rnn_type='lstm'
    ):
        super().__init__()
        
        self.output_dim = output_dim
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.encoder_dim = encoder_dim
        self.attention_dim = attention_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.rnn_type = rnn_type.lower()
        
        # Embedding layer
        self.embedding = nn.Embedding(output_dim, embedding_dim)
        
        # Attention mechanism
        self.attention = Attention(encoder_dim, hidden_dim, attention_dim)
        
        # RNN layer
        if self.rnn_type == 'lstm':
            self.rnn = nn.LSTM(
                embedding_dim + encoder_dim,
                hidden_dim,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0,
                batch_first=True
            )
        elif self.rnn_type == 'gru':
            self.rnn = nn.GRU(
                embedding_dim + encoder_dim,
                hidden_dim,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0,
                batch_first=True
            )
        else:
            raise ValueError(f"Unsupported RNN type: {rnn_type}. Use 'lstm' or 'gru'.")
        
        # Output layer
        self.fc_out = nn.Linear(hidden_dim + encoder_dim + embedding_dim, output_dim)
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input, encoder_outputs, hidden, cell=None, mask=None):
        """
        Forward pass of the decoder for a single time step.
        
        Args:
            input: Target token indices [batch_size, 1]
            encoder_outputs: Sequence of encoder hidden states [batch_size, src_len, encoder_dim]
            hidden: Decoder hidden state [num_layers, batch_size, hidden_dim]
            cell: Decoder cell state for LSTM [num_layers, batch_size, hidden_dim]
            mask: Source padding mask [batch_size, src_len]
            
        Returns:
            prediction: Prediction for target tokens [batch_size, output_dim]
            hidden: New hidden state [num_layers, batch_size, hidden_dim]
            cell: New cell state [num_layers, batch_size, hidden_dim]
            attention: Attention weights [batch_size, src_len]
        """
        # input: [batch_size, 1]
        # encoder_outputs: [batch_size, src_len, encoder_dim]
        # hidden: [num_layers, batch_size, hidden_dim]
        # cell: [num_layers, batch_size, hidden_dim]
        
        # Embed the input
        embedded = self.dropout(self.embedding(input))
        # embedded: [batch_size, 1, embedding_dim]
        
        # Get top layer hidden state
        top_hidden = hidden[-1]
        # top_hidden: [batch_size, hidden_dim]
        
        # Calculate attention
        attention, context = self.attention(encoder_outputs, top_hidden, mask)
        # attention: [batch_size, src_len]
        # context: [batch_size, encoder_dim]
        
        # Reshape context and concatenate with embedded input
        context = context.unsqueeze(1)
        # context: [batch_size, 1, encoder_dim]
        
        rnn_input = torch.cat([embedded, context], dim=2)
        # rnn_input: [batch_size, 1, embedding_dim + encoder_dim]
        
        # Pass through RNN
        if self.rnn_type == 'lstm':
            output, (hidden, cell) = self.rnn(rnn_input, (hidden, cell))
            # output: [batch_size, 1, hidden_dim]
            # hidden: [num_layers, batch_size, hidden_dim]
            # cell: [num_layers, batch_size, hidden_dim]
        else:  # GRU
            output, hidden = self.rnn(rnn_input, hidden)
            # output: [batch_size, 1, hidden_dim]
            # hidden: [num_layers, batch_size, hidden_dim]
        
        # Concatenate output with context and embedded
        output = output.squeeze(1)
        # output: [batch_size, hidden_dim]
        
        context = context.squeeze(1)
        # context: [batch_size, encoder_dim]
        
        embedded = embedded.squeeze(1)
        # embedded: [batch_size, embedding_dim]
        
        prediction_input = torch.cat([output, context, embedded], dim=1)
        # prediction_input: [batch_size, hidden_dim + encoder_dim + embedding_dim]
        
        # Make prediction
        prediction = self.fc_out(prediction_input)
        # prediction: [batch_size, output_dim]
        
        if self.rnn_type == 'lstm':
            return prediction, hidden, cell, attention
        
        return prediction, hidden, attention

class AttentionSeq2Seq(nn.Module):
    """
    Sequence-to-sequence model with attention.
    """
    def __init__(
        self,
        encoder_vocab_size,
        decoder_vocab_size,
        encoder_embedding_dim,
        decoder_embedding_dim,
        hidden_dim,
        attention_dim,
        num_layers,
        encoder_dropout,
        decoder_dropout,
        rnn_type='lstm',
        device=torch.device('cpu')
    ):
        super().__init__()
        
        # Store vocab for later use
        self.decoder_vocab = None
        
        # Store device
        self.device = device
        self.rnn_type = rnn_type.lower()
        
        # Encoder
        self.encoder = Encoder(
            encoder_vocab_size,
            encoder_embedding_dim,
            hidden_dim,
            num_layers,
            encoder_dropout,
            rnn_type
        )
        
        # Decoder
        self.decoder = AttentionDecoder(
            decoder_vocab_size,
            decoder_embedding_dim,
            hidden_dim,
            hidden_dim * 2,  # encoder_dim is 2*hidden_dim due to bidirectionality
            attention_dim,
            num_layers,
            decoder_dropout,
            rnn_type
        )
        
        # Initialize parameters with Glorot / fan_avg
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
        
    def create_mask(self, src, src_lengths):
        """
        Create padding mask for attention.
        
        Args:
            src: Source sequence [batch_size, src_len]
            src_lengths: Lengths of source sequences [batch_size]
            
        Returns:
            mask: Source padding mask [batch_size, src_len]
        """
        batch_size = src.shape[0]
        src_len = src.shape[1]
        
        # Create mask
        mask = torch.zeros(batch_size, src_len, device=self.device)
        
        # Set mask to 1 for actual tokens
        for i, length in enumerate(src_lengths):
            mask[i, :length] = 1
            
        return mask
        
    def forward(self, src, src_lengths, trg, teacher_forcing_ratio=0.5):
        """
        Forward pass of the sequence-to-sequence model with attention.
        
        Args:
            src: Source sequence [batch_size, src_len]
            src_lengths: Lengths of source sequences [batch_size]
            trg: Target sequence [batch_size, trg_len]
            teacher_forcing_ratio: Probability of using teacher forcing
            
        Returns:
            outputs: Sequence of predictions [batch_size, trg_len-1, output_dim]
            attentions: Sequence of attention weights [batch_size, trg_len-1, src_len]
        """
        # src: [batch_size, src_len]
        # trg: [batch_size, trg_len]
        
        batch_size = src.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.output_dim
        src_len = src.shape[1]
        
        # Tensor to store decoder outputs
        outputs = torch.zeros(batch_size, trg_len - 1, trg_vocab_size).to(self.device)
        
        # Tensor to store attention weights
        attentions = torch.zeros(batch_size, trg_len - 1, src_len).to(self.device)
        
        # Create mask for attention
        mask = self.create_mask(src, src_lengths)
        
        # Encode the source sequence
        if self.rnn_type == 'lstm':
            encoder_outputs, (hidden, cell) = self.encoder(src, src_lengths)
        else:  # GRU
            encoder_outputs, hidden = self.encoder(src, src_lengths)
            cell = None
        
        # First input to the decoder is the <SOS> token
        input = trg[:, 0:1]
        
        # Teacher forcing is applied to the whole batch
        use_teacher_forcing = random.random() < teacher_forcing_ratio
        
        # Decode one token at a time
        for t in range(1, trg_len):
            # Pass through decoder
            if self.rnn_type == 'lstm':
                output, hidden, cell, attention = self.decoder(
                    input, encoder_outputs, hidden, cell, mask
                )
            else:  # GRU
                output, hidden, attention = self.decoder(
                    input, encoder_outputs, hidden, mask
                )
            
            # Store output and attention
            outputs[:, t-1] = output
            attentions[:, t-1] = attention
            
            # Teacher forcing: use actual target as next input
            if use_teacher_forcing:
                input = trg[:, t:t+1]
            # No teacher forcing: use predicted token as next input
            else:
                top1 = output.argmax(1)
                input = top1.unsqueeze(1)
        
        return outputs, attentions
    
    def translate_with_attention(self, src, src_lengths, max_length=50):
        """
        Translate a source sequence and return attention weights.
        
        Args:
            src: Source sequence [1, src_len]
            src_lengths: Length of source sequence [1]
            max_length: Maximum length of translation
            
        Returns:
            translation: Sequence of token indices [max_length]
            attention: Sequence of attention weights [max_length, src_len]
        """
        self.eval()
        
        with torch.no_grad():
            # Encode the source sequence
            if self.rnn_type == 'lstm':
                encoder_outputs, (hidden, cell) = self.encoder(src, src_lengths)
            else:  # GRU
                encoder_outputs, hidden = self.encoder(src, src_lengths)
                cell = None
            
            # Create mask for attention that matches encoder_outputs dimensions
            batch_size = encoder_outputs.shape[0]
            src_len = encoder_outputs.shape[1]
            mask = torch.zeros(batch_size, src_len, device=self.device)
            for i, length in enumerate(src_lengths):
                mask[i, :length] = 1
            
            # Start with <SOS> token
            input = torch.tensor([[1]], device=self.device)
            
            # Store translation and attention
            translation = [input.item()]
            all_attention = torch.zeros(max_length, src_len, device=self.device)
            
            # Decode one token at a time
            for t in range(1, max_length):
                # Pass through decoder
                if self.rnn_type == 'lstm':
                    output, hidden, cell, attention = self.decoder(
                        input, encoder_outputs, hidden, cell, mask
                    )
                else:  # GRU
                    output, hidden, attention = self.decoder(
                        input, encoder_outputs, hidden, mask
                    )
                
                # Store attention weights
                all_attention[t-1] = attention
                
                # Get predicted token
                pred_token = output.argmax(1).item()
                translation.append(pred_token)
                
                # Stop if <EOS> token is predicted
                if pred_token == 2:  # <EOS> token
                    break
                
                # Use predicted token as next input
                input = output.argmax(1).unsqueeze(1)
        
        return translation, all_attention[:t]
    
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
        translation, _ = self.translate_with_attention(src, src_lengths, max_length)
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
            
            # Create mask for attention that matches encoder_outputs dimensions
            batch_size = encoder_outputs.shape[0]
            src_len = encoder_outputs.shape[1]
            mask = torch.zeros(batch_size, src_len, device=self.device)
            for i, length in enumerate(src_lengths):
                mask[i, :length] = 1
            
            # Start with <SOS> token
            input = torch.tensor([[1]] * batch_size, device=self.device)
            
            # Store translations
            translations = [[] for _ in range(batch_size)]
            active_idxs = list(range(batch_size))
            
            # Decode one token at a time
            for t in range(1, max_length):
                # Create mask for active sequences
                active_mask = mask[active_idxs]
                
                # Pass through decoder
                if self.rnn_type == 'lstm':
                    output, hidden, cell, attention = self.decoder(
                        input, encoder_outputs[active_idxs], hidden, cell, active_mask
                    )
                else:  # GRU
                    output, hidden, attention = self.decoder(
                        input, encoder_outputs[active_idxs], hidden, active_mask
                    )
                
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
                input = pred_tokens[torch.tensor(new_active_idxs) - torch.tensor(active_idxs)[0]].unsqueeze(1)
                
                # Update hidden and cell states for active sequences
                if new_active_idxs:
                    indices = torch.tensor(new_active_idxs, device=self.device)
                    hidden = hidden[:, indices, :]
                    if self.rnn_type == 'lstm':
                        cell = cell[:, indices, :]
                
                # If all sequences have ended, break
                if not new_active_idxs:
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