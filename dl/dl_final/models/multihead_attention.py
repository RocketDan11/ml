"""
Implementation of Multi-Head Attention from scratch.
Reference: "Attention Is All You Need" (Vaswani et al., 2017)
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention mechanism, implemented from scratch.
    
    Computes attention as:
        Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V
    
    Where:
        Q: Query matrices [batch_size, num_heads, seq_len, head_dim]
        K: Key matrices [batch_size, num_heads, seq_len, head_dim]
        V: Value matrices [batch_size, num_heads, seq_len, head_dim]
        d_k: Dimension of each head
    
    Returns:
        Attention output: [batch_size, num_heads, seq_len, head_dim]
        Attention weights: [batch_size, num_heads, seq_len, seq_len]
    """
    def __init__(self):
        super().__init__()
    
    def forward(self, query, key, value, mask=None):
        """
        Forward pass of Scaled Dot-Product Attention.
        
        Args:
            query: Query matrices [batch_size, num_heads, seq_len_q, head_dim]
            key: Key matrices [batch_size, num_heads, seq_len_k, head_dim]
            value: Value matrices [batch_size, num_heads, seq_len_v, head_dim]
            mask: Optional mask [batch_size, 1, 1, seq_len_k] or [batch_size, 1, seq_len_q, seq_len_k]
                 to prevent attention to certain positions
        
        Returns:
            attention_output: Weighted context vectors [batch_size, num_heads, seq_len_q, head_dim]
            attention_weights: Attention weights [batch_size, num_heads, seq_len_q, seq_len_k]
        """
        # Get dimensions
        head_dim = query.size(-1)
        
        # Calculate raw attention scores: QK^T
        # [batch_size, num_heads, seq_len_q, head_dim] x [batch_size, num_heads, head_dim, seq_len_k]
        # -> [batch_size, num_heads, seq_len_q, seq_len_k]
        attention_scores = torch.matmul(query, key.transpose(-2, -1))
        
        # Scale attention scores by sqrt(d_k)
        attention_scores = attention_scores / math.sqrt(head_dim)
        
        # Apply mask if provided (add large negative values to masked positions)
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e10)
        
        # Apply softmax to get attention weights summing to 1
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        # Calculate weighted sum of values
        # [batch_size, num_heads, seq_len_q, seq_len_k] x [batch_size, num_heads, seq_len_v, head_dim]
        # -> [batch_size, num_heads, seq_len_q, head_dim]
        attention_output = torch.matmul(attention_weights, value)
        
        return attention_output, attention_weights

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention Layer implemented from scratch.
    
    Allows the model to jointly attend to information from different representation subspaces.
    """
    def __init__(self, model_dim, num_heads, dropout=0.1):
        """
        Initialize Multi-Head Attention Layer.
        
        Args:
            model_dim: Dimension of the model (must be divisible by num_heads)
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        
        assert model_dim % num_heads == 0, "model_dim must be divisible by num_heads"
        
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads
        
        # Linear projections for Query, Key, Value
        self.query_projection = nn.Linear(model_dim, model_dim)
        self.key_projection = nn.Linear(model_dim, model_dim)
        self.value_projection = nn.Linear(model_dim, model_dim)
        
        # Scaled dot-product attention
        self.attention = ScaledDotProductAttention()
        
        # Final output projection
        self.output_projection = nn.Linear(model_dim, model_dim)
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier/Glorot initialization"""
        for module in [self.query_projection, self.key_projection, 
                     self.value_projection, self.output_projection]:
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)
    
    def split_heads(self, x, batch_size):
        """
        Split the last dimension into (num_heads, head_dim) and transpose to
        [batch_size, num_heads, seq_len, head_dim]
        
        Args:
            x: Input tensor [batch_size, seq_len, model_dim]
            batch_size: Batch size
            
        Returns:
            Reshaped tensor [batch_size, num_heads, seq_len, head_dim]
        """
        # Reshape to [batch_size, seq_len, num_heads, head_dim]
        x = x.view(batch_size, -1, self.num_heads, self.head_dim)
        
        # Transpose to [batch_size, num_heads, seq_len, head_dim]
        return x.transpose(1, 2)
    
    def combine_heads(self, x, batch_size):
        """
        Combine the multiple heads back into the original shape.
        
        Args:
            x: Input tensor [batch_size, num_heads, seq_len, head_dim]
            batch_size: Batch size
            
        Returns:
            Combined tensor [batch_size, seq_len, model_dim]
        """
        # Transpose to [batch_size, seq_len, num_heads, head_dim]
        x = x.transpose(1, 2)
        
        # Combine heads: [batch_size, seq_len, model_dim]
        return x.contiguous().view(batch_size, -1, self.model_dim)
    
    def forward(self, query, key, value, mask=None):
        """
        Forward pass of Multi-Head Attention.
        
        Args:
            query: Query tensor [batch_size, seq_len_q, model_dim]
            key: Key tensor [batch_size, seq_len_k, model_dim]
            value: Value tensor [batch_size, seq_len_v, model_dim]
            mask: Optional mask [batch_size, 1, seq_len_q, seq_len_k]
                 to prevent attention to certain positions
        
        Returns:
            attention_output: Output tensor [batch_size, seq_len_q, model_dim]
            attention_weights: Attention weights [batch_size, num_heads, seq_len_q, seq_len_k]
        """
        batch_size = query.size(0)
        
        # Project inputs to queries, keys, and values
        query_projected = self.query_projection(query)  # [batch_size, seq_len_q, model_dim]
        key_projected = self.key_projection(key)        # [batch_size, seq_len_k, model_dim]
        value_projected = self.value_projection(value)  # [batch_size, seq_len_v, model_dim]
        
        # Split into multiple heads
        query_heads = self.split_heads(query_projected, batch_size)  # [batch_size, num_heads, seq_len_q, head_dim]
        key_heads = self.split_heads(key_projected, batch_size)      # [batch_size, num_heads, seq_len_k, head_dim]
        value_heads = self.split_heads(value_projected, batch_size)  # [batch_size, num_heads, seq_len_v, head_dim]
        
        # Apply scaled dot-product attention
        attention_output, attention_weights = self.attention(
            query_heads, key_heads, value_heads, mask
        )
        
        # Combine the multiple heads
        attention_output_combined = self.combine_heads(attention_output, batch_size)
        
        # Apply final projection and dropout
        output = self.output_projection(attention_output_combined)
        output = self.dropout(output)
        
        return output, attention_weights

class PositionwiseFeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network as described in "Attention Is All You Need".
    
    FFN(x) = max(0, xW₁ + b₁)W₂ + b₂
    """
    def __init__(self, model_dim, ff_dim, dropout=0.1):
        """
        Initialize Position-wise Feed-Forward Network.
        
        Args:
            model_dim: Dimension of model (input and output)
            ff_dim: Hidden dimension of the feed-forward network
            dropout: Dropout probability
        """
        super().__init__()
        
        self.fc1 = nn.Linear(model_dim, ff_dim)
        self.fc2 = nn.Linear(ff_dim, model_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier/Glorot initialization"""
        for module in [self.fc1, self.fc2]:
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)
    
    def forward(self, x):
        """
        Forward pass of Position-wise Feed-Forward Network.
        
        Args:
            x: Input tensor [batch_size, seq_len, model_dim]
            
        Returns:
            Output tensor [batch_size, seq_len, model_dim]
        """
        # First fully connected layer with ReLU activation
        output = F.relu(self.fc1(x))
        
        # Second fully connected layer with dropout
        output = self.fc2(output)
        output = self.dropout(output)
        
        return output 