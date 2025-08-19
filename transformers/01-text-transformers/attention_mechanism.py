"""
Attention Mechanism Implementation
================================

This module implements the core attention mechanisms used in transformers:
- Scaled Dot-Product Attention
- Multi-Head Attention
- Positional Encoding
- Complete Transformer Layer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Compute scaled dot-product attention
    
    Args:
        Q: Query tensor (batch_size, seq_len, d_k)
        K: Key tensor (batch_size, seq_len, d_k)
        V: Value tensor (batch_size, seq_len, d_v)
        mask: Optional mask tensor
    
    Returns:
        Attention output and attention weights
    """
    # Compute attention scores
    scores = torch.matmul(Q, K.transpose(-2, -1))
    
    # Scale
    d_k = Q.size(-1)
    scores = scores / math.sqrt(d_k)
    
    # Apply mask if provided
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    
    # Apply softmax
    attention_weights = F.softmax(scores, dim=-1)
    
    # Apply to values
    output = torch.matmul(attention_weights, V)
    
    return output, attention_weights


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention mechanism"""
    
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Linear projections
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, Q, K, V, mask=None):
        batch_size = Q.size(0)
        
        # Linear projections and reshape
        Q = self.W_q(Q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(K).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(V).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Apply attention
        attention_output, attention_weights = scaled_dot_product_attention(Q, K, V, mask)
        
        # Concatenate heads
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )
        
        # Final linear projection
        output = self.W_o(attention_output)
        
        return output, attention_weights


class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding as in the original transformer paper"""
    
    def __init__(self, d_model, max_seq_len=5000):
        super().__init__()
        
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len).unsqueeze(1).float()
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class LearnedPositionalEmbedding(nn.Module):
    """Learned positional embeddings"""
    
    def __init__(self, max_seq_len, d_model):
        super().__init__()
        self.embedding = nn.Embedding(max_seq_len, d_model)
        
    def forward(self, x):
        positions = torch.arange(x.size(1), device=x.device)
        return self.embedding(positions).unsqueeze(0)


class TransformerLayer(nn.Module):
    """Complete transformer layer with attention and feed-forward"""
    
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # Self-attention with residual connection
        attn_output, _ = self.self_attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x


def create_padding_mask(seq, pad_idx=0):
    """Create padding mask for attention"""
    return (seq != pad_idx).unsqueeze(1).unsqueeze(2)


def create_causal_mask(size):
    """Create causal mask for autoregressive generation"""
    mask = torch.triu(torch.ones(size, size), diagonal=1).bool()
    return mask


# Example usage
if __name__ == "__main__":
    # Test the attention mechanism
    batch_size = 2
    seq_len = 10
    d_model = 512
    num_heads = 8
    
    # Create sample data
    x = torch.randn(batch_size, seq_len, d_model)
    
    # Test multi-head attention
    attention = MultiHeadAttention(d_model, num_heads)
    output, weights = attention(x, x, x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Attention weights shape: {weights.shape}")
    
    # Test positional encoding
    pos_enc = SinusoidalPositionalEncoding(d_model)
    x_with_pos = pos_enc(x)
    print(f"With positional encoding: {x_with_pos.shape}")
    
    # Test transformer layer
    layer = TransformerLayer(d_model, num_heads, d_model * 4)
    output = layer(x)
    print(f"Transformer layer output: {output.shape}")
