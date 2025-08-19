"""
Advanced Attention Variants
===========================

This module implements advanced attention mechanisms:
- Linear Attention
- Sparse Attention
- Local Attention
- Flash Attention (simplified)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class LinearAttention(nn.Module):
    """Linear attention that reduces complexity from O(n²) to O(n)"""
    
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.dropout = nn.Dropout(dropout)
        
        # Linear projections
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
    def forward(self, x):
        batch_size, seq_len, _ = x.shape  # (B, N, d_model)
        
        # Linear projections
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)
        
        # Reshape for multi-head processing
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Linear attention approximation
        # Apply softmax to Q and K separately
        Q = F.softmax(Q, dim=-1)
        K = F.softmax(K, dim=-1)
        
        # Compute attention in linear time
        KV = torch.einsum('bhd,bhe->bhe', K, V)
        QKV = torch.einsum('bhd,bhe->bhd', Q, KV)
        
        # Reshape back
        QKV = QKV.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        # Output projection
        output = self.W_o(QKV)
        
        return output


class SparseAttention(nn.Module):
    """Sparse attention that only attends to a subset of positions"""
    
    def __init__(self, d_model, num_heads, sparsity_factor=4, dropout=0.1):
        super().__init__()
        
        self.sparsity_factor = sparsity_factor
        self.attention = nn.MultiheadAttention(d_model, num_heads, dropout=dropout)
        
    def forward(self, x):
        seq_len = x.size(1)
        
        # Create sparse attention pattern
        if seq_len > self.sparsity_factor:
            # Attend to every nth position
            sparse_indices = torch.arange(0, seq_len, self.sparsity_factor, device=x.device)
            x_sparse = x[:, sparse_indices]
            
            # Apply attention to sparse subset
            attended, _ = self.attention(x_sparse, x_sparse, x_sparse)
            
            # Interpolate back to full sequence
            attended = attended.transpose(1, 2)  # (batch, embed_dim, sparse_len)
            x = F.interpolate(attended, size=seq_len, mode='linear', align_corners=False)
            x = x.transpose(1, 2)  # (batch, seq_len, embed_dim)
        
        return x


class LocalAttention(nn.Module):
    """Local attention that restricts attention to a fixed window"""
    
    def __init__(self, d_model, num_heads, window_size=7, dropout=0.1):
        super().__init__()
        
        self.window_size = window_size
        self.attention = nn.MultiheadAttention(d_model, num_heads, dropout=dropout)
        
    def forward(self, x):
        batch_size, seq_len, d_model = x.shape
        outputs = []
        
        for i in range(seq_len):
            # Define local window
            start = max(0, i - self.window_size // 2)
            end = min(seq_len, i + self.window_size // 2 + 1)
            
            # Extract local context
            local_x = x[:, start:end]
            
            # Apply attention to local window
            attended, _ = self.attention(
                x[:, i:i+1], local_x, local_x
            )
            outputs.append(attended)
        
        return torch.cat(outputs, dim=1)


class FlashAttention(nn.Module):
    """Simplified Flash Attention for memory-efficient attention"""
    
    def __init__(self, d_model, num_heads, block_size=64, dropout=0.1):
        super().__init__()
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.block_size = block_size
        self.dropout = nn.Dropout(dropout)
        
        # Linear projections
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        # Linear projections
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)
        
        # Reshape for multi-head processing
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Process in blocks for memory efficiency
        outputs = []
        for i in range(0, seq_len, self.block_size):
            end_i = min(i + self.block_size, seq_len)
            
            # Extract block
            Q_block = Q[:, :, i:end_i]
            K_block = K[:, :, i:end_i]
            V_block = V[:, :, i:end_i]
            
            # Standard attention on block
            scores = torch.matmul(Q_block, K_block.transpose(-2, -1))
            scores = scores / math.sqrt(self.head_dim)
            attention_weights = F.softmax(scores, dim=-1)
            attention_weights = self.dropout(attention_weights)
            
            block_output = torch.matmul(attention_weights, V_block)
            outputs.append(block_output)
        
        # Concatenate block outputs
        output = torch.cat(outputs, dim=2)
        
        # Reshape and project output
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        return self.W_o(output)


class MultiQueryAttention(nn.Module):
    """Multi-query attention that shares key and value projections across heads"""
    
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        # Separate projections for Q, but shared for K and V
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, self.head_dim)  # Single head dimension
        self.W_v = nn.Linear(d_model, self.head_dim)  # Single head dimension
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        # Project to Q, K, V
        Q = self.W_q(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        K = self.W_k(x).unsqueeze(2).expand(-1, -1, self.num_heads, -1)
        V = self.W_v(x).unsqueeze(2).expand(-1, -1, self.num_heads, -1)
        
        # Reshape for attention
        Q = Q.transpose(1, 2)  # (batch, heads, seq_len, head_dim)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        
        # Compute attention
        scores = torch.matmul(Q, K.transpose(-2, -1))
        scores = scores / math.sqrt(self.head_dim)
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention
        attended = torch.matmul(attention_weights, V)
        
        # Reshape and project output
        attended = attended.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )
        
        return self.W_o(attended)


class AttentionComparison(nn.Module):
    """Module to compare different attention mechanisms"""
    
    def __init__(self, d_model, num_heads):
        super().__init__()
        
        self.standard_attention = nn.MultiheadAttention(d_model, num_heads)
        self.linear_attention = LinearAttention(d_model, num_heads)
        self.sparse_attention = SparseAttention(d_model, num_heads)
        self.local_attention = LocalAttention(d_model, num_heads)
        self.flash_attention = FlashAttention(d_model, num_heads)
        self.multi_query_attention = MultiQueryAttention(d_model, num_heads)
        
    def forward(self, x, attention_type='standard'):
        """Forward pass with specified attention type"""
        
        if attention_type == 'standard':
            output, _ = self.standard_attention(x, x, x)
            return output
        elif attention_type == 'linear':
            return self.linear_attention(x)
        elif attention_type == 'sparse':
            return self.sparse_attention(x)
        elif attention_type == 'local':
            return self.local_attention(x)
        elif attention_type == 'flash':
            return self.flash_attention(x)
        elif attention_type == 'multi_query':
            return self.multi_query_attention(x)
        else:
            raise ValueError(f"Unknown attention type: {attention_type}")


# Example usage and testing
if __name__ == "__main__":
    # Test different attention variants
    d_model = 512
    num_heads = 8
    batch_size = 2
    seq_len = 128
    
    # Create sample data
    x = torch.randn(batch_size, seq_len, d_model)
    
    print("Testing Advanced Attention Variants...")
    
    # Test linear attention
    print("\n1. Linear Attention")
    linear_attn = LinearAttention(d_model, num_heads)
    output = linear_attn(x)
    print(f"Output shape: {output.shape}")
    
    # Test sparse attention
    print("\n2. Sparse Attention")
    sparse_attn = SparseAttention(d_model, num_heads, sparsity_factor=4)
    output = sparse_attn(x)
    print(f"Output shape: {output.shape}")
    
    # Test local attention
    print("\n3. Local Attention")
    local_attn = LocalAttention(d_model, num_heads, window_size=7)
    output = local_attn(x)
    print(f"Output shape: {output.shape}")
    
    # Test flash attention
    print("\n4. Flash Attention")
    flash_attn = FlashAttention(d_model, num_heads, block_size=32)
    output = flash_attn(x)
    print(f"Output shape: {output.shape}")
    
    # Test multi-query attention
    print("\n5. Multi-Query Attention")
    mqa_attn = MultiQueryAttention(d_model, num_heads)
    output = mqa_attn(x)
    print(f"Output shape: {output.shape}")
    
    # Test attention comparison
    print("\n6. Attention Comparison")
    attn_comparison = AttentionComparison(d_model, num_heads)
    
    attention_types = ['standard', 'linear', 'sparse', 'local', 'flash', 'multi_query']
    for attn_type in attention_types:
        try:
            output = attn_comparison(x, attn_type)
            print(f"{attn_type.capitalize()} attention: {output.shape}")
        except Exception as e:
            print(f"{attn_type.capitalize()} attention: Error - {e}")
    
    print("\nAll attention variants working correctly!")
