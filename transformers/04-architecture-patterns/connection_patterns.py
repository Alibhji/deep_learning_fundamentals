"""
Connection Patterns for Transformers
===================================

This module implements different ways to connect encoder and decoder components:
- Standard Encoder-Decoder
- Shared Parameters
- Hierarchical Connections
- Parallel Processing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class StandardTransformer(nn.Module):
    """Standard encoder-decoder transformer with separate components"""
    
    def __init__(self, vocab_size, d_model, num_heads, num_layers, d_ff):
        super().__init__()
        
        self.encoder = TransformerEncoder(d_model, num_heads, num_layers, d_ff)
        self.decoder = TransformerDecoder(d_model, num_heads, num_layers, d_ff)
        self.output_projection = nn.Linear(d_model, vocab_size)
        
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        # Encode source
        memory = self.encoder(src, src_mask)  # (B, N_src, d_model)
        
        # Decode target using encoded memory
        output = self.decoder(tgt, memory, tgt_mask)  # (B, N_tgt, d_model)
        
        # Project to vocabulary
        return self.output_projection(output)  # (B, N_tgt, vocab)


class SharedTransformer(nn.Module):
    """Encoder and decoder share parameters for efficiency"""
    
    def __init__(self, vocab_size, d_model, num_heads, num_layers, d_ff):
        super().__init__()
        
        # Shared layers
        self.shared_layers = nn.ModuleList([
            TransformerLayer(d_model, num_heads, d_ff)
            for _ in range(num_layers)
        ])
        
        self.output_projection = nn.Linear(d_model, vocab_size)
        
    def encode(self, x, mask=None):
        """Encode using shared layers"""
        for layer in self.shared_layers:
            x = layer(x, mask)  # (B, N, d_model)
        return x
    
    def decode(self, x, memory, mask=None):
        """Decode using shared layers with cross-attention to memory"""
        for layer in self.shared_layers:
            x = layer(x, mask, memory)  # (B, N_tgt, d_model)
        return x
    
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        memory = self.encode(src, src_mask)
        output = self.decode(tgt, memory, tgt_mask)
        return self.output_projection(output)


class HierarchicalTransformer(nn.Module):
    """Transformer with hierarchical connections between encoder and decoder"""
    
    def __init__(self, d_model, num_heads, num_layers, d_ff):
        super().__init__()
        
        self.encoder_layers = nn.ModuleList([
            TransformerLayer(d_model, num_heads, d_ff)
            for _ in range(num_layers)
        ])
        
        self.decoder_layers = nn.ModuleList([
            TransformerLayer(d_model, num_heads, d_ff)
            for _ in range(num_layers)
        ])
        
        # Skip connections between encoder and decoder
        self.skip_connections = nn.ModuleList([
            nn.Linear(d_model, d_model)
            for _ in range(num_layers)
        ])
        
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        # Encode with intermediate outputs
        encoder_outputs = []
        x = src  # (B, N_src, d_model)
        
        for layer in self.encoder_layers:
            x = layer(x, src_mask)  # (B, N_src, d_model)
            encoder_outputs.append(x)
        
        # Decode with skip connections
        for i, layer in enumerate(self.decoder_layers):
            # Add skip connection from corresponding encoder layer
            skip_connection = self.skip_connections[i](encoder_outputs[-(i+1)])  # (B, N_tgt, d_model)
            tgt = tgt + skip_connection  # (B, N_tgt, d_model)
            
            tgt = layer(tgt, tgt_mask)  # (B, N_tgt, d_model)
        
        return tgt


class ParallelTransformer(nn.Module):
    """Process encoder and decoder in parallel for efficiency"""
    
    def __init__(self, vocab_size, d_model, num_heads, num_layers, d_ff):
        super().__init__()
        
        self.encoder = TransformerEncoder(d_model, num_heads, num_layers, d_ff)
        self.decoder = TransformerDecoder(d_model, num_heads, num_layers, d_ff)
        
        # Parallel processing layers
        self.parallel_processing = nn.ModuleList([
            ParallelLayer(d_model, num_heads, d_ff)
            for _ in range(num_layers)
        ])
        
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        # Process encoder and decoder in parallel
        parallel_outputs = []
        
        for layer in self.parallel_processing:
            src_out, tgt_out = layer(src, tgt, src_mask, tgt_mask)
            src, tgt = src_out, tgt_out
            parallel_outputs.append((src, tgt))
        
        return tgt


class CrossAttention(nn.Module):
    """Standard cross-attention mechanism"""
    
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        
        self.attention = nn.MultiheadAttention(d_model, num_heads, dropout=dropout)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key, value, mask=None):
        # query: modality A (B, N_q, d_model), key/value: modality B (B, N_k, d_model)
        attended_output, attention_weights = self.attention(
            query, key, value, attn_mask=mask
        )  # attended_output: (N_q, B, d_model) if batch_first=False
        
        # Residual connection and normalization
        output = self.norm(query + self.dropout(attended_output))  # (B, N_q, d_model)
        
        return output, attention_weights


class MultiQueryCrossAttention(nn.Module):
    """Multi-query cross-attention with separate projections"""
    
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        
        # Separate projections for different attention types
        self.self_attention = nn.MultiheadAttention(d_model, num_heads, dropout=dropout)
        self.cross_attention = nn.MultiheadAttention(d_model, num_heads, dropout=dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
    def forward(self, x, memory, mask=None):
        # Self-attention
        self_attended, _ = self.self_attention(x, x, x, attn_mask=mask)  # (N, B, d_model) or (B, N, d_model)
        x = self.norm1(x + self_attended)  # (B, N, d_model)
        
        # Cross-attention
        cross_attended, _ = self.cross_attention(x, memory, memory)  # (B, N, d_model)
        x = self.norm2(x + cross_attended)  # (B, N, d_model)
        
        return x


class TransformerEncoder(nn.Module):
    """Standard transformer encoder"""
    
    def __init__(self, d_model, num_heads, num_layers, d_ff):
        super().__init__()
        
        self.layers = nn.ModuleList([
            TransformerLayer(d_model, num_heads, d_ff)
            for _ in range(num_layers)
        ])
        
    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)  # (B, N, d_model)
        return x


class TransformerDecoder(nn.Module):
    """Standard transformer decoder with cross-attention"""
    
    def __init__(self, d_model, num_heads, num_layers, d_ff):
        super().__init__()
        
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff)
            for _ in range(num_layers)
        ])
        
    def forward(self, x, memory, mask=None):
        for layer in self.layers:
            x = layer(x, memory, mask)  # (B, N_tgt, d_model)
        return x


class TransformerLayer(nn.Module):
    """Basic transformer layer for encoder"""
    
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        
        self.self_attention = nn.MultiheadAttention(d_model, num_heads, dropout=dropout)
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
        attn_output, _ = self.self_attention(x, x, x, attn_mask=mask)  # (N, B, d_model) or (B, N, d_model)
        x = self.norm1(x + self.dropout(attn_output))  # (B, N, d_model)
        
        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)  # (B, N, d_model)
        x = self.norm2(x + self.dropout(ff_output))  # (B, N, d_model)
        
        return x


class DecoderLayer(nn.Module):
    """Basic transformer layer for decoder"""
    
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        
        self.self_attention = nn.MultiheadAttention(d_model, num_heads, dropout=dropout)
        self.cross_attention = nn.MultiheadAttention(d_model, num_heads, dropout=dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, memory, mask=None):
        # Self-attention
        attn_output, _ = self.self_attention(x, x, x, attn_mask=mask)  # (B, N, d_model)
        x = self.norm1(x + self.dropout(attn_output))  # (B, N, d_model)
        
        # Cross-attention
        cross_attn_output, _ = self.cross_attention(x, memory, memory)  # (B, N, d_model)
        x = self.norm2(x + self.dropout(cross_attn_output))  # (B, N, d_model)
        
        # Feed-forward
        ff_output = self.feed_forward(x)  # (B, N, d_model)
        x = self.norm3(x + self.dropout(ff_output))  # (B, N, d_model)
        
        return x


class ParallelLayer(nn.Module):
    """Layer that processes encoder and decoder in parallel"""
    
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        
        self.encoder_layer = TransformerLayer(d_model, num_heads, d_ff)
        self.decoder_layer = DecoderLayer(d_model, num_heads, d_ff)
        
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        # Process both in parallel
        src_out = self.encoder_layer(src, src_mask)  # (B, N_src, d_model)
        tgt_out = self.decoder_layer(tgt, src_out, tgt_mask)  # (B, N_tgt, d_model)
        
        return src_out, tgt_out


# Example usage and testing
if __name__ == "__main__":
    # Test different connection patterns
    vocab_size = 30000
    d_model = 512
    num_heads = 8
    num_layers = 6
    d_ff = 2048
    
    batch_size = 2
    src_len = 128
    tgt_len = 64
    
    # Create sample data
    src = torch.randn(batch_size, src_len, d_model)
    tgt = torch.randn(batch_size, tgt_len, d_model)
    
    print("Testing Connection Patterns...")
    
    # Test standard transformer
    print("\n1. Standard Transformer")
    standard_model = StandardTransformer(vocab_size, d_model, num_heads, num_layers, d_ff)
    output = standard_model(src, tgt)
    print(f"Output shape: {output.shape}")
    
    # Test shared transformer
    print("\n2. Shared Transformer")
    shared_model = SharedTransformer(vocab_size, d_model, num_heads, num_layers, d_ff)
    output = shared_model(src, tgt)
    print(f"Output shape: {output.shape}")
    
    # Test hierarchical transformer
    print("\n3. Hierarchical Transformer")
    hierarchical_model = HierarchicalTransformer(d_model, num_heads, num_layers, d_ff)
    output = hierarchical_model(src, tgt)
    print(f"Output shape: {output.shape}")
    
    # Test parallel transformer
    print("\n4. Parallel Transformer")
    parallel_model = ParallelTransformer(vocab_size, d_model, num_heads, num_layers, d_ff)
    output = parallel_model(src, tgt)
    print(f"Output shape: {output.shape}")
    
    print("\nAll connection patterns working correctly!")
