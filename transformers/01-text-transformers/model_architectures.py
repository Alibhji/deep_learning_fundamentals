"""
Transformer Model Architectures
==============================

This module implements different transformer architectures:
- Encoder-Only Models (BERT-style)
- Decoder-Only Models (GPT-style)
- Encoder-Decoder Models (T5-style)
"""

import torch
import torch.nn as nn
from attention_mechanism import MultiHeadAttention, SinusoidalPositionalEncoding


class EncoderOnlyTransformer(nn.Module):
    """Encoder-only transformer (BERT-style) for understanding tasks"""
    
    def __init__(self, vocab_size, d_model=768, num_heads=12, num_layers=12, 
                 d_ff=3072, max_seq_len=512, dropout=0.1):
        super().__init__()
        
        self.d_model = d_model
        self.num_layers = num_layers
        
        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = SinusoidalPositionalEncoding(d_model, max_seq_len)
        self.dropout = nn.Dropout(dropout)
        
        # Encoder layers
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # Final normalization
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, input_ids, attention_mask=None):
        # Token embeddings
        x = self.token_embedding(input_ids)
        
        # Add positional encoding
        x = self.positional_encoding(x)
        x = self.dropout(x)
        
        # Apply encoder layers
        for layer in self.encoder_layers:
            x = layer(x, attention_mask)
        
        # Final normalization
        x = self.norm(x)
        
        return x


class DecoderOnlyTransformer(nn.Module):
    """Decoder-only transformer (GPT-style) for generation tasks"""
    
    def __init__(self, vocab_size, d_model=768, num_heads=12, num_layers=12,
                 d_ff=3072, max_seq_len=512, dropout=0.1):
        super().__init__()
        
        self.d_model = d_model
        self.num_layers = num_layers
        
        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = SinusoidalPositionalEncoding(d_model, max_seq_len)
        self.dropout = nn.Dropout(dropout)
        
        # Decoder layers
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # Final normalization and output projection
        self.norm = nn.LayerNorm(d_model)
        self.output_projection = nn.Linear(d_model, vocab_size)
        
    def forward(self, input_ids, attention_mask=None):
        # Token embeddings
        x = self.token_embedding(input_ids)
        
        # Add positional encoding
        x = self.positional_encoding(x)
        x = self.dropout(x)
        
        # Apply decoder layers
        for layer in self.decoder_layers:
            x = layer(x, attention_mask)
        
        # Final normalization and projection
        x = self.norm(x)
        logits = self.output_projection(x)
        
        return logits


class EncoderDecoderTransformer(nn.Module):
    """Encoder-decoder transformer (T5-style) for sequence-to-sequence tasks"""
    
    def __init__(self, vocab_size, d_model=768, num_heads=12, num_layers=6,
                 d_ff=3072, max_seq_len=512, dropout=0.1):
        super().__init__()
        
        self.d_model = d_model
        self.num_layers = num_layers
        
        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = SinusoidalPositionalEncoding(d_model, max_seq_len)
        self.dropout = nn.Dropout(dropout)
        
        # Encoder and decoder
        self.encoder = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        self.decoder = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # Final normalization and output projection
        self.norm = nn.LayerNorm(d_model)
        self.output_projection = nn.Linear(d_model, vocab_size)
        
    def encode(self, input_ids, attention_mask=None):
        """Encode the input sequence"""
        x = self.token_embedding(input_ids)
        x = self.positional_encoding(x)
        x = self.dropout(x)
        
        for layer in self.encoder:
            x = layer(x, attention_mask)
        
        return x
    
    def decode(self, decoder_input_ids, encoder_outputs, attention_mask=None, 
               decoder_attention_mask=None):
        """Decode the sequence using encoder outputs"""
        x = self.token_embedding(decoder_input_ids)
        x = self.positional_encoding(x)
        x = self.dropout(x)
        
        for layer in self.decoder:
            x = layer(x, encoder_outputs, attention_mask, decoder_attention_mask)
        
        x = self.norm(x)
        logits = self.output_projection(x)
        
        return logits
    
    def forward(self, input_ids, decoder_input_ids, attention_mask=None, 
                decoder_attention_mask=None):
        # Encode
        encoder_outputs = self.encode(input_ids, attention_mask)
        
        # Decode
        logits = self.decode(decoder_input_ids, encoder_outputs, 
                           attention_mask, decoder_attention_mask)
        
        return logits


class EncoderLayer(nn.Module):
    """Single encoder layer"""
    
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
        
    def forward(self, x, attention_mask=None):
        # Self-attention
        attn_output, _ = self.self_attention(x, x, x, attention_mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x


class DecoderLayer(nn.Module):
    """Single decoder layer with cross-attention"""
    
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attention = MultiHeadAttention(d_model, num_heads, dropout)
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
        
    def forward(self, x, encoder_outputs, attention_mask=None, 
                decoder_attention_mask=None):
        # Self-attention
        attn_output, _ = self.self_attention(x, x, x, decoder_attention_mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Cross-attention
        cross_attn_output, _ = self.cross_attention(x, encoder_outputs, 
                                                  encoder_outputs, attention_mask)
        x = self.norm2(x + self.dropout(cross_attn_output))
        
        # Feed-forward
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        
        return x


# Example usage and testing
if __name__ == "__main__":
    vocab_size = 30000
    d_model = 768
    num_heads = 12
    num_layers = 6
    max_seq_len = 512
    
    # Test encoder-only model
    print("Testing Encoder-Only Transformer...")
    encoder_model = EncoderOnlyTransformer(vocab_size, d_model, num_heads, num_layers)
    input_ids = torch.randint(0, vocab_size, (2, 128))
    encoder_output = encoder_model(input_ids)
    print(f"Encoder output shape: {encoder_output.shape}")
    
    # Test decoder-only model
    print("\nTesting Decoder-Only Transformer...")
    decoder_model = DecoderOnlyTransformer(vocab_size, d_model, num_heads, num_layers)
    decoder_output = decoder_model(input_ids)
    print(f"Decoder output shape: {decoder_output.shape}")
    
    # Test encoder-decoder model
    print("\nTesting Encoder-Decoder Transformer...")
    enc_dec_model = EncoderDecoderTransformer(vocab_size, d_model, num_heads, num_layers)
    decoder_input_ids = torch.randint(0, vocab_size, (2, 64))
    enc_dec_output = enc_dec_model(input_ids, decoder_input_ids)
    print(f"Encoder-Decoder output shape: {enc_dec_output.shape}")
    
    print("\nAll models working correctly!")
