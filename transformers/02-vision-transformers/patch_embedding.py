"""
Vision Transformer Components
============================

This module implements the core components for vision transformers:
- Patch Embedding
- Vision Transformer (ViT)
- Swin Transformer Block
- Position Embeddings for Images
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange


class PatchEmbedding(nn.Module):
    """Convert image patches to embeddings"""
    
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        
        # Linear projection of flattened patches
        self.projection = nn.Linear(patch_size * patch_size * in_channels, embed_dim)
        
    def forward(self, x):
        batch_size = x.shape[0]
        
        # x: (batch_size, channels, height, width)
        # Create patches: (batch_size, n_patches, patch_size * patch_size * channels)
        patches = x.unfold(2, self.patch_size, self.patch_size).unfold(
            3, self.patch_size, self.patch_size
        ).contiguous()
        
        patches = patches.view(batch_size, -1, self.patch_size * self.patch_size * x.shape[1])
        
        # Project to embedding dimension
        embeddings = self.projection(patches)
        
        return embeddings


class ConvPatchEmbedding(nn.Module):
    """Alternative patch embedding using Conv2d"""
    
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        
        self.n_patches = (img_size // patch_size) ** 2
        
        # Use Conv2d with patch_size stride to create patches
        self.projection = nn.Conv2d(
            in_channels, embed_dim, 
            kernel_size=patch_size, stride=patch_size
        )
        
    def forward(self, x):
        # x: (batch_size, channels, height, width)
        x = self.projection(x)  # (batch_size, embed_dim, H//patch_size, W//patch_size)
        x = x.flatten(2)  # (batch_size, embed_dim, n_patches)
        x = x.transpose(1, 2)  # (batch_size, n_patches, embed_dim)
        
        return x


class VisionTransformer(nn.Module):
    """Vision Transformer (ViT) implementation"""
    
    def __init__(self, img_size=224, patch_size=16, in_channels=3, num_classes=1000,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        self.n_patches = self.patch_embed.n_patches
        
        # Position embeddings
        self.pos_embed = nn.Parameter(torch.randn(1, self.n_patches + 1, embed_dim))
        
        # Class token for classification
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        
        # Transformer encoder
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(embed_dim, num_heads, int(embed_dim * mlp_ratio), dropout)
            for _ in range(depth)
        ])
        
        # Layer normalization
        self.ln_post = nn.LayerNorm(embed_dim)
        
        # Classification head
        self.head = nn.Linear(embed_dim, num_classes)
        
        # Initialize weights
        self.initialize_weights()
        
    def initialize_weights(self):
        # Initialize position embeddings
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        
        # Initialize patch embedding
        nn.init.trunc_normal_(self.patch_embed.projection.weight, std=0.02)
        
    def forward(self, x):
        batch_size = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)  # (batch_size, n_patches, embed_dim)
        
        # Add class token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (batch_size, n_patches + 1, embed_dim)
        
        # Add position embeddings
        x = x + self.pos_embed
        
        # Pass through transformer encoder
        for layer in self.encoder_layers:
            x = layer(x)
        
        # Layer normalization
        x = self.ln_post(x)
        
        # Extract class token for classification
        cls_token = x[:, 0]
        
        # Classification
        logits = self.head(cls_token)
        
        return logits


class EncoderLayer(nn.Module):
    """Single encoder layer for vision transformer"""
    
    def __init__(self, embed_dim, num_heads, mlp_dim, dropout=0.0):
        super().__init__()
        
        self.attention = MultiHeadAttention(embed_dim, num_heads)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        
    def forward(self, x):
        # Self-attention with residual connection
        x = x + self.attention(self.ln1(x))
        
        # MLP with residual connection
        x = x + self.mlp(self.ln2(x))
        
        return x


class MultiHeadAttention(nn.Module):
    """Multi-head attention for vision transformer"""
    
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
        
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        # Linear projections and reshape
        Q = self.W_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # Compute attention
        scores = torch.matmul(Q, K.transpose(-2, -1))
        scores = scores / math.sqrt(self.d_k)
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention
        attended = torch.matmul(attention_weights, V)
        
        # Reshape and project output
        attended = attended.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )
        
        return self.W_o(attended)


def restore_batch_from_flat_embeddings(flat_embeddings: torch.Tensor, batch_size: int, num_patches: int) -> torch.Tensor:
    """
    Restore embeddings shaped as (batch_size * num_patches, embed_dim) back to (batch_size, num_patches, embed_dim)

    This demonstrates the requested einops rearrange usage pattern:
        rearrange(image_embeddings, '(B N) C -> B N C', B=B, N=N)

    Args:
        flat_embeddings: Tensor of shape (B*N, C)
        batch_size: B
        num_patches: N

    Returns:
        Tensor of shape (B, N, C)
    """
    return rearrange(flat_embeddings, '(B N) C -> B N C', B=batch_size, N=num_patches)


class SwinTransformerBlock(nn.Module):
    """Swin Transformer block with window-based attention"""
    
    def __init__(self, dim, num_heads, window_size=7, shift_size=0):
        super().__init__()
        
        self.window_size = window_size
        self.shift_size = shift_size
        
        # Window attention
        self.attention = MultiHeadAttention(dim, num_heads)
        
        # MLP
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )
        
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
    def forward(self, x):
        H, W = x.shape[1], x.shape[2]
        
        # Shift if needed
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        
        # Window partition (simplified)
        x_windows = self.window_partition(x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, x.shape[-1])
        
        # Window attention
        attn_windows = self.attention(x_windows)
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, x.shape[-1])
        
        # Window reverse (simplified)
        x = self.window_reverse(attn_windows, self.window_size, H, W)
        
        # Shift back if needed
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        
        # Residual connections
        x = x + self.attention(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        
        return x
    
    def window_partition(self, x, window_size):
        """Partition input into windows"""
        B, H, W, C = x.shape
        x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.view(-1, window_size, window_size, C)
        return x
    
    def window_reverse(self, windows, window_size, H, W):
        """Reverse window partition"""
        B = windows.shape[0] // (H // window_size * W // window_size)
        x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.view(B, H, W, -1)
        return x


# Example usage and testing
if __name__ == "__main__":
    # Test patch embedding
    print("Testing Patch Embedding...")
    patch_embed = PatchEmbedding(img_size=224, patch_size=16, in_channels=3, embed_dim=768)
    x = torch.randn(2, 3, 224, 224)
    patches = patch_embed(x)
    print(f"Input shape: {x.shape}")
    print(f"Patches shape: {patches.shape}")
    
    # Test Vision Transformer
    print("\nTesting Vision Transformer...")
    vit = VisionTransformer(img_size=224, patch_size=16, num_classes=1000, 
                           embed_dim=768, depth=6, num_heads=12)
    output = vit(x)
    print(f"ViT output shape: {output.shape}")
    
    # Test Swin Transformer Block
    print("\nTesting Swin Transformer Block...")
    swin_block = SwinTransformerBlock(dim=768, num_heads=12, window_size=7)
    # Reshape for Swin (H, W, C format)
    x_swin = x.permute(0, 2, 3, 1)  # (B, H, W, C)
    x_swin = x_swin.mean(dim=0)  # (H, W, C) for single sample
    output_swin = swin_block(x_swin.unsqueeze(0))
    print(f"Swin block output shape: {output_swin.shape}")

    # Demonstrate einops.rearrange usage to restore batched embeddings
    print("\nDemonstrating rearrange to restore (B*N, C) -> (B, N, C)...")
    B, N, C = 2, patch_embed.n_patches, 768
    flat = torch.randn(B * N, C)
    restored = restore_batch_from_flat_embeddings(flat, B, N)
    print(f"Flat shape: {flat.shape} | Restored shape: {restored.shape}")
    
    print("\nAll vision transformer components working correctly!")
