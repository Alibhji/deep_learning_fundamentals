#!/usr/bin/env python3
"""
Transformer Training Example
============================

This script demonstrates how to use the different transformer implementations
for training and inference. It includes examples for:
- Text transformers
- Vision transformers  
- Multimodal transformers
- Different architecture patterns
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm import tqdm

# Note: This script demonstrates the concepts but requires the actual implementation files
# to be in the correct directory structure. For now, we'll use placeholder classes.

# Placeholder classes for demonstration
class MultiHeadAttention(nn.Module):
    """Tiny wrapper around nn.MultiheadAttention for demo purposes.

    Expects input (T, B, D) as required by nn.MultiheadAttention (non-batch_first).
    """
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, num_heads)
    def forward(self, x):
        return self.attention(x, x, x)[0]

class TransformerLayer(nn.Module):
    """Minimal transformer layer: attention + linear FFN (demo)."""
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.ff = nn.Linear(d_model, d_model)
    def forward(self, x):
        return self.ff(self.attention(x))

class EncoderOnlyTransformer(nn.Module):
    """Toy encoder-only model for example training in this script."""
    def __init__(self, vocab_size, d_model, num_heads, num_layers):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([TransformerLayer(d_model, num_heads, d_model*4) for _ in range(num_layers)])
        self.output = nn.Linear(d_model, vocab_size)
    def forward(self, x):
        x = self.embedding(x)
        for layer in self.layers:
            x = x + layer(x)
        return self.output(x)

class VisionTransformer(nn.Module):
    """Toy ViT-like model (heavily simplified) for example training."""
    def __init__(self, img_size=224, patch_size=16, num_classes=1000, embed_dim=768, depth=6, num_heads=12):
        super().__init__()
        self.patch_embed = nn.Linear(3 * patch_size * patch_size, embed_dim)
        self.layers = nn.ModuleList([TransformerLayer(embed_dim, num_heads, embed_dim*4) for _ in range(depth)])
        self.output = nn.Linear(embed_dim, num_classes)
    def forward(self, x):
        # Simplified patch embedding
        batch_size = x.shape[0]
        x = x.view(batch_size, -1, 3 * 16 * 16)  # Assume patch_size=16
        x = self.patch_embed(x)
        for layer in self.layers:
            x = x + layer(x)
        return self.output(x.mean(dim=1))

class CLIP(nn.Module):
    def __init__(self, embed_dim=512):
        super().__init__()
        self.text_encoder = nn.Linear(512, embed_dim)
        self.image_encoder = nn.Linear(3*224*224, embed_dim)
    def forward(self, text_inputs, images):
        text_emb = self.text_encoder(text_inputs['input_ids'].float())
        image_emb = self.image_encoder(images.view(images.shape[0], -1))
        logits = torch.matmul(text_emb, image_emb.T)
        return logits, text_emb, image_emb

class StandardTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, d_ff):
        super().__init__()
        self.encoder = nn.ModuleList([TransformerLayer(d_model, num_heads, d_ff) for _ in range(num_layers)])
        self.decoder = nn.ModuleList([TransformerLayer(d_model, num_heads, d_ff) for _ in range(num_layers)])
        self.output = nn.Linear(d_model, vocab_size)
    def forward(self, src, tgt):
        for layer in self.encoder:
            src = src + layer(src)
        for layer in self.decoder:
            tgt = tgt + layer(tgt)
        return self.output(tgt)

class SharedTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, d_ff):
        super().__init__()
        self.shared_layers = nn.ModuleList([TransformerLayer(d_model, num_heads, d_ff) for _ in range(num_layers)])
        self.output = nn.Linear(d_model, vocab_size)
    def forward(self, src, tgt):
        for layer in self.shared_layers:
            src = src + layer(src)
        for layer in self.shared_layers:
            tgt = tgt + layer(tgt)
        return self.output(tgt)

class LinearAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, num_heads)
    def forward(self, x):
        return self.attention(x, x, x)[0]

class SparseAttention(nn.Module):
    def __init__(self, d_model, num_heads, sparsity_factor=4):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, num_heads)
        self.sparsity_factor = sparsity_factor
    def forward(self, x):
        if x.size(1) > self.sparsity_factor:
            indices = torch.arange(0, x.size(1), self.sparsity_factor, device=x.device)
            x_sparse = x[:, indices]
            attended, _ = self.attention(x_sparse, x_sparse, x_sparse)
            return F.interpolate(attended.transpose(1, 2), size=x.size(1)).transpose(1, 2)
        return x


def create_sample_data(batch_size=4, seq_len=128, vocab_size=30000, d_model=512):
    """Create sample data for training"""
    
    # Text data
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    labels = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    # Vision data
    images = torch.randn(batch_size, 3, 224, 224)
    image_labels = torch.randint(0, 1000, (batch_size,))
    
    # Multimodal data
    text_inputs = {
        'input_ids': input_ids,
        'attention_mask': torch.ones_like(input_ids)
    }
    
    return {
        'text': {'input_ids': input_ids, 'labels': labels},
        'vision': {'images': images, 'labels': image_labels},
        'multimodal': {'text_inputs': text_inputs, 'images': images}
    }


def train_text_transformer():
    """Train a text transformer model"""
    print("Training Text Transformer...")
    
    # Model parameters
    vocab_size = 30000
    d_model = 512
    num_heads = 8
    num_layers = 6
    batch_size = 4
    seq_len = 128
    num_epochs = 3
    
    # Create model
    model = EncoderOnlyTransformer(vocab_size, d_model, num_heads, num_layers)
    
    # Create sample data
    data = create_sample_data(batch_size, seq_len, vocab_size, d_model)
    input_ids = data['text']['input_ids']
    labels = data['text']['labels']
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    
    # Training loop
    model.train()
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(input_ids)
        
        # Reshape for loss computation
        outputs = outputs.view(-1, vocab_size)
        labels = labels.view(-1)
        
        # Compute loss
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")
    
    print("Text transformer training completed!\n")


def train_vision_transformer():
    """Train a vision transformer model"""
    print("Training Vision Transformer...")
    
    # Model parameters
    img_size = 224
    patch_size = 16
    num_classes = 1000
    embed_dim = 768
    depth = 6
    num_heads = 12
    batch_size = 4
    num_epochs = 3
    
    # Create model
    model = VisionTransformer(
        img_size=img_size, patch_size=patch_size, num_classes=num_classes,
        embed_dim=embed_dim, depth=depth, num_heads=num_heads
    )
    
    # Create sample data
    data = create_sample_data(batch_size)
    images = data['vision']['images']
    labels = data['vision']['labels']
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    
    # Training loop
    model.train()
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images)
        
        # Compute loss
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")
    
    print("Vision transformer training completed!\n")


def train_multimodal_transformer():
    """Train a multimodal transformer model (CLIP-style)"""
    print("Training Multimodal Transformer...")
    
    # Model parameters
    embed_dim = 512
    batch_size = 4
    num_epochs = 3
    
    # Create model
    model = CLIP(embed_dim=embed_dim)
    
    # Create sample data
    data = create_sample_data(batch_size)
    text_inputs = data['multimodal']['text_inputs']
    images = data['multimodal']['images']
    
    # Training setup
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    
    # Training loop
    model.train()
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        
        # Forward pass
        logits, text_emb, image_emb = model(text_inputs, images)
        
        # Compute CLIP loss
        loss = clip_loss(logits, None)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")
    
    print("Multimodal transformer training completed!\n")


def clip_loss(logits, labels):
    """Compute CLIP contrastive loss"""
    batch_size = logits.size(0)
    labels = torch.arange(batch_size, device=logits.device)
    
    # Cross-entropy loss for both directions
    loss_i = torch.nn.functional.cross_entropy(logits, labels)
    loss_t = torch.nn.functional.cross_entropy(logits.T, labels)
    
    return (loss_i + loss_t) / 2


def test_architecture_patterns():
    """Test different architecture patterns"""
    print("Testing Architecture Patterns...")
    
    # Model parameters
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
    
    # Test different architectures
    architectures = {
        'Standard': StandardTransformer(vocab_size, d_model, num_heads, num_layers, d_ff),
        'Shared': SharedTransformer(vocab_size, d_model, num_heads, num_layers, d_ff)
    }
    
    for name, model in architectures.items():
        try:
            output = model(src, tgt)
            print(f"{name} Transformer: Output shape {output.shape}")
        except Exception as e:
            print(f"{name} Transformer: Error - {e}")
    
    print("Architecture patterns testing completed!\n")


def test_attention_variants():
    """Test different attention variants"""
    print("Testing Attention Variants...")
    
    # Model parameters
    d_model = 512
    num_heads = 8
    batch_size = 2
    seq_len = 128
    
    # Create sample data
    x = torch.randn(batch_size, seq_len, d_model)
    
    # Test different attention mechanisms
    attention_types = {
        'Linear': LinearAttention(d_model, num_heads),
        'Sparse': SparseAttention(d_model, num_heads),
        'Standard': nn.MultiheadAttention(d_model, num_heads)
    }
    
    for name, attention in attention_types.items():
        try:
            if name == 'Standard':
                output, _ = attention(x, x, x)
            else:
                output = attention(x)
            print(f"{name} Attention: Output shape {output.shape}")
        except Exception as e:
            print(f"{name} Attention: Error - {e}")
    
    print("Attention variants testing completed!\n")


def main():
    """Main function to run all examples"""
    print("=" * 60)
    print("TRANSFORMER IMPLEMENTATION EXAMPLES")
    print("=" * 60)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print()
    
    try:
        # Test text transformers
        train_text_transformer()
        
        # Test vision transformers
        train_vision_transformer()
        
        # Test multimodal transformers
        train_multimodal_transformer()
        
        # Test architecture patterns
        test_architecture_patterns()
        
        # Test attention variants
        test_attention_variants()
        
        print("=" * 60)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
    except Exception as e:
        print(f"Error during execution: {e}")
        print("Some examples may have failed due to missing dependencies or implementation issues.")


if __name__ == "__main__":
    main()
