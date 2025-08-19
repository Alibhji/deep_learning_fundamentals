# Vision Transformers

This module explores how transformers have revolutionized computer vision, from the original Vision Transformer (ViT) to advanced architectures for object detection, segmentation, and video understanding.

## 🎯 Learning Objectives

- Understand how transformers process visual data differently from text
- Learn about patch-based image processing and spatial attention
- Explore different vision transformer architectures and their applications
- Master the implementation of vision transformers for various tasks

## 📖 Table of Contents

1. [From Text to Vision](#from-text-to-vision)
2. [Patch Embedding](#patch-embedding)
3. [Vision Transformer (ViT)](#vision-transformer-vit)
4. [Advanced Vision Architectures](#advanced-vision-architectures)
5. [Object Detection Transformers](#object-detection-transformers)
6. [Segmentation Transformers](#segmentation-transformers)
7. [Video Transformers](#video-transformers)
8. [Practical Implementation](#practical-implementation)

## 🔄 From Text to Vision

### Key Differences

| Aspect | Text Transformers | Vision Transformers |
|--------|------------------|-------------------|
| **Input** | Token sequences | Image patches |
| **Position** | Sequential order | 2D spatial coordinates |
| **Attention** | 1D sequence | 2D spatial relationships |
| **Scale** | Variable length | Fixed grid structure |

### Challenges in Vision

1. **High Dimensionality**: Images have millions of pixels vs. thousands of tokens
2. **Spatial Structure**: 2D relationships vs. 1D sequential order
3. **Local vs. Global**: Vision requires both local features and global context
4. **Computational Cost**: Quadratic attention complexity with image size

## 🧩 Patch Embedding

### Concept

Instead of processing individual pixels, vision transformers divide images into fixed-size patches (e.g., 16×16 or 32×32 pixels) and treat each patch as a "token."

### Benefits

- **Reduced Sequence Length**: 256 patches vs. 1M+ pixels
- **Local Feature Extraction**: Each patch contains meaningful visual information
- **Scalable Attention**: Quadratic complexity becomes manageable
- **Hierarchical Processing**: Patches can be further subdivided

### Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class PatchEmbedding(nn.Module):
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

# Alternative implementation using Conv2d
class ConvPatchEmbedding(nn.Module):
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
```

## 🖼️ Vision Transformer (ViT)

### Architecture Overview

```
Input Image → Patch Embedding → Position Embedding → Transformer Encoder → Classification Head
```

### Key Components

1. **Patch Embedding**: Convert image patches to vectors
2. **Position Embedding**: Add spatial position information
3. **Transformer Encoder**: Process patches with self-attention
4. **Classification Head**: Global average pooling + linear layer

### Implementation

```python
class VisionTransformer(nn.Module):
    def __init__(
        self, 
        img_size=224, 
        patch_size=16, 
        in_channels=3, 
        num_classes=1000,
        embed_dim=768, 
        depth=12, 
        num_heads=12, 
        mlp_ratio=4.0,
        dropout=0.0
    ):
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
```

## 🏗️ Advanced Vision Architectures

### DeiT (Data-efficient Image Transformers)

DeiT introduces knowledge distillation to train vision transformers with less data.

#### Key Innovations

1. **Distillation Token**: Additional token that learns from teacher model
2. **Hard Distillation**: Direct supervision from teacher predictions
3. **Soft Distillation**: KL divergence from teacher logits

```python
class DistillationVisionTransformer(VisionTransformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Distillation token
        self.distillation_token = nn.Parameter(torch.randn(1, 1, self.patch_embed.embed_dim))
        
        # Distillation head
        self.distillation_head = nn.Linear(self.patch_embed.embed_dim, self.num_classes)
        
    def forward(self, x):
        batch_size = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)
        
        # Add class and distillation tokens
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        dist_tokens = self.distillation_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, dist_tokens, x], dim=1)
        
        # Add position embeddings
        x = x + self.pos_embed
        
        # Transformer processing
        for layer in self.encoder_layers:
            x = layer(x)
        
        x = self.ln_post(x)
        
        # Extract tokens
        cls_token = x[:, 0]
        dist_token = x[:, 1]
        
        # Both classification heads
        cls_logits = self.head(cls_token)
        dist_logits = self.distillation_head(dist_token)
        
        return cls_logits, dist_logits
```

### Swin Transformer

Swin Transformer introduces hierarchical processing with shifted windows for efficient attention.

#### Key Features

1. **Window-based Attention**: Process images in non-overlapping windows
2. **Shifted Windows**: Shift windows between layers for cross-window connections
3. **Hierarchical Structure**: Merge patches progressively for multi-scale features

```python
class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, window_size=7, shift_size=0):
        super().__init__()
        
        self.window_size = window_size
        self.shift_size = shift_size
        
        # Window attention
        self.attention = WindowAttention(dim, num_heads, window_size)
        
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
        
        # Window partition
        x_windows = window_partition(x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, x.shape[-1])
        
        # Window attention
        attn_windows = self.attention(x_windows)
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, x.shape[-1])
        
        # Window reverse
        x = window_reverse(attn_windows, self.window_size, H, W)
        
        # Shift back if needed
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        
        # Residual connections
        x = x + self.attention(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        
        return x
```

## 🎯 Object Detection Transformers

### DETR (DEtection TRansformer)

DETR treats object detection as a direct set prediction problem.

#### Key Innovations

1. **Set Prediction**: Predict objects directly without post-processing
2. **Bipartite Matching**: Hungarian algorithm for ground truth assignment
3. **Parallel Decoding**: Generate all predictions simultaneously

```python
class DETR(nn.Module):
    def __init__(self, num_classes, num_queries=100, d_model=256):
        super().__init__()
        
        # Backbone (e.g., ResNet)
        self.backbone = resnet50_backbone()
        
        # Position encoding
        self.position_encoding = PositionalEncoding(d_model)
        
        # Transformer encoder-decoder
        self.transformer = Transformer(d_model, num_heads=8, num_layers=6)
        
        # Object queries
        self.query_embed = nn.Embedding(num_queries, d_model)
        
        # Prediction heads
        self.class_embed = nn.Linear(d_model, num_classes + 1)  # +1 for background
        self.bbox_embed = MLP(d_model, d_model, 4, 3)  # 4 for bbox coordinates
        
    def forward(self, x):
        # Extract features from backbone
        features = self.backbone(x)
        
        # Add position encoding
        features = self.position_encoding(features)
        
        # Prepare queries
        batch_size = x.shape[0]
        query_embed = self.query_embed.weight.unsqueeze(0).repeat(batch_size, 1, 1)
        
        # Transformer processing
        hs = self.transformer(features, query_embed)
        
        # Predictions
        outputs_class = self.class_embed(hs)
        outputs_coords = self.bbox_embed(hs).sigmoid()
        
        return outputs_class, outputs_coords
```

### YOLOS

YOLOS adapts ViT for object detection by treating detection as a sequence-to-sequence task.

## 🔲 Segmentation Transformers

### SegFormer

SegFormer combines hierarchical transformers with lightweight MLP decoders.

#### Architecture

1. **Hierarchical Encoder**: Multi-scale feature extraction
2. **Lightweight Decoder**: MLP-based decoder for efficiency
3. **Overlap Patch Embedding**: Smooth patch boundaries

```python
class SegFormer(nn.Module):
    def __init__(self, num_classes, embed_dims=[64, 128, 256, 512]):
        super().__init__()
        
        # Hierarchical encoder
        self.encoder = HierarchicalEncoder(embed_dims)
        
        # Lightweight decoder
        self.decoder = LightweightDecoder(embed_dims, num_classes)
        
    def forward(self, x):
        # Extract multi-scale features
        features = self.encoder(x)
        
        # Decode to segmentation
        output = self.decoder(features)
        
        return output

class HierarchicalEncoder(nn.Module):
    def __init__(self, embed_dims):
        super().__init__()
        
        self.stages = nn.ModuleList([
            TransformerStage(embed_dims[i], embed_dims[i+1] if i+1 < len(embed_dims) else embed_dims[i])
            for i in range(len(embed_dims))
        ])
        
    def forward(self, x):
        features = []
        
        for stage in self.stages:
            x = stage(x)
            features.append(x)
            
        return features
```

### Mask2Former

Mask2Former unifies object detection, instance segmentation, and semantic segmentation.

## 🎬 Video Transformers

### Video Swin Transformer

Extends Swin Transformer to video by adding temporal dimension.

#### Key Features

1. **3D Windows**: Spatiotemporal attention windows
2. **Temporal Shift**: Shift windows across time for temporal modeling
3. **Hierarchical Processing**: Multi-scale spatiotemporal features

```python
class VideoSwinBlock(nn.Module):
    def __init__(self, dim, num_heads, window_size=(8, 7, 7), shift_size=(0, 0, 0)):
        super().__init__()
        
        self.window_size = window_size
        self.shift_size = shift_size
        
        # 3D window attention
        self.attention = VideoWindowAttention(dim, num_heads, window_size)
        
        # MLP
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )
        
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
    def forward(self, x):
        T, H, W = x.shape[1], x.shape[2], x.shape[3]
        
        # Shift if needed
        if any(s > 0 for s in self.shift_size):
            x = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1], -self.shift_size[2]), 
                          dims=(1, 2, 3))
        
        # 3D window partition
        x_windows = video_window_partition(x, self.window_size)
        
        # Window attention
        attn_windows = self.attention(x_windows)
        
        # Window reverse
        x = video_window_reverse(attn_windows, self.window_size, T, H, W)
        
        # Shift back if needed
        if any(s > 0 for s in self.shift_size):
            x = torch.roll(x, shifts=(self.shift_size[0], self.shift_size[1], self.shift_size[2]), 
                          dims=(1, 2, 3))
        
        # Residual connections
        x = x + self.attention(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        
        return x
```

### TimeSformer

TimeSformer introduces divided space-time attention for efficient video processing.

## 🛠️ Practical Implementation

### Training Vision Transformers

```python
def train_vision_transformer(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    
    for batch in dataloader:
        images, labels = batch['image'].to(device), batch['label'].to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        
        # Calculate loss
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)
```

### Data Augmentation

```python
import torchvision.transforms as transforms

# Standard vision transformer augmentations
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
```

### Model Configuration

```python
# ViT-Base configuration
vit_config = {
    'img_size': 224,
    'patch_size': 16,
    'embed_dim': 768,
    'depth': 12,
    'num_heads': 12,
    'mlp_ratio': 4.0,
    'num_classes': 1000
}

# Swin-T configuration
swin_config = {
    'img_size': 224,
    'patch_size': 4,
    'embed_dim': 96,
    'depths': [2, 2, 6, 2],
    'num_heads': [3, 6, 12, 24],
    'window_size': 7,
    'num_classes': 1000
}
```

## 📊 Performance Comparison

| Model | ImageNet Top-1 | Parameters | FLOPs | Use Case |
|-------|----------------|------------|-------|----------|
| **ViT-Base** | 81.8% | 86M | 17.6G | Classification |
| **DeiT-Base** | 83.1% | 86M | 17.6G | Efficient Training |
| **Swin-T** | 81.3% | 28M | 4.5G | Hierarchical Features |
| **DETR** | 42.0 AP | 41M | 86G | Object Detection |
| **SegFormer-B0** | 37.4 mIoU | 3.7M | 8.4G | Segmentation |

## 🎯 Key Takeaways

1. **Patch-based Processing**: Dividing images into patches makes attention computationally feasible
2. **Positional Encoding**: Spatial position information is crucial for vision tasks
3. **Hierarchical Processing**: Multi-scale features are important for vision
4. **Efficient Attention**: Window-based and local attention reduce computational cost
5. **Task Adaptation**: Different architectures serve different vision tasks

## 🚀 Next Steps

- Explore [Multimodal Transformers](../03-multimodal-transformers/README.md) to see how vision and text are combined
- Study [Architecture Patterns](../04-architecture-patterns/README.md) to learn how to design custom vision architectures
- Dive into [Advanced Concepts](../05-advanced-concepts/README.md) for cutting-edge techniques

## 📚 Further Reading

- [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)
- [Training data-efficient image transformers & distillation through attention](https://arxiv.org/abs/2012.12877)
- [Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](https://arxiv.org/abs/2103.14030)
- [End-to-End Object Detection with Transformers](https://arxiv.org/abs/2005.12872)
- [SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers](https://arxiv.org/abs/2105.15203)

## 🔎 Curated Resources and Further Study

### Official repos & strong tutorials
- ViT: [google-research/vision_transformer](https://github.com/google-research/vision_transformer)
- DeiT: [facebookresearch/deit](https://github.com/facebookresearch/deit)
- Swin: [microsoft/Swin-Transformer](https://github.com/microsoft/Swin-Transformer)
- DETR: [facebookresearch/detr](https://github.com/facebookresearch/detr)
- SegFormer: [NVlabs/SegFormer](https://github.com/NVlabs/SegFormer)
- Great tutorial collection: [NielsRogge/Transformers-Tutorials](https://github.com/NielsRogge/Transformers-Tutorials)

### Papers & surveys
- [An Image is Worth 16x16 Words (ViT)](https://arxiv.org/abs/2010.11929)
- [DETR: End-to-End Object Detection with Transformers](https://arxiv.org/abs/2005.12872)
- [Swin Transformer](https://arxiv.org/abs/2103.14030)
- [Mask2Former: Unified architecture for segmentation](https://arxiv.org/abs/2112.01527)

### Practitioner checklist
- Choose patch size carefully (trade-off between compute and resolution)
- Always augment (RandAugment/ColorJitter/Mixup/CutMix) and tune regularizers
- Warmup + cosine LR; monitor training stability (gradient norm, loss curves)
- For detection/segmentation, use strong backbones or multi-scale features (Deformable DETR/Swin)
- For deployment, profile with FP16/BF16, try ONNX/TensorRT, and consider quantization
