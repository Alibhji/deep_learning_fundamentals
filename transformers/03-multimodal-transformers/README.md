# Multimodal Transformers

This module explores how transformers can process and understand multiple modalities simultaneously, from text-image pairs to complex multimodal interactions across text, vision, audio, and more.

## 🎯 Learning Objectives

- Understand how to combine different modalities in transformer architectures
- Learn about cross-modal attention and fusion strategies
- Explore state-of-the-art multimodal models and their applications
- Master the implementation of multimodal transformer systems

## 📖 Table of Contents

1. [Multimodal Learning Fundamentals](#multimodal-learning-fundamentals)
2. [Text-Image Models](#text-image-models)
3. [Text-Video Models](#text-video-models)
4. [Audio-Visual Models](#audio-visual-models)
5. [Universal Multimodal Models](#universal-multimodal-models)
6. [Cross-Modal Attention](#cross-modal-attention)
7. [Modality Fusion Strategies](#modality-fusion-strategies)
8. [Practical Implementation](#practical-implementation)

## 🔄 Multimodal Learning Fundamentals

### What is Multimodal Learning?

Multimodal learning involves processing and understanding data from multiple modalities (text, image, audio, video) simultaneously to perform tasks that require cross-modal understanding.

### Key Challenges

1. **Modality Alignment**: How to align different types of data
2. **Cross-Modal Understanding**: How to relate information across modalities
3. **Computational Complexity**: Processing multiple modalities efficiently
4. **Data Imbalance**: Different modalities may have varying amounts of data

### Benefits

- **Richer Understanding**: Multiple perspectives on the same concept
- **Robustness**: Less sensitive to single modality failures
- **New Capabilities**: Tasks impossible with single modalities
- **Better Generalization**: Learning cross-modal relationships

## 🖼️ Text-Image Models

### CLIP (Contrastive Language-Image Pre-training)

CLIP learns to associate images and text through contrastive learning.

#### Key Innovations

1. **Contrastive Learning**: Maximize similarity between matched image-text pairs
2. **Dual Encoders**: Separate encoders for text and images
3. **Zero-Shot Transfer**: Generalize to unseen tasks without fine-tuning

#### Architecture

```
Text Input → Text Encoder → Text Features
Image Input → Image Encoder → Image Features
                    ↓
            Contrastive Learning
                    ↓
            Similarity Matrix
```

#### Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class CLIP(nn.Module):
    def __init__(self, text_encoder, image_encoder, embed_dim=512, temperature=0.07):
        super().__init__()
        
        self.text_encoder = text_encoder
        self.image_encoder = image_encoder
        
        # Projection layers to common embedding space
        self.text_projection = nn.Linear(text_encoder.config.hidden_size, embed_dim)
        self.image_projection = nn.Linear(image_encoder.config.hidden_size, embed_dim)
        
        self.temperature = nn.Parameter(torch.ones([]) * temperature)
        
    def forward(self, text_input, image_input):
        # Encode text and images
        text_features = self.text_encoder(**text_input).last_hidden_state
        image_features = self.image_encoder(image_input).pooler_output
        
        # Project to common space
        text_embeddings = self.text_projection(text_features)
        image_embeddings = self.image_projection(image_features)
        
        # Normalize embeddings
        text_embeddings = F.normalize(text_embeddings, dim=-1)
        image_embeddings = F.normalize(image_embeddings, dim=-1)
        
        # Compute similarity matrix
        logits = torch.matmul(text_embeddings, image_embeddings.T) / self.temperature
        
        return logits

def clip_loss(logits, labels):
    """Compute CLIP contrastive loss"""
    # Create labels for positive pairs (diagonal)
    batch_size = logits.size(0)
    labels = torch.arange(batch_size, device=logits.device)
    
    # Cross-entropy loss for both directions
    loss_i = F.cross_entropy(logits, labels)
    loss_t = F.cross_entropy(logits.T, labels)
    
    return (loss_i + loss_t) / 2
```

### ALIGN (Scaling Up Visual and Vision-Language Representation Learning)

ALIGN scales up CLIP with larger datasets and models.

### CoCa (Contrastive Captioners)

CoCa combines contrastive learning with captioning for better multimodal understanding.

## 🎬 Text-Video Models

### VideoCLIP

VideoCLIP extends CLIP to video by processing temporal information.

#### Key Features

1. **Temporal Modeling**: Handle video sequences with temporal attention
2. **Frame Sampling**: Efficient processing of video frames
3. **Cross-Modal Alignment**: Align text descriptions with video content

#### Implementation

```python
class VideoCLIP(nn.Module):
    def __init__(self, text_encoder, video_encoder, embed_dim=512, num_frames=8):
        super().__init__()
        
        self.text_encoder = text_encoder
        self.video_encoder = video_encoder
        self.num_frames = num_frames
        
        # Projection layers
        self.text_projection = nn.Linear(text_encoder.config.hidden_size, embed_dim)
        self.video_projection = nn.Linear(video_encoder.config.hidden_size, embed_dim)
        
        # Temporal attention for video
        self.temporal_attention = nn.MultiheadAttention(embed_dim, num_heads=8)
        
    def forward(self, text_input, video_input):
        # Encode text
        text_features = self.text_encoder(**text_input).last_hidden_state
        text_embeddings = self.text_projection(text_features)
        
        # Encode video frames
        batch_size, num_frames = video_input.shape[:2]
        video_input = video_input.view(-1, *video_input.shape[2:])
        video_features = self.video_encoder(video_input).pooler_output
        video_features = video_features.view(batch_size, num_frames, -1)
        
        # Apply temporal attention
        video_features = video_features.transpose(0, 1)  # (num_frames, batch_size, embed_dim)
        attended_features, _ = self.temporal_attention(
            video_features, video_features, video_features
        )
        attended_features = attended_features.transpose(0, 1)  # (batch_size, num_frames, embed_dim)
        
        # Pool temporal dimension
        video_embeddings = attended_features.mean(dim=1)  # (batch_size, embed_dim)
        video_embeddings = self.video_projection(video_embeddings)
        
        # Normalize and compute similarity
        text_embeddings = F.normalize(text_embeddings, dim=-1)
        video_embeddings = F.normalize(video_embeddings, dim=-1)
        
        logits = torch.matmul(text_embeddings, video_embeddings.T)
        
        return logits
```

### FrozenBiLM

FrozenBiLM uses frozen language models with video encoders for efficient training.

## 🎵 Audio-Visual Models

### AudioCLIP

AudioCLIP extends CLIP to audio by processing spectrograms.

#### Key Features

1. **Spectrogram Processing**: Convert audio to visual-like representations
2. **Cross-Modal Learning**: Learn relationships between audio, vision, and text
3. **Efficient Training**: Leverage pre-trained vision encoders

#### Implementation

```python
class AudioCLIP(nn.Module):
    def __init__(self, text_encoder, vision_encoder, audio_encoder, embed_dim=512):
        super().__init__()
        
        self.text_encoder = text_encoder
        self.vision_encoder = vision_encoder
        self.audio_encoder = audio_encoder
        
        # Projection layers
        self.text_projection = nn.Linear(text_encoder.config.hidden_size, embed_dim)
        self.vision_projection = nn.Linear(vision_encoder.config.hidden_size, embed_dim)
        self.audio_projection = nn.Linear(audio_encoder.config.hidden_size, embed_dim)
        
    def forward(self, text_input, image_input, audio_input):
        # Encode all modalities
        text_features = self.text_encoder(**text_input).last_hidden_state
        image_features = self.vision_encoder(image_input).pooler_output
        audio_features = self.audio_encoder(audio_input).pooler_output
        
        # Project to common space
        text_embeddings = self.text_projection(text_features)
        image_embeddings = self.vision_projection(image_features)
        audio_embeddings = self.audio_projection(audio_features)
        
        # Normalize
        text_embeddings = F.normalize(text_embeddings, dim=-1)
        image_embeddings = F.normalize(image_embeddings, dim=-1)
        audio_embeddings = F.normalize(audio_embeddings, dim=-1)
        
        return text_embeddings, image_embeddings, audio_embeddings

def audio_clip_loss(text_emb, image_emb, audio_emb):
    """Compute multimodal contrastive loss"""
    # Text-image similarity
    ti_sim = torch.matmul(text_emb, image_emb.T)
    
    # Text-audio similarity
    ta_sim = torch.matmul(text_emb, audio_emb.T)
    
    # Image-audio similarity
    ia_sim = torch.matmul(image_emb, audio_emb.T)
    
    # Combined loss
    labels = torch.arange(text_emb.size(0), device=text_emb.device)
    
    loss_ti = F.cross_entropy(ti_sim, labels)
    loss_ta = F.cross_entropy(ta_sim, labels)
    loss_ia = F.cross_entropy(ia_sim, labels)
    
    return (loss_ti + loss_ta + loss_ia) / 3
```

### Perceiver

Perceiver uses cross-attention to process multiple modalities with a single architecture.

## 🌐 Universal Multimodal Models

### PaLM-E (PaLM-Embedded)

PaLM-E embeds multimodal inputs into a language model for general reasoning.

#### Key Features

1. **Modality Embedding**: Convert all modalities to language model tokens
2. **General Reasoning**: Leverage language model capabilities for multimodal tasks
3. **Scalable Architecture**: Scale with language model size

#### Architecture

```
Multimodal Input → Modality Encoders → Token Embeddings → PaLM Language Model → Output
```

#### Implementation

```python
class PaLME(nn.Module):
    def __init__(self, language_model, vision_encoder, audio_encoder, embed_dim=768):
        super().__init__()
        
        self.language_model = language_model
        self.vision_encoder = vision_encoder
        self.audio_encoder = audio_encoder
        
        # Modality projection layers
        self.vision_projection = nn.Linear(vision_encoder.config.hidden_size, embed_dim)
        self.audio_projection = nn.Linear(audio_encoder.config.hidden_size, embed_dim)
        
        # Special tokens for modalities
        self.vision_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.audio_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        
    def forward(self, text_input, image_input=None, audio_input=None):
        # Process text
        text_features = self.language_model(**text_input).last_hidden_state
        
        # Process vision if provided
        if image_input is not None:
            vision_features = self.vision_encoder(image_input).pooler_output
            vision_features = self.vision_projection(vision_features)
            vision_features = self.vision_token.expand(vision_features.size(0), -1, -1)
            
            # Insert vision features into text sequence
            text_features = torch.cat([vision_features, text_features], dim=1)
        
        # Process audio if provided
        if audio_input is not None:
            audio_features = self.audio_encoder(audio_input).pooler_output
            audio_features = self.audio_projection(audio_features)
            audio_features = self.audio_token.expand(audio_features.size(0), -1, -1)
            
            # Insert audio features into sequence
            text_features = torch.cat([audio_features, text_features], dim=1)
        
        # Process with language model
        outputs = self.language_model(inputs_embeds=text_features)
        
        return outputs
```

### GPT-4V (GPT-4 Vision)

GPT-4V extends GPT-4 with vision capabilities through multimodal training.

### Gemini

Gemini is Google's multimodal model designed for reasoning across text, images, audio, and video.

## 🔗 Cross-Modal Attention

### Cross-Attention Mechanism

Cross-attention allows one modality to attend to another, enabling cross-modal understanding.

#### Implementation

```python
class CrossModalAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        
        self.attention = nn.MultiheadAttention(d_model, num_heads, dropout=dropout)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key, value, mask=None):
        # query: modality A, key/value: modality B
        attended_output, attention_weights = self.attention(
            query, key, value, attn_mask=mask
        )
        
        # Residual connection and normalization
        output = self.norm(query + self.dropout(attended_output))
        
        return output, attention_weights

class MultimodalTransformer(nn.Module):
    def __init__(self, d_model, num_heads, num_layers):
        super().__init__()
        
        self.cross_attention_layers = nn.ModuleList([
            CrossModalAttention(d_model, num_heads)
            for _ in range(num_layers)
        ])
        
        self.self_attention_layers = nn.ModuleList([
            nn.MultiheadAttention(d_model, num_heads)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, modality_a, modality_b):
        # Cross-modal attention: A attends to B
        for cross_attn, self_attn in zip(self.cross_attention_layers, self.self_attention_layers):
            # Cross-attention
            modality_a, _ = cross_attn(modality_a, modality_b, modality_b)
            
            # Self-attention within modality A
            modality_a, _ = self_attn(modality_a, modality_a, modality_a)
        
        return self.norm(modality_a)
```

### Multi-Head Cross-Attention

Extend cross-attention to multiple heads for richer cross-modal relationships.

## 🔀 Modality Fusion Strategies

### Early Fusion

Combine modalities at the input level before processing.

```python
class EarlyFusion(nn.Module):
    def __init__(self, text_dim, image_dim, fusion_dim):
        super().__init__()
        
        self.text_projection = nn.Linear(text_dim, fusion_dim)
        self.image_projection = nn.Linear(image_dim, fusion_dim)
        self.fusion_layer = nn.Linear(fusion_dim * 2, fusion_dim)
        
    def forward(self, text_features, image_features):
        # Project to common dimension
        text_proj = self.text_projection(text_features)
        image_proj = self.image_projection(image_features)
        
        # Concatenate and fuse
        combined = torch.cat([text_proj, image_proj], dim=-1)
        fused = self.fusion_layer(combined)
        
        return fused
```

### Late Fusion

Process modalities separately and combine at the output level.

```python
class LateFusion(nn.Module):
    def __init__(self, text_dim, image_dim, output_dim):
        super().__init__()
        
        self.text_encoder = nn.Linear(text_dim, output_dim)
        self.image_encoder = nn.Linear(image_dim, output_dim)
        self.fusion_layer = nn.Linear(output_dim * 2, output_dim)
        
    def forward(self, text_features, image_features):
        # Process separately
        text_encoded = self.text_encoder(text_features)
        image_encoded = self.image_encoder(image_features)
        
        # Combine at output
        combined = torch.cat([text_encoded, image_encoded], dim=-1)
        output = self.fusion_layer(combined)
        
        return output
```

### Cross-Modal Fusion

Use attention mechanisms to dynamically fuse modalities.

```python
class CrossModalFusion(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        
        self.cross_attention = CrossModalAttention(d_model, num_heads)
        self.fusion_gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid()
        )
        
    def forward(self, modality_a, modality_b):
        # Cross-attention
        attended_a, _ = self.cross_attention(modality_a, modality_b, modality_b)
        
        # Gated fusion
        gate = self.fusion_gate(torch.cat([modality_a, attended_a], dim=-1))
        fused = gate * modality_a + (1 - gate) * attended_a
        
        return fused
```

## 🛠️ Practical Implementation

### Multimodal Dataset

```python
class MultimodalDataset(torch.utils.data.Dataset):
    def __init__(self, text_data, image_data, audio_data=None, transform=None):
        self.text_data = text_data
        self.image_data = image_data
        self.audio_data = audio_data
        self.transform = transform
        
    def __len__(self):
        return len(self.text_data)
    
    def __getitem__(self, idx):
        # Load text
        text = self.text_data[idx]
        
        # Load image
        image = Image.open(self.image_data[idx]).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        # Load audio if available
        audio = None
        if self.audio_data is not None:
            audio = torch.load(self.audio_data[idx])
        
        return {
            'text': text,
            'image': image,
            'audio': audio
        }
```

### Training Loop

```python
def train_multimodal_model(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    
    for batch in dataloader:
        # Move data to device
        text_input = {k: v.to(device) for k, v in batch['text'].items()}
        image_input = batch['image'].to(device)
        audio_input = batch['audio'].to(device) if batch['audio'] is not None else None
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(text_input, image_input, audio_input)
        
        # Calculate loss
        loss = criterion(outputs, batch['labels'].to(device))
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)
```

### Evaluation

```python
def evaluate_multimodal_model(model, dataloader, device):
    model.eval()
    total_correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for batch in dataloader:
            text_input = {k: v.to(device) for k, v in batch['text'].items()}
            image_input = batch['image'].to(device)
            audio_input = batch['audio'].to(device) if batch['audio'] is not None else None
            
            outputs = model(text_input, image_input, audio_input)
            predictions = outputs.argmax(dim=-1)
            
            total_correct += (predictions == batch['labels'].to(device)).sum().item()
            total_samples += batch['labels'].size(0)
    
    accuracy = total_correct / total_samples
    return accuracy
```

## 📊 Model Comparison

| Model | Modalities | Architecture | Use Cases |
|-------|------------|--------------|-----------|
| **CLIP** | Text + Image | Dual Encoder | Zero-shot Classification |
| **VideoCLIP** | Text + Video | Temporal + Dual Encoder | Video Understanding |
| **AudioCLIP** | Text + Image + Audio | Triple Encoder | Audio-Visual Tasks |
| **PaLM-E** | Text + Image + Audio + Video | Modality Embedding | General Reasoning |
| **GPT-4V** | Text + Image | Vision + Language Model | Multimodal Chat |

## 🎯 Key Takeaways

1. **Modality Alignment**: Cross-modal attention enables understanding relationships between different data types
2. **Fusion Strategies**: Different fusion approaches serve different use cases
3. **Scalability**: Universal models can handle multiple modalities with a single architecture
4. **Transfer Learning**: Pre-trained encoders enable efficient multimodal learning
5. **Zero-shot Capabilities**: Contrastive learning enables generalization to unseen tasks

## 🚀 Next Steps

- Study [Architecture Patterns](../04-architecture-patterns/README.md) to learn how to design custom multimodal architectures
- Explore [Advanced Concepts](../05-advanced-concepts/README.md) for cutting-edge multimodal techniques
- Build your own multimodal transformer for specific tasks

## 📚 Further Reading

- [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020)
- [Scaling Up Visual and Vision-Language Representation Learning With Noisy Text Supervision](https://arxiv.org/abs/2102.05918)
- [PaLM-E: An Embodied Multimodal Language Model](https://arxiv.org/abs/2303.03378)
- [VideoCLIP: Contrastive Pre-training for Video-Text Understanding](https://arxiv.org/abs/2109.14084)
- [AudioCLIP: Extending CLIP to Image, Text and Audio](https://arxiv.org/abs/2106.13043)
