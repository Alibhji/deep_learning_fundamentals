# Transformers Quick Reference Guide

A concise reference for key transformer concepts, architectures, and implementation patterns.

## 🚀 Quick Start Path

1. **Begin Here**: [Foundation: Text-Based Transformers](./01-text-transformers/README.md)
2. **Add Vision**: [Vision Transformers](./02-vision-transformers/README.md)
3. **Combine Modalities**: [Multimodal Transformers](./03-multimodal-transformers/README.md)
4. **Design Patterns**: [Architecture Patterns](./04-architecture-patterns/README.md)
5. **Advanced Techniques**: [Advanced Concepts](./05-advanced-concepts/README.md)

## 🔑 Core Concepts

### Core notation (at a glance)
- d_model: model/hidden size (width of token vectors)
- H: number of attention heads
- N: sequence length (tokens/patches)
- D_k, D_v: per-head dimensions; typically d_model = H × D_k and D_k = D_v
- Shapes: embeddings (B, N, d_model), Q/K/V (B, H, N, D_k/D_v)

### Attention Mechanism
```
Attention(Q, K, V) = softmax(QK^T / √d_k)V
```

### Three Architecture Types
- **Encoder-Only**: BERT, RoBERTa (understanding tasks)
- **Decoder-Only**: GPT, LLaMA (generation tasks)
- **Encoder-Decoder**: T5, BART (translation, summarization)

### Key Components
- **Self-Attention**: Captures relationships within sequence
- **Multi-Head Attention**: Multiple attention mechanisms in parallel
- **Positional Encoding**: Adds position information to tokens
- **Feed-Forward Networks**: Processes attention outputs

## 🏗️ Architecture Patterns

### Connection Strategies
| Pattern | Use Case | Pros | Cons |
|---------|----------|------|------|
| **Standard** | General purpose | Simple, proven | Fixed architecture |
| **Shared** | Resource-constrained | Efficient, smaller | Less flexible |
| **Hierarchical** | Multi-scale tasks | Rich features | Complex training |
| **Parallel** | Large-scale training | Faster training | Memory intensive |

### Scaling Approaches
- **Width Scaling**: Increase model dimensions
- **Depth Scaling**: Add more layers
- **Compound Scaling**: Scale both simultaneously (recommended)

## 🎯 Implementation Checklist

### Basic Transformer
- [ ] Token embedding layer
- [ ] Positional encoding
- [ ] Multi-head attention
- [ ] Feed-forward network
- [ ] Layer normalization
- [ ] Residual connections

### Vision Transformer
- [ ] Patch embedding
- [ ] Spatial position encoding
- [ ] Image-specific augmentations
- [ ] Classification head

### Multimodal Transformer
- [ ] Modality encoders
- [ ] Cross-modal attention
- [ ] Fusion strategy (early/late/cross)
- [ ] Unified output head

## ⚡ Efficiency Techniques

### Training
- **Mixed Precision**: Use FP16 for faster training
- **Gradient Checkpointing**: Trade compute for memory
- **Distributed Training**: Scale across multiple GPUs

### Inference
- **Quantization**: Convert to INT8 for smaller models
- **Pruning**: Remove unimportant weights
- **Knowledge Distillation**: Train smaller student model

## 🔧 Common Optimizers

| Optimizer | Best For | Memory | Convergence |
|-----------|----------|--------|-------------|
| **AdamW** | General purpose | Medium | Fast |
| **Lion** | Large models | Low | Good |
| **Sophia** | Pre-training | Medium | Excellent |

## 📊 Model Sizes

| Size | Parameters | Use Case |
|------|------------|----------|
| **Small** | <100M | Fine-tuning, prototyping |
| **Medium** | 100M-1B | Production applications |
| **Large** | 1B-10B | Research, advanced tasks |
| **XL** | 10B+ | State-of-the-art performance |

## 🎨 Attention Variants

| Type | Complexity | Memory | Quality |
|------|------------|--------|---------|
| **Standard** | O(n²) | High | Excellent |
| **Linear** | O(n) | Low | Good |
| **Sparse** | O(n) | Low | Good |
| **Local** | O(n×w) | Medium | Good |

## 📍 Positional Encoding

| Method | Extrapolation | Quality | Complexity |
|--------|---------------|---------|------------|
| **Sinusoidal** | Limited | Good | Low |
| **Learned** | None | Good | Low |
| **RoPE** | Excellent | Excellent | Medium |
| **ALiBi** | Good | Good | Low |

## 🚨 Common Pitfalls

1. **Forgetting Positional Encoding**: Models can't understand sequence order
2. **Incorrect Masking**: Causal masks for generation, padding masks for understanding
3. **Poor Initialization**: Use proper weight initialization for stable training
4. **Overfitting**: Regularize with dropout and early stopping
5. **Memory Issues**: Use gradient checkpointing and mixed precision

## 🔍 Debugging Tips

### Training Issues
- **Loss not decreasing**: Check learning rate, data quality, model capacity
- **Gradient explosion**: Use gradient clipping, check initialization
- **Memory errors**: Reduce batch size, use gradient checkpointing

### Inference Issues
- **Poor quality**: Check training data, model size, fine-tuning
- **Slow inference**: Use quantization, model compression
- **Out-of-memory**: Use model sharding, smaller batch sizes

## 📚 Essential Papers

1. **Attention Is All You Need** - Original transformer
2. **BERT** - Bidirectional encoder
3. **GPT** - Generative pre-training
4. **ViT** - Vision transformers
5. **CLIP** - Multimodal learning

## 🛠️ Popular Libraries

- **PyTorch**: Primary deep learning framework
- **Transformers (Hugging Face)**: Pre-trained models and utilities
- **Accelerate**: Distributed training and optimization
- **Optimum**: Model optimization and deployment

## 🎯 Next Steps

1. **Implement**: Build a basic transformer from scratch
2. **Experiment**: Try different architectures and attention mechanisms
3. **Scale**: Apply scaling strategies to your models
4. **Optimize**: Use efficiency techniques for production
5. **Research**: Stay updated with latest developments

---

**Remember**: Start simple, understand the fundamentals, then gradually add complexity. The transformer architecture is powerful but requires careful design and tuning for optimal performance.
