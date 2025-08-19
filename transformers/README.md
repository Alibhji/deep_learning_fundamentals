# Transformers Learning Path

Welcome to the comprehensive learning path for understanding Transformers, Encoders, Decoders, and different architectures. This collection covers the evolution from text-based models to multimodal systems.

## 📚 Learning Path

### 1. [Foundation: Text-Based Transformers](./01-text-transformers/README.md)
- **Attention Mechanism**: Self-attention, multi-head attention
- **Encoder-Only Models**: BERT, RoBERTa, DistilBERT
- **Decoder-Only Models**: GPT series, LLaMA, PaLM
- **Encoder-Decoder Models**: T5, BART, mT5

### 2. [Vision Transformers](./02-vision-transformers/README.md)
- **Image Classification**: ViT, DeiT, Swin Transformer
- **Object Detection**: DETR, YOLOS
- **Segmentation**: SegFormer, Mask2Former
- **Video Understanding**: Video Swin, TimeSformer

### 3. [Multimodal Transformers](./03-multimodal-transformers/README.md)
- **Text-Image**: CLIP, ALIGN, CoCa
- **Text-Video**: VideoCLIP, FrozenBiLM
- **Audio-Visual**: AudioCLIP, Perceiver
- **Universal Models**: PaLM-E, GPT-4V, Gemini

### 4. [Architecture Patterns](./04-architecture-patterns/README.md)
- **Connection Patterns**: How to connect encoders and decoders
- **Scaling Strategies**: Model scaling, data scaling, compute scaling
- **Efficiency Techniques**: Knowledge distillation, quantization, pruning
- **Training Paradigms**: Pre-training, fine-tuning, instruction tuning

### 5. [Advanced Concepts](./05-advanced-concepts/README.md)
- **Attention Variants**: Linear attention, sparse attention, local attention
- **Positional Encoding**: Absolute, relative, rotary, ALiBi
- **Normalization**: LayerNorm, RMSNorm, GroupNorm
- **Optimization**: AdamW, Lion, Sophia

### 6. [SOTA Optimization (Training & Inference)](./06-sota-optimization/README.md)
- **FlashAttention, MQA/GQA, Speculative Decoding**
- **FSDP/ZeRO, Checkpointing, Mixed Precision**
- **Quantization (QLoRA, GPTQ, AWQ), vLLM/TensorRT-LLM**
- **torch.compile/Inductor, BetterTransformer, KV cache**

git ### 7. [Embedded Vision Deployment (32×32 Human/Not-Human)](./07-embedded-vision/README.md)
- **Tiny ViT-like model**: 32×32 input, boolean output
- **Export**: ONNX/TFLite, INT8 quantization
- **Deploy**: MCU (TFLite Micro) or ARM SoC (ONNX Runtime)
- **Notebook**: End-to-end demo

### 8. [Low-level Compute: Pixel→Logit & Memory/Registers](./08-low-level-compute/README.md)
- **Parameter placement** on GPU (HBM/L2/L1/registers)
- **Tensor layout** and GEMM tiling
- **Minimal register/compute units** conceptually
- **Pixel path** from input to final logit with diagrams

## 🎯 Learning Objectives

By the end of this learning path, you will understand:

1. **Core Mechanisms**: How attention works and why it's powerful
2. **Architecture Design**: How to design and connect different components
3. **Modality Integration**: How to combine text, image, and other modalities
4. **Practical Applications**: How to implement and use these models
5. **Recent Advances**: State-of-the-art developments and trends

## 🚀 Quick Start

1. Begin with [Foundation: Text-Based Transformers](./01-text-transformers/README.md)
2. Progress through each section sequentially
3. Complete the practical exercises in each module
4. Build your own transformer-based model

## 📖 Prerequisites

- Basic understanding of neural networks and deep learning
- Familiarity with Python and PyTorch/TensorFlow
- Knowledge of linear algebra and probability
- Understanding of natural language processing concepts

## 🔗 Additional Resources

- [Attention Is All You Need Paper](https://arxiv.org/abs/1706.03762)
- [Hugging Face Transformers Library](https://huggingface.co/docs/transformers)
- [Papers With Code](https://paperswithcode.com/task/transformer)
- [Transformer Visualization](http://jalammar.github.io/illustrated-transformer/)

## 📝 Contributing

Feel free to contribute improvements, corrections, or additional examples to any of these learning materials.

---

**Happy Learning! 🎉**

*This learning path is designed to take you from understanding basic transformer concepts to building sophisticated multimodal AI systems.*
