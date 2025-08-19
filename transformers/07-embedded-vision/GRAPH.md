# Embedded Vision Transformer Flowchart

```mermaid
flowchart TD
    subgraph "Input Processing & Patch Extraction"
        IN["Input Image (B, 3, 32, 32)"] --> VALIDATE["Input Validation<br/>assert H==32 and W==32<br/>Expected: 32×32 RGB image"]
        
        VALIDATE --> PATCH_EXTRACT["Patch Extraction<br/>unfold(2, 4, 4).unfold(3, 4, 4)<br/>(B, 3, 8, 8, 4, 4) → (B, 8, 8, 3, 4, 4)"]
        
        PATCH_EXTRACT --> PATCH_FLAT["Flatten Patches<br/>permute(0, 2, 3, 1, 4, 5)<br/>contiguous().view(B, 64, 3×4×4)<br/>(B, 64, 48)"]
    end
    
    subgraph "Embedding & Tokenization"
        PATCH_FLAT --> PATCH_EMBED["Linear Patch Embedding<br/>Linear(48, embed_dim)<br/>(B, 64, 48) → (B, 64, embed_dim)"]
        
        PATCH_EMBED --> POS_EMBED["+ Positional Embedding<br/>nn.Parameter(1, 64, embed_dim)<br/>Learned spatial positions<br/>xavier_uniform initialization"]
    end
    
    subgraph "Attention Processing"
        POS_EMBED --> NORM1["LayerNorm(embed_dim)<br/>Normalize before attention<br/>(B, 64, embed_dim)"]
        
        NORM1 --> ATTENTION["Multi-Head Attention<br/>nn.MultiheadAttention<br/>num_heads=1 (single head)<br/>batch_first=True<br/>(B, 64, embed_dim)"]
        
        ATTENTION --> RESIDUAL1["+ Residual Connection<br/>tokens + attention_output<br/>Gradient flow preservation"]
    end
    
    subgraph "Classification Head"
        RESIDUAL1 --> POOLING["Global Average Pooling<br/>tokens.mean(dim=1)<br/>(B, 64, embed_dim) → (B, embed_dim)"]
        
        POOLING --> MLP_HEAD["MLP Classification Head<br/>LayerNorm → Linear(embed_dim, embed_dim×2) → GELU → Linear(embed_dim×2, 1)<br/>(B, embed_dim) → (B, 1)"]
        
        MLP_HEAD --> SIGMOID["Sigmoid Activation<br/>torch.sigmoid(logit)<br/>Binary classification probability<br/>(B, 1) → (B, 1)"]
    end
    
    subgraph "Model Architecture Details"
        ARCH["Architecture Specifications<br/>Input: 32×32×3 RGB image<br/>Patches: 4×4 non-overlapping<br/>Tokens: 64 patches (8×8 grid)<br/>Embedding: 64-dimensional<br/>Attention: Single head MHA<br/>Output: Binary probability"]
    end
    
    subgraph "Optimization Techniques"
        OPT["Embedded Optimizations<br/>Single attention head<br/>Minimal embedding dimension<br/>Global average pooling<br/>Lightweight MLP head<br/>Xavier initialization for stability"]
    end
    
    subgraph "Export & Deployment"
        SIGMOID --> ONNX_EXPORT["ONNX Export<br/>torch.onnx.export<br/>Static input shapes<br/>Optimized for inference<br/>Cross-platform compatibility"]
        
        ONNX_EXPORT --> TFLITE["TensorFlow Lite<br/>ONNX → TFLite conversion<br/>Mobile/edge deployment<br/>Quantization support"]
        
        TFLITE --> DEPLOY["Production Deployment<br/>Edge devices<br/>Mobile applications<br/>IoT devices<br/>Real-time inference"]
    end
    
    subgraph "Key Parameters"
        B["B: Batch size"]
        H["H: Image height (32)"]
        W["W: Image width (32)"]
        C["C: Channels (3)"]
        PATCH_SIZE["patch_size: 4×4"]
        N_TOKENS["n_tokens: 64 (8×8 grid)"]
        EMBED_DIM["embed_dim: 64"]
        NUM_HEADS["num_heads: 1"]
    end
    
    subgraph "Memory & Compute"
        MEMORY["Memory Analysis<br/>Parameters: ~50K<br/>Activation memory: ~16KB<br/>Inference time: <1ms<br/>Model size: <200KB"]
        
        COMPUTE["Compute Requirements<br/>FLOPs: ~1M operations<br/>GPU memory: <100MB<br/>CPU: Real-time capable<br/>Edge: Optimized kernels"]
    end
