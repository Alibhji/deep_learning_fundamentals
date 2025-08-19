# Advanced Vision Transformer Architecture Flowchart

```mermaid
flowchart TD
    subgraph "Image Input & Patch Processing"
        IMG["Input Image (B, C, H, W)"] --> PATCH["Patchify Operation\nunfold(2, ps, ps).unfold(3, ps, ps)\n(B, C, H/ps, W/ps, ps, ps)"]
        PATCH --> FLAT["Flatten Patches\n(B, N, ps×ps×C)\nN = (H/ps) × (W/ps)"]
        FLAT --> PROJ["Linear Projection\nLinear(ps×ps×C, d_model)\n(B, N, d_model)"]
    end
    
    subgraph "Token Preparation"
        PROJ --> CLS["[CLS] Token Concat\ncls_token.expand(B, 1, d_model)\n(B, N+1, d_model)"]
        CLS --> POS["+ Position Embedding\nnn.Parameter(1, N+1, d_model)\nLearned spatial positions"]
    end
    
    subgraph "Transformer Encoder Stack"
        POS --> ENC1["Encoder Layer 1\nMulti-Head Attention → FFN → Residual+LN"]
        ENC1 --> ENC2["Encoder Layer 2\nMulti-Head Attention → FFN → Residual+LN"]
        ENC2 --> ENC3["..."]
        ENC3 --> ENCL["Encoder Layer L\nMulti-Head Attention → FFN → Residual+LN"]
    end
    
    subgraph "Classification Head"
        ENCL --> NORM["LayerNorm"]
        NORM --> EXTRACT["Extract [CLS] Token\nx[:, 0, :] → (B, d_model)"]
        EXTRACT --> HEAD["Linear Classification Head\nLinear(d_model, num_classes)"]
        HEAD --> LOGITS["Class Logits (B, num_classes)"]
    end
    
    subgraph "Alternative Patch Embedding"
        IMG --> CONV["Conv2d Patch Embedding\nConv2d(C, d_model, kernel=ps, stride=ps)\n(B, d_model, H/ps, W/ps)"]
        CONV --> FLAT2["Flatten & Transpose\n(B, d_model, N) → (B, N, d_model)"]
    end
    
    subgraph "Key Parameters"
        B["B: Batch size"]
        C["C: Channels (RGB=3)"]
        H["H: Image height"]
        W["W: Image width"]
        PS["ps: Patch size"]
        N["N: Number of patches"]
        D_MODEL["d_model: Embedding dimension"]
        L["L: Number of encoder layers"]
        HEADS["num_heads: Attention heads"]
        MLP_RATIO["mlp_ratio: FFN expansion factor"]
    end
    
    subgraph "Spatial Relationships"
        SPATIAL["2D Spatial Attention\nPatches maintain spatial relationships\nPosition embeddings encode 2D coordinates\nGlobal attention across all patches"]
    end
    
    subgraph "Architecture Variants"
        VIT["ViT: Original Vision Transformer"]
        DEIT["DeiT: Training-efficient ViT"]
        SWIN["Swin: Hierarchical vision transformer"]
        CONV["ConvNeXt: CNN + Transformer hybrid"]
    end
```
