# Advanced Multimodal Transformer Architecture Flowchart

```mermaid
flowchart TD
    subgraph "Text Modality Processing"
        TXT["Text Input (B, N_t, vocab_size)"] --> TOK["Tokenization<br/>BERT/Transformer Tokenizer"]
        TOK --> TE["Text Encoder<br/>BERT/RoBERTa/T5<br/>(B, N_t, d_text)"]
        TE --> TP["Text Projection<br/>Linear(d_text, embed_dim)<br/>(B, N_t, E)"]
        TP --> TNORM["L2 Normalization<br/>F.normalize(., dim=-1)<br/>(B, N_t, E)"]
        TNORM --> TPOOL["Pooling Strategy<br/>[CLS] token or mean<br/>(B, E)"]
    end
    
    subgraph "Image Modality Processing"
        IMG["Image Input (B, C, H, W)"] --> PATCH["Patch Embedding<br/>ViT/Swin/ConvNeXt<br/>(B, N_p, d_vision)"]
        PATCH --> IE["Vision Encoder<br/>Transformer Encoder Stack<br/>(B, N_p, d_vision)"]
        IE --> IP["Vision Projection<br/>Linear(d_vision, embed_dim)<br/>(B, N_p, E)"]
        IP --> INORM["L2 Normalization<br/>F.normalize(., dim=-1)<br/>(B, N_p, E)"]
        INORM --> IPOOL["Pooling Strategy<br/>Global average or [CLS]<br/>(B, E)"]
    end
    
    subgraph "Cross-Modal Learning"
        TPOOL --> SIM["Similarity Matrix<br/>torch.matmul(T, I^T) / temperature<br/>(B, B)"]
        IPOOL --> SIM
        SIM --> CONTRAST["Contrastive Loss<br/>InfoNCE: -log(exp(sim_pos)/Σexp(sim_neg))"]
    end
    
    subgraph "Alternative Architectures"
        TXT --> CROSS["Cross-Modal Attention<br/>Text queries attend to image keys/values"]
        IMG --> CROSS
        CROSS --> FUSION["Modality Fusion<br/>Concatenation, Addition, or Gated Fusion"]
        FUSION --> UNIFIED["Unified Output<br/>Single prediction head"]
    end
    
    subgraph "Training Objectives"
        CONTRAST --> CLIP_LOSS["CLIP Loss<br/>Bidirectional contrastive learning"]
        CONTRAST --> ALIGN_LOSS["ALIGN Loss<br/>Noisy text supervision"]
        CONTRAST --> COCA_LOSS["CoCa Loss<br/>Captioning + contrastive"]
    end
    
    subgraph "Key Components"
        B["B: Batch size"]
        N_T["N_t: Text sequence length"]
        N_P["N_p: Number of image patches"]
        D_TEXT["d_text: Text encoder dimension"]
        D_VISION["d_vision: Vision encoder dimension"]
        E["E: Common embedding dimension"]
        TEMP["temperature: Contrastive temperature"]
    end
    
    subgraph "Advanced Features"
        MULTI["Multi-Modal Variants<br/>AudioCLIP, VideoCLIP, PaLM-E"]
        ZERO["Zero-Shot Transfer<br/>Generalize to unseen tasks"]
        FROZEN["Frozen Encoders<br/>Efficient training with pre-trained models"]
        SCALE["Scalable Architecture<br/>Handle multiple modalities uniformly"]
    end
```
