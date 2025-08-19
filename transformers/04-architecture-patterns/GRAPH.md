# Architecture Patterns Flowchart

```mermaid
flowchart TD
    subgraph "Encoder-Only Architecture"
        SRC1["Source Input (B, N_src, d_model)"] --> ENC1["Encoder Stack<br/>L encoder layers<br/>Self-attention + FFN + Residual+LN"]
        ENC1 --> MEM1["Encoded Memory<br/>(B, N_src, d_model)"]
        MEM1 --> HEAD1["Task-Specific Head<br/>Classification/Regression/Generation"]
    end
    
    subgraph "Decoder-Only Architecture"
        TGT1["Target Input (B, N_tgt, d_model)"] --> MASK1["Causal Mask<br/>Prevent attending to future tokens"]
        MASK1 --> DEC1["Decoder Stack<br/>L decoder layers<br/>Masked self-attention + FFN + Residual+LN"]
        DEC1 --> OUT1["Output Projection<br/>Linear(d_model, vocab_size)"]
    end
    
    subgraph "Encoder-Decoder Architecture"
        SRC2["Source Input (B, N_src, d_model)"] --> ENC2["Encoder Stack<br/>L encoder layers<br/>Self-attention + FFN + Residual+LN"]
        ENC2 --> MEM2["Encoded Memory<br/>(B, N_src, d_model)"]
        TGT2["Target Input (B, N_tgt, d_model)"] --> MASK2["Causal Mask"]
        MASK2 --> DEC2["Decoder Stack<br/>L decoder layers"]
        DEC2 --> CROSS["Cross-Attention<br/>Target queries attend to encoded source<br/>Q: target, K/V: source memory"]
        MEM2 --> CROSS
        CROSS --> FFN2["Feed-Forward Network<br/>Linear + ReLU + Linear"]
        FFN2 --> OUT2["Output Projection<br/>Linear(d_model, vocab_size)"]
    end
    
    subgraph "Connection Patterns"
        RESIDUAL["Residual Connections<br/>x + sublayer(x)<br/>Gradient flow preservation"]
        SKIP["Skip Connections<br/>Bypass multiple layers<br/>DenseNet-style connectivity"]
        SHARED["Shared Parameters<br/>Encoder-decoder weight sharing<br/>Reduced parameter count"]
        PARALLEL["Parallel Processing<br/>Encoder and decoder run simultaneously<br/>Reduced latency"]
    end
    
    subgraph "Cross-Attention Mechanisms"
        STANDARD["Standard Cross-Attention<br/>Q: target, K/V: source<br/>Full attention matrix"]
        MULTI_QUERY["Multi-Query Attention<br/>Shared K/V across heads<br/>Reduced memory usage"]
        SPARSE["Sparse Cross-Attention<br/>Attend to subset of source<br/>Improved efficiency"]
        LOCAL["Local Cross-Attention<br/>Window-based attention<br/>Linear complexity"]
    end
    
    subgraph "Key Parameters"
        B["B: Batch size"]
        N_SRC["N_src: Source sequence length"]
        N_TGT["N_tgt: Target sequence length"]
        D_MODEL["d_model: Model dimension"]
        L["L: Number of layers"]
        HEADS["num_heads: Attention heads"]
        VOCAB["vocab_size: Vocabulary size"]
    end
    
    subgraph "Architecture Variants"
        T5["T5: Encoder-Decoder with relative positions"]
        BART["BART: Denoising autoencoder"]
        GPT["GPT: Decoder-only autoregressive"]
        BERT["BERT: Encoder-only bidirectional"]
        UNIFIED["Unified: Single architecture for all tasks"]
    end
