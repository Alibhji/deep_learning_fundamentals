# Foundation: Text-Based Transformers

This module covers the fundamental concepts of transformers, starting with the attention mechanism and progressing through different architectural patterns for text processing.

### Core notation used in this module

- **d_model**: hidden size/model width used across embeddings, attention, FFN
- **H (num_heads)**: number of attention heads
- **N (sequence length)**: number of tokens per sequence
- **D_k, D_v**: per-head dimensions for queries/keys and values
- Relation: `d_model = H × D_k` and typically `D_k = D_v`
- Common shapes: embeddings `(B, N, d_model)`, Q/K/V `(B, H, N, D_k/D_v)`

## 🎯 Learning Objectives

- Understand the attention mechanism and why it revolutionized NLP
- Learn the differences between encoder-only, decoder-only, and encoder-decoder models
- Master the mathematical foundations of self-attention
- Explore practical implementations and use cases

## 📖 Table of Contents

1. [Attention Mechanism](#attention-mechanism)
2. [Self-Attention Deep Dive](#self-attention-deep-dive)
3. [Multi-Head Attention](#multi-head-attention)
4. [Positional Encoding](#positional-encoding)
5. [Encoder-Only Models](#encoder-only-models)
6. [Decoder-Only Models](#decoder-only-models)
7. [Encoder-Decoder Models](#encoder-decoder-models)
8. [Practical Implementation](#practical-implementation)

## 🔍 Attention Mechanism

### What is Attention?

Attention is a mechanism that allows models to focus on different parts of the input when processing each element. It computes a weighted sum of values, where the weights are determined by the relevance (attention scores) between the current position and all other positions.

### Why Attention?

Before transformers, RNNs and CNNs had limitations:
- **RNNs**: Sequential processing, hard to parallelize, vanishing gradients
- **CNNs**: Limited receptive field, hierarchical feature extraction
- **Attention**: Global dependencies, parallelizable, interpretable

## 🧮 Self-Attention Deep Dive

### Mathematical Foundation

The self-attention mechanism computes three vectors for each input position:
- **Query (Q)**: What am I looking for?
- **Key (K)**: What do I contain?
- **Value (V)**: What do I offer?

#### Attention Formula

```
Attention(Q, K, V) = softmax(QK^T / √d_k)V
```

Where:
- `Q`: Query matrix (n × d_k)
- `K`: Key matrix (n × d_k)  
- `V`: Value matrix (n × d_v)
- `d_k`: Dimension of keys/queries
- `√d_k`: Scaling factor to prevent softmax saturation

#### Step-by-Step Process

1. **Compute Attention Scores**: `QK^T`
2. **Scale**: Divide by `√d_k`
3. **Apply Softmax**: Convert to probabilities
4. **Weight Values**: Multiply with V


The raw scores (from Q @ K.transpose(-2, -1)) are similarity logits between every pair of tokens.

After scaling and softmax, they become attention weights (probabilities).

Shape: [batch, heads, seq_len, seq_len]

### Code Implementation

```python
import torch
import torch.nn.functional as F
import math

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Compute scaled dot-product attention
    
    Args:
        Q: Query tensor (batch_size, seq_len, d_k)
        K: Key tensor (batch_size, seq_len, d_k)
        V: Value tensor (batch_size, seq_len, d_v)
        mask: Optional mask tensor
    
    Returns:
        Attention output and attention weights
    """
    # Compute attention scores
    scores = torch.matmul(Q, K.transpose(-2, -1))
    
    # Scale
    d_k = Q.size(-1)
    scores = scores / math.sqrt(d_k)
    
    # Apply mask if provided
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    
    # Apply softmax
    attention_weights = F.softmax(scores, dim=-1)
    
    # Apply to values
    output = torch.matmul(attention_weights, V)
    
    return output, attention_weights
```

## 🎭 Multi-Head Attention

### Concept

Instead of using a single attention mechanism, multi-head attention allows the model to attend to information from different representation subspaces at different positions simultaneously.

### Benefits

- **Diverse Representations**: Different heads can learn different types of relationships
- **Stability**: Multiple heads provide redundancy and stability
- **Expressiveness**: Captures complex patterns that single attention might miss

### Implementation

```python
class MultiHeadAttention(torch.nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Linear projections
        self.W_q = torch.nn.Linear(d_model, d_model)
        self.W_k = torch.nn.Linear(d_model, d_model)
        self.W_v = torch.nn.Linear(d_model, d_model)
        self.W_o = torch.nn.Linear(d_model, d_model)
        
    def forward(self, Q, K, V, mask=None):
        batch_size = Q.size(0)
        
        # Linear projections and reshape
        Q = self.W_q(Q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(K).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(V).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Apply attention
        attention_output, attention_weights = scaled_dot_product_attention(Q, K, V, mask)
        
        # Concatenate heads
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )
        
        # Final linear projection
        output = self.W_o(attention_output)
        
        return output, attention_weights
```

## 📍 Positional Encoding

### Problem

Self-attention is permutation-invariant - it doesn't know the order of tokens. We need to inject positional information.

### Solutions

#### 1. Sinusoidal Positional Encoding (Original Transformer)

```python
def positional_encoding(seq_len, d_model):
    """Generate sinusoidal positional encodings"""
    pe = torch.zeros(seq_len, d_model)
    
    position = torch.arange(0, seq_len).unsqueeze(1).float()
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                        -(math.log(10000.0) / d_model))
    
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    
    return pe.unsqueeze(0)
```

#### 2. Learned Positional Embeddings

```python
class LearnedPositionalEmbedding(torch.nn.Module):
    def __init__(self, max_seq_len, d_model):
        super().__init__()
        self.embedding = torch.nn.Embedding(max_seq_len, d_model)
        
    def forward(self, x):
        positions = torch.arange(x.size(1), device=x.device)
        return self.embedding(positions).unsqueeze(0)
```

#### 3. Relative Positional Encoding (RPE)

RPE encodes relative distances between positions rather than absolute positions.

## 🏗️ Encoder-Only Models

### Architecture

Encoder-only models process input sequences bidirectionally and are excellent for understanding tasks.

```
Input → Token Embedding + Positional Encoding → N × Encoder Layers → Output
```

### Key Models

#### BERT (Bidirectional Encoder Representations from Transformers)

- **Architecture**: 12-24 encoder layers
- **Pre-training**: Masked Language Modeling (MLM) + Next Sentence Prediction (NSP)
- **Use Cases**: Classification, NER, Question Answering

#### RoBERTa

- **Improvements**: Larger batch size, longer sequences, dynamic masking
- **Performance**: Better than BERT on most tasks

#### DistilBERT

- **Goal**: Distillation of BERT for faster inference
- **Method**: Knowledge distillation + cosine similarity loss

### Implementation Example

```python
class EncoderLayer(torch.nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        
        self.self_attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = torch.nn.Sequential(
            torch.nn.Linear(d_model, d_ff),
            torch.nn.ReLU(),
            torch.nn.Linear(d_ff, d_model)
        )
        
        self.norm1 = torch.nn.LayerNorm(d_model)
        self.norm2 = torch.nn.LayerNorm(d_model)
        self.dropout = torch.nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # Self-attention with residual connection
        attn_output, _ = self.self_attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x
```

## 🚀 Decoder-Only Models

### Architecture

Decoder-only models generate text autoregressively, one token at a time.

```
Input → Token Embedding + Positional Encoding → N × Decoder Layers → Output
```

### Key Models

#### GPT (Generative Pre-trained Transformer)

- **Architecture**: 12-175B decoder layers
- **Pre-training**: Next token prediction
- **Use Cases**: Text generation, completion, few-shot learning

#### LLaMA

- **Innovations**: RMSNorm, SwiGLU, rotary positional embeddings
- **Scaling**: 7B to 70B parameters

#### PaLM

- **Architecture**: Parallel attention and MLP
- **Scaling**: Up to 540B parameters

### Implementation Example

```python
class DecoderLayer(torch.nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        
        self.self_attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = torch.nn.Sequential(
            torch.nn.Linear(d_model, d_ff),
            torch.nn.ReLU(),
            torch.nn.Linear(d_ff, d_model)
        )
        
        self.norm1 = torch.nn.LayerNorm(d_model)
        self.norm2 = torch.nn.LayerNorm(d_model)
        self.dropout = torch.nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # Self-attention with causal mask
        attn_output, _ = self.self_attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x

def causal_mask(size):
    """Create causal mask for autoregressive generation"""
    mask = torch.triu(torch.ones(size, size), diagonal=1).bool()
    return mask
```

## 🔄 Encoder-Decoder Models

### Architecture

Encoder-decoder models first encode the input, then decode it to generate output.

```
Input → Encoder → Encoded Representation → Decoder → Output
```

### Key Models

#### T5 (Text-to-Text Transfer Transformer)

- **Unified Framework**: All NLP tasks as text-to-text
- **Architecture**: Encoder-decoder with relative positional encoding
- **Scaling**: Up to 11B parameters

#### BART (Bidirectional and Auto-Regressive Transformers)

- **Pre-training**: Denoising autoencoder
- **Use Cases**: Summarization, translation, generation

#### mT5 (Multilingual T5)

- **Multilingual**: 101 languages
- **Architecture**: Same as T5 but multilingual pre-training

### Implementation Example

```python
class EncoderDecoderModel(torch.nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, d_ff):
        super().__init__()
        
        self.embedding = torch.nn.Embedding(vocab_size, d_model)
        self.pos_encoding = LearnedPositionalEmbedding(512, d_model)
        
        self.encoder = torch.nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff) 
            for _ in range(num_layers)
        ])
        
        self.decoder = torch.nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff) 
            for _ in range(num_layers)
        ])
        
        self.output_projection = torch.nn.Linear(d_model, vocab_size)
        
    def encode(self, src, src_mask=None):
        x = self.embedding(src) + self.pos_encoding(src)
        
        for layer in self.encoder:
            x = layer(x, src_mask)
            
        return x
    
    def decode(self, tgt, memory, tgt_mask=None, memory_mask=None):
        x = self.embedding(tgt) + self.pos_encoding(tgt)
        
        for layer in self.decoder:
            x = layer(x, tgt_mask)
            
        return self.output_projection(x)
    
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        memory = self.encode(src, src_mask)
        output = self.decode(tgt, memory, tgt_mask)
        return output
```

## 🛠️ Practical Implementation

### Training Loop

```python
def train_transformer(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    
    for batch in dataloader:
        src, tgt = batch['src'].to(device), batch['tgt'].to(device)
        
        # Create masks
        src_mask = create_padding_mask(src)
        tgt_mask = create_causal_mask(tgt.size(1))
        
        # Forward pass
        optimizer.zero_grad()
        output = model(src, tgt[:, :-1], src_mask, tgt_mask)
        
        # Calculate loss
        loss = criterion(output.reshape(-1, output.size(-1)), tgt[:, 1:].reshape(-1))
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)
```

### Inference

```python
def generate_text(model, src, max_length, device):
    model.eval()
    
    with torch.no_grad():
        memory = model.encode(src.to(device))
        
        # Start with start token
        tgt = torch.tensor([[START_TOKEN]], device=device)
        
        for _ in range(max_length):
            output = model.decode(tgt, memory)
            next_token = output[:, -1:].argmax(dim=-1)
            tgt = torch.cat([tgt, next_token], dim=1)
            
            if next_token.item() == END_TOKEN:
                break
                
    return tgt
```

## 📊 Model Comparison

| Model Type | Bidirectional | Autoregressive | Use Cases |
|------------|---------------|----------------|-----------|
| **Encoder-Only** | ✅ | ❌ | Understanding, Classification |
| **Decoder-Only** | ❌ | ✅ | Generation, Completion |
| **Encoder-Decoder** | ✅ | ✅ | Translation, Summarization |

## 🎯 Key Takeaways

1. **Attention is the core**: Self-attention allows models to capture long-range dependencies
2. **Architecture matters**: Different architectures serve different purposes
3. **Positional encoding is crucial**: Without it, transformers can't understand order
4. **Scaling works**: Larger models generally perform better
5. **Pre-training is powerful**: Models learn rich representations from unlabeled data

## 🚀 Next Steps

- Move to [Vision Transformers](../02-vision-transformers/README.md) to understand how transformers process images
- Explore [Multimodal Transformers](../03-multimodal-transformers/README.md) to see how different modalities are combined
- Study [Architecture Patterns](../04-architecture-patterns/README.md) to learn how to design custom architectures

## 📚 Further Reading

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805)
- [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- [Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://arxiv.org/abs/1910.10683)

## 🔎 Curated Resources and Further Study

### Official docs & high-quality guides
- [Hugging Face Transformers documentation](https://huggingface.co/docs/transformers)
- [The Annotated Transformer (HarvardNLP)](http://nlp.seas.harvard.edu/annotated-transformer/)
- [fast.ai: A Code-First Intro to NLP](https://www.fast.ai/2020/07/25/fastai-nlp/)

### Reference implementations & libraries
- [facebookresearch/fairseq](https://github.com/facebookresearch/fairseq)
- [NVIDIA/Megatron-LM](https://github.com/NVIDIA/Megatron-LM)
- [HuggingFace/transformers](https://github.com/huggingface/transformers)

### Selected papers beyond the basics
- [GPT-3: Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165)
- [RoBERTa: A Robustly Optimized BERT Pretraining Approach](https://arxiv.org/abs/1907.11692)
- [T5: Exploring the Limits of Transfer Learning](https://arxiv.org/abs/1910.10683)
- [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361)

### Practitioner checklist
- Understand masking (padding vs causal) and how it affects attention
- Validate tokenization and special tokens (pad, bos/eos, cls/sep)
- Start with a strong pretrained checkpoint; fine-tune with proper LR schedule (warmup + cosine/linear decay)
- Track metrics (perplexity, accuracy, F1, BLEU/ROUGE as appropriate)
- Profile memory and speed (batch size, sequence length, gradient accumulation)
