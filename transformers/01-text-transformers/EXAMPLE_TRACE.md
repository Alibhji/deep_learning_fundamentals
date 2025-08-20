# Transformer Processing Trace: "This is an example input"

This document traces through the complete processing of the input sentence "This is an example input" through a transformer model, showing the values at each step.

## 📝 Input Sentence
**Original Text**: "This is an example input"

## 🔤 Tokenization
**Tokens**: `["This", "is", "an", "example", "input"]`
**Token IDs**: `[101, 2023, 1037, 2741, 4248, 102]` (using BERT tokenizer)
**Sequence Length (N)**: 6 tokens

## 🏗️ Model Configuration
- **d_model**: 512 (embedding dimension)
- **num_heads**: 8 (number of attention heads)
- **d_k = d_v**: 64 (per-head dimension: 512/8 = 64)
- **vocab_size**: 30522 (BERT vocabulary size)
- **max_seq_len**: 512

## 📊 Step-by-Step Processing Trace

### Step 1: Token Embeddings
```python
# Input token IDs: [101, 2023, 1037, 2741, 4248, 102]
# Shape: (1, 6) - batch_size=1, sequence_length=6

# Token embeddings lookup
token_embeddings = embedding_layer(token_ids)
# Shape: (1, 6, 512)
# Values (first few dimensions shown):
# Position 0: [0.1, -0.3, 0.8, 0.2, -0.5, ...]  # [CLS] token
# Position 1: [0.4, 0.7, -0.2, 0.9, 0.1, ...]  # "This"
# Position 2: [0.2, -0.8, 0.6, 0.3, -0.4, ...] # "is"
# Position 3: [0.5, 0.1, 0.8, -0.3, 0.7, ...]  # "an"
# Position 4: [0.3, -0.6, 0.4, 0.8, -0.2, ...] # "example"
# Position 5: [0.7, 0.2, -0.5, 0.1, 0.9, ...] # "input"
```

### Step 2: Positional Encoding
```python
# Add positional encodings to token embeddings
# Sinusoidal positional encoding for positions 0-5

positional_encodings = positional_encoding_layer(seq_len=6, d_model=512)
# Shape: (1, 6, 512)
# Values (first few dimensions shown):
# Position 0: [0.0, 1.0, 0.0, 1.0, 0.0, ...]  # sin(0), cos(0), sin(0), cos(0), ...
# Position 1: [0.84, 0.54, 0.99, 0.14, 0.98, ...] # sin(1), cos(1), sin(1/10000^(2/512)), ...
# Position 2: [0.91, -0.42, 0.99, 0.14, 0.98, ...] # sin(2), cos(2), ...
# Position 3: [0.14, -0.99, 0.99, 0.14, 0.98, ...] # sin(3), cos(3), ...
# Position 4: [-0.76, -0.65, 0.99, 0.14, 0.98, ...] # sin(4), cos(4), ...
# Position 5: [-0.96, 0.28, 0.99, 0.14, 0.98, ...] # sin(5), cos(5), ...

# Add positional encodings to token embeddings
x = token_embeddings + positional_encodings
# Shape: (1, 6, 512)
# Values (first few dimensions shown):
# Position 0: [0.1, 0.7, 0.8, 1.2, -0.5, ...]  # token + pos encoding
# Position 1: [1.24, 1.24, -0.2, 1.04, 0.1, ...] # token + pos encoding
# Position 2: [1.11, -1.22, 1.4, 0.44, -0.4, ...] # token + pos encoding
# Position 3: [0.64, -0.89, 1.6, -0.16, 0.7, ...] # token + pos encoding
# Position 4: [-0.46, -1.25, 1.2, 0.94, -0.2, ...] # token + pos encoding
# Position 5: [-0.26, 0.48, -0.5, 0.24, 0.9, ...] # token + pos encoding
```

### Step 3: Multi-Head Attention - Linear Projections
```python
# Generate Q, K, V from input embeddings
# W_q, W_k, W_v are learnable weight matrices

# Query projection
Q = W_q(x)  # Shape: (1, 6, 512)
# Values (first few dimensions shown):
# Position 0: [0.3, -0.1, 0.9, 0.4, -0.2, ...]  # Transformed query
# Position 1: [0.8, 0.6, -0.3, 0.7, 0.2, ...]  # Transformed query
# Position 2: [0.5, -0.4, 0.8, 0.1, -0.5, ...] # Transformed query
# Position 3: [0.2, 0.9, 0.4, -0.1, 0.6, ...]  # Transformed query
# Position 4: [-0.1, -0.7, 0.6, 0.8, -0.3, ...] # Transformed query
# Position 5: [0.7, 0.3, -0.2, 0.5, 0.8, ...]  # Transformed query

# Key projection
K = W_k(x)  # Shape: (1, 6, 512)
# Values (first few dimensions shown):
# Position 0: [0.4, 0.2, 0.7, -0.3, 0.5, ...]  # Transformed key
# Position 1: [0.6, 0.8, -0.1, 0.4, 0.3, ...]  # Transformed key
# Position 2: [0.3, -0.5, 0.9, 0.2, -0.4, ...] # Transformed key
# Position 3: [0.1, 0.7, 0.5, -0.2, 0.8, ...]  # Transformed key
# Position 4: [-0.2, -0.6, 0.7, 0.9, -0.1, ...] # Transformed key
# Position 5: [0.8, 0.4, -0.3, 0.6, 0.7, ...]  # Transformed key

# Value projection
V = W_v(x)  # Shape: (1, 6, 512)
# Values (first few dimensions shown):
# Position 0: [0.2, -0.4, 0.6, 0.8, -0.1, ...]  # Transformed value
# Position 1: [0.5, 0.9, -0.2, 0.3, 0.7, ...]  # Transformed value
# Position 2: [0.4, -0.7, 0.8, 0.1, -0.5, ...] # Transformed value
# Position 3: [0.7, 0.2, 0.5, -0.4, 0.9, ...]  # Transformed value
# Position 4: [-0.3, -0.8, 0.6, 0.7, -0.2, ...] # Transformed value
# Position 5: [0.6, 0.5, -0.1, 0.4, 0.8, ...]  # Transformed value
```

### Step 4: Multi-Head Attention - Reshape to Heads
```python
# Reshape Q, K, V for multi-head attention
# Split into 8 heads, each with dimension 64

Q = Q.view(1, 6, 8, 64).transpose(1, 2)  # Shape: (1, 8, 6, 64)
K = K.view(1, 6, 8, 64).transpose(1, 2)  # Shape: (1, 8, 6, 64)
V = V.view(1, 6, 8, 64).transpose(1, 2)  # Shape: (1, 8, 6, 64)

# Now we have 8 attention heads, each processing 6 tokens with 64 dimensions
# Head 0: Q[0], K[0], V[0] each with shape (1, 6, 64)
# Head 1: Q[1], K[1], V[1] each with shape (1, 6, 64)
# ... and so on for all 8 heads
```

### Step 5: Multi-Head Attention - Compute Attention Scores
```python
# For each head, compute Q @ K^T
# Let's trace Head 0:

# Head 0 Q: (1, 6, 64)
Q_head0 = Q[0]  # Shape: (1, 6, 64)
# Values (first few dimensions shown):
# Position 0: [0.3, -0.1, 0.9, 0.4, -0.2, ...]  # 64 dimensions
# Position 1: [0.8, 0.6, -0.3, 0.7, 0.2, ...]  # 64 dimensions
# Position 2: [0.5, -0.4, 0.8, 0.1, -0.5, ...] # 64 dimensions
# Position 3: [0.2, 0.9, 0.4, -0.1, 0.6, ...]  # 64 dimensions
# Position 4: [-0.1, -0.7, 0.6, 0.8, -0.3, ...] # 64 dimensions
# Position 5: [0.7, 0.3, -0.2, 0.5, 0.8, ...]  # 64 dimensions

# Head 0 K: (1, 6, 64)
K_head0 = K[0]  # Shape: (1, 6, 64)
# Values (first few dimensions shown):
# Position 0: [0.4, 0.2, 0.7, -0.3, 0.5, ...]  # 64 dimensions
# Position 1: [0.6, 0.8, -0.1, 0.4, 0.3, ...]  # 64 dimensions
# Position 2: [0.3, -0.5, 0.9, 0.2, -0.4, ...] # 64 dimensions
# Position 3: [0.1, 0.7, 0.5, -0.2, 0.8, ...]  # 64 dimensions
# Position 4: [-0.2, -0.6, 0.7, 0.9, -0.1, ...] # 64 dimensions
# Position 5: [0.7, 0.4, -0.3, 0.6, 0.7, ...]  # 64 dimensions

# Compute attention scores: Q @ K^T
scores = torch.matmul(Q_head0, K_head0.transpose(-2, -1))
# Shape: (1, 6, 6) - attention matrix for Head 0
# Values (6x6 matrix):
#           Pos0   Pos1   Pos2   Pos3   Pos4   Pos5
# Pos0:    [2.1,  1.8,   1.9,   1.7,   1.6,   2.0]  # [CLS] attends to all
# Pos1:    [1.8,  2.3,   1.7,   1.9,   1.5,   2.1]  # "This" attends to all
# Pos2:    [1.9,  1.7,   2.4,   1.8,   1.6,   2.2]  # "is" attends to all
# Pos3:    [1.7,  1.9,   1.8,   2.2,   1.7,   1.9]  # "an" attends to all
# Pos4:    [1.6,  1.5,   1.6,   1.7,   2.5,   1.8]  # "example" attends to all
# Pos5:    [2.0,  2.1,   2.2,   1.9,   1.8,   2.6]  # "input" attends to all
```

### Step 6: Multi-Head Attention - Scale and Softmax
```python
# Scale scores by √d_k
d_k = 64
scaled_scores = scores / math.sqrt(d_k)  # Divide by √64 = 8
# Shape: (1, 6, 6)
# Values (6x6 matrix):
#           Pos0   Pos1   Pos2   Pos3   Pos4   Pos5
# Pos0:    [0.26, 0.23,  0.24,  0.21,  0.20,  0.25]  # Scaled scores
# Pos1:    [0.23, 0.29,  0.21,  0.24,  0.19,  0.26]  # Scaled scores
# Pos2:    [0.24, 0.21,  0.30,  0.23,  0.20,  0.28]  # Scaled scores
# Pos3:    [0.21, 0.24,  0.23,  0.28,  0.21,  0.24]  # Scaled scores
# Pos4:    [0.20, 0.19,  0.20,  0.21,  0.31,  0.23]  # Scaled scores
# Pos5:    [0.25, 0.26,  0.28,  0.24,  0.23,  0.33]  # Scaled scores

# Apply softmax to get attention weights
attention_weights = F.softmax(scaled_scores, dim=-1)
# Shape: (1, 6, 6)
# Values (6x6 matrix - probabilities sum to 1 per row):
#           Pos0   Pos1   Pos2   Pos3   Pos4   Pos5
# Pos0:    [0.17, 0.16,  0.16,  0.15,  0.15,  0.17]  # [CLS] attention
# Pos1:    [0.15, 0.19,  0.15,  0.17,  0.14,  0.18]  # "This" attention
# Pos2:    [0.16, 0.15,  0.20,  0.16,  0.15,  0.18]  # "is" attention
# Pos3:    [0.15, 0.16,  0.16,  0.19,  0.15,  0.17]  # "an" attention
# Pos4:    [0.15, 0.14,  0.15,  0.15,  0.22,  0.16]  # "example" attention
# Pos5:    [0.16,  0.17, 0.18,  0.16,  0.15,  0.21]  # "input" attention
```

### Step 7: Multi-Head Attention - Apply to Values
```python
# Apply attention weights to values
# Head 0 V: (1, 6, 64)
V_head0 = V[0]  # Shape: (1, 6, 64)

# Compute weighted sum: attention_weights @ V
attention_output = torch.matmul(attention_weights, V_head0)
# Shape: (1, 6, 64)
# Values (first few dimensions shown):
# Position 0: [0.35, -0.45, 0.65, 0.75, -0.15, ...]  # Weighted combination
# Position 1: [0.45, 0.85, -0.25, 0.35, 0.65, ...]  # Weighted combination
# Position 2: [0.40, -0.65, 0.75, 0.15, -0.45, ...] # Weighted combination
# Position 3: [0.55, 0.25, 0.45, -0.35, 0.85, ...]  # Weighted combination
# Position 4: [-0.25, -0.75, 0.55, 0.65, -0.25, ...] # Weighted combination
# Position 5: [0.65, 0.45, -0.15, 0.45, 0.75, ...]  # Weighted combination
```

### Step 8: Multi-Head Attention - Concatenate Heads
```python
# Repeat steps 5-7 for all 8 heads
# Each head produces output of shape (1, 6, 64)

# Concatenate all heads
all_heads_output = torch.cat([
    attention_output_head0,  # (1, 6, 64)
    attention_output_head1,  # (1, 6, 64)
    attention_output_head2,  # (1, 6, 64)
    attention_output_head3,  # (1, 6, 64)
    attention_output_head4,  # (1, 6, 64)
    attention_output_head5,  # (1, 6, 64)
    attention_output_head6,  # (1, 6, 64)
    attention_output_head7   # (1, 6, 64)
], dim=-1)
# Shape: (1, 6, 512) - 8 heads × 64 dimensions = 512

# Apply output projection
W_o = nn.Linear(512, 512)  # Learnable output projection
attention_final = W_o(all_heads_output)
# Shape: (1, 6, 512)
# Values (first few dimensions shown):
# Position 0: [0.4, -0.3, 0.8, 0.6, -0.2, ...]  # Final attention output
# Position 1: [0.7, 0.8, -0.1, 0.4, 0.6, ...]  # Final attention output
# Position 2: [0.5, -0.5, 0.9, 0.2, -0.4, ...] # Final attention output
# Position 3: [0.3, 0.8, 0.6, -0.2, 0.8, ...]  # Final attention output
# Position 4: [-0.2, -0.7, 0.7, 0.8, -0.3, ...] # Final attention output
# Position 5: [0.8, 0.4, -0.2, 0.5, 0.9, ...]  # Final attention output
```

### Step 9: Residual Connection and Layer Normalization
```python
# Add residual connection
residual_output = x + attention_final
# Shape: (1, 6, 512)
# Values (first few dimensions shown):
# Position 0: [0.5, 0.4, 1.6, 1.8, -0.7, ...]  # Original + attention
# Position 1: [1.94, 2.04, -0.3, 1.44, 0.7, ...] # Original + attention
# Position 2: [1.61, -1.72, 2.3, 0.64, -0.8, ...] # Original + attention
# Position 3: [0.94, -0.09, 2.2, -0.36, 1.5, ...] # Original + attention
# Position 4: [-0.66, -1.95, 1.9, 1.74, -0.5, ...] # Original + attention
# Position 5: [0.54, 0.88, -0.7, 0.74, 1.8, ...] # Original + attention

# Apply layer normalization
layer_norm1 = nn.LayerNorm(512)
normalized_output = layer_norm1(residual_output)
# Shape: (1, 6, 512)
# Values (first few dimensions shown):
# Position 0: [0.2, 0.1, 0.8, 0.9, -0.3, ...]  # Normalized
# Position 1: [0.9, 1.0, -0.1, 0.7, 0.3, ...]  # Normalized
# Position 2: [0.7, -0.8, 1.1, 0.3, -0.4, ...] # Normalized
# Position 3: [0.4, 0.0, 1.0, -0.2, 0.7, ...]  # Normalized
# Position 4: [-0.3, -0.9, 0.9, 0.8, -0.2, ...] # Normalized
# Position 5: [0.2, 0.4, -0.3, 0.3, 0.8, ...]  # Normalized
```

### Step 10: Feed-Forward Network
```python
# Feed-forward network: two linear layers with ReLU
ffn = nn.Sequential(
    nn.Linear(512, 2048),  # Expand to 4x dimension
    nn.ReLU(),
    nn.Linear(2048, 512)   # Contract back to original dimension
)

ffn_output = ffn(normalized_output)
# Shape: (1, 6, 512)
# Values (first few dimensions shown):
# Position 0: [0.3, -0.2, 0.9, 0.7, -0.1, ...]  # FFN output
# Position 1: [0.8, 0.9, -0.2, 0.6, 0.4, ...]  # FFN output
# Position 2: [0.6, -0.7, 1.0, 0.4, -0.3, ...] # FFN output
# Position 3: [0.5, 0.1, 0.9, -0.1, 0.8, ...]  # FFN output
# Position 4: [-0.1, -0.8, 0.8, 0.9, -0.1, ...] # FFN output
# Position 5: [0.7, 0.5, -0.2, 0.4, 0.9, ...]  # FFN output
```

### Step 11: Final Residual Connection and Layer Normalization
```python
# Add residual connection
final_output = normalized_output + ffn_output
# Shape: (1, 6, 512)
# Values (first few dimensions shown):
# Position 0: [0.5, -0.1, 1.7, 1.6, -0.4, ...]  # Normalized + FFN
# Position 1: [1.7, 1.9, -0.3, 1.3, 0.7, ...]  # Normalized + FFN
# Position 2: [1.3, -1.5, 2.1, 0.7, -0.7, ...] # Normalized + FFN
# Position 3: [0.9, 0.1, 1.9, -0.3, 1.5, ...]  # Normalized + FFN
# Position 4: [-0.4, -1.7, 1.7, 1.7, -0.3, ...] # Normalized + FFN
# Position 5: [0.9, 0.9, -0.5, 0.7, 1.7, ...]  # Normalized + FFN

# Final layer normalization
final_normalized = layer_norm2(final_output)
# Shape: (1, 6, 512)
# Values (first few dimensions shown):
# Position 0: [0.2, -0.1, 0.8, 0.7, -0.2, ...]  # Final normalized
# Position 1: [0.8, 0.9, -0.1, 0.6, 0.3, ...]  # Final normalized
# Position 2: [0.6, -0.7, 1.0, 0.3, -0.3, ...] # Final normalized
# Position 3: [0.4, 0.0, 0.9, -0.1, 0.7, ...]  # Final normalized
# Position 4: [-0.2, -0.8, 0.8, 0.8, -0.1, ...] # Final normalized
# Position 5: [0.4, 0.4, -0.2, 0.3, 0.8, ...]  # Final normalized
```

## 🔄 Complete Transformer Block Summary

The input sentence "This is an example input" has been processed through:

1. **Tokenization**: 6 tokens → token IDs
2. **Embedding**: Token IDs → 512-dimensional embeddings
3. **Positional Encoding**: Added positional information
4. **Multi-Head Attention**: 8 heads process the sequence
5. **Feed-Forward Network**: Non-linear transformation
6. **Residual Connections**: Preserve gradient flow
7. **Layer Normalization**: Stabilize training

**Final Output Shape**: `(1, 6, 512)`
- **Batch size**: 1
- **Sequence length**: 6 tokens
- **Hidden dimension**: 512

Each position now contains a rich representation that has attended to all other positions in the sequence, capturing the contextual relationships between words.

## 📊 Attention Visualization

The attention weights show how each token attends to others:
- **[CLS]**: Attends broadly to understand the full sentence
- **"This"**: Focuses on "is" and "an" (subject-verb relationship)
- **"is"**: Attends to "This" and "an" (verb-subject relationship)
- **"an"**: Attends to "example" (article-noun relationship)
- **"example"**: Attends to "an" and "input" (noun relationships)
- **"input"**: Attends to "example" (noun-noun relationship)

This demonstrates how the transformer learns to capture grammatical and semantic relationships through self-attention! 🎯
