#!/usr/bin/env python3
"""
Transformer Processing Demo: Complete Trace of "It is a example input"
This file demonstrates the complete transformer processing pipeline with detailed
matrix multiplications and step-by-step transformations.

MATHEMATICAL FOUNDATIONS:
- Self-Attention: Attention(Q,K,V) = softmax(QK^T/√d_k)V
- Multi-Head: Concatenate multiple attention heads and project with W_o
- Positional Encoding: sin(pos/10000^(2i/d_model)) and cos(pos/10000^(2i/d_model))
- Layer Normalization: Normalize across the last dimension (d_model)
- Residual Connections: x + sublayer(x) for gradient flow preservation

TENSOR SHAPES EXPLAINED:
- Input: (batch_size, seq_len, d_model)
- Q,K,V: (batch_size, seq_len, d_model) before reshaping
- Q,K,V reshaped: (batch_size, num_heads, seq_len, d_k) where d_k = d_model/num_heads
- Attention scores: (batch_size, num_heads, seq_len, seq_len)
- Attention weights: (batch_size, num_heads, seq_len, seq_len) - probabilities summing to 1 per row
- Head outputs: (batch_size, num_heads, seq_len, d_v) where d_v = d_k typically
- Concatenated: (batch_size, seq_len, num_heads * d_v) = (batch_size, seq_len, d_model)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

# Set random seed for reproducible results
torch.manual_seed(42)
np.random.seed(42)

class TransformerTraceDemo:
    """
    Complete transformer implementation with detailed tracing of all operations.
    This class shows every step of the transformer processing pipeline.
    
    MATHEMATICAL CONCEPTS:
    - Self-Attention: Allows each position to attend to all positions in the sequence
    - Multi-Head: Multiple attention heads learn different types of relationships
    - Positional Encoding: Injects sequence order information (transformers are permutation-invariant)
    - Residual Connections: Help with gradient flow in deep networks
    - Layer Normalization: Stabilizes training by normalizing activations
    """
    
    def __init__(self, d_model=512, num_heads=8, vocab_size=30522, max_seq_len=512):
        """
        Initialize the transformer demo with configurable parameters.
        
        MATHEMATICAL RELATIONSHIPS:
        - d_k = d_v = d_model / num_heads (ensures proper dimensionality)
        - Total parameters in attention: 4 * d_model * d_model (W_q, W_k, W_v, W_o)
        - FFN typically expands to 4x d_model then contracts back
        
        Args:
            d_model: Hidden dimension - determines the width of all representations
            num_heads: Number of attention heads - each head learns different patterns
            vocab_size: Vocabulary size - number of unique tokens the model can handle
            max_seq_len: Maximum sequence length - determines positional encoding size
        """
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # Per-head dimension for queries and keys
        self.d_v = d_model // num_heads  # Per-head dimension for values (usually equals d_k)
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        
        # LEARNABLE PARAMETERS - These are what the model learns during training
        
        # Token Embeddings: Maps token IDs to dense vectors
        # Shape: (vocab_size, d_model) - each token gets a unique d_model-dimensional vector
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        # Positional Encodings: Learnable position representations
        # Shape: (1, max_seq_len, d_model) - each position gets a unique encoding
        self.positional_encoding = nn.Parameter(torch.randn(1, max_seq_len, d_model))
        
        # MULTI-HEAD ATTENTION PROJECTIONS - These transform the input into Q, K, V spaces
        
        # W_q: Query projection - learns what information each position is looking for
        # Shape: (d_model, d_model) - transforms input to query space
        self.W_q = nn.Linear(d_model, d_model)
        
        # W_k: Key projection - learns what information each position can provide
        # Shape: (d_model, d_model) - transforms input to key space
        self.W_k = nn.Linear(d_model, d_model)
        
        # W_v: Value projection - learns the actual content/information to be retrieved
        # Shape: (d_model, d_model) - transforms input to value space
        self.W_v = nn.Linear(d_model, d_model)
        
        # W_o: Output projection - learns how to combine information from all heads
        # Shape: (d_model, d_model) - transforms concatenated head outputs back to model space
        self.W_o = nn.Linear(d_model, d_model)
        
        # FEED-FORWARD NETWORK - Adds non-linearity and increases model capacity
        
        # Typically expands to 4x d_model then contracts back
        # This allows the model to learn complex non-linear transformations
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),  # Expand: (batch, seq, d_model) → (batch, seq, 4*d_model)
            nn.ReLU(),                        # Non-linearity: max(0, x)
            nn.Linear(d_model * 4, d_model)   # Contract: (batch, seq, 4*d_model) → (batch, seq, d_model)
        )
        
        # LAYER NORMALIZATION - Stabilizes training by normalizing activations
        
        # Normalizes across the last dimension (d_model) for each position independently
        # This helps with training stability and convergence
        self.layer_norm1 = nn.LayerNorm(d_model)  # After attention
        self.layer_norm2 = nn.LayerNorm(d_model)  # After FFN
        
        # Initialize weights for better visualization and understanding
        self._initialize_weights()
    
    def _initialize_weights(self):
        """
        Initialize weights with small values for better visualization.
        
        WEIGHT INITIALIZATION STRATEGIES:
        - Xavier/He initialization: Helps with gradient flow in deep networks
        - Small initial values: Prevents saturation of activation functions
        - Sinusoidal positional encoding: Provides unique position representations
        """
        # Initialize attention projection weights with Xavier uniform
        # This ensures proper scaling of the input to prevent vanishing/exploding gradients
        for module in [self.W_q, self.W_k, self.W_v, self.W_o]:
            nn.init.xavier_uniform_(module.weight, gain=0.1)  # Small gain for visualization
            nn.init.zeros_(module.bias)  # Start with zero bias
        
        # Initialize token embeddings with small normal distribution
        # This gives each token a unique starting representation
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.1)
        
        # Initialize positional encodings with sinusoidal pattern
        # This provides unique position representations that can generalize to unseen positions
        with torch.no_grad():
            for pos in range(self.max_seq_len):
                for i in range(0, self.d_model, 2):
                    # sin(pos / 10000^(2i/d_model)) for even dimensions
                    self.positional_encoding[0, pos, i] = math.sin(pos / (10000 ** (i / self.d_model)))
                    # cos(pos / 10000^(2i/d_model)) for odd dimensions
                    if i + 1 < self.d_model:
                        self.positional_encoding[0, pos, i + 1] = math.cos(pos / (10000 ** (i / self.d_model)))
    
    def tokenize_input(self, text):
        """
        Simple tokenization for demonstration purposes.
        In practice, you would use a proper tokenizer like BERT's WordPiece.
        
        TOKENIZATION PROCESS:
        1. Split text into words/subwords
        2. Add special tokens: [CLS] (start) and [SEP] (end)
        3. Convert tokens to numerical IDs using vocabulary
        4. Handle unknown tokens (OOV - Out of Vocabulary)
        
        Args:
            text: Input text string
            
        Returns:
            token_ids: Tensor of token IDs with shape (1, seq_len)
            tokens: List of token strings for visualization
        """
        # Simple word-based tokenization for demo
        # In practice, BERT uses WordPiece tokenization which can split words into subwords
        tokens = ["[CLS]"] + text.split() + ["[SEP]"]
        
        # Create a simple vocabulary mapping for demo
        # In practice, this would be a large vocabulary learned from training data
        vocab = {"[CLS]": 101, "[SEP]": 102, "It": 2023, "is": 1037, 
                "a": 2741, "example": 4248, "input": 5000}
        
        # Convert tokens to IDs, using 1000 for unknown tokens (OOV)
        token_ids = [vocab.get(token, 1000) for token in tokens]
        
        print(f"🔤 Tokenization:")
        print(f"   Text: '{text}'")
        print(f"   Tokens: {tokens}")
        print(f"   Token IDs: {token_ids}")
        print(f"   Sequence length: {len(tokens)}")
        print(f"   Special tokens: [CLS] marks start, [SEP] marks end")
        print()
        
        return torch.tensor([token_ids]), tokens
    
    def get_token_embeddings(self, token_ids):
        """
        Convert token IDs to dense vector representations (embeddings).
        
        EMBEDDING PROCESS:
        1. Each token ID is used as an index into the embedding table
        2. The embedding table is a learnable matrix of shape (vocab_size, d_model)
        3. Each token gets a unique d_model-dimensional vector
        4. These vectors capture semantic meaning of tokens
        
        MATHEMATICAL OPERATION:
        - embedding_table[token_id] → vector of shape (d_model,)
        - For batch: embedding_table[token_ids] → tensor of shape (batch_size, seq_len, d_model)
        
        Args:
            token_ids: Tensor of token IDs with shape (batch_size, seq_len)
            
        Returns:
            embeddings: Token embeddings with shape (batch_size, seq_len, d_model)
        """
        # Lookup token embeddings from the learnable embedding table
        # This is essentially a matrix lookup: embedding_table[token_ids]
        embeddings = self.token_embedding(token_ids)
        
        print(f"📊 Step 1: Token Embeddings")
        print(f"   Input shape: {token_ids.shape}")
        print(f"   Output shape: {embeddings.shape}")
        print(f"   Mathematical operation: embedding_table[token_ids]")
        print(f"   Each token now has a {self.d_model}-dimensional representation")
        print(f"   Sample values (first 5 dimensions of first 3 positions):")
        for i in range(min(3, embeddings.shape[1])):
            pos_name = ["[CLS]", "It", "is"][i] if i < 3 else f"Pos{i}"
            print(f"     {pos_name}: {embeddings[0, i, :5].tolist()}")
        print(f"   These vectors will be learned during training to capture semantic meaning")
        print()
        
        return embeddings
    
    def add_positional_encoding(self, embeddings):
        """
        Add positional encodings to token embeddings.
        
        WHY POSITIONAL ENCODING IS NEEDED:
        - Transformers are permutation-invariant (order doesn't matter)
        - Without positional information, "It is" and "is It" would be identical
        - Positional encoding injects sequence order information
        
        MATHEMATICAL OPERATION:
        - final_embedding = token_embedding + positional_encoding
        - This is element-wise addition (broadcasting)
        
        POSITIONAL ENCODING TYPES:
        1. Sinusoidal (original transformer): sin(pos/10000^(2i/d_model))
        2. Learned: nn.Parameter that learns optimal position representations
        3. Relative: Encodes relative distances between positions
        
        Args:
            embeddings: Token embeddings with shape (batch_size, seq_len, d_model)
            
        Returns:
            encoded: Embeddings with positional information added
        """
        seq_len = embeddings.shape[1]
        # Extract positional encodings for the current sequence length
        pos_encodings = self.positional_encoding[:, :seq_len, :]
        
        # Add positional encodings to token embeddings
        # This is element-wise addition: embeddings + pos_encodings
        # Broadcasting ensures proper addition across all dimensions
        encoded = embeddings + pos_encodings
        
        print(f"📍 Step 2: Positional Encoding")
        print(f"   Input embeddings shape: {embeddings.shape}")
        print(f"   Positional encoding shape: {pos_encodings.shape}")
        print(f"   Output shape: {encoded.shape}")
        print(f"   Mathematical operation: embeddings + positional_encodings")
        print(f"   Each position now has unique positional information")
        print(f"   Sample values after adding positional encoding (first 5 dimensions):")
        for i in range(min(3, encoded.shape[1])):
            pos_name = ["[CLS]", "It", "is"][i] if i < 3 else f"Pos{i}"
            print(f"     {pos_name}: {embeddings[0, i, :5].tolist()} + {pos_encodings[0, i, :5].tolist()} = {encoded[0, i, :5].tolist()}")
        print(f"   Now each token has both semantic (from embedding) and positional information")
        print()
        
        return encoded
    
    def multi_head_attention(self, x):
        """
        Multi-head attention mechanism with detailed tracing.
        
        MULTI-HEAD ATTENTION OVERVIEW:
        - Instead of one attention mechanism, use multiple (num_heads) attention mechanisms
        - Each head learns different types of relationships
        - Heads run in parallel and their outputs are concatenated
        - Final output projection (W_o) learns how to combine information from all heads
        
        MATHEMATICAL STEPS:
        1. Linear projections: Q = W_q(x), K = W_k(x), V = W_v(x)
        2. Reshape to heads: Split d_model into num_heads × d_k
        3. For each head: Attention(Q_h, K_h, V_h) = softmax(Q_h K_h^T/√d_k) V_h
        4. Concatenate all head outputs
        5. Final projection: output = W_o(concatenated_heads)
        
        Args:
            x: Input tensor with shape (batch_size, seq_len, d_model)
            
        Returns:
            output: Attention output with shape (batch_size, seq_len, d_model)
            attention_weights: Attention weights for visualization
        """
        batch_size, seq_len, _ = x.shape
        
        print(f"🎭 Step 3-8: Multi-Head Attention")
        print(f"   Input shape: {x.shape}")
        print(f"   Number of heads: {self.num_heads}")
        print(f"   Per-head dimension (d_k, d_v): {self.d_k}")
        print(f"   Total attention parameters: 4 × {self.d_model} × {self.d_model} = {4 * self.d_model * self.d_model:,}")
        print()
        
        # Step 3: Linear projections to generate Q, K, V
        print(f"   🔍 Step 3: Linear Projections")
        print(f"     Mathematical operations:")
        print(f"       Q = W_q × x  (Query: what am I looking for?)")
        print(f"       K = W_k × x  (Key: what do I contain?)")
        print(f"       V = W_v × x  (Value: what do I offer?)")
        print(f"     Where W_q, W_k, W_v are learnable weight matrices")
        
        Q = self.W_q(x)  # Shape: (batch_size, seq_len, d_model)
        K = self.W_k(x)  # Shape: (batch_size, seq_len, d_model)
        V = self.W_v(x)  # Shape: (batch_size, seq_len, d_model)
        
        print(f"     Q shape: {Q.shape}")
        print(f"     K shape: {K.shape}")
        print(f"     V shape: {V.shape}")
        print(f"     Sample Q values (first 5 dimensions of first 3 positions):")
        for i in range(min(3, Q.shape[1])):
            pos_name = ["[CLS]", "It", "is"][i] if i < 3 else f"Pos{i}"
            print(f"       {pos_name}: {Q[0, i, :5].tolist()}")
        print(f"     These Q, K, V are learned transformations of the input")
        print()
        
        # Step 4: Reshape for multi-head attention
        print(f"   🔄 Step 4: Reshape to Heads")
        print(f"     Mathematical operation:")
        print(f"       Reshape (batch_size, seq_len, d_model) → (batch_size, seq_len, num_heads, d_k)")
        print(f"       Then transpose to (batch_size, num_heads, seq_len, d_k)")
        print(f"     This allows each head to process the sequence independently")
        
        # Reshape: split d_model dimension into num_heads × d_k
        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.d_v).transpose(1, 2)
        
        print(f"     Q reshaped: {Q.shape}")
        print(f"     K reshaped: {K.shape}")
        print(f"     V reshaped: {V.shape}")
        print(f"     Now each head has its own Q, K, V tensors")
        print()
        
        # Step 5-7: Compute attention for each head
        print(f"   🧮 Step 5-7: Attention Computation")
        print(f"     For each head, compute: Attention(Q_h, K_h, V_h) = softmax(Q_h K_h^T/√d_k) V_h")
        print(f"     This happens in parallel for all heads")
        
        attention_outputs = []
        all_attention_weights = []
        
        for head in range(self.num_heads):
            print(f"     Head {head}:")
            
            # Extract Q, K, V for this head
            Q_head = Q[:, head, :, :]  # Shape: (batch_size, seq_len, d_k)
            K_head = K[:, head, :, :]  # Shape: (batch_size, seq_len, d_k)
            V_head = V[:, head, :, :]  # Shape: (batch_size, seq_len, d_v)
            
            # Step 5: Compute attention scores: Q @ K^T
            # This computes similarity between every query and every key
            # Shape: (batch_size, seq_len, seq_len) - attention matrix
            scores = torch.matmul(Q_head, K_head.transpose(-2, -1))
            print(f"       Attention scores shape: {scores.shape}")
            print(f"       Mathematical operation: scores = Q_head @ K_head^T")
            print(f"       Each score[i,j] represents how much position i should attend to position j")
            print(f"       Sample scores (first 3x3 positions):")
            print(f"         {scores[0, :3, :3].tolist()}")
            
            # Step 6: Scale scores by √d_k
            # This prevents softmax from saturating when d_k is large
            # Mathematical reason: variance of dot product grows with d_k
            scaled_scores = scores / math.sqrt(self.d_k)
            print(f"       Scaled scores (first 3x3 positions):")
            print(f"         {scaled_scores[0, :3, :3].tolist()}")
            print(f"       Scaling factor: √{self.d_k} = {math.sqrt(self.d_k):.2f}")
            print(f"       This prevents softmax saturation and stabilizes gradients")
            
            # Apply softmax to get attention weights
            # This converts scores to probabilities that sum to 1 for each position
            attention_weights = F.softmax(scaled_scores, dim=-1)
            print(f"       Attention weights (first 3x3 positions):")
            print(f"         {attention_weights[0, :3, :3].tolist()}")
            print(f"       Each row sums to 1: {attention_weights[0, :3, :3].sum(dim=-1).tolist()}")
            print(f"       These weights determine how much each position attends to others")
            
            # Step 7: Apply attention weights to values
            # This creates a weighted combination of values based on attention weights
            head_output = torch.matmul(attention_weights, V_head)
            print(f"       Head output shape: {head_output.shape}")
            print(f"       Mathematical operation: output = attention_weights @ V_head")
            print(f"       Each position gets a weighted sum of all values")
            print()
            
            attention_outputs.append(head_output)
            all_attention_weights.append(attention_weights)
        
        # Step 8: Concatenate all heads
        print(f"   🔗 Step 8: Concatenate Heads")
        print(f"     Mathematical operation:")
        print(f"       Concatenate all head outputs along the last dimension")
        print(f"       Shape: {self.num_heads} × (batch_size, seq_len, d_v) → (batch_size, seq_len, {self.num_heads * self.d_v})")
        
        # Concatenate all head outputs
        attention_output = torch.cat(attention_outputs, dim=-1)
        print(f"     Concatenated shape: {attention_output.shape}")
        print(f"     Now we have information from all {self.num_heads} attention heads")
        
        # Apply output projection to learn how to combine information from all heads
        output = self.W_o(attention_output)
        print(f"     Final output shape: {output.shape}")
        print(f"     W_o learns optimal combination of head outputs")
        print(f"     Sample output values (first 5 dimensions of first 3 positions):")
        for i in range(min(3, output.shape[1])):
            pos_name = ["[CLS]", "It", "is"][i] if i < 3 else f"Pos{i}"
            print(f"       {pos_name}: {output[0, i, :5].tolist()}")
        print(f"     This output contains rich contextual information from all positions")
        print()
        
        # Return average attention weights across heads for visualization
        avg_attention_weights = torch.stack(all_attention_weights).mean(dim=0)
        
        return output, avg_attention_weights
    
    def feed_forward_network(self, x):
        """
        Feed-forward network with detailed tracing.
        
        FEED-FORWARD NETWORK PURPOSE:
        - Adds non-linearity to the model
        - Increases model capacity and expressiveness
        - Processes each position independently (position-wise)
        - Typically expands to 4x dimension then contracts back
        
        MATHEMATICAL OPERATION:
        - FFN(x) = W2 × ReLU(W1 × x + b1) + b2
        - Where W1: (d_model, 4*d_model), W2: (4*d_model, d_model)
        
        Args:
            x: Input tensor with shape (batch_size, seq_len, d_model)
            
        Returns:
            output: FFN output with shape (batch_size, seq_len, d_model)
        """
        print(f"🔄 Step 10: Feed-Forward Network")
        print(f"   Input shape: {x.shape}")
        print(f"   Mathematical operation:")
        print(f"     FFN(x) = W2 × ReLU(W1 × x + b1) + b2")
        print(f"     Where W1: ({self.d_model}, {self.d_model * 4}), W2: ({self.d_model * 4}, {self.d_model})")
        print(f"     This expands to 4x dimension then contracts back")
        print(f"     Each position is processed independently (position-wise)")
        
        # Apply feed-forward network
        output = self.ffn(x)
        
        print(f"   Output shape: {output.shape}")
        print(f"   Sample output values (first 5 dimensions of first 3 positions):")
        for i in range(min(3, output.shape[1])):
            pos_name = ["[CLS]", "It", "is"][i] if i < 3 else f"Pos{i}"
            print(f"     {pos_name}: {output[0, i, :5].tolist()}")
        print(f"   The FFN adds non-linearity and increases model capacity")
        print()
        
        return output
    
    def transformer_block(self, x):
        """
        Complete transformer block with residual connections and layer normalization.
        
        TRANSFORMER BLOCK ARCHITECTURE:
        1. Multi-Head Attention
        2. Add & Norm (residual connection + layer normalization)
        3. Feed-Forward Network
        4. Add & Norm (residual connection + layer normalization)
        
        RESIDUAL CONNECTIONS:
        - Help with gradient flow in deep networks
        - Allow the model to learn incremental changes
        - Mathematical: output = x + sublayer(x)
        
        LAYER NORMALIZATION:
        - Normalizes activations across the last dimension (d_model)
        - Helps with training stability and convergence
        - Applied after each sublayer
        
        Args:
            x: Input tensor with shape (batch_size, seq_len, d_model)
            
        Returns:
            output: Transformer block output
            attention_weights: Attention weights for visualization
        """
        print(f"🏗️  Complete Transformer Block Processing")
        print(f"   Input shape: {x.shape}")
        print(f"   Architecture: Attention → Add & Norm → FFN → Add & Norm")
        print()
        
        # Store original input for residual connection
        original_x = x
        
        # Multi-Head Attention
        attention_output, attention_weights = self.multi_head_attention(x)
        
        # Step 9: Residual connection and layer normalization
        print(f"🔗 Step 9: Residual Connection and Layer Normalization")
        print(f"     Mathematical operation: residual = x + attention_output")
        print(f"     This preserves the original information and helps with gradient flow")
        
        residual_output = original_x + attention_output
        print(f"     Residual connection shape: {residual_output.shape}")
        
        # Apply layer normalization
        normalized_output = self.layer_norm1(residual_output)
        print(f"     After layer norm shape: {normalized_output.shape}")
        print(f"     Layer norm normalizes across the last dimension (d_model) for each position")
        print(f"     This stabilizes training and helps with convergence")
        print(f"     Sample values (first 5 dimensions of first 3 positions):")
        for i in range(min(3, normalized_output.shape[1])):
            pos_name = ["[CLS]", "It", "is"][i] if i < 3 else f"Pos{i}"
            print(f"     {pos_name}: {normalized_output[0, i, :5].tolist()}")
        print()
        
        # Feed-forward network
        ffn_output = self.feed_forward_network(normalized_output)
        
        # Step 11: Final residual connection and layer normalization
        print(f"🔗 Step 11: Final Residual Connection and Layer Normalization")
        print(f"     Mathematical operation: final_residual = normalized_output + ffn_output")
        print(f"     This is the second residual connection in the transformer block")
        
        final_residual = normalized_output + ffn_output
        print(f"     Final residual shape: {final_residual.shape}")
        
        # Final layer normalization
        final_output = self.layer_norm2(final_residual)
        print(f"     Final output shape: {final_output.shape}")
        print(f"     This completes the transformer block")
        print(f"     Sample final values (first 5 dimensions of first 3 positions):")
        for i in range(min(3, final_output.shape[1])):
            pos_name = ["[CLS]", "It", "is"][i] if i < 3 else f"Pos{i}"
            print(f"     {pos_name}: {final_output[0, i, :5].tolist()}")
        print(f"     Each position now contains rich contextual information from the entire sequence")
        print()
        
        return final_output, attention_weights
    
    def process_sentence(self, text):
        """
        Complete pipeline from text input to transformer output.
        
        COMPLETE PIPELINE OVERVIEW:
        1. Tokenization: Text → Tokens → Token IDs
        2. Embedding: Token IDs → Dense vectors
        3. Positional Encoding: Add sequence order information
        4. Transformer Block: Multi-head attention + FFN + residual connections
        5. Output: Rich contextual representations for each position
        
        MATHEMATICAL FLOW:
        - Input: text string
        - Tokenization: text → [token_ids]
        - Embedding: [token_ids] → embedding_table[token_ids]
        - Positional: embeddings + positional_encoding
        - Attention: softmax(QK^T/√d_k)V
        - FFN: W2 × ReLU(W1 × x + b1) + b2
        - Residual: x + sublayer(x)
        - Output: contextual representations
        
        Args:
            text: Input text string
            
        Returns:
            output: Final transformer output
            attention_weights: Attention weights for visualization
            tokens: List of processed tokens
        """
        print(f"🚀 Complete Transformer Processing Pipeline")
        print(f"   Input text: '{text}'")
        print(f"   This will trace through the complete transformation:")
        print(f"   Text → Tokens → IDs → Embeddings → Positional → Attention → FFN → Output")
        print(f"=" * 80)
        print()
        
        # Step 1: Tokenization
        token_ids, tokens = self.tokenize_input(text)
        
        # Step 2: Token embeddings
        embeddings = self.get_token_embeddings(token_ids)
        
        # Step 3: Add positional encoding
        encoded = self.add_positional_encoding(embeddings)
        
        # Step 4-11: Transformer block
        output, attention_weights = self.transformer_block(encoded)
        
        print(f"🎯 Processing Complete!")
        print(f"   Final output shape: {output.shape}")
        print(f"   Attention weights shape: {attention_weights.shape}")
        print(f"   Each position now contains contextual information from the entire sequence")
        print(f"   The model has learned to attend to relevant parts of the input")
        print(f"=" * 80)
        
        return output, attention_weights, tokens
    
    def visualize_attention(self, attention_weights, tokens):
        """
        Visualize attention weights as a matrix.
        
        ATTENTION MATRIX INTERPRETATION:
        - Rows: Query positions (what each position is looking for)
        - Columns: Key positions (what each position can provide)
        - Values: Attention weights (how much attention is paid)
        - Each row sums to 1 (softmax normalization)
        - Higher values indicate stronger attention
        
        ATTENTION PATTERNS:
        - Diagonal: Self-attention (position attends to itself)
        - Off-diagonal: Cross-position attention
        - Global: Position attends to all other positions
        - Local: Position attends to nearby positions
        
        Args:
            attention_weights: Attention weights tensor
            tokens: List of token strings
        """
        print(f"📊 Attention Visualization")
        print(f"   Attention matrix shape: {attention_weights.shape}")
        print(f"   This shows how each position attends to all other positions")
        print()
        
        # Convert to numpy for easier printing
        attn_np = attention_weights[0].detach().numpy()
        
        # Print attention matrix
        print("   Attention weights (rows = query, columns = key):")
        print("   " + " " * 8 + "".join([f"{token:>8}" for token in tokens]))
        
        for i, token in enumerate(tokens):
            row = attn_np[i, :]
            print(f"   {token:>6} {row}")
        
        print()
        print("   Interpretation:")
        print("   - Each row shows how a token attends to all other tokens")
        print("   - Higher values (closer to 1) indicate stronger attention")
        print("   - Values in each row sum to 1 (softmax normalization)")
        print("   - This reveals what relationships the model has learned")
        print("   - For example, 'It' might attend strongly to 'is' (subject-verb relationship)")
        print()

def main():
    """Main function to run the transformer trace demo."""
    print("🤖 Transformer Processing Trace Demo")
    print("=" * 80)
    print("This demo shows the complete transformer processing pipeline with detailed")
    print("explanations of all mathematical operations and matrix transformations.")
    print()
    
    # Initialize the transformer demo
    transformer = TransformerTraceDemo(d_model=512, num_heads=8)
    
    # Process the example sentence
    text = "It is a example input"
    output, attention_weights, tokens = transformer.process_sentence(text)
    
    # Visualize attention
    transformer.visualize_attention(attention_weights, tokens)
    
    print("✅ Demo completed successfully!")
    print("\n📚 Key Mathematical Concepts:")
    print("   1. Self-Attention: Attention(Q,K,V) = softmax(QK^T/√d_k)V")
    print("   2. Multi-Head: Multiple attention heads learn different relationships")
    print("   3. Positional Encoding: Injects sequence order information")
    print("   4. Residual Connections: x + sublayer(x) for gradient flow")
    print("   5. Layer Normalization: Stabilizes training across d_model dimension")
    print("   6. Feed-Forward: W2 × ReLU(W1 × x + b1) + b2 for non-linearity")
    print()
    print("🎯 What Happened:")
    print("   - Text was tokenized into numerical IDs")
    print("   - IDs were converted to dense vector embeddings")
    print("   - Positional information was added")
    print("   - Multi-head attention computed relationships between all positions")
    print("   - Feed-forward networks added non-linearity")
    print("   - Final output contains rich contextual representations!")
    print()
    print("🔬 Next Steps:")
    print("   - Try different input sentences")
    print("   - Experiment with different model configurations")
    print("   - Analyze attention patterns for different inputs")
    print("   - Understand how the model learns relationships")

if __name__ == "__main__":
    main()
