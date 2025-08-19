# Advanced Concepts

This module covers cutting-edge transformer techniques including attention variants, advanced positional encoding, normalization methods, and optimization strategies.

## 🎯 Learning Objectives

- Master advanced attention mechanisms beyond standard self-attention
- Understand sophisticated positional encoding techniques
- Learn modern normalization and optimization methods
- Explore state-of-the-art transformer innovations

## 📖 Table of Contents

1. [Attention Variants](#attention-variants)
2. [Positional Encoding](#positional-encoding)
3. [Normalization Techniques](#normalization-techniques)
4. [Optimization Strategies](#optimization-strategies)
5. [Recent Innovations](#recent-innovations)

## 🔍 Attention Variants

### Linear Attention

Linear attention reduces quadratic complexity to linear by approximating softmax.

```python
class LinearAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
    def forward(self, Q, K, V):
        # Linear attention approximation
        Q = Q.softmax(dim=-1)
        K = K.softmax(dim=-1)
        
        # Compute attention in linear time
        KV = torch.einsum('bhd,bhe->bhe', K, V)
        QKV = torch.einsum('bhd,bhe->bhd', Q, KV)
        
        return QKV
```

### Sparse Attention

Sparse attention only attends to a subset of positions.

```python
class SparseAttention(nn.Module):
    def __init__(self, d_model, num_heads, sparsity_factor=4):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, num_heads)
        self.sparsity_factor = sparsity_factor
        
    def forward(self, x):
        seq_len = x.size(1)
        
        # Create sparse attention pattern
        if seq_len > self.sparsity_factor:
            # Attend to every nth position
            sparse_indices = torch.arange(0, seq_len, self.sparsity_factor)
            x_sparse = x[:, sparse_indices]
            
            # Apply attention to sparse subset
            attended, _ = self.attention(x_sparse, x_sparse, x_sparse)
            
            # Interpolate back to full sequence
            x = F.interpolate(attended.transpose(1, 2), size=seq_len).transpose(1, 2)
        
        return x
```

### Local Attention

Local attention restricts attention to a fixed window around each position.

```python
class LocalAttention(nn.Module):
    def __init__(self, d_model, num_heads, window_size=7):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, num_heads)
        self.window_size = window_size
        
    def forward(self, x):
        seq_len = x.size(1)
        outputs = []
        
        for i in range(seq_len):
            # Define local window
            start = max(0, i - self.window_size // 2)
            end = min(seq_len, i + self.window_size // 2 + 1)
            
            # Extract local context
            local_x = x[:, start:end]
            
            # Apply attention to local window
            attended, _ = self.attention(
                x[:, i:i+1], local_x, local_x
            )
            outputs.append(attended)
        
        return torch.cat(outputs, dim=1)
```

## 📍 Positional Encoding

### Rotary Positional Embedding (RoPE)

RoPE encodes relative positions through rotation in complex space.

```python
class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_seq_len=2048):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        
        # Generate rotation matrices
        self.register_buffer('cos', self._get_cos_embeddings())
        self.register_buffer('sin', self._get_sin_embeddings())
        
    def _get_cos_embeddings(self):
        position = torch.arange(0, self.max_seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2) * 
                           -(math.log(10000.0) / self.d_model))
        return torch.cos(position * div_term)
    
    def _get_sin_embeddings(self):
        position = torch.arange(0, self.max_seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2) * 
                           -(math.log(10000.0) / self.d_model))
        return torch.sin(position * div_term)
    
    def forward(self, x, seq_len):
        # Apply rotary embeddings
        cos = self.cos[:seq_len]
        sin = self.sin[:seq_len]
        
        # Rotate embeddings
        x_rot = x * cos + self._rotate_half(x) * sin
        return x_rot
    
    def _rotate_half(self, x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)
```

### ALiBi (Attention with Linear Biases)

ALiBi adds learnable position biases to attention scores.

```python
class ALiBiPositionalEmbedding(nn.Module):
    def __init__(self, num_heads, max_seq_len=2048):
        super().__init__()
        self.num_heads = num_heads
        
        # Learnable position biases
        slopes = torch.Tensor(self._get_slopes(num_heads))
        self.register_buffer('slopes', slopes)
        
        # Create position indices
        pos_indices = torch.arange(max_seq_len)
        self.register_buffer('pos_indices', pos_indices)
        
    def _get_slopes(self, num_heads):
        # Generate slopes for different heads
        slopes = []
        for i in range(num_heads):
            slope = 2 ** (-8 * (i + 1) / num_heads)
            slopes.append(slope)
        return slopes
    
    def forward(self, attention_scores):
        # Add ALiBi biases
        seq_len = attention_scores.size(-1)
        pos_indices = self.pos_indices[:seq_len]
        
        # Compute position biases
        pos_biases = pos_indices.unsqueeze(0) - pos_indices.unsqueeze(1)
        pos_biases = pos_biases.unsqueeze(0).expand(self.num_heads, -1, -1)
        
        # Apply slopes
        pos_biases = pos_biases * self.slopes.unsqueeze(-1).unsqueeze(-1)
        
        return attention_scores + pos_biases
```

## 🔧 Normalization Techniques

### RMSNorm

RMSNorm normalizes using root mean square instead of mean and variance.

```python
class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))
        
    def forward(self, x):
        # Compute RMS
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        
        # Normalize and scale
        return x * rms * self.weight
```

### GroupNorm

GroupNorm normalizes across groups of channels.

```python
class GroupNorm(nn.Module):
    def __init__(self, d_model, num_groups=32, eps=1e-6):
        super().__init__()
        self.num_groups = num_groups
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))
        
    def forward(self, x):
        # Reshape for group normalization
        batch_size, seq_len, d_model = x.shape
        x = x.view(batch_size, seq_len, self.num_groups, -1)
        
        # Compute group statistics
        mean = x.mean(dim=(1, 3), keepdim=True)
        var = x.var(dim=(1, 3), keepdim=True, unbiased=False)
        
        # Normalize
        x = (x - mean) / torch.sqrt(var + self.eps)
        
        # Reshape back and apply affine transformation
        x = x.view(batch_size, seq_len, d_model)
        return x * self.weight + self.bias
```

## 🚀 Optimization Strategies

### AdamW

AdamW with proper weight decay implementation.

```python
class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)
        
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                state = self.state[p]
                
                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                
                state['step'] += 1
                
                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                # Bias correction
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                
                # Update parameters
                step_size = group['lr'] / bias_correction1
                bias_correction2_sqrt = math.sqrt(bias_correction2)
                
                # Apply weight decay
                p.data.add_(p.data, alpha=-group['weight_decay'] * group['lr'])
                
                # Apply Adam update
                p.data.addcdiv_(exp_avg, exp_avg_sq.sqrt() + group['eps'], value=-step_size)
        
        return loss
```

### Lion Optimizer

Lion is a memory-efficient optimizer that uses sign-based updates.

```python
class Lion(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)
        
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                state = self.state[p]
                
                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                
                exp_avg = state['exp_avg']
                beta1, beta2 = group['betas']
                
                state['step'] += 1
                
                # Update momentum
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                
                # Lion update: use sign of momentum
                update = torch.sign(exp_avg)
                
                # Apply weight decay
                if group['weight_decay'] != 0:
                    update.add_(p.data, alpha=group['weight_decay'])
                
                # Update parameters
                p.data.add_(update, alpha=-group['lr'])
        
        return loss
```

### Sophia Optimizer

Sophia uses second-order information for better optimization.

```python
class Sophia(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-4, betas=(0.965, 0.99), rho=0.01, weight_decay=1e-1):
        defaults = dict(lr=lr, betas=betas, rho=rho, weight_decay=weight_decay)
        super().__init__(params, defaults)
        
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                state = self.state[p]
                
                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['hessian'] = torch.zeros_like(p.data)
                
                exp_avg, hessian = state['exp_avg'], state['hessian']
                beta1, beta2 = group['betas']
                rho = group['rho']
                
                state['step'] += 1
                
                # Update momentum
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                
                # Update Hessian estimate
                hessian.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                # Compute update
                update = exp_avg / (hessian.sqrt() + rho)
                
                # Apply weight decay
                if group['weight_decay'] != 0:
                    update.add_(p.data, alpha=group['weight_decay'])
                
                # Update parameters
                p.data.add_(update, alpha=-group['lr'])
        
        return loss
```

## 🆕 Recent Innovations

### Flash Attention

Flash Attention reduces memory usage and improves speed.

```python
class FlashAttention(nn.Module):
    def __init__(self, d_model, num_heads, block_size=64):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.block_size = block_size
        
    def forward(self, Q, K, V):
        # Flash attention implementation
        # This is a simplified version - full implementation is complex
        batch_size, seq_len, _ = Q.shape
        
        # Process in blocks for memory efficiency
        outputs = []
        for i in range(0, seq_len, self.block_size):
            end_i = min(i + self.block_size, seq_len)
            
            # Extract block
            Q_block = Q[:, i:end_i]
            K_block = K[:, i:end_i]
            V_block = V[:, i:end_i]
            
            # Standard attention on block
            scores = torch.matmul(Q_block, K_block.transpose(-2, -1))
            scores = scores / math.sqrt(self.head_dim)
            attention_weights = F.softmax(scores, dim=-1)
            
            block_output = torch.matmul(attention_weights, V_block)
            outputs.append(block_output)
        
        return torch.cat(outputs, dim=1)
```

### Multi-Query Attention

Multi-query attention shares key and value projections across heads.

```python
class MultiQueryAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        # Separate projections for Q, but shared for K and V
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, self.head_dim)  # Single head dimension
        self.W_v = nn.Linear(d_model, self.head_dim)  # Single head dimension
        self.W_o = nn.Linear(d_model, d_model)
        
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        # Project to Q, K, V
        Q = self.W_q(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        K = self.W_k(x).unsqueeze(2).expand(-1, -1, self.num_heads, -1)
        V = self.W_v(x).unsqueeze(2).expand(-1, -1, self.num_heads, -1)
        
        # Reshape for attention
        Q = Q.transpose(1, 2)  # (batch, heads, seq_len, head_dim)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        
        # Compute attention
        scores = torch.matmul(Q, K.transpose(-2, -1))
        scores = scores / math.sqrt(self.head_dim)
        attention_weights = F.softmax(scores, dim=-1)
        
        # Apply attention
        attended = torch.matmul(attention_weights, V)
        
        # Reshape and project output
        attended = attended.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )
        
        return self.W_o(attended)
```

## 📊 Performance Comparison

| Technique | Memory | Speed | Quality | Use Case |
|-----------|--------|-------|---------|----------|
| **Linear Attention** | Low | Fast | Good | Long sequences |
| **Sparse Attention** | Low | Fast | Good | Large models |
| **Local Attention** | Low | Fast | Good | Local patterns |
| **RoPE** | Low | Fast | Excellent | Relative positions |
| **ALiBi** | Low | Fast | Good | Extrapolation |
| **RMSNorm** | Low | Fast | Good | Efficiency |
| **Flash Attention** | Low | Very Fast | Excellent | Production |

## 🎯 Key Takeaways

1. **Attention Variants**: Different attention mechanisms trade off quality for efficiency
2. **Positional Encoding**: Modern methods like RoPE and ALiBi improve generalization
3. **Normalization**: RMSNorm and GroupNorm offer alternatives to LayerNorm
4. **Optimization**: New optimizers like Lion and Sophia show promising results
5. **Innovation**: Flash attention and multi-query attention enable larger models

## 🚀 Next Steps

- Implement these advanced techniques in your transformer models
- Experiment with different attention variants for your specific use case
- Benchmark optimization strategies on your datasets
- Stay updated with the latest transformer research

## 📚 Further Reading

- [Efficient Transformers: A Survey](https://arxiv.org/abs/2009.06732)
- [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)
- [Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation](https://arxiv.org/abs/2108.12409)
- [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135)
- [Sophia: A Scalable Stochastic Second-order Optimizer for Language Model Pre-training](https://arxiv.org/abs/2305.14342)

## 🔎 Curated Resources and Further Study

### Libraries & kernels
- [HazyResearch/flash-attention](https://github.com/HazyResearch/flash-attention)
- [facebookresearch/xformers](https://github.com/facebookresearch/xformers)

### Key papers
- [RoPE / RoFormer](https://arxiv.org/abs/2104.09864)
- [ALiBi](https://arxiv.org/abs/2108.12409)
- [FlashAttention v2](https://arxiv.org/abs/2307.08691)
- [Longformer / BigBird (long context)](https://arxiv.org/abs/2004.05150), (https://arxiv.org/abs/2007.14062)

### Practitioner checklist
- Prefer BF16 where possible for stability; use FP16 with care (loss scaling)
- Use fused kernels (FlashAttention/xFormers) when supported; validate numerics
- Verify extrapolation behavior when changing context windows (RoPE scaling/ALiBi)
