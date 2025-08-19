# Architecture Patterns

This module explores advanced architectural patterns for designing and connecting transformer components, scaling strategies, efficiency techniques, and training paradigms.

## 🎯 Learning Objectives

- Master different ways to connect encoders and decoders
- Understand scaling strategies for models, data, and compute
- Learn efficiency techniques for production deployment
- Explore advanced training paradigms and optimization strategies

## 📖 Table of Contents

1. [Connection Patterns](#connection-patterns)
2. [Scaling Strategies](#scaling-strategies)
3. [Efficiency Techniques](#efficiency-techniques)
4. [Training Paradigms](#training-paradigms)
5. [Architecture Design Principles](#architecture-design-principles)
6. [Custom Architectures](#custom-architectures)
7. [Production Considerations](#production-considerations)

## 🔗 Connection Patterns

### 1. Standard Encoder-Decoder

The classic transformer architecture with separate encoder and decoder.

```
Input → Encoder → Encoded Representation → Decoder → Output
```

#### Implementation

```python
class StandardTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, d_ff):
        super().__init__()
        
        self.encoder = TransformerEncoder(d_model, num_heads, num_layers, d_ff)
        self.decoder = TransformerDecoder(d_model, num_heads, num_layers, d_ff)
        self.output_projection = nn.Linear(d_model, vocab_size)
        
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        # Encode source
        memory = self.encoder(src, src_mask)
        
        # Decode target using encoded memory
        output = self.decoder(tgt, memory, tgt_mask)
        
        # Project to vocabulary
        return self.output_projection(output)
```

### 2. Shared Encoder-Decoder

Encoder and decoder share parameters for efficiency.

```python
class SharedTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, d_ff):
        super().__init__()
        
        # Shared layers
        self.shared_layers = nn.ModuleList([
            TransformerLayer(d_model, num_heads, d_ff)
            for _ in range(num_layers)
        ])
        
        self.output_projection = nn.Linear(d_model, vocab_size)
        
    def encode(self, x, mask=None):
        for layer in self.shared_layers:
            x = layer(x, mask)
        return x
    
    def decode(self, x, memory, mask=None):
        # Use shared layers for decoding
        for layer in self.shared_layers:
            x = layer(x, mask, memory)
        return x
    
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        memory = self.encode(src, src_mask)
        output = self.decode(tgt, memory, tgt_mask)
        return self.output_projection(output)
```

### 3. Cross-Attention Variants

Different ways to implement cross-attention between encoder and decoder.

#### Standard Cross-Attention

```python
class CrossAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        
        self.attention = nn.MultiheadAttention(d_model, num_heads)
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x, memory, mask=None):
        # x: decoder input, memory: encoder output
        attended_output, _ = self.attention(x, memory, memory, attn_mask=mask)
        return self.norm(x + attended_output)
```

#### Multi-Query Cross-Attention

```python
class MultiQueryCrossAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        
        # Separate projections for different attention types
        self.self_attention = nn.MultiheadAttention(d_model, num_heads)
        self.cross_attention = nn.MultiheadAttention(d_model, num_heads)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
    def forward(self, x, memory, mask=None):
        # Self-attention
        self_attended, _ = self.self_attention(x, x, x, attn_mask=mask)
        x = self.norm1(x + self_attended)
        
        # Cross-attention
        cross_attended, _ = self.cross_attention(x, memory, memory)
        x = self.norm2(x + cross_attended)
        
        return x
```

### 4. Hierarchical Connections

Multi-scale connections for better feature propagation.

```python
class HierarchicalTransformer(nn.Module):
    def __init__(self, d_model, num_heads, num_layers, d_ff):
        super().__init__()
        
        self.encoder_layers = nn.ModuleList([
            TransformerLayer(d_model, num_heads, d_ff)
            for _ in range(num_layers)
        ])
        
        self.decoder_layers = nn.ModuleList([
            TransformerLayer(d_model, num_heads, d_ff)
            for _ in range(num_layers)
        ])
        
        # Skip connections between encoder and decoder
        self.skip_connections = nn.ModuleList([
            nn.Linear(d_model, d_model)
            for _ in range(num_layers)
        ])
        
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        # Encode with intermediate outputs
        encoder_outputs = []
        x = src
        
        for layer in self.encoder_layers:
            x = layer(x, src_mask)
            encoder_outputs.append(x)
        
        # Decode with skip connections
        for i, layer in enumerate(self.decoder_layers):
            # Add skip connection from corresponding encoder layer
            skip_connection = self.skip_connections[i](encoder_outputs[-(i+1)])
            tgt = tgt + skip_connection
            
            tgt = layer(tgt, tgt_mask)
        
        return tgt
```

### 5. Parallel Connections

Process encoder and decoder in parallel for efficiency.

```python
class ParallelTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, d_ff):
        super().__init__()
        
        self.encoder = TransformerEncoder(d_model, num_heads, num_layers, d_ff)
        self.decoder = TransformerDecoder(d_model, num_heads, num_layers, d_ff)
        
        # Parallel processing
        self.parallel_processing = nn.ModuleList([
            ParallelLayer(d_model, num_heads, d_ff)
            for _ in range(num_layers)
        ])
        
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        # Process encoder and decoder in parallel
        parallel_outputs = []
        
        for layer in self.parallel_processing:
            src_out, tgt_out = layer(src, tgt, src_mask, tgt_mask)
            src, tgt = src_out, tgt_out
            parallel_outputs.append((src, tgt))
        
        return tgt
```

## 📈 Scaling Strategies

### 1. Model Scaling

#### Width Scaling (Increasing Model Dimensions)

```python
class ScalableTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, d_ff, scale_factor=1.0):
        super().__init__()
        
        # Scale dimensions
        self.d_model = int(d_model * scale_factor)
        self.num_heads = int(num_heads * scale_factor)
        self.d_ff = int(d_ff * scale_factor)
        
        # Ensure num_heads divides d_model
        self.num_heads = max(1, self.num_heads)
        if self.d_model % self.num_heads != 0:
            self.d_model = (self.d_model // self.num_heads) * self.num_heads
        
        self.encoder = TransformerEncoder(self.d_model, self.num_heads, num_layers, self.d_ff)
        self.decoder = TransformerDecoder(self.d_model, self.num_heads, num_layers, self.d_ff)
        
    def get_model_size(self):
        """Calculate total parameters"""
        total_params = sum(p.numel() for p in self.parameters())
        return total_params
```

#### Depth Scaling (Increasing Number of Layers)

```python
class DeepTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, d_ff, depth_multiplier=1.0):
        super().__init__()
        
        # Scale depth
        self.num_layers = int(6 * depth_multiplier)  # Base: 6 layers
        
        # Progressive layer scaling
        self.encoder_layers = nn.ModuleList()
        self.decoder_layers = nn.ModuleList()
        
        for i in range(self.num_layers):
            # Scale dimensions progressively
            layer_scale = 1.0 + (i / self.num_layers) * 0.5
            layer_d_model = int(d_model * layer_scale)
            layer_d_ff = int(d_ff * layer_scale)
            
            self.encoder_layers.append(
                TransformerLayer(layer_d_model, num_heads, layer_d_ff)
            )
            self.decoder_layers.append(
                TransformerLayer(layer_d_model, num_heads, layer_d_ff)
            )
```

#### Compound Scaling

```python
class CompoundScaledTransformer(nn.Module):
    def __init__(self, vocab_size, base_d_model, base_num_heads, base_num_layers, 
                 width_multiplier=1.0, depth_multiplier=1.0):
        super().__init__()
        
        # Compound scaling (EfficientNet style)
        self.d_model = int(base_d_model * width_multiplier)
        self.num_heads = int(base_num_heads * width_multiplier)
        self.num_layers = int(base_num_layers * depth_multiplier)
        
        # Ensure divisibility
        self.num_heads = max(1, self.num_heads)
        if self.d_model % self.num_heads != 0:
            self.d_model = (self.d_model // self.num_heads) * self.num_heads
        
        # Scale feed-forward dimension
        self.d_ff = int(self.d_model * 4 * width_multiplier)
        
        self.encoder = TransformerEncoder(self.d_model, self.num_heads, self.num_layers, self.d_ff)
        self.decoder = TransformerDecoder(self.d_model, self.num_heads, self.num_layers, self.d_ff)
```

### 2. Data Scaling

#### Progressive Data Loading

```python
class ProgressiveDataLoader:
    def __init__(self, dataset, batch_sizes=[32, 64, 128, 256], epochs_per_stage=5):
        self.dataset = dataset
        self.batch_sizes = batch_sizes
        self.epochs_per_stage = epochs_per_stage
        
    def get_dataloader(self, stage):
        """Get dataloader for current training stage"""
        batch_size = self.batch_sizes[min(stage, len(self.batch_sizes) - 1)]
        
        return DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4
        )
    
    def get_training_schedule(self):
        """Get training schedule with progressive batch sizes"""
        schedule = []
        for stage, batch_size in enumerate(self.batch_sizes):
            schedule.extend([stage] * self.epochs_per_stage)
        return schedule
```

#### Curriculum Learning

```python
class CurriculumDataLoader:
    def __init__(self, dataset, difficulty_scores):
        self.dataset = dataset
        self.difficulty_scores = difficulty_scores
        
    def get_curriculum_batch(self, epoch, max_difficulty=1.0):
        """Get batch with curriculum-based sampling"""
        # Gradually increase difficulty
        current_difficulty = min(epoch / 100, max_difficulty)
        
        # Filter samples by difficulty
        easy_samples = [
            i for i, score in enumerate(self.difficulty_scores)
            if score <= current_difficulty
        ]
        
        if len(easy_samples) == 0:
            easy_samples = list(range(len(self.dataset)))
        
        # Sample from easy examples
        selected_indices = random.sample(easy_samples, min(32, len(easy_samples)))
        return [self.dataset[i] for i in selected_indices]
```

### 3. Compute Scaling

#### Distributed Training

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def setup_distributed_training():
    """Setup distributed training environment"""
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(dist.get_rank())
    
def create_distributed_model(model, device):
    """Wrap model for distributed training"""
    model = model.to(device)
    model = DDP(model, device_ids=[device])
    return model

def distributed_training_step(model, batch, optimizer, criterion, device):
    """Single training step for distributed training"""
    model.train()
    
    # Move batch to device
    batch = {k: v.to(device) for k, v in batch.items()}
    
    # Forward pass
    optimizer.zero_grad()
    outputs = model(**batch)
    loss = criterion(outputs, batch['labels'])
    
    # Backward pass
    loss.backward()
    optimizer.step()
    
    return loss.item()
```

#### Mixed Precision Training

```python
from torch.cuda.amp import autocast, GradScaler

class MixedPrecisionTrainer:
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer
        self.scaler = GradScaler()
        
    def training_step(self, batch):
        """Training step with mixed precision"""
        self.optimizer.zero_grad()
        
        # Forward pass with autocast
        with autocast():
            outputs = self.model(**batch)
            loss = self.criterion(outputs, batch['labels'])
        
        # Backward pass with gradient scaling
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        
        return loss.item()
```

## ⚡ Efficiency Techniques

### 1. Knowledge Distillation

#### Teacher-Student Distillation

```python
class DistillationTrainer:
    def __init__(self, teacher_model, student_model, temperature=4.0, alpha=0.7):
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.temperature = temperature
        self.alpha = alpha
        
    def distillation_loss(self, student_logits, teacher_logits, labels):
        """Compute distillation loss"""
        # Hard labels loss
        hard_loss = F.cross_entropy(student_logits, labels)
        
        # Soft labels loss (knowledge distillation)
        soft_loss = F.kl_div(
            F.log_softmax(student_logits / self.temperature, dim=-1),
            F.softmax(teacher_logits / self.temperature, dim=-1),
            reduction='batchmean'
        ) * (self.temperature ** 2)
        
        # Combined loss
        total_loss = (1 - self.alpha) * hard_loss + self.alpha * soft_loss
        return total_loss
    
    def train_step(self, batch):
        """Single training step with distillation"""
        # Get teacher predictions
        with torch.no_grad():
            teacher_logits = self.teacher_model(**batch)
        
        # Get student predictions
        student_logits = self.student_model(**batch)
        
        # Compute distillation loss
        loss = self.distillation_loss(
            student_logits, teacher_logits, batch['labels']
        )
        
        return loss
```

### 2. Quantization

#### Post-Training Quantization

```python
import torch.quantization as quantization

class QuantizedTransformer(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        
    def quantize(self):
        """Quantize the model"""
        # Prepare for quantization
        self.model.eval()
        
        # Quantize to int8
        quantized_model = quantization.quantize_dynamic(
            self.model,
            {nn.Linear, nn.LSTM, nn.LSTMCell, nn.RNNCell, nn.GRUCell},
            dtype=torch.qint8
        )
        
        return quantized_model
    
    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)
```

#### Quantization-Aware Training

```python
class QATTrainer:
    def __init__(self, model):
        self.model = model
        
    def prepare_qat(self):
        """Prepare model for quantization-aware training"""
        # Fuse operations
        self.model = quantization.fuse_modules(self.model, ['conv', 'bn', 'relu'])
        
        # Insert fake quantization
        self.model.train()
        self.model = quantization.prepare_qat(self.model)
        
        return self.model
    
    def convert_to_quantized(self):
        """Convert to quantized model after training"""
        self.model.eval()
        quantized_model = quantization.convert(self.model)
        return quantized_model
```

### 3. Pruning

#### Structured Pruning

```python
class StructuredPruner:
    def __init__(self, model, pruning_ratio=0.3):
        self.model = model
        self.pruning_ratio = pruning_ratio
        
    def prune_attention_heads(self):
        """Prune attention heads based on importance"""
        for name, module in self.model.named_modules():
            if isinstance(module, nn.MultiheadAttention):
                # Calculate head importance
                head_importance = self.calculate_head_importance(module)
                
                # Prune least important heads
                num_heads_to_prune = int(module.num_heads * self.pruning_ratio)
                indices_to_prune = torch.argsort(head_importance)[:num_heads_to_prune]
                
                # Apply pruning
                self.prune_heads(module, indices_to_prune)
    
    def calculate_head_importance(self, attention_module):
        """Calculate importance of each attention head"""
        # Use gradient-based importance
        importance = torch.zeros(attention_module.num_heads)
        
        for i in range(attention_module.num_heads):
            # Mask head and compute gradient
            with torch.no_grad():
                original_output = attention_module.forward(...)
                masked_output = self.mask_head(attention_module, i)
                importance[i] = torch.norm(original_output - masked_output)
        
        return importance
```

#### Dynamic Pruning

```python
class DynamicPruner:
    def __init__(self, model, sparsity_target=0.8):
        self.model = model
        self.sparsity_target = sparsity_target
        
    def apply_dynamic_pruning(self, epoch, total_epochs):
        """Apply dynamic pruning based on training progress"""
        # Gradually increase sparsity
        current_sparsity = self.sparsity_target * (epoch / total_epochs)
        
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                # Calculate threshold for current sparsity
                weights = module.weight.data
                threshold = torch.quantile(torch.abs(weights), current_sparsity)
                
                # Create mask
                mask = torch.abs(weights) > threshold
                
                # Apply mask
                module.weight.data *= mask
```

## 🎓 Training Paradigms

### 1. Pre-training Strategies

#### Masked Language Modeling (MLM)

```python
class MLMTrainer:
    def __init__(self, model, mask_ratio=0.15):
        self.model = model
        self.mask_ratio = mask_ratio
        
    def create_masked_inputs(self, input_ids):
        """Create masked inputs for MLM"""
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Create mask
        mask = torch.rand(batch_size, seq_len, device=device) < self.mask_ratio
        
        # Create masked inputs
        masked_inputs = input_ids.clone()
        masked_inputs[mask] = self.tokenizer.mask_token_id
        
        # Create labels (only for masked positions)
        labels = input_ids.clone()
        labels[~mask] = -100  # Ignore non-masked positions
        
        return masked_inputs, labels, mask
    
    def mlm_loss(self, outputs, labels):
        """Compute MLM loss"""
        # Only compute loss on masked positions
        active_loss = labels.view(-1) != -100
        active_logits = outputs.view(-1, outputs.size(-1))
        active_labels = labels.view(-1)[active_loss]
        
        loss = F.cross_entropy(active_logits, active_labels)
        return loss
```

#### Next Sentence Prediction (NSP)

```python
class NSPTrainer:
    def __init__(self, model):
        self.model = model
        
    def create_nsp_inputs(self, sentence_a, sentence_b, labels):
        """Create inputs for NSP task"""
        # Concatenate sentences with separator
        inputs = []
        for a, b in zip(sentence_a, sentence_b):
            combined = a + [self.tokenizer.sep_token_id] + b
            inputs.append(combined)
        
        # Pad sequences
        inputs = self.tokenizer.pad({'input_ids': inputs})
        
        return inputs, labels
    
    def nsp_loss(self, outputs, labels):
        """Compute NSP loss"""
        # NSP is a binary classification task
        nsp_logits = outputs.nsp_logits
        loss = F.cross_entropy(nsp_logits, labels)
        return loss
```

### 2. Fine-tuning Strategies

#### Task-Specific Fine-tuning

```python
class TaskSpecificFineTuner:
    def __init__(self, base_model, task_config):
        self.base_model = base_model
        self.task_config = task_config
        
    def add_task_head(self, task_type):
        """Add task-specific head to base model"""
        if task_type == 'classification':
            head = nn.Linear(self.base_model.config.hidden_size, self.task_config.num_classes)
        elif task_type == 'regression':
            head = nn.Linear(self.base_model.config.hidden_size, 1)
        elif task_type == 'sequence_labeling':
            head = nn.Linear(self.base_model.config.hidden_size, self.task_config.num_labels)
        
        return head
    
    def freeze_base_layers(self, num_frozen_layers):
        """Freeze base model layers"""
        for i, layer in enumerate(self.base_model.encoder.layers):
            if i < num_frozen_layers:
                for param in layer.parameters():
                    param.requires_grad = False
```

#### Adapter Fine-tuning

```python
class AdapterLayer(nn.Module):
    def __init__(self, d_model, adapter_size=64):
        super().__init__()
        
        self.down_projection = nn.Linear(d_model, adapter_size)
        self.up_projection = nn.Linear(adapter_size, d_model)
        self.activation = nn.GELU()
        
    def forward(self, x):
        # Adapter forward pass
        adapter_output = self.down_projection(x)
        adapter_output = self.activation(adapter_output)
        adapter_output = self.up_projection(adapter_output)
        
        # Residual connection
        return x + adapter_output

class AdapterTransformer(nn.Module):
    def __init__(self, base_model, adapter_size=64):
        super().__init__()
        
        self.base_model = base_model
        self.adapters = nn.ModuleDict()
        
        # Add adapters to each layer
        for name, module in self.base_model.named_modules():
            if isinstance(module, (nn.Linear, nn.MultiheadAttention)):
                self.adapters[name] = AdapterLayer(module.out_features, adapter_size)
    
    def forward(self, *args, **kwargs):
        # Forward pass with adapters
        outputs = self.base_model(*args, **kwargs)
        return outputs
```

### 3. Instruction Tuning

#### Instruction Following

```python
class InstructionTuner:
    def __init__(self, model, instruction_template):
        self.model = model
        self.instruction_template = instruction_template
        
    def format_instruction(self, instruction, input_text):
        """Format instruction with input"""
        return self.instruction_template.format(
            instruction=instruction,
            input=input_text
        )
    
    def instruction_loss(self, outputs, targets):
        """Compute instruction following loss"""
        # Only compute loss on target tokens
        active_loss = targets.view(-1) != -100
        active_logits = outputs.view(-1, outputs.size(-1))
        active_labels = targets.view(-1)[active_loss]
        
        loss = F.cross_entropy(active_logits, active_labels)
        return loss
```

## 🏗️ Architecture Design Principles

### 1. Modularity

```python
class ModularTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        # Modular components
        self.embedding = self.create_embedding(config)
        self.encoder = self.create_encoder(config)
        self.decoder = self.create_decoder(config)
        self.output_head = self.create_output_head(config)
        
    def create_embedding(self, config):
        """Create embedding module based on config"""
        if config.embedding_type == 'learned':
            return LearnedEmbedding(config.vocab_size, config.d_model)
        elif config.embedding_type == 'sinusoidal':
            return SinusoidalEmbedding(config.d_model)
        else:
            raise ValueError(f"Unknown embedding type: {config.embedding_type}")
    
    def create_encoder(self, config):
        """Create encoder based on config"""
        if config.encoder_type == 'standard':
            return StandardEncoder(config)
        elif config.encoder_type == 'hierarchical':
            return HierarchicalEncoder(config)
        else:
            raise ValueError(f"Unknown encoder type: {config.encoder_type}")
```

### 2. Scalability

```python
class ScalableArchitecture(nn.Module):
    def __init__(self, base_config, scale_config):
        super().__init__()
        
        # Scale dimensions based on config
        self.d_model = self.scale_dimension(base_config.d_model, scale_config.width_scale)
        self.num_layers = self.scale_dimension(base_config.num_layers, scale_config.depth_scale)
        self.num_heads = self.scale_dimension(base_config.num_heads, scale_config.width_scale)
        
        # Ensure divisibility
        self.num_heads = max(1, self.num_heads)
        if self.d_model % self.num_heads != 0:
            self.d_model = (self.d_model // self.num_heads) * self.num_heads
        
        # Build scaled architecture
        self.build_architecture()
    
    def scale_dimension(self, base_dim, scale_factor):
        """Scale dimension with constraints"""
        scaled_dim = int(base_dim * scale_factor)
        return max(1, scaled_dim)
    
    def build_architecture(self):
        """Build the scaled architecture"""
        # Implementation depends on specific architecture
        pass
```

### 3. Efficiency

```python
class EfficientArchitecture(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        # Use efficient components
        self.attention = self.create_efficient_attention(config)
        self.ffn = self.create_efficient_ffn(config)
        self.normalization = self.create_efficient_norm(config)
        
    def create_efficient_attention(self, config):
        """Create efficient attention mechanism"""
        if config.attention_type == 'linear':
            return LinearAttention(config.d_model, config.num_heads)
        elif config.attention_type == 'sparse':
            return SparseAttention(config.d_model, config.num_heads)
        elif config.attention_type == 'local':
            return LocalAttention(config.d_model, config.num_heads, config.window_size)
        else:
            return StandardAttention(config.d_model, config.num_heads)
    
    def create_efficient_ffn(self, config):
        """Create efficient feed-forward network"""
        if config.ffn_type == 'mlp_mixer':
            return MLPMixer(config.d_model, config.d_ff)
        elif config.ffn_type == 'gated':
            return GatedFFN(config.d_model, config.d_ff)
        else:
            return StandardFFN(config.d_model, config.d_ff)
```

## 🎨 Custom Architectures

### 1. Hybrid Architectures

```python
class HybridTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        # Combine different architectural elements
        self.cnn_backbone = self.create_cnn_backbone(config)
        self.transformer_encoder = self.create_transformer_encoder(config)
        self.rnn_decoder = self.create_rnn_decoder(config)
        
    def forward(self, x):
        # CNN feature extraction
        cnn_features = self.cnn_backbone(x)
        
        # Transformer processing
        transformer_features = self.transformer_encoder(cnn_features)
        
        # RNN decoding
        output = self.rnn_decoder(transformer_features)
        
        return output
```

### 2. Attention Variants

```python
class MultiScaleAttention(nn.Module):
    def __init__(self, d_model, num_heads, scales=[1, 2, 4]):
        super().__init__()
        
        self.scales = scales
        self.attention_heads = nn.ModuleList([
            nn.MultiheadAttention(d_model, num_heads)
            for _ in scales
        ])
        
    def forward(self, x):
        outputs = []
        
        for scale, attention in zip(self.scales, self.attention_heads):
            # Apply attention at different scales
            if scale > 1:
                # Downsample input
                scaled_x = F.avg_pool1d(x.transpose(1, 2), scale).transpose(1, 2)
            else:
                scaled_x = x
            
            # Apply attention
            attended, _ = attention(scaled_x, scaled_x, scaled_x)
            
            # Upsample if needed
            if scale > 1:
                attended = F.interpolate(attended.transpose(1, 2), size=x.size(1)).transpose(1, 2)
            
            outputs.append(attended)
        
        # Combine multi-scale outputs
        return torch.stack(outputs).mean(dim=0)
```

## 🚀 Production Considerations

### 1. Model Optimization

```python
class ProductionOptimizer:
    def __init__(self, model):
        self.model = model
        
    def optimize_for_inference(self):
        """Optimize model for production inference"""
        # Fuse operations
        self.model = torch.jit.script(self.model)
        
        # Quantize if needed
        if self.quantize:
            self.model = self.quantize_model()
        
        # Optimize memory usage
        self.model = self.optimize_memory()
        
        return self.model
    
    def optimize_memory(self):
        """Optimize memory usage"""
        # Use gradient checkpointing
        self.model.gradient_checkpointing_enable()
        
        # Use mixed precision
        self.model = self.model.half()
        
        return self.model
```

### 2. Deployment Strategies

```python
class ModelDeployer:
    def __init__(self, model, deployment_config):
        self.model = model
        self.config = deployment_config
        
    def deploy(self):
        """Deploy model based on configuration"""
        if self.config.deployment_type == 'torchserve':
            return self.deploy_torchserve()
        elif self.config.deployment_type == 'triton':
            return self.deploy_triton()
        elif self.config.deployment_type == 'onnx':
            return self.deploy_onnx()
        else:
            raise ValueError(f"Unknown deployment type: {self.config.deployment_type}")
    
    def deploy_onnx(self):
        """Deploy as ONNX model"""
        # Export to ONNX
        dummy_input = torch.randn(1, self.config.sequence_length, self.config.d_model)
        
        torch.onnx.export(
            self.model,
            dummy_input,
            "model.onnx",
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size', 1: 'sequence_length'},
                'output': {0: 'batch_size', 1: 'sequence_length'}
            }
        )
        
        return "model.onnx"
```

## 📊 Architecture Comparison

| Pattern | Pros | Cons | Use Cases |
|---------|------|------|-----------|
| **Standard Encoder-Decoder** | Simple, proven | Fixed architecture | Translation, summarization |
| **Shared Parameters** | Efficient, smaller model | Less flexible | Resource-constrained scenarios |
| **Hierarchical** | Multi-scale features | Complex training | Vision tasks, long sequences |
| **Parallel** | Faster training | Memory intensive | Large-scale training |
| **Hybrid** | Best of multiple worlds | Complex design | Specialized applications |

## 🎯 Key Takeaways

1. **Connection Patterns**: Different ways to connect components serve different purposes
2. **Scaling Strategies**: Compound scaling often works better than scaling single dimensions
3. **Efficiency Techniques**: Knowledge distillation, quantization, and pruning are essential for production
4. **Training Paradigms**: Pre-training, fine-tuning, and instruction tuning serve different stages
5. **Design Principles**: Modularity, scalability, and efficiency guide good architecture design

## 🚀 Next Steps

- Explore [Advanced Concepts](../05-advanced-concepts/README.md) for cutting-edge techniques
- Build your own custom transformer architecture
- Experiment with different connection patterns and scaling strategies
- Implement production-ready optimization techniques

## 📚 Further Reading

- [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361)
- [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946)
- [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531)
- [Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference](https://arxiv.org/abs/1712.05877)
- [The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks](https://arxiv.org/abs/1803.03635)
