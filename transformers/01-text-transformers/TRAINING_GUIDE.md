# 🚀 Transformer Training Guide: Complete Pipeline

This guide walks you through the complete process of training a transformer model from scratch, including dataset preparation, labeling, model setup, and training execution.

## 📚 Table of Contents
1. [Dataset Preparation](#dataset-preparation)
2. [Data Labeling](#data-labeling)
3. [Model Architecture](#model-architecture)
4. [Training Pipeline](#training-pipeline)
5. [Hyperparameter Tuning](#hyperparameter-tuning)
6. [Monitoring & Evaluation](#monitoring--evaluation)
7. [Practical Examples](#practical-examples)

---

## 🗂️ Dataset Preparation

### **Text Classification Dataset Structure**

```python
# Example dataset structure for sentiment analysis
dataset = {
    "text": [
        "This movie is absolutely fantastic!",
        "I really didn't enjoy this film.",
        "The plot was okay but nothing special.",
        "Amazing performance by all actors!",
        "Terrible waste of time and money."
    ],
    "labels": [1, 0, 2, 1, 0],  # 1=positive, 0=negative, 2=neutral
    "metadata": {
        "length": [8, 7, 9, 7, 8],  # token counts
        "source": ["review", "review", "review", "review", "review"]
    }
}
```

### **Dataset Loading & Preprocessing**

```python
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import pandas as pd

class TextClassificationDataset(Dataset):
    """
    Custom dataset for text classification tasks.
    
    Args:
        texts (List[str]): List of input text strings
        labels (List[int]): List of corresponding labels
        tokenizer: HuggingFace tokenizer
        max_length (int): Maximum sequence length
    """
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # Tokenize and encode text
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# Load and prepare dataset
def prepare_dataset(data_path, tokenizer_name="bert-base-uncased"):
    """
    Prepare dataset from CSV file.
    
    Args:
        data_path (str): Path to CSV file
        tokenizer_name (str): Name of pretrained tokenizer
    
    Returns:
        train_dataset, val_dataset, test_dataset
    """
    # Load data
    df = pd.read_csv(data_path)
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    # Split data
    train_df = df[df['split'] == 'train']
    val_df = df[df['split'] == 'val']
    test_df = df[df['split'] == 'test']
    
    # Create datasets
    train_dataset = TextClassificationDataset(
        train_df['text'].tolist(),
        train_df['label'].tolist(),
        tokenizer
    )
    
    val_dataset = TextClassificationDataset(
        val_df['text'].tolist(),
        val_df['label'].tolist(),
        tokenizer
    )
    
    test_dataset = TextClassificationDataset(
        test_df['text'].tolist(),
        test_df['label'].tolist(),
        tokenizer
    )
    
    return train_dataset, val_dataset, test_dataset
```

### **Data Augmentation Techniques**

```python
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.sentence as nas

def augment_text_data(texts, labels, augmentation_factor=2):
    """
    Apply data augmentation to increase dataset size.
    
    Args:
        texts (List[str]): Original texts
        labels (List[int]): Original labels
        augmentation_factor (int): How many augmented samples per original
    
    Returns:
        augmented_texts, augmented_labels
    """
    # Synonym replacement
    syn_aug = naw.SynonymAug(aug_src='wordnet')
    
    # Back translation (English -> German -> English)
    back_trans_aug = naw.BackTranslationAug(
        from_model_name='facebook/wmt19-en-de',
        to_model_name='facebook/wmt19-de-en'
    )
    
    # Sentence augmentation
    sentence_aug = nas.RandomSentAug()
    
    augmented_texts = texts.copy()
    augmented_labels = labels.copy()
    
    for i, (text, label) in enumerate(zip(texts, labels)):
        # Generate augmented samples
        for _ in range(augmentation_factor):
            # Randomly choose augmentation method
            if torch.rand(1) < 0.5:
                aug_text = syn_aug.augment(text)[0]
            else:
                aug_text = back_trans_aug.augment(text)[0]
            
            augmented_texts.append(aug_text)
            augmented_labels.append(label)
    
    return augmented_texts, augmented_labels
```

---

## 🏷️ Data Labeling

### **Labeling Strategies**

#### **1. Manual Labeling**
```python
# Example labeling interface
def manual_labeling_interface(texts):
    """
    Simple command-line labeling interface.
    """
    labels = []
    label_map = {0: "negative", 1: "positive", 2: "neutral"}
    
    print("Label each text (0=negative, 1=positive, 2=neutral):")
    print("-" * 50)
    
    for i, text in enumerate(texts):
        print(f"\nText {i+1}: {text}")
        while True:
            try:
                label = int(input("Label (0/1/2): "))
                if label in [0, 1, 2]:
                    labels.append(label)
                    break
                else:
                    print("Invalid label. Use 0, 1, or 2.")
            except ValueError:
                print("Please enter a number.")
    
    return labels
```

#### **2. Rule-Based Labeling**
```python
import re
from textblob import TextBlob

def rule_based_labeling(texts):
    """
    Apply rule-based labeling using sentiment analysis.
    """
    labels = []
    
    for text in texts:
        # Use TextBlob for sentiment analysis
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        
        # Define thresholds
        if polarity > 0.1:
            labels.append(1)  # positive
        elif polarity < -0.1:
            labels.append(0)  # negative
        else:
            labels.append(2)  # neutral
    
    return labels
```

#### **3. Active Learning for Labeling**
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

def active_learning_labeling(unlabeled_texts, labeled_texts, labeled_labels, 
                           sample_size=100):
    """
    Use active learning to identify most informative samples for labeling.
    """
    # Train a simple classifier on labeled data
    vectorizer = TfidfVectorizer(max_features=1000)
    X_labeled = vectorizer.fit_transform(labeled_texts)
    
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_labeled, labeled_labels)
    
    # Vectorize unlabeled data
    X_unlabeled = vectorizer.transform(unlabeled_texts)
    
    # Get prediction probabilities
    probs = clf.predict_proba(X_unlabeled)
    
    # Calculate uncertainty (entropy)
    import numpy as np
    entropy = -np.sum(probs * np.log(probs + 1e-10), axis=1)
    
    # Select most uncertain samples
    uncertain_indices = np.argsort(entropy)[-sample_size:]
    
    return [unlabeled_texts[i] for i in uncertain_indices]
```

---

## 🏗️ Model Architecture

### **Custom Transformer Model**

```python
import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for transformer models.
    """
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class TransformerClassifier(nn.Module):
    """
    Transformer-based text classifier.
    
    Args:
        vocab_size (int): Size of vocabulary
        d_model (int): Model dimension
        nhead (int): Number of attention heads
        num_layers (int): Number of transformer layers
        num_classes (int): Number of output classes
        dropout (float): Dropout rate
    """
    def __init__(self, vocab_size, d_model=512, nhead=8, num_layers=6, 
                 num_classes=3, dropout=0.1):
        super().__init__()
        
        self.d_model = d_model
        
        # Token embedding
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """
        Initialize model weights using proper initialization strategies.
        
        Weight initialization is crucial for training stability and convergence.
        Different layers require different initialization strategies:
        
        - Linear layers: Xavier/Glorot uniform initialization for balanced gradients
        - Embedding layers: Small normal distribution to prevent large initial values
        - Bias terms: Zero initialization to start from neutral position
        
        This initialization helps with:
        1. Gradient flow during backpropagation
        2. Preventing vanishing/exploding gradients
        3. Faster convergence during training
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Xavier/Glorot uniform initialization for linear layers
                # This initialization maintains variance across layers
                # Formula: std = sqrt(2 / (fan_in + fan_out))
                nn.init.xavier_uniform_(module.weight)
                
                # Initialize bias to zero for neutral starting point
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
                    
            elif isinstance(module, nn.Embedding):
                # Normal distribution for embedding weights
                # Small standard deviation (0.02) prevents large initial values
                # This helps with training stability
                nn.init.normal_(module.weight, mean=0, std=0.02)
    
    def forward(self, input_ids, attention_mask=None):
        """
        Forward pass through the transformer classifier.
        
        This method processes input token IDs through the complete model pipeline:
        1. Token Embedding: Convert token IDs to dense vectors
        2. Positional Encoding: Add sequence position information
        3. Transformer Processing: Apply self-attention and feed-forward layers
        4. Sequence Pooling: Aggregate sequence information into fixed-size representation
        5. Classification: Generate final class predictions
        
        Mathematical Flow:
        - Input: (batch_size, seq_len) token IDs
        - Embeddings: (batch_size, seq_len, d_model) dense vectors
        - Transformer: (batch_size, seq_len, d_model) contextual representations
        - Pooling: (batch_size, d_model) sequence summary
        - Output: (batch_size, num_classes) class logits
        
        Args:
            input_ids (torch.Tensor): Input token IDs with shape (batch_size, seq_len)
            attention_mask (torch.Tensor, optional): Attention mask with shape (batch_size, seq_len)
                                                   1 = attend to this position, 0 = ignore
        
        Returns:
            logits (torch.Tensor): Raw classification scores with shape (batch_size, num_classes)
                                 Higher values indicate higher probability for that class
        
        Example:
            >>> input_ids = torch.randint(0, 30522, (16, 128))  # 16 samples, 128 tokens each
            >>> attention_mask = torch.ones(16, 128)  # Attend to all positions
            >>> logits = model(input_ids, attention_mask)  # Shape: (16, 3)
        """
        # Step 1: Token Embedding
        # Convert integer token IDs to dense vector representations
        # Shape: (batch_size, seq_len) → (batch_size, seq_len, d_model)
        x = self.embedding(input_ids)
        
        # Scale embeddings by sqrt(d_model) to prevent large values
        # This scaling helps maintain variance across the network
        # Formula from "Attention Is All You Need" paper
        x = x * math.sqrt(self.d_model)
        
        # Step 2: Positional Encoding
        # Add information about token positions in the sequence
        # This allows the model to understand word order and relationships
        # Shape: (batch_size, seq_len, d_model) + (seq_len, d_model) → (batch_size, seq_len, d_model)
        x = self.pos_encoding(x)
        
        # Step 3: Prepare Attention Mask for Transformer
        # The transformer needs to know which positions to attend to
        # PyTorch transformer expects: True = ignore, False = attend
        if attention_mask is not None:
            # Convert our mask (1=attend, 0=ignore) to transformer format (True=ignore, False=attend)
            src_key_padding_mask = (attention_mask == 0)
        else:
            # If no mask provided, attend to all positions
            src_key_padding_mask = None
        
        # Step 4: Transformer Encoder Processing
        # Pass through all transformer layers (self-attention + feed-forward)
        # Each layer processes the sequence and captures contextual relationships
        # Shape: (batch_size, seq_len, d_model) → (batch_size, seq_len, d_model)
        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
        
        # Step 5: Sequence Pooling (Aggregation)
        # Convert variable-length sequence to fixed-size representation
        # We need to aggregate information from all positions into a single vector
        
        if attention_mask is not None:
            # Masked Average Pooling: Only average over attended positions
            # This handles variable-length sequences properly
            
            # Expand mask to match tensor dimensions: (batch_size, seq_len) → (batch_size, seq_len, 1)
            mask_expanded = attention_mask.unsqueeze(-1)
            
            # Apply mask and sum: (batch_size, seq_len, d_model) * (batch_size, seq_len, 1)
            masked_sum = (x * mask_expanded).sum(dim=1)  # Sum over sequence dimension
            
            # Normalize by number of attended positions
            mask_sum = attention_mask.sum(dim=1, keepdim=True)  # Count attended positions per batch
            x = masked_sum / mask_sum  # Shape: (batch_size, d_model)
            
        else:
            # Simple average pooling: average over all positions
            # Shape: (batch_size, seq_len, d_model) → (batch_size, d_model)
            x = x.mean(dim=1)
        
        # Step 6: Classification
        # Pass the sequence representation through the classification head
        # This produces raw logits (unscaled probabilities) for each class
        # Shape: (batch_size, d_model) → (batch_size, num_classes)
        logits = self.classifier(x)
        
        return logits
```

---

## 🚂 Training Pipeline

### **Training Configuration**

```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class TrainingConfig:
    """
    Comprehensive configuration class for transformer training.
    
    This dataclass centralizes all training hyperparameters and settings,
    making it easy to experiment with different configurations and
    ensure reproducibility across training runs.
    
    The configuration is organized into logical groups:
    - Model Architecture: Transformer structure parameters
    - Training Process: Learning rates, epochs, batch sizes
    - Optimization: Regularization and optimization settings
    - Data Handling: Sequence lengths and data splits
    - Monitoring: Logging and evaluation intervals
    
    Usage:
        >>> config = TrainingConfig(
        ...     d_model=768,           # Larger model
        ...     num_layers=12,        # More layers
        ...     learning_rate=5e-5,   # Lower learning rate
        ...     batch_size=16         # Smaller batch size
        ... )
        >>> trainer = TransformerTrainer(model, train_loader, val_loader, config)
    
    Note: All parameters have sensible defaults based on transformer best practices.
    """
    
    # ===== MODEL ARCHITECTURE PARAMETERS =====
    # These control the structure and capacity of the transformer model
    
    vocab_size: int = 30522
    """Vocabulary size - number of unique tokens the model can process.
    Default: 30522 (BERT base vocabulary size)
    Larger vocab = more token types, but requires more memory."""
    
    d_model: int = 512
    """Model dimension - hidden size of embeddings and transformer layers.
    Default: 512 (standard transformer size)
    This is the core dimension that affects model capacity and performance.
    Common values: 256 (small), 512 (medium), 768 (large), 1024 (xlarge)"""
    
    nhead: int = 8
    """Number of attention heads in multi-head attention.
    Default: 8 (standard for d_model=512)
    Must divide d_model evenly: d_model % nhead == 0
    More heads = more parallel attention patterns, but requires more computation."""
    
    num_layers: int = 6
    """Number of transformer encoder layers.
    Default: 6 (standard transformer size)
    More layers = deeper model = more complex patterns, but risk of overfitting.
    Common values: 4 (shallow), 6 (standard), 12 (deep), 24 (very deep)"""
    
    num_classes: int = 3
    """Number of output classes for classification.
    Default: 3 (e.g., positive, negative, neutral for sentiment)
    Must match your dataset's label cardinality."""
    
    # ===== TRAINING PROCESS PARAMETERS =====
    # These control how the model learns during training
    
    batch_size: int = 32
    """Number of samples processed together in each training step.
    Default: 32 (good balance of memory usage and training stability)
    Larger batches = more stable gradients, but require more memory.
    Smaller batches = more noisy gradients, but use less memory.
    Rule of thumb: Use largest batch size that fits in GPU memory."""
    
    learning_rate: float = 1e-4
    """Initial learning rate for the optimizer.
    Default: 1e-4 (0.0001) - standard for transformer fine-tuning
    Learning rate is crucial for training success:
    - Too high: training may diverge or oscillate
    - Too low: training may be very slow
    - Just right: stable convergence to good performance
    Common values: 1e-5 (very conservative), 1e-4 (standard), 5e-4 (aggressive)"""
    
    num_epochs: int = 10
    """Total number of training epochs.
    Default: 10 (reasonable starting point)
    An epoch = one complete pass through all training data.
    More epochs = more training time, but may lead to overfitting.
    Use early stopping or validation performance to determine optimal epochs."""
    
    warmup_steps: int = 1000
    """Number of steps to gradually increase learning rate.
    Default: 1000 (standard for transformer training)
    Warmup is crucial for transformer stability:
    - Start with very low LR (close to 0)
    - Gradually increase to target LR over warmup_steps
    - Then decay LR over remaining training steps
    Rule of thumb: warmup_steps = 10% of total training steps"""
    
    max_grad_norm: float = 1.0
    """Maximum gradient norm for gradient clipping.
    Default: 1.0 (standard for transformer training)
    Gradient clipping prevents exploding gradients:
    - If ||gradient|| > max_grad_norm, scale gradient down
    - This stabilizes training, especially for deep models
    Common values: 0.5 (conservative), 1.0 (standard), 2.0 (aggressive)"""
    
    # ===== OPTIMIZATION PARAMETERS =====
    # These control regularization and optimization behavior
    
    weight_decay: float = 0.01
    """L2 regularization strength for weight decay.
    Default: 0.01 (1%) - standard for transformer training
    Weight decay helps prevent overfitting by penalizing large weights.
    Higher values = stronger regularization, but may hurt performance.
    Common values: 0.001 (weak), 0.01 (standard), 0.1 (strong)"""
    
    dropout: float = 0.1
    """Dropout rate for regularization.
    Default: 0.1 (10%) - standard for transformer training
    Dropout randomly zeros some neurons during training:
    - Prevents overfitting by reducing co-adaptation
    - Forces model to be robust to missing information
    - Disabled during evaluation (model.eval())
    Common values: 0.0 (no dropout), 0.1 (standard), 0.2 (strong)"""
    
    # ===== DATA HANDLING PARAMETERS =====
    # These control how data is processed and split
    
    max_length: int = 512
    """Maximum sequence length for input texts.
    Default: 512 (standard transformer length)
    Longer sequences = more context, but require more memory and computation.
    Shorter sequences = faster training, but may lose important context.
    Must match your tokenizer's maximum length."""
    
    train_split: float = 0.8
    """Fraction of data to use for training.
    Default: 0.8 (80% of data for training)
    Remaining 20% is split between validation and test.
    Typical splits: 70/15/15, 80/10/10, 90/5/5 (train/val/test)"""
    
    val_split: float = 0.1
    """Fraction of data to use for validation.
    Default: 0.1 (10% of data for validation)
    Validation data is used to monitor training progress and prevent overfitting.
    Should be large enough to give reliable performance estimates."""
    
    # ===== MONITORING PARAMETERS =====
    # These control logging and evaluation frequency
    
    log_interval: int = 100
    """How often to log training metrics (in batches).
    Default: 100 (log every 100 batches)
    More frequent logging = better monitoring, but slower training.
    Less frequent logging = faster training, but less visibility."""
    
    eval_interval: int = 500
    """How often to run validation (in batches).
    Default: 500 (validate every 500 batches)
    More frequent validation = better overfitting detection, but slower training.
    Less frequent validation = faster training, but may miss overfitting."""
    
    save_interval: int = 1000
    """How often to save model checkpoints (in batches).
    Default: 1000 (save every 1000 batches)
    More frequent saving = better recovery from failures, but uses more disk space.
    Less frequent saving = less disk usage, but may lose progress on failures."""
```

### **Training Loop**

```python
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import get_linear_schedule_with_warmup
from tqdm import tqdm
import wandb

class TransformerTrainer:
    """
    Comprehensive trainer class for transformer models with advanced training features.
    
    This trainer handles the complete training pipeline including:
    - Model optimization with AdamW optimizer
    - Learning rate scheduling with warmup
    - Training and validation loops
    - Progress tracking and logging
    - Model checkpointing
    - Performance monitoring
    
    The trainer implements best practices for transformer training:
    1. Gradient clipping to prevent exploding gradients
    2. Learning rate warmup for stable early training
    3. Proper device management (GPU/CPU)
    4. Comprehensive logging and monitoring
    5. Automatic model saving for best performance
    
    Args:
        model: The transformer model to train
        train_dataloader: DataLoader for training data
        val_dataloader: DataLoader for validation data
        config: Training configuration object
    
    Example:
        >>> trainer = TransformerTrainer(model, train_loader, val_loader, config)
        >>> trainer.train()  # Start training
    """
    def __init__(self, model, train_dataloader, val_dataloader, config):
        # Store references to model and data
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.config = config
        
        # Step 1: Setup Optimizer (AdamW)
        # AdamW is an improved version of Adam with better weight decay
        # It's the standard choice for transformer training
        self.optimizer = AdamW(
            model.parameters(),           # All trainable parameters
            lr=config.learning_rate,     # Initial learning rate
            weight_decay=config.weight_decay  # L2 regularization strength
        )
        
        # Step 2: Setup Learning Rate Scheduler
        # Linear warmup followed by linear decay
        # This is crucial for transformer training stability
        
        # Calculate total training steps for scheduler
        total_steps = len(train_dataloader) * config.num_epochs
        
        # Create scheduler with warmup
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,                    # The optimizer to schedule
            num_warmup_steps=config.warmup_steps,  # Steps to gradually increase LR
            num_training_steps=total_steps         # Total steps for LR decay
        )
        
        # Step 3: Setup Loss Function
        # CrossEntropyLoss is standard for classification tasks
        # It combines softmax + negative log likelihood
        # No need to apply softmax manually - the loss function handles it
        self.criterion = nn.CrossEntropyLoss()
        
        # Step 4: Device Management
        # Automatically detect and use GPU if available
        # This significantly speeds up training
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Move model to the appropriate device
        self.model.to(self.device)
        
        # Step 5: Initialize Experiment Tracking
        # Weights & Biases (wandb) for experiment tracking
        # This helps monitor training progress and compare experiments
        wandb.init(project="transformer-training", config=vars(config))
    
    def train_epoch(self, epoch):
        """
        Train the model for one complete epoch.
        
        An epoch consists of processing all training samples once. This method:
        1. Sets the model to training mode (enables dropout, batch norm updates)
        2. Iterates through all training batches
        3. Performs forward pass, loss calculation, and backward pass
        4. Updates model weights using the optimizer
        5. Tracks training metrics and logs progress
        
        Training Process for Each Batch:
        - Forward Pass: Input → Model → Predictions
        - Loss Calculation: Compare predictions with true labels
        - Backward Pass: Compute gradients for all parameters
        - Gradient Clipping: Prevent exploding gradients
        - Weight Update: Apply gradients to model parameters
        - Learning Rate Update: Adjust learning rate according to schedule
        
        Args:
            epoch (int): Current epoch number (0-indexed)
        
        Returns:
            tuple: (average_loss, accuracy_percentage) for the epoch
        
        Example:
            >>> avg_loss, accuracy = trainer.train_epoch(0)
            >>> print(f"Epoch 0: Loss={avg_loss:.4f}, Accuracy={accuracy:.2f}%")
        """
        # Step 1: Set Model to Training Mode
        # This enables:
        # - Dropout layers (randomly zero some neurons)
        # - Batch normalization updates (track running statistics)
        # - Gradient computation (required for backpropagation)
        self.model.train()
        
        # Step 2: Initialize Epoch Statistics
        total_loss = 0          # Accumulate loss across all batches
        correct = 0             # Count correct predictions
        total = 0               # Count total predictions
        
        # Step 3: Create Progress Bar
        # Shows real-time training progress with current metrics
        progress_bar = tqdm(self.train_dataloader, desc=f'Epoch {epoch}')
        
        # Step 4: Iterate Through Training Batches
        for batch_idx, batch in enumerate(progress_bar):
            # Step 4a: Move Data to Device
            # Transfer tensors to GPU/CPU for processing
            # This is crucial for performance - GPU operations are much faster
            input_ids = batch['input_ids'].to(self.device)           # Token IDs
            attention_mask = batch['attention_mask'].to(self.device) # Attention mask
            labels = batch['labels'].to(self.device)                 # True labels
            
            # Step 4b: Forward Pass
            # Clear previous gradients before computing new ones
            # This prevents gradient accumulation across batches
            self.optimizer.zero_grad()
            
            # Pass data through the model to get predictions
            # Shape: (batch_size, num_classes) - raw logits for each class
            logits = self.model(input_ids, attention_mask)
            
            # Calculate loss between predictions and true labels
            # CrossEntropyLoss automatically applies softmax and computes NLL
            loss = self.criterion(logits, labels)
            
            # Step 4c: Backward Pass
            # Compute gradients for all model parameters
            # This is the core of backpropagation
            loss.backward()
            
            # Step 4d: Gradient Clipping
            # Prevent exploding gradients by limiting gradient magnitude
            # This is especially important for transformers (RNNs too)
            # Formula: if ||g|| > threshold, then g = g * threshold / ||g||
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),     # All model parameters
                self.config.max_grad_norm    # Maximum gradient norm (e.g., 1.0)
            )
            
            # Step 4e: Update Model Weights
            # Apply computed gradients to update model parameters
            # This is where the model actually learns
            self.optimizer.step()
            
            # Step 4f: Update Learning Rate
            # Adjust learning rate according to the schedule
            # Warmup: gradually increase LR, then decay
            self.scheduler.step()
            
            # Step 4g: Update Training Statistics
            # Accumulate loss for epoch average
            total_loss += loss.item()
            
            # Calculate accuracy for this batch
            # Get predicted class (highest logit value)
            _, predicted = torch.max(logits, 1)  # dim=1: max over classes
            total += labels.size(0)              # Add batch size to total
            correct += (predicted == labels).sum().item()  # Count correct predictions
            
            # Step 4h: Update Progress Bar
            # Show real-time metrics: current loss and cumulative accuracy
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',                    # Current batch loss
                'Acc': f'{100 * correct / total:.2f}%'           # Cumulative accuracy
            })
            
            # Step 4i: Log Metrics to WandB
            # Log training metrics at specified intervals
            # This enables experiment tracking and visualization
            if batch_idx % self.config.log_interval == 0:
                wandb.log({
                    'train_loss': loss.item(),                    # Current batch loss
                    'train_accuracy': 100 * correct / total,      # Cumulative accuracy
                    'learning_rate': self.scheduler.get_last_lr()[0],  # Current LR
                    'epoch': epoch,                               # Current epoch
                    'step': epoch * len(self.train_dataloader) + batch_idx  # Global step
                })
        
        # Step 5: Calculate Epoch Results
        # Return average loss and final accuracy for the epoch
        avg_loss = total_loss / len(self.train_dataloader)
        accuracy = 100 * correct / total
        
        return avg_loss, accuracy
    
    def validate(self, epoch):
        """
        Validate the model on the validation dataset.
        
        Validation is crucial for:
        1. Monitoring model performance on unseen data
        2. Detecting overfitting (training accuracy >> validation accuracy)
        3. Selecting the best model checkpoint
        4. Early stopping decisions
        
        Key Differences from Training:
        - Model is in evaluation mode (no dropout, no gradient computation)
        - No weight updates (gradients are not computed)
        - No learning rate updates
        - Metrics are logged but not used for optimization
        
        Args:
            epoch (int): Current epoch number for logging purposes
        
        Returns:
            tuple: (average_validation_loss, validation_accuracy_percentage)
        
        Example:
            >>> val_loss, val_acc = trainer.validate(0)
            >>> print(f"Validation: Loss={val_loss:.4f}, Accuracy={val_acc:.2f}%")
        """
        # Step 1: Set Model to Evaluation Mode
        # This disables:
        # - Dropout layers (use full network capacity)
        # - Batch normalization updates (use running statistics)
        # - Gradient computation (save memory and computation)
        self.model.eval()
        
        # Step 2: Initialize Validation Statistics
        total_loss = 0          # Accumulate validation loss
        correct = 0             # Count correct predictions
        total = 0               # Count total predictions
        
        # Step 3: Disable Gradient Computation
        # This saves memory and computation since we don't need gradients for validation
        # torch.no_grad() is a context manager that disables gradient tracking
        with torch.no_grad():
            # Step 4: Iterate Through Validation Batches
            for batch in self.val_dataloader:
                # Step 4a: Move Data to Device
                # Same as training - transfer tensors to GPU/CPU
                input_ids = batch['input_ids'].to(self.device)           # Token IDs
                attention_mask = batch['attention_mask'].to(self.device) # Attention mask
                labels = batch['labels'].to(self.device)                 # True labels
                
                # Step 4b: Forward Pass (No Backward Pass)
                # Pass data through the model to get predictions
                # No need to compute gradients - just get predictions
                logits = self.model(input_ids, attention_mask)
                
                # Calculate loss for monitoring (not for optimization)
                # This helps track validation performance over time
                loss = self.criterion(logits, labels)
                
                # Step 4c: Update Validation Statistics
                # Accumulate loss for epoch average
                total_loss += loss.item()
                
                # Calculate accuracy for this batch
                # Get predicted class (highest logit value)
                _, predicted = torch.max(logits, 1)  # dim=1: max over classes
                total += labels.size(0)              # Add batch size to total
                correct += (predicted == labels).sum().item()  # Count correct predictions
        
        # Step 5: Calculate Final Validation Metrics
        # Compute average loss and accuracy across all validation batches
        avg_loss = total_loss / len(self.val_dataloader)
        accuracy = 100 * correct / total
        
        # Step 6: Log Validation Metrics to WandB
        # Track validation performance over time
        # This helps identify overfitting and model convergence
        wandb.log({
            'val_loss': avg_loss,        # Average validation loss for the epoch
            'val_accuracy': accuracy,    # Validation accuracy for the epoch
            'epoch': epoch               # Current epoch number
        })
        
        return avg_loss, accuracy
    
    def train(self):
        """
        Execute the complete training loop for all epochs.
        
        This is the main training method that orchestrates the entire training process:
        1. Iterates through all epochs
        2. Calls train_epoch() for each epoch
        3. Calls validate() after each training epoch
        4. Saves the best model based on validation performance
        5. Tracks and reports training progress
        
        Training Loop Structure:
        For each epoch:
        - Train the model on all training data
        - Validate the model on validation data
        - Compare validation performance with best so far
        - Save model if it's the best performer
        - Log metrics and progress
        
        Model Checkpointing:
        - Saves the best model based on validation accuracy
        - Includes model weights, optimizer state, and configuration
        - Enables training resumption and model deployment
        
        Args:
            None (uses instance variables set in __init__)
        
        Returns:
            None (saves best model to disk)
        
        Example:
            >>> trainer = TransformerTrainer(model, train_loader, val_loader, config)
            >>> trainer.train()  # Start complete training process
        """
        # Step 1: Initialize Best Performance Tracking
        # Track the best validation accuracy seen so far
        # This is used to determine when to save model checkpoints
        best_val_acc = 0
        
        # Step 2: Main Training Loop
        # Iterate through all epochs specified in the configuration
        for epoch in range(self.config.num_epochs):
            print(f"\n{'='*60}")
            print(f"Starting Epoch {epoch + 1}/{self.config.num_epochs}")
            print(f"{'='*60}")
            
            # Step 2a: Training Phase
            # Train the model on all training data for this epoch
            # This updates model weights based on training loss
            print(f"Training phase...")
            train_loss, train_acc = self.train_epoch(epoch)
            
            # Step 2b: Validation Phase
            # Evaluate the model on validation data
            # This helps monitor performance on unseen data
            print(f"Validation phase...")
            val_loss, val_acc = self.validate(epoch)
            
            # Step 2c: Epoch Summary
            # Print comprehensive results for this epoch
            print(f'\n📊 Epoch {epoch + 1} Results:')
            print(f'  🚂 Training:   Loss: {train_loss:.4f}, Accuracy: {train_acc:.2f}%')
            print(f'  ✅ Validation: Loss: {val_loss:.4f}, Accuracy: {val_acc:.2f}%')
            
            # Step 2d: Model Checkpointing
            # Save the model if it achieves the best validation accuracy so far
            # This implements a simple form of model selection
            if val_acc > best_val_acc:
                # Update best accuracy record
                best_val_acc = val_acc
                
                # Create comprehensive checkpoint
                checkpoint = {
                    'epoch': epoch,                                    # Current epoch
                    'model_state_dict': self.model.state_dict(),       # Model weights
                    'optimizer_state_dict': self.optimizer.state_dict(), # Optimizer state
                    'val_accuracy': val_acc,                           # Best validation accuracy
                    'config': self.config,                             # Training configuration
                    'training_stats': {                                # Additional metadata
                        'train_loss': train_loss,
                        'train_accuracy': train_acc,
                        'val_loss': val_loss,
                        'best_val_accuracy': best_val_acc
                    }
                }
                
                # Save checkpoint to disk
                torch.save(checkpoint, 'best_model.pth')
                
                print(f'  🎯 New best model saved!')
                print(f'     Validation accuracy: {val_acc:.2f}%')
                print(f'     Previous best: {best_val_acc:.2f}%')
                print(f'     Checkpoint: best_model.pth')
                
                # Log to wandb for experiment tracking
                wandb.log({
                    'best_val_accuracy': best_val_acc,
                    'epoch_best_achieved': epoch
                })
            else:
                print(f'  📈 No improvement (Best: {best_val_acc:.2f}%)')
            
            # Step 2e: Progress Summary
            # Show overall training progress
            progress = (epoch + 1) / self.config.num_epochs * 100
            print(f'  📈 Overall Progress: {progress:.1f}% ({epoch + 1}/{self.config.num_epochs})')
            print()
        
        # Step 3: Training Completion
        print(f"{'='*60}")
        print(f"🎉 Training Completed Successfully!")
        print(f"{'='*60}")
        print(f"📊 Final Results:")
        print(f"  🏆 Best validation accuracy: {best_val_acc:.2f}%")
        print(f"  💾 Best model saved as: best_model.pth")
        print(f"  📁 Checkpoint includes: model weights, optimizer state, config")
        print(f"  🔄 To resume training: load checkpoint and continue")
        
        # Step 4: Cleanup
        # Finalize wandb experiment tracking
        wandb.finish()
        
        print(f"\n✅ Training pipeline completed. Model ready for deployment!")
```

---

## 🎛️ Hyperparameter Tuning

### **Grid Search Example**

```python
from itertools import product

def hyperparameter_grid_search():
    """Perform grid search over hyperparameters."""
    
    # Define hyperparameter ranges
    param_grid = {
        'd_model': [256, 512, 768],
        'nhead': [4, 8, 12],
        'num_layers': [4, 6, 8],
        'learning_rate': [1e-5, 1e-4, 5e-4],
        'batch_size': [16, 32, 64]
    }
    
    # Generate all combinations
    param_combinations = [dict(zip(param_grid.keys(), v)) 
                         for v in product(*param_grid.values())]
    
    best_config = None
    best_score = 0
    
    for i, config in enumerate(param_combinations):
        print(f"\n--- Testing Configuration {i+1}/{len(param_combinations)} ---")
        print(f"Config: {config}")
        
        # Create model with current config
        model = TransformerClassifier(
            vocab_size=30522,
            d_model=config['d_model'],
            nhead=config['nhead'],
            num_layers=config['num_layers']
        )
        
        # Train and evaluate
        score = train_and_evaluate(model, config)
        
        if score > best_score:
            best_score = score
            best_config = config
            print(f"New best configuration found! Score: {score:.4f}")
    
    print(f"\n=== Best Configuration ===")
    print(f"Config: {best_config}")
    print(f"Score: {best_score:.4f}")
    
    return best_config
```

### **Learning Rate Finder**

```python
class LRFinder:
    """Learning rate finder for optimal learning rate selection."""
    
    def __init__(self, model, optimizer, criterion, device):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        
        # Save original learning rate
        self.original_lr = optimizer.param_groups[0]['lr']
        
        # Initialize
        self.best_loss = None
        self.smoothed_loss = None
        self.lr_mult = 1.0
        
    def range_test(self, train_loader, start_lr=1e-7, end_lr=10, num_iter=100):
        """Find optimal learning rate range."""
        
        # Reset learning rate
        self.optimizer.param_groups[0]['lr'] = start_lr
        
        # Calculate multiplier
        self.lr_mult = (end_lr / start_lr) ** (1.0 / num_iter)
        
        # Training loop
        self.model.train()
        losses = []
        log_lrs = []
        
        for iteration, batch in enumerate(train_loader):
            # Move to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            logits = self.model(input_ids, attention_mask)
            loss = self.criterion(logits, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Update learning rate
            for param_group in self.optimizer.param_groups:
                param_group['lr'] *= self.lr_mult
            
            # Store results
            losses.append(loss.item())
            log_lrs.append(math.log10(self.optimizer.param_groups[0]['lr']))
            
            if iteration >= num_iter:
                break
        
        # Reset learning rate
        self.optimizer.param_groups[0]['lr'] = self.original_lr
        
        return log_lrs, losses
```

---

## 📊 Monitoring & Evaluation

### **Training Metrics**

```python
import matplotlib.pyplot as plt
import seaborn as sns

class TrainingMonitor:
    """Monitor and visualize training progress."""
    
    def __init__(self):
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.learning_rates = []
    
    def update(self, train_loss, val_loss, train_acc, val_acc, lr):
        """Update metrics."""
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.train_accuracies.append(train_acc)
        self.val_accuracies.append(val_acc)
        self.learning_rates.append(lr)
    
    def plot_training_curves(self):
        """Plot training and validation curves."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss curves
        ax1.plot(self.train_losses, label='Train Loss', color='blue')
        ax1.plot(self.val_losses, label='Validation Loss', color='red')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy curves
        ax2.plot(self.train_accuracies, label='Train Accuracy', color='blue')
        ax2.plot(self.val_accuracies, label='Validation Accuracy', color='red')
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True)
        
        # Learning rate schedule
        ax3.plot(self.learning_rates, color='green')
        ax3.set_title('Learning Rate Schedule')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Learning Rate')
        ax3.grid(True)
        
        # Loss difference (overfitting detection)
        loss_diff = [abs(t - v) for t, v in zip(self.train_losses, self.val_losses)]
        ax4.plot(loss_diff, color='orange')
        ax4.set_title('Train-Val Loss Difference (Overfitting Indicator)')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Loss Difference')
        ax4.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def plot_confusion_matrix(self, y_true, y_pred, class_names):
        """Plot confusion matrix."""
        from sklearn.metrics import confusion_matrix
        
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.show()
```

---

## 🎯 Practical Examples

### **Complete Training Script**

```python
def main():
    """
    Complete training pipeline for transformer text classification.
    
    This function demonstrates the end-to-end process of training a transformer model:
    1. Configuration Setup: Define all training hyperparameters
    2. Data Preparation: Load and prepare datasets
    3. Model Initialization: Create and configure the transformer model
    4. Training Execution: Run the complete training loop
    5. Model Evaluation: Test the trained model on unseen data
    6. Results Summary: Report final performance metrics
    
    The pipeline follows transformer training best practices:
    - Proper data splitting (train/val/test)
    - Learning rate warmup and scheduling
    - Gradient clipping for stability
    - Model checkpointing for best performance
    - Comprehensive logging and monitoring
    
    Usage:
        >>> python training_script.py
        # This will run the complete training pipeline
    
    Output:
        - Trained model saved as 'best_model.pth'
        - Training logs and metrics
        - Final test performance results
    """
    
    print("🚀 Starting Transformer Training Pipeline")
    print("=" * 60)
    
    # ===== STEP 1: CONFIGURATION SETUP =====
    # Define all training hyperparameters in one place
    # This makes it easy to experiment with different settings
    
    print("📋 Step 1: Setting up training configuration...")
    config = TrainingConfig(
        # Model Architecture
        vocab_size=30522,      # BERT vocabulary size
        d_model=512,           # Hidden dimension (standard transformer size)
        nhead=8,               # Number of attention heads
        num_layers=6,          # Number of transformer layers
        num_classes=3,         # 3 classes: positive, negative, neutral
        
        # Training Process
        batch_size=32,         # Samples per batch (balance memory vs stability)
        learning_rate=1e-4,    # Initial learning rate (standard for fine-tuning)
        num_epochs=10,         # Total training epochs
        
        # Optimization
        warmup_steps=1000,     # Learning rate warmup steps
        max_grad_norm=1.0,     # Gradient clipping threshold
        weight_decay=0.01,     # L2 regularization strength
        dropout=0.1            # Dropout rate for regularization
    )
    
    print(f"   ✅ Configuration created:")
    print(f"      Model: {config.d_model}d, {config.nhead} heads, {config.num_layers} layers")
    print(f"      Training: {config.batch_size} batch size, {config.learning_rate} LR, {config.num_epochs} epochs")
    print(f"      Optimization: {config.warmup_steps} warmup, {config.max_grad_norm} grad clip")
    
    # ===== STEP 2: DATA PREPARATION =====
    # Load and prepare the dataset for training
    # This includes data loading, preprocessing, and splitting
    
    print("\n📁 Step 2: Loading and preparing dataset...")
    try:
        train_dataset, val_dataset, test_dataset = prepare_dataset(
            "sentiment_dataset.csv",    # Path to your dataset
            "bert-base-uncased"         # Tokenizer to use
        )
        print(f"   ✅ Dataset loaded successfully:")
        print(f"      Training samples: {len(train_dataset):,}")
        print(f"      Validation samples: {len(val_dataset):,}")
        print(f"      Test samples: {len(test_dataset):,}")
    except FileNotFoundError:
        print("   ❌ Error: Dataset file not found!")
        print("      Please ensure 'sentiment_dataset.csv' exists in the current directory")
        return
    except Exception as e:
        print(f"   ❌ Error loading dataset: {e}")
        return
    
    # ===== STEP 3: CREATE DATA LOADERS =====
    # DataLoaders handle batching, shuffling, and parallel processing
    # They're essential for efficient training
    
    print("\n🔄 Step 3: Creating data loaders...")
    
    # Training DataLoader: shuffle=True for better training dynamics
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.batch_size,    # Use batch size from config
        shuffle=True,                     # Shuffle data for each epoch
        num_workers=4,                    # Parallel data loading (adjust based on CPU cores)
        pin_memory=True                   # Faster data transfer to GPU
    )
    
    # Validation DataLoader: shuffle=False for consistent evaluation
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.batch_size,    # Same batch size for consistency
        shuffle=False,                    # No shuffling needed for validation
        num_workers=4,                    # Parallel loading
        pin_memory=True                   # Faster GPU transfer
    )
    
    # Test DataLoader: shuffle=False for consistent evaluation
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config.batch_size,    # Same batch size
        shuffle=False,                    # No shuffling for testing
        num_workers=4,                    # Parallel loading
        pin_memory=True                   # Faster GPU transfer
    )
    
    print(f"   ✅ Data loaders created:")
    print(f"      Training batches: {len(train_loader)}")
    print(f"      Validation batches: {len(val_loader)}")
    print(f"      Test batches: {len(test_loader)}")
    
    # ===== STEP 4: MODEL INITIALIZATION =====
    # Create the transformer model with specified architecture
    # This is where we define the model structure
    
    print("\n🏗️  Step 4: Initializing transformer model...")
    try:
        model = TransformerClassifier(
            # Model Architecture
            vocab_size=config.vocab_size,      # Vocabulary size
            d_model=config.d_model,            # Hidden dimension
            nhead=config.nhead,                # Number of attention heads
            num_layers=config.num_layers,      # Number of transformer layers
            num_classes=config.num_classes,    # Output classes
            
            # Regularization
            dropout=config.dropout             # Dropout rate
        )
        
        # Calculate model parameters for information
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"   ✅ Model initialized successfully:")
        print(f"      Total parameters: {total_params:,}")
        print(f"      Trainable parameters: {trainable_params:,}")
        print(f"      Model size: {total_params * 4 / 1024 / 1024:.1f} MB (float32)")
        
    except Exception as e:
        print(f"   ❌ Error initializing model: {e}")
        return
    
    # ===== STEP 5: TRAINER INITIALIZATION =====
    # Create the trainer that handles the training loop
    # This includes optimizer, scheduler, and training logic
    
    print("\n🎯 Step 5: Initializing trainer...")
    try:
        trainer = TransformerTrainer(model, train_loader, val_loader, config)
        print("   ✅ Trainer initialized successfully")
        print("   ✅ Optimizer: AdamW with weight decay")
        print("   ✅ Scheduler: Linear warmup + decay")
        print("   ✅ Loss function: CrossEntropyLoss")
        print("   ✅ Device: GPU" if torch.cuda.is_available() else "   ✅ Device: CPU")
    except Exception as e:
        print(f"   ❌ Error initializing trainer: {e}")
        return
    
    # ===== STEP 6: TRAINING EXECUTION =====
    # Run the complete training loop
    # This is where the model actually learns from the data
    
    print("\n🚂 Step 6: Starting training...")
    print("   This may take a while depending on your dataset size and hardware")
    print("   Monitor progress with the progress bars and logged metrics")
    
    try:
        trainer.train()
        print("   ✅ Training completed successfully!")
    except KeyboardInterrupt:
        print("\n   ⚠️  Training interrupted by user")
        print("   💾 Model checkpoint saved (if available)")
        return
    except Exception as e:
        print(f"   ❌ Error during training: {e}")
        return
    
    # ===== STEP 7: MODEL EVALUATION =====
    # Load the best model and evaluate on test data
    # This gives us the final performance metrics
    
    print("\n🧪 Step 7: Evaluating best model...")
    try:
        # Load the best model checkpoint
        print("   Loading best model checkpoint...")
        checkpoint = torch.load('best_model.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Evaluate on test set
        print("   Running evaluation on test set...")
        test_loss, test_acc = trainer.evaluate(test_loader)
        
        print(f"   ✅ Final Test Results:")
        print(f"      Loss: {test_loss:.4f}")
        print(f"      Accuracy: {test_acc:.2f}%")
        print(f"      Best validation accuracy: {checkpoint['val_accuracy']:.2f}%")
        
    except FileNotFoundError:
        print("   ❌ Error: Best model checkpoint not found!")
        print("      Training may have failed or no model was saved")
        return
    except Exception as e:
        print(f"   ❌ Error during evaluation: {e}")
        return
    
    # ===== STEP 8: FINAL SUMMARY =====
    # Provide a comprehensive summary of the training run
    
    print("\n" + "=" * 60)
    print("🎉 TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print("📊 Final Results Summary:")
    print(f"   🏆 Test Accuracy: {test_acc:.2f}%")
    print(f"   📈 Best Validation Accuracy: {checkpoint['val_accuracy']:.2f}%")
    print(f"   💾 Model saved as: best_model.pth")
    print(f"   🔧 Configuration used: {config}")
    print(f"   📁 Checkpoint includes: model weights, optimizer state, config")
    
    print("\n🚀 Next Steps:")
    print("   1. Use the trained model for inference")
    print("   2. Deploy the model in production")
    print("   3. Experiment with different hyperparameters")
    print("   4. Try different model architectures")
    
    print("\n✅ Training pipeline completed. Model ready for deployment!")

if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()
```

### **Quick Start Example**

```python
# Quick start for sentiment analysis
def quick_sentiment_training():
    """Quick start example for sentiment analysis."""
    
    # Sample data
    texts = [
        "I love this movie!",
        "This is terrible.",
        "It's okay, nothing special.",
        "Amazing performance!",
        "Waste of time."
    ]
    
    labels = [1, 0, 2, 1, 0]  # 1=positive, 0=negative, 2=neutral
    
    # Simple tokenization
    from sklearn.feature_extraction.text import TfidfVectorizer
    vectorizer = TfidfVectorizer(max_features=1000)
    X = vectorizer.fit_transform(texts)
    
    # Simple classifier
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression()
    clf.fit(X, labels)
    
    # Test
    test_texts = ["This is great!", "I don't like it"]
    test_X = vectorizer.transform(test_texts)
    predictions = clf.predict(test_X)
    
    print("Predictions:", predictions)
    return clf, vectorizer
```

---

## 🔧 Troubleshooting Common Issues

### **1. Overfitting**
```python
# Solutions:
# - Increase dropout
# - Reduce model capacity
# - Early stopping
# - Data augmentation
# - Regularization
```

### **2. Underfitting**
```python
# Solutions:
# - Increase model capacity
# - Reduce regularization
# - Train longer
# - Better feature engineering
```

### **3. Memory Issues**
```python
# Solutions:
# - Reduce batch size
# - Gradient accumulation
# - Mixed precision training
# - Model parallelism
```

### **4. Slow Training**
```python
# Solutions:
# - Use GPU
# - Optimize data loading
# - Mixed precision
# - Distributed training
```

---

## 📚 Additional Resources

- **Papers**: "Attention Is All You Need", "BERT: Pre-training of Deep Bidirectional Transformers"
- **Libraries**: HuggingFace Transformers, PyTorch Lightning, TensorFlow
- **Datasets**: GLUE, SuperGLUE, SQuAD, IMDB
- **Tutorials**: PyTorch tutorials, HuggingFace courses

---

## 🎉 Conclusion

This guide provides a comprehensive framework for training transformer models. Key takeaways:

1. **Data Quality**: Clean, well-labeled data is crucial
2. **Model Architecture**: Start simple, scale up as needed
3. **Hyperparameter Tuning**: Use systematic approaches
4. **Monitoring**: Track metrics to detect issues early
5. **Iteration**: Training is iterative - experiment and improve

Remember: **"Garbage in, garbage out"** - the quality of your data directly impacts model performance! 🚀
