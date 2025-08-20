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
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.02)
    
    def forward(self, input_ids, attention_mask=None):
        """
        Forward pass through the model.
        
        Args:
            input_ids (torch.Tensor): Input token IDs (B, N)
            attention_mask (torch.Tensor): Attention mask (B, N)
        
        Returns:
            logits (torch.Tensor): Classification logits (B, num_classes)
        """
        # Embedding + positional encoding
        x = self.embedding(input_ids) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        
        # Create attention mask for transformer
        if attention_mask is not None:
            # Convert to transformer format: True = attend, False = ignore
            src_key_padding_mask = (attention_mask == 0)
        else:
            src_key_padding_mask = None
        
        # Pass through transformer
        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
        
        # Global average pooling over sequence dimension
        if attention_mask is not None:
            # Masked average pooling
            x = (x * attention_mask.unsqueeze(-1)).sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
        else:
            x = x.mean(dim=1)
        
        # Classification
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
    """Configuration for training."""
    # Model parameters
    vocab_size: int = 30522  # BERT vocab size
    d_model: int = 512
    nhead: int = 8
    num_layers: int = 6
    num_classes: int = 3
    
    # Training parameters
    batch_size: int = 32
    learning_rate: float = 1e-4
    num_epochs: int = 10
    warmup_steps: int = 1000
    max_grad_norm: float = 1.0
    
    # Optimization
    weight_decay: float = 0.01
    dropout: float = 0.1
    
    # Data parameters
    max_length: int = 512
    train_split: float = 0.8
    val_split: float = 0.1
    
    # Logging
    log_interval: int = 100
    eval_interval: int = 500
    save_interval: int = 1000
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
    Trainer class for transformer models.
    """
    def __init__(self, model, train_dataloader, val_dataloader, config):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.config = config
        
        # Setup optimizer and scheduler
        self.optimizer = AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        total_steps = len(train_dataloader) * config.num_epochs
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=config.warmup_steps,
            num_training_steps=total_steps
        )
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Initialize wandb
        wandb.init(project="transformer-training", config=vars(config))
    
    def train_epoch(self, epoch):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        progress_bar = tqdm(self.train_dataloader, desc=f'Epoch {epoch}')
        
        for batch_idx, batch in enumerate(progress_bar):
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
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            
            # Update weights
            self.optimizer.step()
            self.scheduler.step()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100 * correct / total:.2f}%'
            })
            
            # Log to wandb
            if batch_idx % self.config.log_interval == 0:
                wandb.log({
                    'train_loss': loss.item(),
                    'train_accuracy': 100 * correct / total,
                    'learning_rate': self.scheduler.get_last_lr()[0],
                    'epoch': epoch,
                    'step': epoch * len(self.train_dataloader) + batch_idx
                })
        
        return total_loss / len(self.train_dataloader), 100 * correct / total
    
    def validate(self, epoch):
        """Validate the model."""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in self.val_dataloader:
                # Move to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass
                logits = self.model(input_ids, attention_mask)
                loss = self.criterion(logits, labels)
                
                # Statistics
                total_loss += loss.item()
                _, predicted = torch.max(logits, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        avg_loss = total_loss / len(self.val_dataloader)
        accuracy = 100 * correct / total
        
        # Log to wandb
        wandb.log({
            'val_loss': avg_loss,
            'val_accuracy': accuracy,
            'epoch': epoch
        })
        
        return avg_loss, accuracy
    
    def train(self):
        """Complete training loop."""
        best_val_acc = 0
        
        for epoch in range(self.config.num_epochs):
            # Train
            train_loss, train_acc = self.train_epoch(epoch)
            
            # Validate
            val_loss, val_acc = self.validate(epoch)
            
            print(f'Epoch {epoch}:')
            print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            print(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_accuracy': val_acc,
                    'config': self.config
                }, 'best_model.pth')
                print(f'  New best model saved with validation accuracy: {val_acc:.2f}%')
            
            print()
        
        wandb.finish()
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
    """Complete training pipeline."""
    
    # Configuration
    config = TrainingConfig(
        vocab_size=30522,
        d_model=512,
        nhead=8,
        num_layers=6,
        num_classes=3,
        batch_size=32,
        learning_rate=1e-4,
        num_epochs=10
    )
    
    # Load and prepare data
    print("Loading dataset...")
    train_dataset, val_dataset, test_dataset = prepare_dataset(
        "sentiment_dataset.csv",
        "bert-base-uncased"
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.batch_size, 
        shuffle=True,
        num_workers=4
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.batch_size, 
        shuffle=False,
        num_workers=4
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config.batch_size, 
        shuffle=False,
        num_workers=4
    )
    
    # Initialize model
    print("Initializing model...")
    model = TransformerClassifier(
        vocab_size=config.vocab_size,
        d_model=config.d_model,
        nhead=config.nhead,
        num_layers=config.num_layers,
        num_classes=config.num_classes,
        dropout=config.dropout
    )
    
    # Initialize trainer
    trainer = TransformerTrainer(model, train_loader, val_loader, config)
    
    # Train model
    print("Starting training...")
    trainer.train()
    
    # Load best model for evaluation
    print("Loading best model for evaluation...")
    checkpoint = torch.load('best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Evaluate on test set
    print("Evaluating on test set...")
    test_loss, test_acc = trainer.evaluate(test_loader)
    print(f"Test Accuracy: {test_acc:.2f}%")
    
    print("Training completed!")

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
