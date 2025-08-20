# 📊 Data Preparation Guide for Transformer Training

This comprehensive guide focuses on **data preparation** - the crucial first step in training transformer models. Proper data preparation can make or break your model's performance.

## 🎯 Why Data Preparation Matters

> **"Data preparation is 80% of the work in machine learning"** - Andrew Ng

- **Quality data** leads to better model performance
- **Clean preprocessing** prevents training issues
- **Proper formatting** ensures compatibility with transformer architectures
- **Data augmentation** increases effective dataset size

---

## 📁 Dataset Structure & Organization

### **Recommended Directory Structure**

```
project/
├── data/
│   ├── raw/                    # Original, unprocessed data
│   │   ├── text_files/
│   │   ├── csv_files/
│   │   └── json_files/
│   ├── processed/              # Cleaned and processed data
│   │   ├── train/
│   │   ├── validation/
│   │   └── test/
│   ├── augmented/              # Data augmentation results
│   └── metadata/               # Dataset statistics and info
├── scripts/
│   ├── data_cleaning.py
│   ├── data_augmentation.py
│   └── data_validation.py
└── configs/
    └── data_config.yaml
```

### **Data File Formats**

#### **1. CSV Format (Recommended)**
```csv
text,label,split,metadata
"This movie is fantastic!",1,train,{"length": 6, "source": "review"}
"I didn't enjoy this film.",0,train,{"length": 7, "source": "review"}
"The plot was okay.",2,validation,{"length": 4, "source": "review"}
```

#### **2. JSON Format**
```json
[
  {
    "text": "This movie is fantastic!",
    "label": 1,
    "split": "train",
    "metadata": {
      "length": 6,
      "source": "review",
      "timestamp": "2024-01-15T10:30:00Z"
    }
  }
]
```

#### **3. JSONL Format (Large Datasets)**
```jsonl
{"text": "This movie is fantastic!", "label": 1, "split": "train"}
{"text": "I didn't enjoy this film.", "label": 0, "split": "train"}
{"text": "The plot was okay.", "label": 2, "split": "validation"}
```

---

## 🧹 Data Cleaning & Preprocessing

### **Text Cleaning Pipeline**

```python
import re
import string
import unicodedata
from typing import List, Dict, Any
import pandas as pd
import numpy as np

class TextCleaner:
    """Comprehensive text cleaning for transformer training."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.setup_patterns()
    
    def setup_patterns(self):
        """Setup regex patterns for cleaning."""
        # URLs
        self.url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        
        # Emails
        self.email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
        
        # HTML tags
        self.html_pattern = re.compile(r'<[^>]+>')
        
        # Multiple spaces
        self.space_pattern = re.compile(r'\s+')
        
        # Special characters
        self.special_char_pattern = re.compile(r'[^\w\s]')
    
    def clean_text(self, text: str) -> str:
        """
        Clean a single text string.
        
        Args:
            text (str): Raw text input
            
        Returns:
            str: Cleaned text
        """
        if not isinstance(text, str) or pd.isna(text):
            return ""
        
        # Convert to lowercase if specified
        if self.config.get('lowercase', True):
            text = text.lower()
        
        # Remove HTML tags
        if self.config.get('remove_html', True):
            text = self.html_pattern.sub('', text)
        
        # Remove URLs
        if self.config.get('remove_urls', True):
            text = self.url_pattern.sub('[URL]', text)
        
        # Remove emails
        if self.config.get('remove_emails', True):
            text = self.email_pattern.sub('[EMAIL]', text)
        
        # Remove special characters
        if self.config.get('remove_special_chars', False):
            text = self.special_char_pattern.sub('', text)
        
        # Normalize unicode
        if self.config.get('normalize_unicode', True):
            text = unicodedata.normalize('NFKC', text)
        
        # Remove extra whitespace
        text = self.space_pattern.sub(' ', text)
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def clean_dataset(self, df: pd.DataFrame, text_column: str = 'text') -> pd.DataFrame:
        """
        Clean entire dataset.
        
        Args:
            df (pd.DataFrame): Input dataframe
            text_column (str): Name of text column
            
        Returns:
            pd.DataFrame: Cleaned dataframe
        """
        print(f"Cleaning {len(df)} texts...")
        
        # Clean text column
        df[f'{text_column}_cleaned'] = df[text_column].apply(self.clean_text)
        
        # Remove empty texts
        initial_count = len(df)
        df = df[df[f'{text_column}_cleaned'].str.len() > 0]
        removed_count = initial_count - len(df)
        
        print(f"Removed {removed_count} empty texts after cleaning")
        
        return df

# Configuration for text cleaning
cleaning_config = {
    'lowercase': True,
    'remove_html': True,
    'remove_urls': True,
    'remove_emails': True,
    'remove_special_chars': False,
    'normalize_unicode': True
}

# Usage example
cleaner = TextCleaner(cleaning_config)
cleaned_df = cleaner.clean_dataset(raw_df, 'text')
```

### **Data Quality Checks**

```python
class DataQualityChecker:
    """Check data quality and identify issues."""
    
    def __init__(self):
        self.issues = []
    
    def check_dataset(self, df: pd.DataFrame, text_column: str = 'text') -> Dict[str, Any]:
        """
        Comprehensive data quality check.
        
        Returns:
            Dict with quality metrics and issues
        """
        results = {
            'total_samples': len(df),
            'missing_values': {},
            'duplicates': 0,
            'text_length_stats': {},
            'label_distribution': {},
            'issues': []
        }
        
        # Check missing values
        for col in df.columns:
            missing_count = df[col].isnull().sum()
            if missing_count > 0:
                results['missing_values'][col] = missing_count
                results['issues'].append(f"Missing values in {col}: {missing_count}")
        
        # Check duplicates
        duplicates = df.duplicated().sum()
        results['duplicates'] = duplicates
        if duplicates > 0:
            results['issues'].append(f"Duplicate rows: {duplicates}")
        
        # Text length statistics
        if text_column in df.columns:
            text_lengths = df[text_column].str.len()
            results['text_length_stats'] = {
                'min': int(text_lengths.min()),
                'max': int(text_lengths.max()),
                'mean': float(text_lengths.mean()),
                'median': float(text_lengths.median()),
                'std': float(text_lengths.std())
            }
            
            # Check for extremely short/long texts
            short_texts = (text_lengths < 10).sum()
            long_texts = (text_lengths > 1000).sum()
            
            if short_texts > 0:
                results['issues'].append(f"Very short texts (<10 chars): {short_texts}")
            if long_texts > 0:
                results['issues'].append(f"Very long texts (>1000 chars): {long_texts}")
        
        # Label distribution
        if 'label' in df.columns:
            label_counts = df['label'].value_counts()
            results['label_distribution'] = label_counts.to_dict()
            
            # Check for class imbalance
            total = len(df)
            for label, count in label_counts.items():
                percentage = (count / total) * 100
                if percentage < 5:  # Less than 5% of data
                    results['issues'].append(f"Class {label} has only {percentage:.1f}% of data")
        
        return results
    
    def generate_report(self, results: Dict[str, Any]) -> str:
        """Generate human-readable quality report."""
        report = "=" * 50 + "\n"
        report += "DATA QUALITY REPORT\n"
        report += "=" * 50 + "\n\n"
        
        report += f"Total samples: {results['total_samples']:,}\n"
        report += f"Duplicate rows: {results['duplicates']:,}\n\n"
        
        if results['missing_values']:
            report += "Missing values:\n"
            for col, count in results['missing_values'].items():
                report += f"  {col}: {count:,}\n"
            report += "\n"
        
        if results['text_length_stats']:
            stats = results['text_length_stats']
            report += "Text length statistics:\n"
            report += f"  Min: {stats['min']} characters\n"
            report += f"  Max: {stats['max']} characters\n"
            report += f"  Mean: {stats['mean']:.1f} characters\n"
            report += f"  Median: {stats['median']:.1f} characters\n"
            report += f"  Std: {stats['std']:.1f} characters\n\n"
        
        if results['label_distribution']:
            report += "Label distribution:\n"
            total = results['total_samples']
            for label, count in results['label_distribution'].items():
                percentage = (count / total) * 100
                report += f"  Class {label}: {count:,} ({percentage:.1f}%)\n"
            report += "\n"
        
        if results['issues']:
            report += "Issues found:\n"
            for issue in results['issues']:
                report += f"  ⚠️  {issue}\n"
        else:
            report += "✅ No major issues found!\n"
        
        return report

# Usage
checker = DataQualityChecker()
quality_results = checker.check_dataset(cleaned_df, 'text_cleaned')
report = checker.generate_report(quality_results)
print(report)
```

---

## 🔄 Data Augmentation

### **Text Augmentation Techniques**

```python
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.sentence as nas
import nlpaug.augmenter.char as nac
from typing import List, Tuple
import random

class TextAugmenter:
    """Advanced text augmentation for transformer training."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.setup_augmenters()
    
    def setup_augmenters(self):
        """Initialize augmentation methods."""
        self.augmenters = {}
        
        # Word-level augmentation
        if self.config.get('synonym_replacement', True):
            self.augmenters['synonym'] = naw.SynonymAug(
                aug_src='wordnet',
                aug_p=0.3  # 30% probability of replacement
            )
        
        if self.config.get('random_insertion', True):
            self.augmenters['insertion'] = naw.RandomWordAug(
                action="insert",
                aug_p=0.1  # 10% probability of insertion
            )
        
        if self.config.get('random_deletion', True):
            self.augmenters['deletion'] = naw.RandomWordAug(
                action="delete",
                aug_p=0.1  # 10% probability of deletion
            )
        
        if self.config.get('random_swap', True):
            self.augmenters['swap'] = naw.RandomWordAug(
                action="swap",
                aug_p=0.1  # 10% probability of swapping
            )
        
        # Sentence-level augmentation
        if self.config.get('back_translation', True):
            self.augmenters['back_translation'] = naw.BackTranslationAug(
                from_model_name='facebook/wmt19-en-de',
                to_model_name='facebook/wmt19-de-en'
            )
        
        # Character-level augmentation
        if self.config.get('character_typo', True):
            self.augmenters['typo'] = nac.KeyboardAug(
                aug_char_p=0.1,  # 10% probability of character change
                aug_word_p=0.1   # 10% probability of word change
            )
    
    def augment_text(self, text: str, method: str = 'random') -> str:
        """
        Augment a single text using specified method.
        
        Args:
            text (str): Input text
            method (str): Augmentation method or 'random'
            
        Returns:
            str: Augmented text
        """
        if method == 'random':
            # Randomly select augmentation method
            method = random.choice(list(self.augmenters.keys()))
        
        if method not in self.augmenters:
            return text
        
        try:
            augmented = self.augmenters[method].augment(text)
            return augmented[0] if isinstance(augmented, list) else augmented
        except Exception as e:
            print(f"Error in {method} augmentation: {e}")
            return text
    
    def augment_dataset(self, df: pd.DataFrame, text_column: str, 
                       label_column: str, augmentation_factor: int = 2,
                       balance_classes: bool = True) -> pd.DataFrame:
        """
        Augment entire dataset.
        
        Args:
            df: Input dataframe
            text_column: Name of text column
            label_column: Name of label column
            augmentation_factor: How many augmented samples per original
            balance_classes: Whether to balance class distribution
            
        Returns:
            Augmented dataframe
        """
        augmented_data = []
        
        if balance_classes and label_column in df.columns:
            # Balance classes by augmenting minority classes more
            label_counts = df[label_column].value_counts()
            max_count = label_counts.max()
            
            for label in df[label_column].unique():
                class_data = df[df[label_column] == label]
                current_count = len(class_data)
                
                if current_count < max_count:
                    # Calculate how many augmented samples needed
                    needed = max_count - current_count
                    samples_per_original = max(1, needed // current_count)
                    
                    for _, row in class_data.iterrows():
                        # Add original sample
                        augmented_data.append(row.to_dict())
                        
                        # Add augmented samples
                        for _ in range(samples_per_original):
                            aug_text = self.augment_text(row[text_column])
                            aug_row = row.copy()
                            aug_row[text_column] = aug_text
                            aug_row['augmented'] = True  # Mark as augmented
                            augmented_data.append(aug_row)
                else:
                    # Majority class - just add original samples
                    for _, row in class_data.iterrows():
                        augmented_data.append(row.to_dict())
        else:
            # Simple augmentation without balancing
            for _, row in df.iterrows():
                # Add original sample
                augmented_data.append(row.to_dict())
                
                # Add augmented samples
                for _ in range(augmentation_factor):
                    aug_text = self.augment_text(row[text_column])
                    aug_row = row.copy()
                    aug_row[text_column] = aug_text
                    aug_row['augmented'] = True
                    augmented_data.append(aug_row)
        
        return pd.DataFrame(augmented_data)

# Augmentation configuration
aug_config = {
    'synonym_replacement': True,
    'random_insertion': True,
    'random_deletion': True,
    'random_swap': True,
    'back_translation': True,
    'character_typo': False  # Disable for cleaner augmentation
}

# Usage
augmenter = TextAugmenter(aug_config)
augmented_df = augmenter.augment_dataset(
    cleaned_df, 
    'text_cleaned', 
    'label', 
    augmentation_factor=2,
    balance_classes=True
)
```

---

## 📊 Data Splitting & Validation

### **Stratified Data Splitting**

```python
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold

class DataSplitter:
    """Intelligent data splitting for transformer training."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
    
    def split_data(self, df: pd.DataFrame, text_column: str, 
                   label_column: str, test_size: float = 0.2,
                   val_size: float = 0.1, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into train, validation, and test sets.
        
        Args:
            df: Input dataframe
            text_column: Name of text column
            label_column: Name of label column
            test_size: Fraction for test set
            val_size: Fraction for validation set (from remaining data)
            random_state: Random seed for reproducibility
            
        Returns:
            train_df, val_df, test_df
        """
        # First split: separate test set
        train_val_df, test_df = train_test_split(
            df, 
            test_size=test_size, 
            random_state=random_state,
            stratify=df[label_column] if label_column in df.columns else None
        )
        
        # Second split: separate validation set from remaining data
        val_size_adjusted = val_size / (1 - test_size)
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=val_size_adjusted,
            random_state=random_state,
            stratify=train_val_df[label_column] if label_column in train_val_df.columns else None
        )
        
        # Add split information
        train_df['split'] = 'train'
        val_df['split'] = 'validation'
        test_df['split'] = 'test'
        
        print(f"Data split complete:")
        print(f"  Train: {len(train_df):,} samples ({len(train_df)/len(df)*100:.1f}%)")
        print(f"  Validation: {len(val_df):,} samples ({len(val_df)/len(df)*100:.1f}%)")
        print(f"  Test: {len(test_df):,} samples ({len(test_df)/len(df)*100:.1f}%)")
        
        return train_df, val_df, test_df
    
    def cross_validation_split(self, df: pd.DataFrame, label_column: str, 
                              n_splits: int = 5, random_state: int = 42) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Create cross-validation splits.
        
        Returns:
            List of (train_df, val_df) tuples
        """
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        splits = []
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(df, df[label_column])):
            train_df = df.iloc[train_idx].copy()
            val_df = df.iloc[val_idx].copy()
            
            train_df['fold'] = fold
            val_df['fold'] = fold
            
            splits.append((train_df, val_df))
            print(f"Fold {fold}: Train={len(train_df):,}, Val={len(val_df):,}")
        
        return splits

# Usage
splitter = DataSplitter({})
train_df, val_df, test_df = splitter.split_data(
    augmented_df, 
    'text_cleaned', 
    'label',
    test_size=0.2,
    val_size=0.1
)
```

---

## 🔧 Data Formatting for Training

### **PyTorch Dataset Creation**

```python
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import numpy as np

class TransformerDataset(Dataset):
    """PyTorch dataset for transformer training."""
    
    def __init__(self, df: pd.DataFrame, tokenizer, text_column: str = 'text',
                 label_column: str = 'label', max_length: int = 512,
                 truncation: str = 'longest_first', padding: str = 'max_length'):
        self.df = df
        self.tokenizer = tokenizer
        self.text_column = text_column
        self.label_column = label_column
        self.max_length = max_length
        self.truncation = truncation
        self.padding = padding
        
        # Validate data
        self._validate_data()
    
    def _validate_data(self):
        """Validate dataset integrity."""
        required_columns = [self.text_column, self.label_column]
        missing_columns = [col for col in required_columns if col not in self.df.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Check for empty texts
        empty_texts = self.df[self.text_column].isna().sum()
        if empty_texts > 0:
            print(f"Warning: {empty_texts} texts are NaN")
        
        # Check label types
        if not pd.api.types.is_numeric_dtype(self.df[self.label_column]):
            print("Warning: Labels are not numeric, converting...")
            self.df[self.label_column] = pd.to_numeric(self.df[self.label_column], errors='coerce')
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        """Get a single sample."""
        row = self.df.iloc[idx]
        
        # Get text and label
        text = str(row[self.text_column])
        label = int(row[self.label_column])
        
        # Tokenize text
        encoding = self.tokenizer(
            text,
            truncation=self.truncation,
            padding=self.padding,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Prepare sample
        sample = {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }
        
        # Add token type IDs if available
        if 'token_type_ids' in encoding:
            sample['token_type_ids'] = encoding['token_type_ids'].flatten()
        
        return sample
    
    def get_label_distribution(self):
        """Get label distribution for analysis."""
        return self.df[self.label_column].value_counts().to_dict()

# Create datasets
def create_datasets(train_df, val_df, test_df, tokenizer_name="bert-base-uncased", 
                   max_length=512):
    """Create PyTorch datasets for training."""
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    # Add special tokens if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create datasets
    train_dataset = TransformerDataset(
        train_df, tokenizer, 'text_cleaned', 'label', max_length
    )
    
    val_dataset = TransformerDataset(
        val_df, tokenizer, 'text_cleaned', 'label', max_length
    )
    
    test_dataset = TransformerDataset(
        test_df, tokenizer, 'text_cleaned', 'label', max_length
    )
    
    return train_dataset, val_dataset, test_dataset, tokenizer

# Create data loaders
def create_dataloaders(train_dataset, val_dataset, test_dataset, 
                      batch_size=32, num_workers=4):
    """Create PyTorch data loaders."""
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader

# Usage
train_dataset, val_dataset, test_dataset, tokenizer = create_datasets(
    train_df, val_df, test_df, "bert-base-uncased", max_length=512
)

train_loader, val_loader, test_loader = create_dataloaders(
    train_dataset, val_dataset, test_dataset, batch_size=32
)
```

---

## 📈 Data Analysis & Visualization

### **Dataset Statistics & Insights**

```python
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

class DataAnalyzer:
    """Analyze and visualize dataset characteristics."""
    
    def __init__(self, df: pd.DataFrame, text_column: str = 'text', 
                 label_column: str = 'label'):
        self.df = df
        self.text_column = text_column
        self.label_column = label_column
    
    def analyze_text_lengths(self):
        """Analyze text length distribution."""
        lengths = self.df[self.text_column].str.len()
        
        plt.figure(figsize=(12, 4))
        
        # Histogram
        plt.subplot(1, 2, 1)
        plt.hist(lengths, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        plt.xlabel('Text Length (characters)')
        plt.ylabel('Frequency')
        plt.title('Text Length Distribution')
        plt.grid(True, alpha=0.3)
        
        # Box plot
        plt.subplot(1, 2, 2)
        plt.boxplot(lengths)
        plt.ylabel('Text Length (characters)')
        plt.title('Text Length Box Plot')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Statistics
        stats = {
            'mean': lengths.mean(),
            'median': lengths.median(),
            'std': lengths.std(),
            'min': lengths.min(),
            'max': lengths.max(),
            'q25': lengths.quantile(0.25),
            'q75': lengths.quantile(0.75)
        }
        
        print("Text Length Statistics:")
        for key, value in stats.items():
            print(f"  {key.upper()}: {value:.1f}")
    
    def analyze_labels(self):
        """Analyze label distribution."""
        label_counts = self.df[self.label_column].value_counts()
        
        plt.figure(figsize=(10, 6))
        
        # Bar plot
        plt.subplot(1, 2, 1)
        bars = plt.bar(label_counts.index, label_counts.values, 
                      color=['lightcoral', 'lightblue', 'lightgreen'])
        plt.xlabel('Label')
        plt.ylabel('Count')
        plt.title('Label Distribution')
        plt.xticks(label_counts.index)
        
        # Add value labels on bars
        for bar, count in zip(bars, label_counts.values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01*max(label_counts.values),
                    str(count), ha='center', va='bottom')
        
        # Pie chart
        plt.subplot(1, 2, 2)
        plt.pie(label_counts.values, labels=label_counts.index, autopct='%1.1f%%',
                colors=['lightcoral', 'lightblue', 'lightgreen'])
        plt.title('Label Distribution (%)')
        
        plt.tight_layout()
        plt.show()
        
        print("Label Distribution:")
        total = len(self.df)
        for label, count in label_counts.items():
            percentage = (count / total) * 100
            print(f"  Label {label}: {count:,} ({percentage:.1f}%)")
    
    def generate_wordcloud(self, label: int = None):
        """Generate word cloud for specific label or all data."""
        if label is not None:
            texts = self.df[self.df[self.label_column] == label][self.text_column]
            title = f'Word Cloud - Label {label}'
        else:
            texts = self.df[self.text_column]
            title = 'Word Cloud - All Data'
        
        # Combine all texts
        combined_text = ' '.join(texts.astype(str))
        
        # Generate word cloud
        wordcloud = WordCloud(
            width=800, 
            height=400, 
            background_color='white',
            max_words=100,
            colormap='viridis'
        ).generate(combined_text)
        
        plt.figure(figsize=(12, 6))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(title)
        plt.show()
    
    def analyze_vocabulary(self):
        """Analyze vocabulary characteristics."""
        # Get all unique words
        all_words = []
        for text in self.df[self.text_column]:
            words = str(text).lower().split()
            all_words.extend(words)
        
        # Calculate statistics
        unique_words = set(all_words)
        word_freq = {}
        for word in all_words:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        # Sort by frequency
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        
        print("Vocabulary Analysis:")
        print(f"  Total words: {len(all_words):,}")
        print(f"  Unique words: {len(unique_words):,}")
        print(f"  Vocabulary size: {len(unique_words):,}")
        print(f"  Average words per text: {len(all_words) / len(self.df):.1f}")
        
        # Top 20 most frequent words
        print("\nTop 20 Most Frequent Words:")
        for i, (word, freq) in enumerate(sorted_words[:20]):
            print(f"  {i+1:2d}. {word:15s}: {freq:,}")
        
        return word_freq

# Usage
analyzer = DataAnalyzer(train_df, 'text_cleaned', 'label')
analyzer.analyze_text_lengths()
analyzer.analyze_labels()
analyzer.generate_wordcloud()  # All data
analyzer.generate_wordcloud(label=1)  # Positive class
word_freq = analyzer.analyze_vocabulary()
```

---

## 💾 Data Export & Persistence

### **Save Processed Data**

```python
import pickle
import json
from pathlib import Path

class DataExporter:
    """Export processed data in various formats."""
    
    def __init__(self, output_dir: str = "processed_data"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
    
    def save_csv(self, df: pd.DataFrame, filename: str, split: str = None):
        """Save dataframe as CSV."""
        if split:
            filename = f"{split}_{filename}"
        
        filepath = self.output_dir / filename
        df.to_csv(filepath, index=False)
        print(f"Saved CSV: {filepath}")
    
    def save_json(self, df: pd.DataFrame, filename: str, split: str = None):
        """Save dataframe as JSON."""
        if split:
            filename = f"{split}_{filename}"
        
        filepath = self.output_dir / filename
        df.to_json(filepath, orient='records', indent=2)
        print(f"Saved JSON: {filepath}")
    
    def save_pickle(self, obj, filename: str):
        """Save object as pickle file."""
        filepath = self.output_dir / filename
        with open(filepath, 'wb') as f:
            pickle.dump(obj, f)
        print(f"Saved pickle: {filepath}")
    
    def save_tokenizer(self, tokenizer, tokenizer_dir: str = "tokenizer"):
        """Save tokenizer for later use."""
        tokenizer_path = self.output_dir / tokenizer_dir
        tokenizer.save_pretrained(tokenizer_path)
        print(f"Saved tokenizer: {tokenizer_path}")
    
    def save_metadata(self, metadata: Dict[str, Any], filename: str = "metadata.json"):
        """Save dataset metadata."""
        filepath = self.output_dir / filename
        with open(filepath, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"Saved metadata: {filepath}")
    
    def export_all(self, train_df, val_df, test_df, tokenizer, 
                   base_filename: str = "dataset"):
        """Export all data and metadata."""
        
        # Save splits
        self.save_csv(train_df, f"{base_filename}.csv", "train")
        self.save_csv(val_df, f"{base_filename}.csv", "val")
        self.save_csv(test_df, f"{base_filename}.csv", "test")
        
        # Save tokenizer
        self.save_tokenizer(tokenizer)
        
        # Save metadata
        metadata = {
            'dataset_info': {
                'total_samples': len(train_df) + len(val_df) + len(test_df),
                'train_samples': len(train_df),
                'val_samples': len(val_df),
                'test_samples': len(test_df),
                'num_classes': train_df['label'].nunique(),
                'class_distribution': train_df['label'].value_counts().to_dict()
            },
            'text_info': {
                'max_length': train_df['text_cleaned'].str.len().max(),
                'min_length': train_df['text_cleaned'].str.len().min(),
                'avg_length': train_df['text_cleaned'].str.len().mean()
            },
            'processing_info': {
                'cleaned': True,
                'augmented': 'augmented' in train_df.columns,
                'tokenized': True
            }
        }
        
        self.save_metadata(metadata)

# Usage
exporter = DataExporter("processed_data")
exporter.export_all(train_df, val_df, test_df, tokenizer, "sentiment_dataset")
```

---

## 🚀 Complete Data Preparation Pipeline

### **Main Pipeline Script**

```python
def main_data_preparation_pipeline():
    """Complete data preparation pipeline."""
    
    print("🚀 Starting Data Preparation Pipeline")
    print("=" * 50)
    
    # 1. Load raw data
    print("\n📁 Step 1: Loading raw data...")
    raw_df = pd.read_csv("data/raw/sentiment_data.csv")
    print(f"   Loaded {len(raw_df):,} samples")
    
    # 2. Clean data
    print("\n🧹 Step 2: Cleaning data...")
    cleaner = TextCleaner(cleaning_config)
    cleaned_df = cleaner.clean_dataset(raw_df, 'text')
    print(f"   Cleaned {len(cleaned_df):,} samples")
    
    # 3. Check data quality
    print("\n🔍 Step 3: Checking data quality...")
    checker = DataQualityChecker()
    quality_results = checker.check_dataset(cleaned_df, 'text_cleaned')
    report = checker.generate_report(quality_results)
    print(report)
    
    # 4. Augment data
    print("\n🔄 Step 4: Augmenting data...")
    augmenter = TextAugmenter(aug_config)
    augmented_df = augmenter.augment_dataset(
        cleaned_df, 'text_cleaned', 'label', 
        augmentation_factor=2, balance_classes=True
    )
    print(f"   Augmented to {len(augmented_df):,} samples")
    
    # 5. Split data
    print("\n✂️  Step 5: Splitting data...")
    splitter = DataSplitter({})
    train_df, val_df, test_df = splitter.split_data(
        augmented_df, 'text_cleaned', 'label'
    )
    
    # 6. Create datasets
    print("\n🏗️  Step 6: Creating PyTorch datasets...")
    train_dataset, val_dataset, test_dataset, tokenizer = create_datasets(
        train_df, val_df, test_df, "bert-base-uncased"
    )
    
    # 7. Analyze data
    print("\n📊 Step 7: Analyzing data...")
    analyzer = DataAnalyzer(train_df, 'text_cleaned', 'label')
    analyzer.analyze_text_lengths()
    analyzer.analyze_labels()
    analyzer.generate_wordcloud()
    
    # 8. Export data
    print("\n💾 Step 8: Exporting data...")
    exporter = DataExporter("processed_data")
    exporter.export_all(train_df, val_df, test_df, tokenizer, "sentiment_dataset")
    
    print("\n✅ Data preparation pipeline completed successfully!")
    print(f"   Final dataset sizes:")
    print(f"     Train: {len(train_df):,} samples")
    print(f"     Validation: {len(val_df):,} samples")
    print(f"     Test: {len(test_df):,} samples")
    
    return train_df, val_df, test_df, tokenizer

if __name__ == "__main__":
    train_df, val_df, test_df, tokenizer = main_data_preparation_pipeline()
```

---

## 📋 Data Preparation Checklist

### **Before Training Checklist**

- [ ] **Data Loading**: Raw data successfully loaded
- [ ] **Data Cleaning**: Text preprocessing completed
- [ ] **Quality Check**: No major data issues found
- [ ] **Data Augmentation**: Applied if needed for class balance
- [ ] **Data Splitting**: Train/val/test splits created
- [ ] **Dataset Creation**: PyTorch datasets ready
- [ ] **Data Analysis**: Dataset characteristics understood
- [ ] **Data Export**: Processed data saved
- [ ] **Tokenization**: Text properly tokenized
- [ ] **Label Encoding**: Labels properly formatted

### **Common Pitfalls to Avoid**

1. **Data Leakage**: Ensure no test data leaks into training
2. **Class Imbalance**: Check and address if necessary
3. **Text Length**: Ensure consistent text lengths
4. **Missing Values**: Handle missing data appropriately
5. **Encoding Issues**: Use proper text encoding (UTF-8)
6. **Memory Issues**: Monitor dataset size and memory usage

---

## 🎯 Key Takeaways

1. **Data Quality is Paramount**: Clean, well-structured data leads to better models
2. **Systematic Approach**: Follow a structured pipeline for consistency
3. **Validation at Each Step**: Check data quality throughout the process
4. **Documentation**: Keep track of all preprocessing steps
5. **Reproducibility**: Use random seeds and save intermediate results

---

## 📚 Additional Resources

- **Text Preprocessing**: NLTK, spaCy, TextBlob
- **Data Augmentation**: nlpaug, TextAugment
- **Data Validation**: Great Expectations, Pandas Profiling
- **Visualization**: Matplotlib, Seaborn, Plotly
- **Best Practices**: Google ML Guide, Microsoft ML Best Practices

---

## 🎉 Conclusion

Proper data preparation is the foundation of successful transformer training. This guide provides a comprehensive framework for:

- **Cleaning and preprocessing** text data
- **Augmenting datasets** for better performance
- **Splitting data** intelligently
- **Creating training-ready** PyTorch datasets
- **Analyzing and visualizing** data characteristics
- **Exporting processed data** for training

Remember: **"The best model architecture won't help if your data is poor quality!"** 🚀

Invest time in data preparation - it will pay dividends in model performance! 💪
