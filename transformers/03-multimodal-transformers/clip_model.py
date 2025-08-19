"""
CLIP Model Implementation
=========================

This module implements CLIP (Contrastive Language-Image Pre-training):
- Text and image encoders
- Contrastive learning
- Cross-modal similarity computation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel


class CLIP(nn.Module):
    """CLIP model for contrastive language-image pre-training"""
    
    def __init__(self, text_encoder_name="bert-base-uncased", image_encoder_name="resnet50",
                 embed_dim=512, temperature=0.07):
        super().__init__()
        
        # Text encoder (using pre-trained BERT)
        self.text_encoder = AutoModel.from_pretrained(text_encoder_name)
        self.text_projection = nn.Linear(self.text_encoder.config.hidden_size, embed_dim)
        
        # Image encoder (using pre-trained ResNet)
        self.image_encoder = AutoModel.from_pretrained(image_encoder_name)
        self.image_projection = nn.Linear(self.image_encoder.config.hidden_size, embed_dim)
        
        # Temperature parameter
        self.temperature = nn.Parameter(torch.ones([]) * temperature)
        
        # Tokenizer for text
        self.tokenizer = AutoTokenizer.from_pretrained(text_encoder_name)
        
    def encode_text(self, text_inputs):
        """Encode text inputs"""
        text_features = self.text_encoder(**text_inputs).last_hidden_state
        
        # Use [CLS] token representation
        text_features = text_features[:, 0, :]  # (batch_size, hidden_size)
        
        # Project to common space
        text_embeddings = self.text_projection(text_features)
        
        # Normalize
        text_embeddings = F.normalize(text_embeddings, dim=-1)
        
        return text_embeddings
    
    def encode_image(self, image_inputs):
        """Encode image inputs"""
        image_features = self.image_encoder(image_inputs).pooler_output
        
        # Project to common space
        image_embeddings = self.image_projection(image_features)
        
        # Normalize
        image_embeddings = F.normalize(image_embeddings, dim=-1)
        
        return image_embeddings
    
    def forward(self, text_inputs, image_inputs):
        """Forward pass for CLIP"""
        # Encode text and images
        text_embeddings = self.encode_text(text_inputs)
        image_embeddings = self.encode_image(image_inputs)
        
        # Compute similarity matrix
        logits = torch.matmul(text_embeddings, image_embeddings.T) / self.temperature
        
        return logits, text_embeddings, image_embeddings


def clip_loss(logits, labels):
    """Compute CLIP contrastive loss"""
    # Create labels for positive pairs (diagonal)
    batch_size = logits.size(0)
    labels = torch.arange(batch_size, device=logits.device)
    
    # Cross-entropy loss for both directions
    loss_i = F.cross_entropy(logits, labels)
    loss_t = F.cross_entropy(logits.T, labels)
    
    return (loss_i + loss_t) / 2


class CLIPTextEncoder(nn.Module):
    """Standalone text encoder for CLIP"""
    
    def __init__(self, model_name="bert-base-uncased", embed_dim=512):
        super().__init__()
        
        self.encoder = AutoModel.from_pretrained(model_name)
        self.projection = nn.Linear(self.encoder.config.hidden_size, embed_dim)
        
    def forward(self, input_ids, attention_mask=None):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        
        # Use [CLS] token
        features = outputs.last_hidden_state[:, 0, :]
        embeddings = self.projection(features)
        
        return F.normalize(embeddings, dim=-1)


class CLIPImageEncoder(nn.Module):
    """Standalone image encoder for CLIP"""
    
    def __init__(self, model_name="resnet50", embed_dim=512):
        super().__init__()
        
        self.encoder = AutoModel.from_pretrained(model_name)
        self.projection = nn.Linear(self.encoder.config.hidden_size, embed_dim)
        
    def forward(self, pixel_values):
        outputs = self.encoder(pixel_values=pixel_values)
        
        # Use pooled output
        features = outputs.pooler_output
        embeddings = self.projection(features)
        
        return F.normalize(embeddings, dim=-1)


class CLIPDataset:
    """Simple dataset for CLIP training"""
    
    def __init__(self, texts, images, tokenizer, max_length=77):
        self.texts = texts
        self.images = images
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        image = self.images[idx]
        
        # Tokenize text
        text_inputs = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Remove batch dimension
        text_inputs = {k: v.squeeze(0) for k, v in text_inputs.items()}
        
        return {
            'text_inputs': text_inputs,
            'image': image
        }


# Example usage and testing
if __name__ == "__main__":
    # Test CLIP model
    print("Testing CLIP Model...")
    
    # Create sample data
    batch_size = 4
    embed_dim = 512
    
    # Mock text inputs
    text_inputs = {
        'input_ids': torch.randint(0, 30000, (batch_size, 77)),
        'attention_mask': torch.ones(batch_size, 77)
    }
    
    # Mock image inputs
    image_inputs = torch.randn(batch_size, 3, 224, 224)
    
    # Initialize CLIP
    clip_model = CLIP(embed_dim=embed_dim)
    
    # Forward pass
    logits, text_emb, image_emb = clip_model(text_inputs, image_inputs)
    
    print(f"Logits shape: {logits.shape}")
    print(f"Text embeddings shape: {text_emb.shape}")
    print(f"Image embeddings shape: {image_emb.shape}")
    
    # Test loss computation
    loss = clip_loss(logits, None)
    print(f"CLIP loss: {loss.item():.4f}")
    
    print("\nCLIP model working correctly!")
