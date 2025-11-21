"""
PyTorch-based Transformer model for ASL sign language recognition
Captures temporal patterns in hand movements
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


class PositionalEncoding(nn.Module):
    """Add positional information to the input embeddings"""
    
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        
        # Create positional encoding matrix
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Register as buffer (not a parameter, but part of state)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # x shape: (batch_size, sequence_length, d_model)
        seq_len = x.size(1)
        return x + self.pe[:seq_len, :].unsqueeze(0)


class TransformerBlock(nn.Module):
    """Single Transformer encoder block"""
    
    def __init__(self, d_model, num_heads, ff_dim, dropout_rate=0.1):
        super().__init__()
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout_rate,
            batch_first=True
        )
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(ff_dim, d_model)
        )
        
        # Layer normalization
        self.layernorm1 = nn.LayerNorm(d_model)
        self.layernorm2 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)
    
    def forward(self, x):
        # Multi-head attention with residual connection
        attn_output, _ = self.attention(x, x, x)
        attn_output = self.dropout1(attn_output)
        out1 = self.layernorm1(x + attn_output)
        
        # Feed-forward network with residual connection
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        out2 = self.layernorm2(out1 + ffn_output)
        
        return out2


class TransformerModel(nn.Module):
    """
    Transformer model for ASL sign recognition
    """
    
    def __init__(
        self,
        sequence_length=30,
        feature_dim=126,
        num_classes=10,
        d_model=128,
        num_heads=8,
        num_transformer_blocks=4,
        ff_dim=256,
        dropout_rate=0.1
    ):
        super().__init__()
        
        self.sequence_length = sequence_length
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        
        # Project input features to d_model dimensions
        self.input_projection = nn.Linear(feature_dim, d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_len=sequence_length)
        
        # Stack transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, ff_dim, dropout_rate)
            for _ in range(num_transformer_blocks)
        ])
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(d_model, ff_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(ff_dim // 2, num_classes)
        )
    
    def forward(self, x):
        # x shape: (batch_size, sequence_length, feature_dim)
        
        # Project to d_model
        x = self.input_projection(x)
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Pass through transformer blocks
        for block in self.transformer_blocks:
            x = block(x)
        
        # Global average pooling
        x = x.mean(dim=1)  # (batch_size, d_model)
        
        # Classification
        logits = self.classifier(x)
        
        return logits


def create_transformer_model(
    sequence_length=30,
    feature_dim=126,
    num_classes=10,
    d_model=128,
    num_heads=8,
    num_transformer_blocks=4,
    ff_dim=256,
    dropout_rate=0.1
):
    """
    Create a Transformer model for ASL sign recognition
    
    Args:
        sequence_length: Number of frames in a sequence (e.g., 30 frames)
        feature_dim: Number of features per frame (126 for 2 hands)
        num_classes: Number of sign classes to predict
        d_model: Dimension of the model (embedding size)
        num_heads: Number of attention heads
        num_transformer_blocks: Number of transformer encoder blocks
        ff_dim: Dimension of feed-forward network
        dropout_rate: Dropout rate for regularization
    
    Returns:
        PyTorch model
    """
    
    model = TransformerModel(
        sequence_length=sequence_length,
        feature_dim=feature_dim,
        num_classes=num_classes,
        d_model=d_model,
        num_heads=num_heads,
        num_transformer_blocks=num_transformer_blocks,
        ff_dim=ff_dim,
        dropout_rate=dropout_rate
    )
    
    return model


def create_lightweight_transformer(
    sequence_length=30,
    feature_dim=126,
    num_classes=10
):
    """
    Create a lightweight Transformer model for faster inference
    Good for real-time applications
    """
    return create_transformer_model(
        sequence_length=sequence_length,
        feature_dim=feature_dim,
        num_classes=num_classes,
        d_model=64,          # Smaller model
        num_heads=4,         # Fewer heads
        num_transformer_blocks=2,  # Fewer blocks
        ff_dim=128,          # Smaller FF network
        dropout_rate=0.1
    )


def create_powerful_transformer(
    sequence_length=30,
    feature_dim=126,
    num_classes=10
):
    """
    Create a powerful Transformer model for maximum accuracy
    Slower but more accurate
    """
    return create_transformer_model(
        sequence_length=sequence_length,
        feature_dim=feature_dim,
        num_classes=num_classes,
        d_model=256,         # Larger model
        num_heads=8,         # More heads
        num_transformer_blocks=6,  # More blocks
        ff_dim=512,          # Larger FF network
        dropout_rate=0.2
    )


def count_parameters(model):
    """Count the number of trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test model creation
    print("Creating lightweight Transformer model...")
    model = create_lightweight_transformer(
        sequence_length=30,
        feature_dim=126,
        num_classes=10
    )
    
    print(f"\nModel architecture:")
    print(model)
    
    print(f"\nTotal parameters: {count_parameters(model):,}")
    
    # Test with dummy data
    dummy_input = torch.randn(2, 30, 126)  # batch_size=2
    print(f"\nInput shape: {dummy_input.shape}")
    
    model.eval()
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"Output shape: {output.shape}")
    print(f"Output logits: {output}")
    
    # Apply softmax to get probabilities
    probs = F.softmax(output, dim=1)
    print(f"\nProbabilities: {probs}")
    print(f"Sum of probabilities: {probs.sum(dim=1)}")
    
    print("\n✓ Model test successful!")
