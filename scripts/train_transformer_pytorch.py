"""
Train PyTorch Transformer model for ASL sign recognition
"""

import os
import sys
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from transformer.transformer_model_pytorch import create_lightweight_transformer, create_powerful_transformer, count_parameters
from config import TEST_SIZE, RANDOM_STATE, SEQUENCES_PICKLE_PATH, TRANSFORMER_MODEL_PATH, TRANSFORMER_LABEL_ENCODER_PATH, EPOCHS, BATCH_SIZE, LEARNING_RATE, MODEL_TYPE

DEVICE = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')  # Use Apple Silicon GPU if available


class SequenceDataset(Dataset):
    """PyTorch Dataset for sequences"""
    
    def __init__(self, sequences, labels):
        self.sequences = torch.FloatTensor(sequences)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


def load_sequences(sequences_path=SEQUENCES_PICKLE_PATH):
    """Load sequences from pickle file"""
    print(f"Loading sequences from: {sequences_path}")
    
    with open(sequences_path, 'rb') as f:
        data_dict = pickle.load(f)
    
    sequences = np.array(data_dict['sequences'], dtype=np.float32)
    labels = np.array(data_dict['labels'])
    
    print(f"✓ Loaded {len(sequences)} sequences")
    print(f"✓ Sequence shape: {sequences.shape}")
    print(f"✓ Unique labels: {np.unique(labels)}")
    
    return sequences, labels


def prepare_data(sequences, labels, test_size=TEST_SIZE, random_state=RANDOM_STATE):
    """Prepare data for training"""
    
    # Encode labels
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)
    
    print(f"\nLabel mapping:")
    for i, label in enumerate(label_encoder.classes_):
        print(f"  {i}: {label}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        sequences, labels_encoded,
        test_size=test_size,
        random_state=random_state,
        stratify=labels_encoded
    )
    
    print(f"\nDataset split:")
    print(f"  Training: {len(X_train)} sequences")
    print(f"  Testing: {len(X_test)} sequences")
    
    # Create datasets
    train_dataset = SequenceDataset(X_train, y_train)
    test_dataset = SequenceDataset(X_test, y_test)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    return train_loader, test_loader, label_encoder


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for sequences, labels in tqdm(train_loader, desc="Training", leave=False):
        sequences, labels = sequences.to(device), labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(sequences)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    avg_loss = total_loss / len(train_loader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy


def evaluate(model, test_loader, criterion, device):
    """Evaluate model on test set"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for sequences, labels in tqdm(test_loader, desc="Evaluating", leave=False):
            sequences, labels = sequences.to(device), labels.to(device)
            
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    avg_loss = total_loss / len(test_loader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy


def train_model(
    train_loader, test_loader,
    model_type=MODEL_TYPE,
    epochs=EPOCHS,
    learning_rate=LEARNING_RATE,
    device=DEVICE
):
    """Train the Transformer model"""
    
    # Get data info
    sample_batch = next(iter(train_loader))
    sequence_length = sample_batch[0].shape[1]
    feature_dim = sample_batch[0].shape[2]
    num_classes = len(torch.unique(sample_batch[1]))
    
    print(f"\n{'='*60}")
    print(f"MODEL CONFIGURATION")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"Model type: {model_type}")
    print(f"Sequence length: {sequence_length}")
    print(f"Feature dimension: {feature_dim}")
    print(f"Number of classes: {num_classes}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Learning rate: {learning_rate}")
    print(f"{'='*60}\n")
    
    # Create model
    if model_type == 'lightweight':
        model = create_lightweight_transformer(
            sequence_length=sequence_length,
            feature_dim=feature_dim,
            num_classes=num_classes
        )
    else:
        model = create_powerful_transformer(
            sequence_length=sequence_length,
            feature_dim=feature_dim,
            num_classes=num_classes
        )
    
    model = model.to(device)
    print(f"Model parameters: {count_parameters(model):,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # Training loop
    print(f"\n{'='*60}")
    print(f"TRAINING")
    print(f"{'='*60}\n")
    
    best_test_acc = 0
    best_model_state = None
    patience_counter = 0
    patience = 10
    
    for epoch in range(epochs):
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Evaluate
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        
        # Learning rate scheduling
        scheduler.step(test_loss)
        
        # Print progress
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Test Loss:  {test_loss:.4f} | Test Acc:  {test_acc:.2f}%")
        
        # Save best model
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_model_state = model.state_dict().copy()
            patience_counter = 0
            print(f"  ✓ New best model! (Test Acc: {test_acc:.2f}%)")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= patience:
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            break
        
        print()
    
    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model, best_test_acc


def evaluate_per_class(model, test_loader, label_encoder, device):
    """Evaluate per-class accuracy"""
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for sequences, labels in test_loader:
            sequences = sequences.to(device)
            outputs = model(sequences)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    print(f"\n{'='*60}")
    print(f"PER-CLASS ACCURACY")
    print(f"{'='*60}\n")
    
    for i, label in enumerate(label_encoder.classes_):
        mask = all_labels == i
        if mask.sum() > 0:
            class_accuracy = (all_preds[mask] == all_labels[mask]).mean() * 100
            print(f"  {label}: {class_accuracy:.2f}%")


def save_model(model, label_encoder, model_path=TRANSFORMER_MODEL_PATH, encoder_path=TRANSFORMER_LABEL_ENCODER_PATH):
    """Save model and label encoder"""
    
    # Save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': {
            'sequence_length': model.sequence_length,
            'feature_dim': model.feature_dim,
            'num_classes': model.num_classes
        }
    }, model_path)
    print(f"\n✓ Model saved to: {model_path}")
    
    # Save label encoder
    with open(encoder_path, 'wb') as f:
        pickle.dump(label_encoder, f)
    print(f"✓ Label encoder saved to: {encoder_path}")


def main():
    """Main training function"""
    
    print(f"\n{'='*60}")
    print(f"ASL TRANSFORMER MODEL TRAINING (PyTorch)")
    print(f"{'='*60}\n")
    
    print(f"Device: {DEVICE}")
    if DEVICE.type == 'mps':
        print("✓ Using Apple Silicon GPU acceleration!")
    
    # Load sequences
    sequences, labels = load_sequences(SEQUENCES_PICKLE_PATH)
    
    # Check if we have enough data
    num_classes = len(np.unique(labels))
    if num_classes < 2:
        print("\n⚠️  WARNING: Only 1 class detected!")
        print("   You need at least 2 different signs to train a classifier.")
        print("   Collect more signs with: python scripts/collect_sequences.py A B C hello")
        print("\n   Continuing anyway for testing purposes...\n")
    
    if len(sequences) < 20:
        print(f"\n⚠️  WARNING: Only {len(sequences)} sequences collected!")
        print("   Recommended: At least 50-100 sequences total (20+ per sign)")
        print("   Current data may not be enough for good accuracy.\n")
    
    # Prepare data
    train_loader, test_loader, label_encoder = prepare_data(sequences, labels)
    
    # Train model
    model, best_test_acc = train_model(
        train_loader, test_loader,
        model_type=MODEL_TYPE,
        epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        device=DEVICE
    )
    
    # Evaluate per-class accuracy
    evaluate_per_class(model, test_loader, label_encoder, DEVICE)
    
    # Save model and encoder
    save_model(model, label_encoder)
    
    print(f"\n{'='*60}")
    print(f"TRAINING COMPLETE!")
    print(f"{'='*60}")
    print(f"Best test accuracy: {best_test_acc:.2f}%")
    print(f"Model saved to: {TRANSFORMER_MODEL_PATH}")
    print(f"Label encoder saved to: {TRANSFORMER_LABEL_ENCODER_PATH}")
    print(f"{'='*60}\n")
    print(f"Next step: Update app.py to use the PyTorch Transformer model")


if __name__ == "__main__":
    main()
