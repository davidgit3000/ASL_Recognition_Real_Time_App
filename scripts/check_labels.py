#!/usr/bin/env python3
"""
Check what labels/signs are in each model's dataset
Usage: python scripts/check_labels.py
"""

import pickle
import os
import sys
from collections import Counter

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MODEL_PATH, DATA_PICKLE_PATH, SEQUENCES_PICKLE_PATH, TRANSFORMER_LABEL_ENCODER_PATH


def check_random_forest_data():
    """Check Random Forest dataset"""
    print("=" * 60)
    print("📊 RANDOM FOREST MODEL")
    print("=" * 60)
    
    if not os.path.exists(DATA_PICKLE_PATH):
        print(f"⚠️  File not found: {DATA_PICKLE_PATH}")
        print("   Run 'python scripts/create_dataset.py' to create it.")
        return
    
    try:
        with open(DATA_PICKLE_PATH, 'rb') as f:
            rf_data = pickle.load(f)
        
        labels = rf_data['labels']
        unique_labels = sorted(set(labels))
        label_counts = Counter(labels)
        
        print(f"📁 File: {DATA_PICKLE_PATH}")
        print(f"📈 Total samples: {len(labels)}")
        print(f"🏷️  Total unique signs: {len(unique_labels)}")
        print(f"\n📋 Signs and sample counts:")
        
        for i, label in enumerate(unique_labels, 1):
            count = label_counts[label]
            print(f"  {i:2d}. {label:25s} - {count:4d} samples")
        
        print(f"\n✅ Random Forest dataset loaded successfully!")
        return unique_labels
    except Exception as e:
        print(f"❌ Error loading Random Forest data: {e}")
        return None


def check_transformer_data():
    """Check Transformer dataset"""
    print("\n")
    print("=" * 60)
    print("🤖 TRANSFORMER MODEL")
    print("=" * 60)
    
    if not os.path.exists(SEQUENCES_PICKLE_PATH):
        print(f"⚠️  File not found: {SEQUENCES_PICKLE_PATH}")
        print("   Run 'python scripts/collect_sequences.py' to create it.")
        return
    
    try:
        with open(SEQUENCES_PICKLE_PATH, 'rb') as f:
            tf_data = pickle.load(f)
        
        labels = tf_data['labels']
        sequences = tf_data['sequences']
        unique_labels = sorted(set(labels))
        label_counts = Counter(labels)
        
        print(f"📁 File: {SEQUENCES_PICKLE_PATH}")
        print(f"📈 Total sequences: {len(labels)}")
        print(f"🏷️  Total unique signs: {len(unique_labels)}")
        print(f"🎬 Frames per sequence: {len(sequences[0]) if sequences else 0}")
        print(f"\n📋 Signs and sequence counts:")
        
        for i, label in enumerate(unique_labels, 1):
            count = label_counts[label]
            print(f"  {i:2d}. {label:25s} - {count:4d} sequences")
        
        print(f"\n✅ Transformer dataset loaded successfully!")
        return unique_labels
    except Exception as e:
        print(f"❌ Error loading Transformer data: {e}")
        return None


def check_label_encoders():
    """Check label encoders"""
    print("\n")
    print("=" * 60)
    print("🔤 LABEL ENCODERS")
    print("=" * 60)
    
    # Random Forest model labels
    if os.path.exists(MODEL_PATH):
        try:
            with open(MODEL_PATH, 'rb') as f:
                rf_model_data = pickle.load(f)
            
            if 'labels' in rf_model_data:
                labels = sorted(rf_model_data['labels'])
                print(f"\n📊 Random Forest Model Labels ({len(labels)} signs):")
                for i, label in enumerate(labels, 1):
                    print(f"  {i:2d}. {label}")
        except Exception as e:
            print(f"⚠️  Could not read RF model labels: {e}")
    
    # Transformer label encoder
    if os.path.exists(TRANSFORMER_LABEL_ENCODER_PATH):
        try:
            with open(TRANSFORMER_LABEL_ENCODER_PATH, 'rb') as f:
                tf_encoder = pickle.load(f)
            
            labels = sorted(tf_encoder.classes_)
            print(f"\n🤖 Transformer Label Encoder ({len(labels)} signs):")
            for i, label in enumerate(labels, 1):
                print(f"  {i:2d}. {label}")
            
            print(f"\n✅ Transformer label encoder loaded successfully!")
        except Exception as e:
            print(f"❌ Error loading Transformer encoder: {e}")
    else:
        print(f"\n⚠️  File not found: {TRANSFORMER_LABEL_ENCODER_PATH}")
        print("   Run 'python scripts/train_transformer_pytorch.py' to create it.")


def compare_datasets(rf_labels, tf_labels):
    """Compare labels between datasets"""
    if rf_labels is None or tf_labels is None:
        return
    
    print("\n")
    print("=" * 60)
    print("🔄 DATASET COMPARISON")
    print("=" * 60)
    
    rf_set = set(rf_labels)
    tf_set = set(tf_labels)
    
    # Signs in both
    common = rf_set & tf_set
    if common:
        print(f"\n✅ Signs in BOTH models ({len(common)}):")
        for label in sorted(common):
            print(f"  • {label}")
    
    # Signs only in RF
    rf_only = rf_set - tf_set
    if rf_only:
        print(f"\n📊 Signs ONLY in Random Forest ({len(rf_only)}):")
        for label in sorted(rf_only):
            print(f"  • {label}")
    
    # Signs only in Transformer
    tf_only = tf_set - rf_set
    if tf_only:
        print(f"\n🤖 Signs ONLY in Transformer ({len(tf_only)}):")
        for label in sorted(tf_only):
            print(f"  • {label}")
    
    # Summary
    print(f"\n📈 Summary:")
    print(f"  Random Forest: {len(rf_set)} signs")
    print(f"  Transformer: {len(tf_set)} signs")
    print(f"  Common: {len(common)} signs")
    print(f"  RF only: {len(rf_only)} signs")
    print(f"  TF only: {len(tf_only)} signs")


def main():
    """Main function"""
    print("=" * 60)
    print("ASL Sign Language Translator - Label Checker")
    print("=" * 60)
    print()
    
    # Check datasets
    rf_labels = check_random_forest_data()
    tf_labels = check_transformer_data()
    
    # Check encoders
    check_label_encoders()
    
    # Compare datasets
    compare_datasets(rf_labels, tf_labels)
    
    # Tips
    print("\n")
    print("=" * 60)
    print("✅ Label check complete!")
    print("=" * 60)
    print("\n💡 Tips:")
    print("  - Add to Random Forest: python scripts/add_new_sign.py <sign_name>")
    print("  - Add to Transformer: python scripts/add_new_signs_tf.py <sign_name>")
    print("  - Remove signs: python scripts/remove_signs.py <sign_name>")
    print()


if __name__ == "__main__":
    main()
