#!/bin/bash

# Script to check what labels/signs are in each model's dataset
# Usage: ./check_labels.sh

echo "=================================================="
echo "ASL Sign Language Translator - Label Checker"
echo "=================================================="
echo ""

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# Create Python script to check labels
python3 << 'EOF'
import pickle
import os
from collections import Counter

print("🔍 Checking labels in datasets...\n")

# Check Random Forest model
print("=" * 60)
print("📊 RANDOM FOREST MODEL")
print("=" * 60)

rf_data_path = "./models/rf_model/data.pickle"
if os.path.exists(rf_data_path):
    try:
        with open(rf_data_path, 'rb') as f:
            rf_data = pickle.load(f)
        
        labels = rf_data['labels']
        unique_labels = sorted(set(labels))
        label_counts = Counter(labels)
        
        print(f"📁 File: {rf_data_path}")
        print(f"📈 Total samples: {len(labels)}")
        print(f"🏷️  Total unique signs: {len(unique_labels)}")
        print(f"\n📋 Signs and sample counts:")
        
        for i, label in enumerate(unique_labels, 1):
            count = label_counts[label]
            print(f"  {i:2d}. {label:25s} - {count:4d} samples")
        
        print(f"\n✅ Random Forest dataset loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading Random Forest data: {e}")
else:
    print(f"⚠️  File not found: {rf_data_path}")
    print("   Run 'python scripts/create_dataset.py' to create it.")

print("\n")

# Check Transformer model
print("=" * 60)
print("🤖 TRANSFORMER MODEL")
print("=" * 60)

tf_data_path = "./models/tf_model/sequences.pickle"
if os.path.exists(tf_data_path):
    try:
        with open(tf_data_path, 'rb') as f:
            tf_data = pickle.load(f)
        
        labels = tf_data['labels']
        sequences = tf_data['sequences']
        unique_labels = sorted(set(labels))
        label_counts = Counter(labels)
        
        print(f"📁 File: {tf_data_path}")
        print(f"📈 Total sequences: {len(labels)}")
        print(f"🏷️  Total unique signs: {len(unique_labels)}")
        print(f"🎬 Frames per sequence: {len(sequences[0]) if sequences else 0}")
        print(f"\n📋 Signs and sequence counts:")
        
        for i, label in enumerate(unique_labels, 1):
            count = label_counts[label]
            print(f"  {i:2d}. {label:25s} - {count:4d} sequences")
        
        print(f"\n✅ Transformer dataset loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading Transformer data: {e}")
else:
    print(f"⚠️  File not found: {tf_data_path}")
    print("   Run 'python scripts/collect_sequences.py' to create it.")

print("\n")

# Check label encoders
print("=" * 60)
print("🔤 LABEL ENCODERS")
print("=" * 60)

# Random Forest label encoder (if exists)
rf_model_path = "./models/rf_model/model.pickle"
if os.path.exists(rf_model_path):
    try:
        with open(rf_model_path, 'rb') as f:
            rf_model_data = pickle.load(f)
        
        if 'labels' in rf_model_data:
            labels = sorted(rf_model_data['labels'])
            print(f"\n📊 Random Forest Model Labels ({len(labels)} signs):")
            for i, label in enumerate(labels, 1):
                print(f"  {i:2d}. {label}")
    except Exception as e:
        print(f"⚠️  Could not read RF model labels: {e}")

# Transformer label encoder
tf_encoder_path = "./models/tf_model/label_encoder.pickle"
if os.path.exists(tf_encoder_path):
    try:
        with open(tf_encoder_path, 'rb') as f:
            tf_encoder = pickle.load(f)
        
        labels = sorted(tf_encoder.classes_)
        print(f"\n🤖 Transformer Label Encoder ({len(labels)} signs):")
        for i, label in enumerate(labels, 1):
            print(f"  {i:2d}. {label}")
        
        print(f"\n✅ Transformer label encoder loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading Transformer encoder: {e}")
else:
    print(f"\n⚠️  File not found: {tf_encoder_path}")
    print("   Run 'python scripts/train_transformer_pytorch.py' to create it.")

print("\n")
print("=" * 60)
print("✅ Label check complete!")
print("=" * 60)

EOF

echo ""
echo "💡 Tips:"
echo "  - To add signs to Random Forest: python scripts/add_new_sign.py <sign_name>"
echo "  - To add signs to Transformer: python scripts/add_new_signs_tf.py <sign_name>"
echo "  - To remove signs: python scripts/remove_signs.py <sign_name>"
echo ""
