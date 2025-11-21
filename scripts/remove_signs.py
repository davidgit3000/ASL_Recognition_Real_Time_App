"""
Remove specific signs from existing sequence data
"""

import os
import sys
import pickle
import argparse

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import SEQUENCES_PICKLE_PATH

def load_sequences(filepath=SEQUENCES_PICKLE_PATH):
    """Load existing sequences"""
    if not os.path.exists(filepath):
        print(f"❌ File not found: {filepath}")
        return None
    
    try:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        print(f"✓ Loaded data from: {filepath}")
        print(f"  Total sequences: {len(data['sequences'])}")
        print(f"  Signs: {sorted(set(data['labels']))}")
        
        return data
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None


def remove_signs(data, signs_to_remove):
    """Remove specific signs from dataset"""
    
    sequences = data['sequences']
    labels = data['labels']
    
    # Count before removal
    before_count = len(sequences)
    before_signs = set(labels)
    
    print(f"\n{'='*60}")
    print(f"REMOVING SIGNS")
    print(f"{'='*60}")
    
    # Filter out the signs to remove
    filtered_sequences = []
    filtered_labels = []
    removed_count = 0
    
    for seq, label in zip(sequences, labels):
        if label in signs_to_remove:
            removed_count += 1
        else:
            filtered_sequences.append(seq)
            filtered_labels.append(label)
    
    # Create new data dict
    new_data = {
        'sequences': filtered_sequences,
        'labels': filtered_labels
    }
    
    # Print summary
    after_count = len(filtered_sequences)
    after_signs = set(filtered_labels)
    
    print(f"\nBefore removal:")
    print(f"  Total sequences: {before_count}")
    print(f"  Signs: {sorted(before_signs)}")
    
    print(f"\nRemoved:")
    for sign in signs_to_remove:
        count = labels.count(sign)
        if count > 0:
            print(f"  {sign}: {count} sequences")
        else:
            print(f"  {sign}: Not found in dataset")
    
    print(f"\nAfter removal:")
    print(f"  Total sequences: {after_count}")
    print(f"  Signs: {sorted(after_signs)}")
    print(f"  Removed: {removed_count} sequences")
    
    print(f"\nSequences per sign:")
    for sign in sorted(after_signs):
        count = filtered_labels.count(sign)
        print(f"  {sign}: {count}")
    
    return new_data


def save_sequences(data, filepath=SEQUENCES_PICKLE_PATH, backup=True):
    """Save sequences to pickle file with optional backup"""
    
    # Create backup if requested
    if backup and os.path.exists(filepath):
        backup_path = filepath + '.backup'
        import shutil
        shutil.copy2(filepath, backup_path)
        print(f"\n✓ Backup created: {backup_path}")
    
    # Save new data
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)
    
    print(f"✓ Updated data saved to: {filepath}")


def main():
    """Main function to remove signs"""
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Remove specific signs from ASL sequence data')
    parser.add_argument('signs', nargs='+', help='Signs to remove (e.g., rest "bad sign" A)')
    parser.add_argument('-f', '--file', type=str, default=SEQUENCES_PICKLE_PATH,
                        help='Path to sequences file (default: ./models/tf_model/sequences.pickle)')
    parser.add_argument('--no-backup', action='store_true',
                        help='Do not create backup before modifying')
    
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print(f"REMOVE SIGNS FROM DATASET")
    print(f"{'='*60}\n")
    
    # Load existing data
    data = load_sequences(args.file)
    if data is None:
        return
    
    # Check if signs exist
    existing_labels = set(data['labels'])
    signs_to_remove = args.signs
    
    not_found = [sign for sign in signs_to_remove if sign not in existing_labels]
    if not_found:
        print(f"\n⚠️  WARNING: The following signs were not found in the dataset:")
        for sign in not_found:
            print(f"  - {sign}")
        print(f"\nAvailable signs: {sorted(existing_labels)}")
    
    found_signs = [sign for sign in signs_to_remove if sign in existing_labels]
    if not found_signs:
        print("\n❌ No valid signs to remove. Exiting.")
        return
    
    # Confirm removal
    print(f"\n⚠️  You are about to remove the following signs:")
    for sign in found_signs:
        count = data['labels'].count(sign)
        print(f"  - {sign}: {count} sequences")
    
    response = input("\nAre you sure you want to continue? (y/n): ")
    if response.lower() != 'y':
        print("Operation cancelled.")
        return
    
    # Remove signs
    new_data = remove_signs(data, found_signs)
    
    # Check if dataset is empty
    if len(new_data['sequences']) == 0:
        print("\n❌ ERROR: Removal would result in empty dataset!")
        print("   Operation cancelled.")
        return
    
    # Save updated data
    save_sequences(new_data, args.file, backup=not args.no_backup)
    
    print(f"\n{'='*60}")
    print(f"REMOVAL COMPLETE!")
    print(f"{'='*60}\n")
    print(f"Next step: Run 'python scripts/train_transformer_pytorch.py' to retrain the model")


if __name__ == "__main__":
    main()
