"""
Add new signs to existing sequence data without starting from scratch
Loads existing sequences.pickle, collects new signs, and merges them
"""

import os
import sys
import cv2
import pickle
import argparse
import mediapipe as mp
import numpy as np

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MIN_DETECTION_CONFIDENCE, MAX_NUM_HANDS, SEQUENCE_LENGTH, SEQUENCES_PER_SIGN, SEQUENCES_PICKLE_PATH

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


def load_existing_sequences(filepath=SEQUENCES_PICKLE_PATH):
    """Load existing sequences if they exist"""
    if not os.path.exists(filepath):
        print(f"⚠️  No existing data found at {filepath}")
        print("   Starting with empty dataset")
        return {'sequences': [], 'labels': []}
    
    try:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        print(f"✓ Loaded existing data from: {filepath}")
        print(f"  Existing sequences: {len(data['sequences'])}")
        print(f"  Existing signs: {set(data['labels'])}")
        
        return data
    except Exception as e:
        print(f"❌ Error loading existing data: {e}")
        print("   Starting with empty dataset")
        return {'sequences': [], 'labels': []}


def extract_features_from_frame(results):
    """Extract hand landmarks from a single frame"""
    if not results.multi_hand_landmarks:
        return None
    
    data_aux = []
    
    # Process up to 2 hands
    for hand_idx in range(min(2, len(results.multi_hand_landmarks))):
        hand_landmarks = results.multi_hand_landmarks[hand_idx]
        
        x_coords = [lm.x for lm in hand_landmarks.landmark]
        y_coords = [lm.y for lm in hand_landmarks.landmark]
        z_coords = [lm.z for lm in hand_landmarks.landmark]
        
        min_x, min_y, min_z = min(x_coords), min(y_coords), min(z_coords)
        
        for lm in hand_landmarks.landmark:
            data_aux.append(lm.x - min_x)
            data_aux.append(lm.y - min_y)
            data_aux.append(lm.z - min_z)
    
    # Pad with zeros if only one hand detected
    if len(results.multi_hand_landmarks) == 1:
        data_aux.extend([0.0] * 63)
    
    return data_aux if len(data_aux) == 126 else None


def collect_sequences_for_sign(sign, num_sequences, sequence_length=SEQUENCE_LENGTH):
    """Collect sequences for a specific sign"""
    
    print(f"\n{'='*60}")
    print(f"Collecting sequences for sign: '{sign}'")
    print(f"{'='*60}")
    print(f"Sequences to collect: {num_sequences}")
    print(f"Frames per sequence: {sequence_length}")
    print(f"Total frames: {num_sequences * sequence_length}")
    print(f"\nInstructions:")
    print(f"1. Press SPACE to start recording a sequence")
    print(f"2. Perform the sign '{sign}' naturally")
    print(f"3. Sequence will be captured automatically")
    print(f"4. Press 'q' to quit early")
    print(f"{'='*60}\n")
    
    # Initialize MediaPipe
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=MAX_NUM_HANDS,
        min_detection_confidence=MIN_DETECTION_CONFIDENCE
    )
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    sequences = []
    current_sequence = []
    recording = False
    sequence_count = 0
    
    while sequence_count < num_sequences:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Flip frame horizontally for mirror view
        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process frame with MediaPipe
        results = hands.process(frame_rgb)
        
        # Draw hand landmarks
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                )
        
        # Display instructions
        status_text = f"Sign: {sign} | Sequence: {sequence_count + 1}/{num_sequences}"
        cv2.putText(frame, status_text, (10, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        if recording:
            # Extract features and add to sequence
            features = extract_features_from_frame(results)
            
            if features is not None:
                current_sequence.append(features)
                
                # Show recording progress
                progress = len(current_sequence)
                cv2.putText(frame, f"Recording: {progress}/{sequence_length}", (10, 80),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.circle(frame, (frame.shape[1] - 50, 50), 20, (0, 0, 255), -1)
                
                # Check if sequence is complete
                if len(current_sequence) >= sequence_length:
                    sequences.append(current_sequence[:sequence_length])
                    sequence_count += 1
                    print(f"✓ Sequence {sequence_count}/{num_sequences} captured!")
                    current_sequence = []
                    recording = False
            else:
                # No hand detected, show warning
                cv2.putText(frame, "No hand detected!", (10, 120),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            cv2.putText(frame, "Press SPACE to start recording", (10, 80),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        cv2.imshow(f'Collecting: {sign}', frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' ') and not recording:
            recording = True
            current_sequence = []
            print(f"Recording sequence {sequence_count + 1}...")
        elif key == ord('q'):
            print("\nCollection interrupted by user")
            break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    hands.close()
    
    return sequences


def save_sequences(data_dict, filepath=SEQUENCES_PICKLE_PATH):
    """Save sequences to pickle file"""
    with open(filepath, 'wb') as f:
        pickle.dump(data_dict, f)
    print(f"\n✓ Sequences saved to: {filepath}")


def main():
    """Main function to add new signs"""
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Add new signs to existing ASL sequence data')
    parser.add_argument('signs', nargs='+', help='New signs to add (e.g., C D hello)')
    parser.add_argument('-n', '--sequences', type=int, default=SEQUENCES_PER_SIGN,
                        help=f'Number of sequences per sign (default: {SEQUENCES_PER_SIGN})')
    parser.add_argument('-l', '--length', type=int, default=SEQUENCE_LENGTH,
                        help=f'Frames per sequence (default: {SEQUENCE_LENGTH})')
    parser.add_argument('-f', '--file', type=str, default=SEQUENCES_PICKLE_PATH,
                        help='Path to sequences file (default: ./models/tf_model/sequences.pickle)')
    
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print(f"ADD NEW SIGNS TO ASL DATASET")
    print(f"{'='*60}")
    
    # Load existing data
    existing_data = load_existing_sequences(args.file)
    existing_sequences = existing_data['sequences']
    existing_labels = existing_data['labels']
    existing_signs = set(existing_labels)
    
    # Check for duplicates
    new_signs = args.signs
    duplicates = [sign for sign in new_signs if sign in existing_signs]
    
    if duplicates:
        print(f"\n⚠️  WARNING: The following signs already exist: {duplicates}")
        response = input("Do you want to add more sequences for these signs? (y/n): ")
        if response.lower() != 'y':
            new_signs = [sign for sign in new_signs if sign not in existing_signs]
            if not new_signs:
                print("No new signs to add. Exiting.")
                return
    
    print(f"\nSigns to add: {new_signs}")
    print(f"Sequences per sign: {args.sequences}")
    print(f"Frames per sequence: {args.length}")
    print(f"{'='*60}\n")
    
    # Collect new sequences
    new_sequences = []
    new_labels = []
    
    for sign in new_signs:
        sequences = collect_sequences_for_sign(sign, args.sequences, args.length)
        new_sequences.extend(sequences)
        new_labels.extend([sign] * len(sequences))
        
        print(f"\n✓ Collected {len(sequences)} sequences for '{sign}'")
    
    # Merge with existing data
    all_sequences = existing_sequences + new_sequences
    all_labels = existing_labels + new_labels
    
    # Save merged data
    merged_data = {
        'sequences': all_sequences,
        'labels': all_labels
    }
    save_sequences(merged_data, args.file)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"DATASET UPDATE COMPLETE!")
    print(f"{'='*60}")
    print(f"Previous total: {len(existing_sequences)} sequences")
    print(f"New sequences: {len(new_sequences)} sequences")
    print(f"Updated total: {len(all_sequences)} sequences")
    print(f"\nAll signs in dataset: {sorted(set(all_labels))}")
    print(f"\nSequences per sign:")
    for sign in sorted(set(all_labels)):
        count = all_labels.count(sign)
        print(f"  {sign}: {count}")
    print(f"{'='*60}\n")
    print(f"Next step: Run 'python scripts/train_transformer_pytorch.py' to retrain the model")


if __name__ == "__main__":
    main()
