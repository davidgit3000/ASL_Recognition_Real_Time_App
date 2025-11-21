"""
Collect sequences of frames for temporal sign recognition
Instead of single frames, collect 30-60 frames per sign sample
"""

import os
import sys
import cv2
import pickle
import mediapipe as mp
import numpy as np
import argparse

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_DIR, MIN_DETECTION_CONFIDENCE, MAX_NUM_HANDS, SEQUENCE_LENGTH, SEQUENCES_PER_SIGN, SEQUENCES_PICKLE_PATH

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


def collect_sequences_for_sign(sign_name, num_sequences=SEQUENCES_PER_SIGN):
    """Collect sequences of frames for a specific sign"""
    
    print(f"\n{'='*60}")
    print(f"Collecting sequences for sign: '{sign_name}'")
    print(f"{'='*60}")
    print(f"Sequences to collect: {num_sequences}")
    print(f"Frames per sequence: {SEQUENCE_LENGTH}")
    print(f"Total frames: {num_sequences * SEQUENCE_LENGTH}")
    print(f"\nInstructions:")
    print(f"1. Press SPACE to start recording a sequence")
    print(f"2. Perform the sign '{sign_name}' naturally")
    print(f"3. Sequence will be captured automatically")
    print(f"4. Press 'q' to quit early")
    print(f"{'='*60}\n")
    
    # Initialize MediaPipe
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
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
    sequences_collected = 0
    
    while sequences_collected < num_sequences:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        
        # Flip frame horizontally for mirror view
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process frame with MediaPipe
        results = hands.process(frame_rgb)
        
        # Draw hand landmarks
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                )
        
        # Display status
        status_text = f"Sign: {sign_name} | Collected: {sequences_collected}/{num_sequences}"
        if recording:
            status_text += f" | Recording: {len(current_sequence)}/{SEQUENCE_LENGTH}"
            cv2.putText(frame, "RECORDING", (50, 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
        else:
            cv2.putText(frame, "Press SPACE to start", (50, 100),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.putText(frame, status_text, (50, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Show frame
        cv2.imshow(f'Collecting: {sign_name}', frame)
        
        # Handle recording
        if recording:
            features = extract_features_from_frame(results)
            if features is not None:
                current_sequence.append(features)
                
                # Check if sequence is complete
                if len(current_sequence) >= SEQUENCE_LENGTH:
                    sequences.append(current_sequence[:SEQUENCE_LENGTH])
                    sequences_collected += 1
                    current_sequence = []
                    recording = False
                    print(f"✓ Sequence {sequences_collected}/{num_sequences} captured!")
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' ') and not recording:
            recording = True
            current_sequence = []
            print(f"Recording sequence {sequences_collected + 1}...")
        elif key == ord('q'):
            print("\nQuitting early...")
            break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    hands.close()
    
    return sequences


def save_sequences(sequences_dict, output_path=SEQUENCES_PICKLE_PATH):
    """Save collected sequences to a pickle file"""
    with open(output_path, 'wb') as f:
        pickle.dump(sequences_dict, f)
    print(f"\n✓ Sequences saved to: {output_path}")


def main():
    """Main function to collect sequences for multiple signs"""
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Collect sequences for ASL signs')
    parser.add_argument('signs', nargs='*', help='Signs to collect (e.g., A B C hello)')
    parser.add_argument('-n', '--sequences', type=int, default=SEQUENCES_PER_SIGN,
                        help=f'Number of sequences per sign (default: {SEQUENCES_PER_SIGN})')
    parser.add_argument('-l', '--length', type=int, default=SEQUENCE_LENGTH,
                        help=f'Frames per sequence (default: {SEQUENCE_LENGTH})')
    
    args = parser.parse_args()
    
    # Signs to collect
    if args.signs:
        signs_to_collect = args.signs
    else:
        from config import SIGNS_TO_COLLECT
        signs_to_collect = SIGNS_TO_COLLECT
    
    num_sequences = args.sequences
    
    print(f"\n{'='*60}")
    print(f"ASL SEQUENCE COLLECTION")
    print(f"{'='*60}")
    print(f"Signs to collect: {signs_to_collect}")
    print(f"Sequences per sign: {num_sequences}")
    print(f"Frames per sequence: {args.length}")
    print(f"{'='*60}\n")
    
    all_sequences = {}
    all_labels = {}
    
    for sign in signs_to_collect:
        sequences = collect_sequences_for_sign(sign, num_sequences)
        all_sequences[sign] = sequences
        all_labels[sign] = [sign] * len(sequences)
        
        print(f"\n✓ Collected {len(sequences)} sequences for '{sign}'")
    
    # Convert to flat lists
    flat_sequences = []
    flat_labels = []
    for sign in signs_to_collect:
        flat_sequences.extend(all_sequences[sign])
        flat_labels.extend(all_labels[sign])
    
    # Save to pickle
    data_dict = {
        'sequences': flat_sequences,
        'labels': flat_labels
    }
    save_sequences(data_dict, SEQUENCES_PICKLE_PATH)
    
    print(f"\n{'='*60}")
    print(f"COLLECTION COMPLETE!")
    print(f"{'='*60}")
    print(f"Total sequences: {len(flat_sequences)}")
    print(f"Total signs: {len(signs_to_collect)}")
    print(f"Sequences per sign: {dict((sign, flat_labels.count(sign)) for sign in signs_to_collect)}")
    print(f"{'='*60}\n")
    print(f"Next step: Run 'python scripts/train_transformer_pytorch.py' to train the model")


if __name__ == "__main__":
    main()
