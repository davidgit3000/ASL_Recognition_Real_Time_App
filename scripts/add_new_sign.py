import os
import sys
import pickle
import cv2
import mediapipe as mp

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_DIR, MIN_DETECTION_CONFIDENCE, MAX_NUM_HANDS, DATA_PICKLE_PATH

def process_new_sign_images(sign_name):
    """Process images for a new sign and extract landmarks"""
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=True, max_num_hands=MAX_NUM_HANDS, 
                           min_detection_confidence=MIN_DETECTION_CONFIDENCE)
    
    data = []
    labels = []
    
    sign_dir = os.path.join(DATA_DIR, sign_name)
    
    if not os.path.isdir(sign_dir):
        print(f"Error: Directory {sign_dir} does not exist!")
        return None, None
    
    print(f"\nProcessing sign: {sign_name}")
    image_files = [f for f in os.listdir(sign_dir) if not f.startswith('.')]
    total_images = len(image_files)
    
    for idx, image_file in enumerate(image_files, 1):
        print(f"  [{idx}/{total_images}] Processing {image_file}...", end='\r')
        
        img_path = os.path.join(sign_dir, image_file)
        img = cv2.imread(img_path)
        
        if img is None:
            print(f"\n  Warning: Could not read image {img_path}")
            continue
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)
        
        if results.multi_hand_landmarks:
            data_aux = []
            
            # Process up to 2 hands
            for hand_idx in range(min(2, len(results.multi_hand_landmarks))):
                hand_landmarks = results.multi_hand_landmarks[hand_idx]
                x_coords = []
                y_coords = []
                z_coords = []
                
                for landmark in hand_landmarks.landmark:
                    x_coords.append(landmark.x)
                    y_coords.append(landmark.y)
                    z_coords.append(landmark.z)
                
                min_x, min_y, min_z = min(x_coords), min(y_coords), min(z_coords)
                for landmark in hand_landmarks.landmark:
                    data_aux.append(landmark.x - min_x)
                    data_aux.append(landmark.y - min_y)
                    data_aux.append(landmark.z - min_z)
            
            # Pad with zeros if only one hand detected (to ensure consistent 126 features)
            if len(results.multi_hand_landmarks) == 1:
                data_aux.extend([0.0] * 63)  # Add 63 zeros for missing second hand
            
            data.append(data_aux)
            labels.append(sign_name)
        else:
            print(f"\n  Warning: No hand detected in {img_path}")
    
    print()  # New line after processing
    hands.close()
    
    return data, labels


def add_new_signs_to_dataset(new_signs):
    """Add new signs to existing dataset"""
    
    # Load existing dataset
    if os.path.exists(DATA_PICKLE_PATH):
        print(f"Loading existing dataset from '{DATA_PICKLE_PATH}'...")
        with open(DATA_PICKLE_PATH, 'rb') as f:
            dataset = pickle.load(f)
        
        existing_data = dataset['data']
        existing_labels = dataset['labels']
        
        print(f"Existing dataset: {len(existing_data)} samples")
        print(f"Existing labels: {set(existing_labels)}")
    else:
        print("No existing dataset found. Creating new dataset...")
        existing_data = []
        existing_labels = []
    
    # Process new signs
    all_new_data = []
    all_new_labels = []
    
    for sign in new_signs:
        new_data, new_labels = process_new_sign_images(sign)
        
        if new_data and new_labels:
            all_new_data.extend(new_data)
            all_new_labels.extend(new_labels)
            print(f"Added {len(new_data)} samples for sign '{sign}'")
        else:
            print(f"No data added for sign '{sign}'")
    
    if not all_new_data:
        print("\nNo new data to add!")
        return
    
    # Combine datasets
    combined_data = existing_data + all_new_data
    combined_labels = existing_labels + all_new_labels
    
    # Save updated dataset
    with open(DATA_PICKLE_PATH, 'wb') as f:
        pickle.dump({'data': combined_data, 'labels': combined_labels}, f)
    
    print(f"\n{'='*60}")
    print(f"Dataset updated successfully!")
    print(f"Total samples: {len(combined_data)}")
    print(f"All labels: {set(combined_labels)}")
    print(f"Samples per label: {dict((label, combined_labels.count(label)) for label in set(combined_labels))}")
    print(f"Saved to: '{DATA_PICKLE_PATH}'")
    print(f"{'='*60}")
    print(f"\nNext step: Run 'python scripts/train_classifier.py' to retrain the model")


if __name__ == "__main__":
    # Specify which new signs to add
    # You can modify this list or pass as command line arguments
    
    if len(sys.argv) > 1:
        # Use command line arguments if provided
        new_signs_to_add = sys.argv[1:]
    else:
        # Default: check config for new signs
        from config import SIGNS_TO_COLLECT
        
        # Load existing dataset to find which signs are already there
        if os.path.exists(DATA_PICKLE_PATH):
            with open(DATA_PICKLE_PATH, 'rb') as f:
                dataset = pickle.load(f)
            existing_signs = set(dataset['labels'])
        else:
            existing_signs = set()
        
        # Find new signs
        new_signs_to_add = [sign for sign in SIGNS_TO_COLLECT if sign not in existing_signs]
        
        if not new_signs_to_add:
            print("No new signs to add!")
            print(f"Existing signs: {existing_signs}")
            print(f"Signs in config: {set(SIGNS_TO_COLLECT)}")
            exit(0)
        
        print(f"New signs to add: {new_signs_to_add}")
    
    add_new_signs_to_dataset(new_signs_to_add)
