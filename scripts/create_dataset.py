import os
import sys
import pickle

import cv2
import mediapipe as mp

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_DIR, MIN_DETECTION_CONFIDENCE, MAX_NUM_HANDS, DATA_PICKLE_PATH

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=MAX_NUM_HANDS, min_detection_confidence=MIN_DETECTION_CONFIDENCE)

data = []
labels = []

# Count total images for progress tracking
total_images = 0
for sign in os.listdir(DATA_DIR):
    sign_dir = os.path.join(DATA_DIR, sign)
    if os.path.isdir(sign_dir):
        total_images += len([f for f in os.listdir(sign_dir) if not f.startswith('.')])

print(f"Processing {total_images} images...")
processed = 0

for sign in os.listdir(DATA_DIR):
    sign_dir = os.path.join(DATA_DIR, sign)
    
    # Skip if not a directory (e.g., .DS_Store files)
    if not os.path.isdir(sign_dir):
        continue
    
    print(f"\nProcessing sign: {sign}")
    
    for image_file in os.listdir(sign_dir):
        # Skip hidden files
        if image_file.startswith('.'):
            continue
        
        processed += 1
        print(f"  [{processed}/{total_images}] Processing {image_file}...", end='\r')
            
        img_path = os.path.join(sign_dir, image_file)
        img = cv2.imread(img_path)
        
        if img is None:
            print(f"Warning: Could not read image {img_path}")
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
                
                # Collect all landmark coordinates
                for landmark in hand_landmarks.landmark:
                    x_coords.append(landmark.x)
                    y_coords.append(landmark.y)
                    z_coords.append(landmark.z)
                
                # Normalize coordinates relative to the bounding box
                min_x, min_y, min_z = min(x_coords), min(y_coords), min(z_coords)
                for landmark in hand_landmarks.landmark:
                    data_aux.append(landmark.x - min_x)
                    data_aux.append(landmark.y - min_y)
                    data_aux.append(landmark.z - min_z)
            
            # Pad with zeros if only one hand detected (to ensure consistent 126 features)
            if len(results.multi_hand_landmarks) == 1:
                data_aux.extend([0.0] * 63)  # Add 63 zeros for missing second hand
            
            data.append(data_aux)
            labels.append(sign)
        else:
            print(f"\n  Warning: No hand detected in {img_path}")
    
    print()  # New line after each sign

hands.close()

# Save the dataset
with open(DATA_PICKLE_PATH, 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)

print(f"Dataset created successfully! and saved as '{DATA_PICKLE_PATH}'")
print(f"Total samples: {len(data)}")
print(f"Labels: {set(labels)}")
print(f"Samples per label: {dict((label, labels.count(label)) for label in set(labels))}")
