import pickle
import cv2
import mediapipe as mp
import numpy as np
import os
import sys
from datetime import datetime

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (MODEL_PATH, SAVE_DIR, CAMERA_INDEX, MIN_DETECTION_CONFIDENCE,
                    MAX_NUM_HANDS, CONFIDENCE_THRESHOLD_HIGH, CONFIDENCE_THRESHOLD_MEDIUM,
                    CONFIDENCE_THRESHOLD_UNKNOWN)

# Load the trained model
print("Loading model...")
with open(MODEL_PATH, 'rb') as f:
    model_dict = pickle.load(f)
model = model_dict['model']
print("Model loaded successfully!")

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=False, max_num_hands=MAX_NUM_HANDS, min_detection_confidence=MIN_DETECTION_CONFIDENCE)

# Initialize camera
cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    print("Error: Could not open camera. Trying camera index 1...")
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("Error: No camera available. Please check your camera connection and permissions.")
        exit(1)

# Create directory for saved frames
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

print("Camera initialized. Press 'q' to quit, 's' to save frame.")

# Variable to track current prediction for saving
current_prediction = "Unknown"

while True:
    ret, frame = cap.read()
    
    if not ret or frame is None:
        print("Error: Failed to capture frame from camera.")
        break
    
    # Convert BGR to RGB (no flip to match training data)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the frame with MediaPipe
    results = hands.process(frame_rgb)
    
    if results.multi_hand_landmarks:
        # Draw all detected hands
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )
        
        # Extract features from up to 2 hands for prediction
        data_aux = []
        all_x_coords = []
        all_y_coords = []
        
        for hand_idx in range(min(2, len(results.multi_hand_landmarks))):
            hand_landmarks = results.multi_hand_landmarks[hand_idx]
            x_coords = []
            y_coords = []
            z_coords = []
            
            for landmark in hand_landmarks.landmark:
                x_coords.append(landmark.x)
                y_coords.append(landmark.y)
                z_coords.append(landmark.z)
                all_x_coords.append(landmark.x)
                all_y_coords.append(landmark.y)
            
            # Normalize coordinates
            min_x, min_y, min_z = min(x_coords), min(y_coords), min(z_coords)
            for landmark in hand_landmarks.landmark:
                data_aux.append(landmark.x - min_x)
                data_aux.append(landmark.y - min_y)
                data_aux.append(landmark.z - min_z)
        
        # Pad with zeros if only one hand detected (to ensure consistent 126 features)
        if len(results.multi_hand_landmarks) == 1:
            data_aux.extend([0.0] * 63)
        
        # Make prediction
        prediction = model.predict([np.asarray(data_aux)])
        predicted_sign = prediction[0]
        
        # Get prediction probability
        prediction_proba = model.predict_proba([np.asarray(data_aux)])
        confidence = np.max(prediction_proba) * 100
        
        # Use all hands' coordinates for bounding box
        x_coords = all_x_coords
        y_coords = all_y_coords
        
        # Calculate bounding box for text placement
        h, w, _ = frame.shape
        x_min = int(min(x_coords) * w) - 10
        y_min = int(min(y_coords) * h) - 10
        x_max = int(max(x_coords) * w) + 10
        y_max = int(max(y_coords) * h) + 10
        
        # Check if confidence is too low (unknown sign)
        if confidence < CONFIDENCE_THRESHOLD_UNKNOWN:
            predicted_sign = "Unknown sign"
            current_prediction = "Unknown"
            color = (64, 64, 64)  # Dark gray - Unknown sign
            text = f"{predicted_sign} ({confidence:.1f}%)"
        else:
            current_prediction = predicted_sign
            # Color code based on confidence
            if confidence >= CONFIDENCE_THRESHOLD_HIGH:
                color = (0, 255, 0)  # Green - High confidence
            elif confidence >= CONFIDENCE_THRESHOLD_MEDIUM:
                color = (0, 165, 255)  # Orange - Medium confidence
            else:
                color = (0, 0, 255)  # Red - Low confidence
            text = f"{predicted_sign} ({confidence:.1f}%)"
        
        # Draw bounding box
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
        
        # Display prediction and confidence
        cv2.putText(frame, text, (x_min, y_min - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
    else:
        # No hand detected - display "Unknown"
        current_prediction = "NoHand"
        h, w, _ = frame.shape
        cv2.putText(frame, "Unknown - No hand detected", (10, h - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
    
    # Display instructions
    cv2.putText(frame, "Press 'q' to quit | 's' to save", (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    
    # Show the frame
    cv2.imshow('ASL Sign Language Detection', frame)
    
    # Handle key presses
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('q'):
        break
    elif key == ord('s'):
        # Save the current frame with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(SAVE_DIR, f"prediction_{current_prediction}_{timestamp}.jpg")
        cv2.imwrite(filename, frame)
        print(f"Frame saved: {filename}")

# Cleanup
cap.release()
cv2.destroyAllWindows()
hands.close()
print("Application closed.")
