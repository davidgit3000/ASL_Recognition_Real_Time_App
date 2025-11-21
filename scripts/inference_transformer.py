"""
Real-time inference using PyTorch Transformer model
Captures sequences and predicts ASL signs
"""

import os
import sys
import cv2
import pickle
import mediapipe as mp
import numpy as np
import torch
import torch.nn.functional as F

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from transformer.transformer_model_pytorch import create_lightweight_transformer
from config import (
    MIN_DETECTION_CONFIDENCE,
    MAX_NUM_HANDS,
    SEQUENCE_LENGTH,
    CONFIDENCE_THRESHOLD_UNKNOWN, 
    TRANSFORMER_MODEL_PATH, 
    TRANSFORMER_LABEL_ENCODER_PATH
)

# Settings
MODEL_PATH = TRANSFORMER_MODEL_PATH
ENCODER_PATH = TRANSFORMER_LABEL_ENCODER_PATH
DEVICE = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')


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


def load_model_and_encoder():
    """Load trained model and label encoder"""
    
    # Load checkpoint
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    model_config = checkpoint['model_config']
    
    # Create model
    model = create_lightweight_transformer(
        sequence_length=model_config['sequence_length'],
        feature_dim=model_config['feature_dim'],
        num_classes=model_config['num_classes']
    )
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(DEVICE)
    model.eval()
    
    # Load label encoder
    with open(ENCODER_PATH, 'rb') as f:
        label_encoder = pickle.load(f)
    
    print(f"✓ Model loaded from: {MODEL_PATH}")
    print(f"✓ Device: {DEVICE}")
    print(f"✓ Classes: {label_encoder.classes_}")
    
    return model, label_encoder


def predict_sequence(model, label_encoder, sequence, confidence_threshold=CONFIDENCE_THRESHOLD_UNKNOWN):
    """Predict sign from a sequence of frames"""
    
    # Convert to tensor
    sequence_tensor = torch.FloatTensor([sequence]).to(DEVICE)
    
    # Predict
    with torch.no_grad():
        logits = model(sequence_tensor)
        probs = F.softmax(logits, dim=1)
        predicted_class = torch.argmax(probs, dim=1).item()
        confidence = probs[0][predicted_class].item() * 100
        
        # Calculate entropy for uncertainty detection
        # High entropy = model is uncertain (probabilities are similar)
        entropy = -torch.sum(probs * torch.log(probs + 1e-10)).item()
        max_entropy = np.log(len(probs[0]))  # Maximum possible entropy
        normalized_entropy = entropy / max_entropy  # 0 to 1
        
        # If entropy is high (>0.8), model is very uncertain
        # This happens when probabilities are similar (e.g., 55% vs 45%)
        uncertainty_threshold = 0.8
    
    # Get label
    # Show "Unknown" if:
    # 1. Confidence is below threshold, OR
    # 2. Model is very uncertain (high entropy)
    if confidence < confidence_threshold or normalized_entropy > uncertainty_threshold:
        prediction = "Unknown"
    else:
        prediction = label_encoder.inverse_transform([predicted_class])[0]
    
    return prediction, confidence


def main():
    """Main inference loop"""
    
    print(f"\n{'='*60}")
    print(f"ASL TRANSFORMER INFERENCE")
    print(f"{'='*60}\n")
    
    # Load model
    try:
        model, label_encoder = load_model_and_encoder()
    except FileNotFoundError:
        print("❌ Model not found!")
        print("   Train a model first: python scripts/train_transformer_pytorch.py")
        return
    
    print(f"\nSequence length: {SEQUENCE_LENGTH} frames")
    print(f"Buffer will fill before predictions start.\n")
    print(f"{'='*60}")
    print(f"Controls:")
    print(f"  - Perform signs naturally")
    print(f"  - Press 'q' to quit")
    print(f"  - Press 'r' to reset buffer")
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
    
    # Frame buffer for sequences
    frame_buffer = []
    
    # Prediction state
    current_prediction = "Collecting frames..."
    current_confidence = 0.0
    
    print("Starting camera...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
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
            
            # Extract features
            features = extract_features_from_frame(results)
            
            if features is not None:
                # Add to buffer
                frame_buffer.append(features)
                
                # Keep only last SEQUENCE_LENGTH frames
                if len(frame_buffer) > SEQUENCE_LENGTH:
                    frame_buffer.pop(0)
                
                # Predict if buffer is full
                if len(frame_buffer) == SEQUENCE_LENGTH:
                    prediction, confidence = predict_sequence(
                        model, label_encoder, frame_buffer
                    )
                    current_prediction = prediction
                    current_confidence = confidence
        
        # Display info
        buffer_status = f"Buffer: {len(frame_buffer)}/{SEQUENCE_LENGTH}"
        
        # Color based on confidence
        if current_confidence >= 80:
            color = (0, 255, 0)  # Green
        elif current_confidence >= 60:
            color = (0, 165, 255)  # Orange
        else:
            color = (0, 0, 255)  # Red
        
        # Draw prediction
        cv2.putText(frame, f"Prediction: {current_prediction}", (10, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
        cv2.putText(frame, f"Confidence: {current_confidence:.1f}%", (10, 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(frame, buffer_status, (10, 120),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Show frame
        cv2.imshow('ASL Transformer Inference', frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            frame_buffer = []
            current_prediction = "Buffer reset"
            current_confidence = 0.0
            print("Buffer reset")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    hands.close()
    
    print("\n✓ Inference stopped")


if __name__ == "__main__":
    main()
