import os
import pickle
import base64
from io import BytesIO
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import mediapipe as mp
import numpy as np
import cv2
from PIL import Image
import torch
import torch.nn.functional as F
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from config import (MODEL_PATH, DATA_DIR,
                    DATA_PICKLE_PATH, TEST_SIZE, RANDOM_STATE, N_ESTIMATORS, 
                    MAX_DEPTH, MIN_SAMPLES_SPLIT, MIN_SAMPLES_LEAF, 
                    CONFIDENCE_THRESHOLD_UNKNOWN, SEQUENCE_LENGTH, TRANSFORMER_MODEL_PATH, TRANSFORMER_LABEL_ENCODER_PATH)

# Import existing scripts
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from scripts.add_new_sign import process_new_sign_images
from transformer.transformer_model_pytorch import create_lightweight_transformer

app = Flask(__name__)
CORS(app)

# Model type: 'random_forest' or 'transformer'
CURRENT_MODEL_TYPE = 'random_forest'  # Default model

# PyTorch device
DEVICE = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

# Global variables
rf_model = None
transformer_model = None
transformer_encoder = None
frame_buffer = []  # For Transformer sequence buffering
last_transformer_prediction_time = 0  # Cooldown for Transformer predictions
last_transformer_prediction = None  # Store last prediction to show during cooldown

# Load Random Forest model
try:
    with open(MODEL_PATH, 'rb') as f:
        model_dict = pickle.load(f)
    rf_model = model_dict['model']
    print(f"✅ Random Forest model loaded successfully")
except Exception as e:
    print(f"⚠️  Random Forest model not found: {e}")

# Load Transformer model
try:
    checkpoint = torch.load(TRANSFORMER_MODEL_PATH, map_location=DEVICE)
    model_config = checkpoint['model_config']
    
    transformer_model = create_lightweight_transformer(
        sequence_length=model_config['sequence_length'],
        feature_dim=model_config['feature_dim'],
        num_classes=model_config['num_classes']
    )
    transformer_model.load_state_dict(checkpoint['model_state_dict'])
    transformer_model = transformer_model.to(DEVICE)
    transformer_model.eval()
    
    with open(TRANSFORMER_LABEL_ENCODER_PATH, 'rb') as f:
        transformer_encoder = pickle.load(f)
    
    print(f"✅ Transformer model loaded successfully")
    print(f"   Device: {DEVICE}")
    print(f"   Classes: {transformer_encoder.classes_}")
except Exception as e:
    print(f"⚠️  Transformer model not found: {e}")

# Frame buffer for Transformer (stores recent frames)
frame_buffer = []

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    global frame_buffer
    
    try:
        data = request.json
        landmarks_list = data['landmarks']
        
        # Extract features from landmarks
        features = extract_features(landmarks_list)
        
        if features is None:
            return jsonify({
                'success': True,
                'prediction': 'No hand detected',
                'confidence': 0.0,
                'model_type': CURRENT_MODEL_TYPE
            })
        
        # Route to appropriate model
        if CURRENT_MODEL_TYPE == 'transformer':
            return predict_transformer(features)
        else:
            return predict_random_forest(features)
        
    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({'success': False, 'error': str(e)})

def predict_random_forest(features):
    """Predict using Random Forest model (single frame)"""
    if rf_model is None:
        return jsonify({'success': False, 'error': 'Random Forest model not loaded'})
    
    try:
        # Make prediction
        prediction = rf_model.predict([features])[0]
        probabilities = rf_model.predict_proba([features])[0]
        confidence = float(np.max(probabilities) * 100)
        
        # Check confidence threshold
        if confidence < CONFIDENCE_THRESHOLD_UNKNOWN:
            prediction = 'Unknown sign'
        
        return jsonify({
            'success': True,
            'prediction': str(prediction),
            'confidence': confidence,
            'model_type': 'random_forest'
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

def predict_transformer(features):
    """Predict using Transformer model (sequence of frames)"""
    global frame_buffer, last_transformer_prediction_time, last_transformer_prediction
    
    if transformer_model is None or transformer_encoder is None:
        return jsonify({'success': False, 'error': 'Transformer model not loaded'})
    
    try:
        # Add frame to buffer
        frame_buffer.append(features)
        
        # Keep only last SEQUENCE_LENGTH frames
        if len(frame_buffer) > SEQUENCE_LENGTH:
            frame_buffer.pop(0)
        
        # Need full buffer to predict
        if len(frame_buffer) < SEQUENCE_LENGTH:
            return jsonify({
                'success': True,
                'prediction': f'Buffering... {len(frame_buffer)}/{SEQUENCE_LENGTH}',
                'confidence': 0.0,
                'model_type': 'transformer',
                'buffer_status': f'{len(frame_buffer)}/{SEQUENCE_LENGTH}'
            })
        
        # Cooldown mechanism: Only predict every 1.5 seconds to avoid too-sensitive detection
        import time
        current_time = time.time()
        if current_time - last_transformer_prediction_time < 2:
            # Return last prediction without recalculating
            if last_transformer_prediction:
                return jsonify({
                    'success': True,
                    'prediction': last_transformer_prediction['prediction'],
                    'confidence': last_transformer_prediction['confidence'],
                    'model_type': 'transformer',
                    'buffer_status': f'{len(frame_buffer)}/{SEQUENCE_LENGTH}',
                    'cooldown': True
                })
            else:
                return jsonify({
                    'success': True,
                    'prediction': 'Processing...',
                    'confidence': 0.0,
                    'model_type': 'transformer',
                    'buffer_status': f'{len(frame_buffer)}/{SEQUENCE_LENGTH}',
                    'cooldown': True
                })
        
        # Make prediction
        sequence_tensor = torch.FloatTensor([frame_buffer]).to(DEVICE)
        
        with torch.no_grad():
            logits = transformer_model(sequence_tensor)
            probs = F.softmax(logits, dim=1)
            predicted_class = torch.argmax(probs, dim=1).item()
            confidence = probs[0][predicted_class].item() * 100
            
            # Calculate entropy for uncertainty detection
            entropy = -torch.sum(probs * torch.log(probs + 1e-10)).item()
            max_entropy = np.log(len(probs[0]))
            normalized_entropy = entropy / max_entropy
        
        # Get label
        if confidence < CONFIDENCE_THRESHOLD_UNKNOWN or normalized_entropy > 0.8:
            prediction = 'Unknown sign'
        else:
            prediction = transformer_encoder.inverse_transform([predicted_class])[0]
        
        # Update last prediction time and store prediction
        last_transformer_prediction_time = current_time
        last_transformer_prediction = {
            'prediction': str(prediction),
            'confidence': float(confidence)
        }
        
        return jsonify({
            'success': True,
            'prediction': str(prediction),
            'confidence': float(confidence),
            'model_type': 'transformer',
            'buffer_status': f'{len(frame_buffer)}/{SEQUENCE_LENGTH}'
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/model_info', methods=['GET'])
def model_info():
    try:
        if CURRENT_MODEL_TYPE == 'transformer':
            if transformer_encoder is None:
                return jsonify({'success': False, 'error': 'Transformer model not loaded'})
            return jsonify({
                'success': True,
                'model_type': 'transformer',
                'num_signs': len(transformer_encoder.classes_),
                'signs': transformer_encoder.classes_.tolist(),
                'sequence_length': SEQUENCE_LENGTH,
                'device': str(DEVICE)
            })
        else:
            if rf_model is None:
                return jsonify({'success': False, 'error': 'Random Forest model not loaded'})
            return jsonify({
                'success': True,
                'model_type': 'random_forest',
                'num_signs': len(rf_model.classes_),
                'signs': rf_model.classes_.tolist()
            })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/toggle_model', methods=['POST'])
def toggle_model():
    """Toggle between Random Forest and Transformer models"""
    global CURRENT_MODEL_TYPE, frame_buffer, last_transformer_prediction, last_transformer_prediction_time
    
    try:
        data = request.json
        new_model_type = data.get('model_type', 'random_forest')
        
        if new_model_type not in ['random_forest', 'transformer']:
            return jsonify({'success': False, 'error': 'Invalid model type'})
        
        # Check if requested model is available
        if new_model_type == 'random_forest' and rf_model is None:
            return jsonify({'success': False, 'error': 'Random Forest model not loaded'})
        
        if new_model_type == 'transformer' and transformer_model is None:
            return jsonify({'success': False, 'error': 'Transformer model not loaded'})
        
        # Switch model
        CURRENT_MODEL_TYPE = new_model_type
        
        # Reset frame buffer and prediction when switching to transformer
        if new_model_type == 'transformer':
            frame_buffer = []
            last_transformer_prediction = None
            last_transformer_prediction_time = 0
        
        return jsonify({
            'success': True,
            'model_type': CURRENT_MODEL_TYPE,
            'message': f'Switched to {CURRENT_MODEL_TYPE} model'
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/reset_buffer', methods=['POST'])
def reset_buffer():
    """Reset the frame buffer for Transformer model"""
    global frame_buffer, last_transformer_prediction, last_transformer_prediction_time
    frame_buffer = []
    last_transformer_prediction = None
    last_transformer_prediction_time = 0
    return jsonify({'success': True, 'message': 'Buffer reset'})

def extract_features(landmarks_list):
    """Extract features from hand landmarks (up to 2 hands)"""
    if not landmarks_list or len(landmarks_list) == 0:
        return None
    
    data_aux = []
    
    # Process up to 2 hands
    for hand_idx in range(min(2, len(landmarks_list))):
        hand_landmarks = landmarks_list[hand_idx]
        
        # Extract x, y, z coordinates
        x_coords = [lm['x'] for lm in hand_landmarks]
        y_coords = [lm['y'] for lm in hand_landmarks]
        z_coords = [lm['z'] for lm in hand_landmarks]
        
        # Normalize coordinates
        min_x = min(x_coords)
        min_y = min(y_coords)
        min_z = min(z_coords)
        
        for lm in hand_landmarks:
            data_aux.append(lm['x'] - min_x)
            data_aux.append(lm['y'] - min_y)
            data_aux.append(lm['z'] - min_z)
    
    # Pad with zeros if only one hand detected
    if len(landmarks_list) == 1:
        data_aux.extend([0.0] * 63)
    
    return data_aux

def train_model_from_dataset():
    """Train model from dataset (reuses train_classifier.py logic)"""
    # Load dataset
    with open(DATA_PICKLE_PATH, 'rb') as f:
        data_dict = pickle.load(f)
    
    data_array = np.asarray(data_dict['data'])
    labels_array = np.asarray(data_dict['labels'])
    
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        data_array, labels_array,
        test_size=TEST_SIZE,
        shuffle=True,
        stratify=labels_array,
        random_state=RANDOM_STATE
    )
    
    # Train model
    new_model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        min_samples_split=MIN_SAMPLES_SPLIT,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    
    new_model.fit(X_train, y_train)
    
    # Save model
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump({'model': new_model}, f)
    
    return new_model

@app.route('/add_new_sign', methods=['POST'])
def add_new_sign():
    """Add a new sign to the dataset and retrain the model"""
    global rf_model, transformer_model, transformer_encoder, frame_buffer
    
    try:
        data = request.json
        sign_name = data['sign_name'].strip()
        model_type = data.get('model_type', CURRENT_MODEL_TYPE)
        
        if not sign_name:
            return jsonify({'success': False, 'error': 'Sign name is required'})
        
        if model_type == 'random_forest':
            # Random Forest: Add from images
            images_base64 = data.get('images', [])
            
            if not images_base64 or len(images_base64) == 0:
                return jsonify({'success': False, 'error': 'No images provided'})
            
            # Create directory for sign
            sign_dir = os.path.join(DATA_DIR, sign_name)
            if not os.path.exists(sign_dir):
                os.makedirs(sign_dir)
            
            # Save images from base64
            for idx, img_base64 in enumerate(images_base64):
                img_data = base64.b64decode(img_base64.split(',')[1])
                img = Image.open(BytesIO(img_data))
                img_array = np.array(img)
                
                # Convert RGB to BGR for OpenCV
                if len(img_array.shape) == 3 and img_array.shape[2] == 3:
                    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                else:
                    img_bgr = img_array
                
                # Save image
                img_path = os.path.join(sign_dir, f"{idx}.jpg")
                cv2.imwrite(img_path, img_bgr)
            
            # Process images and extract features
            new_data, new_labels = process_new_sign_images(sign_name)
            
            if not new_data:
                return jsonify({'success': False, 'error': 'No hands detected in images'})
            
            # Load existing dataset and combine
            if os.path.exists(DATA_PICKLE_PATH):
                with open(DATA_PICKLE_PATH, 'rb') as f:
                    dataset = pickle.load(f)
                existing_data = dataset['data']
                existing_labels = dataset['labels']
            else:
                existing_data = []
                existing_labels = []
            
            combined_data = existing_data + new_data
            combined_labels = existing_labels + new_labels
            
            # Save updated dataset
            with open(DATA_PICKLE_PATH, 'wb') as f:
                pickle.dump({'data': combined_data, 'labels': combined_labels}, f)
            
            # Retrain Random Forest model
            rf_model = train_model_from_dataset()
            
            return jsonify({
                'success': True,
                'message': f'Successfully added {len(new_data)} samples for sign "{sign_name}"',
                'total_samples': len(combined_data),
                'total_signs': len(set(combined_labels)),
                'model_type': 'random_forest'
            })
            
        else:  # transformer
            return jsonify({
                'success': False,
                'error': 'Adding signs to Transformer model requires collecting sequences. Use: python scripts/add_new_signs.py <sign_name>',
                'instructions': [
                    '1. Run: python scripts/add_new_signs.py ' + sign_name + ' --sequences 15',
                    '2. Run: python scripts/train_transformer_pytorch.py',
                    '3. Restart the Flask app to load the updated model'
                ]
            })
        
    except Exception as e:
        print(f"Error adding new sign: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 ASL SIGN LANGUAGE TRANSLATOR WEB APP")
    print("="*60)
    print(f"📡 Server: http://localhost:5001")
    print(f"🤖 Current Model: {CURRENT_MODEL_TYPE}")
    print(f"\nAvailable Models:")
    print(f"  • Random Forest: {'✅ Loaded' if rf_model else '❌ Not found'}")
    print(f"  • Transformer: {'✅ Loaded' if transformer_model else '❌ Not found'}")
    if transformer_model:
        print(f"    - Device: {DEVICE}")
        print(f"    - Signs: {len(transformer_encoder.classes_)}")
    print(f"\n📚 API Endpoints:")
    print(f"  POST /predict - Make prediction")
    print(f"  POST /toggle_model - Switch between models")
    print(f"  POST /reset_buffer - Reset Transformer buffer")
    print(f"  GET  /model_info - Get current model info")
    print(f"  POST /add_new_sign - Add new sign (Random Forest only)")
    print("="*60)
    print("Press Ctrl+C to stop\n")
    app.run(debug=True, host='0.0.0.0', port=5001)
