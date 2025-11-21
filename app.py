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
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from config import (MODEL_PATH, DATA_DIR,
                    DATA_PICKLE_PATH, TEST_SIZE, RANDOM_STATE, N_ESTIMATORS, 
                    MAX_DEPTH, MIN_SAMPLES_SPLIT, MIN_SAMPLES_LEAF, 
                    CONFIDENCE_THRESHOLD_UNKNOWN)

# Import existing scripts
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from scripts.add_new_sign import process_new_sign_images

app = Flask(__name__)
CORS(app)

# Load model
try:
    with open(MODEL_PATH, 'rb') as f:
        model_dict = pickle.load(f)
    model = model_dict['model']
    print(f"✅ Model loaded successfully")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'success': False, 'error': 'Model not loaded'})
    
    try:
        data = request.json
        landmarks_list = data['landmarks']
        
        # Extract features from landmarks
        features = extract_features(landmarks_list)
        
        if features is None:
            return jsonify({
                'success': True,
                'prediction': 'No hand detected',
                'confidence': 0.0
            })
        
        # Make prediction
        prediction = model.predict([features])[0]
        probabilities = model.predict_proba([features])[0]
        confidence = float(np.max(probabilities) * 100)
        
        # Check confidence threshold
        if confidence < CONFIDENCE_THRESHOLD_UNKNOWN:
            prediction = 'Unknown sign'
        
        return jsonify({
            'success': True,
            'prediction': str(prediction),
            'confidence': confidence
        })
        
    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/model_info', methods=['GET'])
def model_info():
    if model is None:
        return jsonify({'success': False, 'error': 'Model not loaded'})
    
    try:
        num_signs = len(model.classes_)
        return jsonify({
            'success': True,
            'num_signs': num_signs,
            'signs': model.classes_.tolist()
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

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
    global model
    
    try:
        data = request.json
        sign_name = data['sign_name'].strip().lower()
        images_base64 = data['images']
        
        if not sign_name:
            return jsonify({'success': False, 'error': 'Sign name is required'})
        
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
        
        # Reuse process_new_sign_images from scripts/add_new_sign.py
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
        
        # Retrain model (reuses train_classifier.py logic)
        model = train_model_from_dataset()
        
        return jsonify({
            'success': True,
            'message': f'Successfully added {len(new_data)} samples for sign "{sign_name}"',
            'total_samples': len(combined_data),
            'total_signs': len(set(combined_labels))
        })
        
    except Exception as e:
        print(f"Error adding new sign: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    print("🚀 Starting ASL Sign Language Translator Web App")
    print("📡 Server running at http://localhost:5001")
    print("Press Ctrl+C to stop")
    app.run(debug=True, host='0.0.0.0', port=5001)
