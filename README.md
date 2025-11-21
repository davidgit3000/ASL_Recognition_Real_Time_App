# 🤟 ASL Sign Language Translator

A comprehensive machine learning project for real-time American Sign Language (ASL) recognition using **dual model architecture**: Random Forest for static signs and PyTorch Transformer for dynamic motion-based signs.

## ✨ Features

- 🎯 **Dual Model System**: Toggle between Random Forest (static signs) and Transformer (dynamic signs)
- 📹 **Real-Time Detection**: Live camera feed with MediaPipe hand tracking
- 🔊 **Text-to-Speech**: Automatic sentence building and voice output
- 🌐 **Web Interface**: Beautiful Flask-based UI with model switching
- 🤖 **PyTorch Transformer**: Sequence-based recognition for motion signs
- 🌲 **Random Forest**: Fast, accurate static sign classification
- ➕ **Add New Signs**: Web UI for Random Forest, CLI for Transformer
- 📊 **Model Analytics**: Real-time confidence scores and buffer status
- 🎨 **Modern UI**: Responsive design with TailwindCSS

## 🏗️ Project Structure

```
asl_sign_model/
├── app.py                          # Flask web application (dual model)
├── config.py                       # Central configuration
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
├── CHANGELOG.md                    # Version history
│
├── shell_cmd/                      # Shell scripts
│   ├── run_webapp.sh               # Quick start Flask app
│   └── check_labels.sh             # Check dataset labels
│
├── scripts/                        # Training & data collection
│   ├── collect_images.py           # Collect static images (RF)
│   ├── collect_sequences.py        # Collect sequences (Transformer)
│   ├── add_new_sign.py             # Add signs to RF dataset
│   ├── add_new_signs_tf.py         # Add signs to Transformer dataset
│   ├── remove_signs.py             # Remove signs from dataset
│   ├── check_labels.py             # Check dataset labels (Python)
│   ├── create_dataset.py           # Process RF training data
│   ├── train_classifier.py         # Train Random Forest
│   ├── train_transformer_pytorch.py # Train Transformer
│   ├── inference_classifier.py     # CLI inference (RF)
│   └── inference_transformer.py    # CLI inference (Transformer)
│
├── transformer/                    # Transformer model architecture
│   └── transformer_model_pytorch.py
│
├── models/                         # Saved models
│   ├── rf_model/                   # Random Forest models
│   │   ├── model.pickle
│   │   └── data.pickle
│   ├── tf_model/                   # Transformer models
│   │   ├── transformer_model.pth
│   │   ├── label_encoder.pickle
│   │   └── sequences.pickle
│   └── backup/                     # Model backups
│
├── templates/                      # HTML templates
│   └── index.html                  # Main web interface
│
├── static/                         # Frontend assets
│   └── app.js                      # JavaScript (model toggle, UI)
│
├── docs/                           # Documentation
│   └── DATASET_MANAGEMENT.md       # Dataset management guide
│
├── data/                           # Training images
├── saved_predictions/              # Saved screenshots
└── confusion_matrix/               # Model evaluation plots

```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Create virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

### 2. Launch Web Application

```bash
# Quick start (recommended)
./shell_cmd/run_webapp.sh

# Or manually
python app.py
```

Open browser: **http://localhost:5001**

### 3. Use the Application

1. **Start Camera**: Click "▶️ Start Camera"
2. **Choose Model**:
   - Random Forest: Instant static sign detection (A, B, C)
   - Transformer: Motion-based dynamic signs (hello, good morning)
3. **Perform Signs**: Show hand signs to camera
4. **Build Sentences**: Signs auto-add at high confidence
5. **Speak**: Click "🔊 Speak Sentence" for text-to-speech

---

## 📊 Dual Model System

### Random Forest (Static Signs)

**Best for**: Letters (A-Z), numbers, static gestures

**Features**:

- ⚡ Instant predictions (no buffering)
- 🎯 High accuracy for static signs
- 🔄 Add new signs via web interface
- 📸 Single-frame classification

**Usage**:

```bash
# Train Random Forest
python scripts/train_classifier.py

# CLI inference
python scripts/inference_classifier.py
```

### PyTorch Transformer (Dynamic Signs)

**Best for**: Motion-based signs (hello, how are you, good morning)

**Features**:

- 🎬 Sequence-based (30 frames)
- 🧠 Captures temporal patterns
- 🎯 Better for dynamic motions
- ⏱️ 2-second cooldown between predictions

**Usage**:

```bash
# Train Transformer
python scripts/train_transformer_pytorch.py

# CLI inference
python scripts/inference_transformer.py
```

### Model Comparison

| Feature                | Random Forest | Transformer       |
| ---------------------- | ------------- | ----------------- |
| **Speed**              | Instant       | 3-sec buffer      |
| **Input**              | 1 frame       | 30 frames         |
| **Best For**           | Static (A-Z)  | Dynamic (hello)   |
| **Add Sign**           | Web UI ✅     | CLI only          |
| **Accuracy (Static)**  | Excellent     | Good              |
| **Accuracy (Dynamic)** | Poor          | Excellent         |
| **Device**             | CPU           | Apple Silicon GPU |

---

## 🎓 Training Workflows

### Random Forest Training (Static Signs)

```bash
# 1. Collect images
python scripts/collect_images.py

# 2. Create dataset
python scripts/create_dataset.py

# 3. Train model
python scripts/train_classifier.py

# 4. Test inference
python scripts/inference_classifier.py
```

### Transformer Training (Dynamic Signs)

```bash
# 1. Collect sequences (30 frames each)
python scripts/collect_sequences.py A B C hello "good morning"

# 2. Train model
python scripts/train_transformer_pytorch.py

# 3. Test inference
python scripts/inference_transformer.py
```

---

## ➕ Adding New Signs

### Random Forest (Web Interface)

1. Open web app: `http://localhost:5001`
2. Click "➕ Add New Sign"
3. Enter sign name
4. Collect 100 images via webcam
5. Model retrains automatically ✅

### Random Forest (CLI)

```bash
# Add new sign incrementally
python scripts/add_new_sign.py <sign_name>

# Retrain model
python scripts/train_classifier.py
```

### Transformer (CLI Only)

```bash
# Add new sign with sequences
python scripts/add_new_signs_tf.py <sign_name> --sequences 15

# Retrain model
python scripts/train_transformer_pytorch.py

# Restart Flask app to load new model
```

---

## 🗑️ Removing Signs

```bash
# Remove specific signs from dataset
python scripts/remove_signs.py <sign1> <sign2>

# Example
python scripts/remove_signs.py "bad sign" rest

# Retrain model after removal
python scripts/train_transformer_pytorch.py  # or train_classifier.py
```

---

## 🔍 Check Dataset Labels

View all signs in your datasets:

```bash
# Python script (detailed)
python scripts/check_labels.py

# Bash script (quick)
./shell_cmd/check_labels.sh
```

**Output includes**:

- 📊 Random Forest signs and sample counts
- 🤖 Transformer signs and sequence counts
- 🔄 Comparison showing common/unique signs
- 📈 Dataset statistics

---

## ⚙️ Configuration

All settings in `config.py`:

### Data Collection

```python
SIGNS_TO_COLLECT = ["A", "B", "C", "hello"]
DATASET_SIZE = 200  # Images per sign (RF)
SEQUENCES_PER_SIGN = 20  # Sequences per sign (Transformer)
SEQUENCE_LENGTH = 30  # Frames per sequence
```

### Model Settings

```python
# Random Forest
N_ESTIMATORS = 100
MAX_DEPTH = 10

# Transformer
EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 0.001
MODEL_TYPE = 'lightweight'  # or 'powerful'
```

### Detection Settings

```python
MIN_DETECTION_CONFIDENCE = 0.5
CONFIDENCE_THRESHOLD_HIGH = 80  # Green
CONFIDENCE_THRESHOLD_MEDIUM = 60  # Orange
CONFIDENCE_THRESHOLD_UNKNOWN = 50  # Unknown
```

### File Paths

```python
# Random Forest
MODEL_PATH = "./models/rf_model/model.pickle"
DATA_PICKLE_PATH = "./models/rf_model/data.pickle"

# Transformer
TRANSFORMER_MODEL_PATH = "./models/tf_model/transformer_model.pth"
TRANSFORMER_LABEL_ENCODER_PATH = "./models/tf_model/label_encoder.pickle"
SEQUENCES_PICKLE_PATH = "./models/tf_model/sequences.pickle"
```

---

## 🌐 Web API Endpoints

### Prediction

```bash
POST /predict
Content-Type: application/json

{
  "landmarks": [[...]]  # MediaPipe hand landmarks
}

Response:
{
  "success": true,
  "prediction": "hello",
  "confidence": 92.5,
  "model_type": "transformer",
  "buffer_status": "30/30"
}
```

### Toggle Model

```bash
POST /toggle_model
Content-Type: application/json

{
  "model_type": "transformer"  # or "random_forest"
}
```

### Reset Buffer

```bash
POST /reset_buffer

Response:
{
  "success": true,
  "message": "Buffer reset"
}
```

### Model Info

```bash
GET /model_info

Response:
{
  "success": true,
  "model_type": "random_forest",
  "num_signs": 13,
  "signs": ["A", "B", "C", ...]
}
```

---

## 🎯 Model Performance

### Random Forest

- **Test Accuracy**: **98.73%** 🎯
- **Per-Class Accuracy**:
  - A: 100% (precision: 1.00, recall: 1.00)
  - B: 100% (precision: 1.00, recall: 1.00)
  - C: 100% (precision: 1.00, recall: 1.00)
  - a: 99% (precision: 1.00, recall: 0.97)
  - d: 100% (precision: 1.00, recall: 1.00)
  - fine: 99% (precision: 0.97, recall: 1.00)
  - good morning: 99% (precision: 1.00, recall: 0.97)
  - good night: 96% (precision: 0.99, recall: 0.94)
  - hello: 100% (precision: 1.00, recall: 1.00)
  - how are you?: 99% (precision: 1.00, recall: 0.99)
  - i: 97% (precision: 1.00, recall: 0.95)
  - i love you: 98% (precision: 0.95, recall: 1.00)
  - my name is: 97% (precision: 0.94, recall: 1.00)
  - nice to meet you: 100% (precision: 1.00, recall: 1.00)
  - v: 100% (precision: 1.00, recall: 1.00)
  - yes: 99% (precision: 0.98, recall: 1.00)
- **Features**: 126 (2 hands × 21 landmarks × 3 coords)
- **Speed**: Instant (<10ms)
- **Device**: CPU
- **Training**: 16 signs, 4716 samples

### Transformer

- **Test Accuracy**: **98.72%** 🎯
- **Per-Class Accuracy**:
  - A: 100.00%
  - B: 100.00%
  - C: 100.00%
  - I love you: 100.00%
  - d: 83.33%
  - good morning: 100.00%
  - good night: 100.00%
  - hello: 100.00%
  - how are you: 100.00%
  - i: 100.00%
  - nice to meet you: 100.00%
  - v: 100.00%
  - yes: 83.33%
- **Parameters**: ~50K (lightweight) or ~200K (powerful)
- **Speed**: 3-second buffer + 2-second cooldown
- **Device**: Apple Silicon GPU (MPS)
- **Training**: 13 signs, 50 epochs

---

## 💡 Tips for Best Results

### Data Collection

1. **Consistent lighting** - same environment for training/testing
2. **Clear hand visibility** - avoid occlusion
3. **Steady camera** - mount or stabilize
4. **Varied angles** - slight variations improve generalization
5. **Multiple sequences** - 15-20 for dynamic signs, 10 for static

### Model Selection

- **Static signs** (A-Z, numbers): Use Random Forest
- **Dynamic signs** (hello, goodbye): Use Transformer
- **Mixed dataset**: Train both models

### Inference

- **Random Forest**: Hold sign steady for instant detection
- **Transformer**: Perform full motion naturally
- **Confidence**: Green (≥80%) is reliable, Red (<60%) may be wrong
- **Unknown signs**: Shows when model is uncertain

---

## 🛠️ Troubleshooting

### "Processing..." stuck on Transformer

- **Fixed**: Cooldown mechanism stores last prediction
- Wait 2 seconds between predictions

### Model not loading

- Check file paths in `config.py`
- Ensure models exist in `./models/` directory
- Retrain if necessary

### Low accuracy

- Collect more training data
- Ensure consistent lighting
- Check hand visibility in camera
- Verify correct model for sign type

### Camera not working

- Check `CAMERA_INDEX` in `config.py`
- Try `CAMERA_INDEX = 1` for external camera
- Grant camera permissions

---

## 📦 Requirements

```
opencv-python
mediapipe
numpy
scikit-learn
scipy
pandas
matplotlib
seaborn
flask
flask-cors
torch>=2.0.0
torchvision>=0.15.0
```

**Platform**: macOS (Apple Silicon GPU support), Windows, Linux

---

## 🚀 Future Enhancements

- [ ] Web UI for Transformer sign addition (with WebSocket progress)
- [ ] More sign languages (BSL, ISL, etc.)
- [ ] Mobile app (iOS/Android)
- [ ] Real-time translation mode
- [ ] Sign language learning mode
- [ ] Multi-hand gesture recognition
- [ ] Export trained models (ONNX, CoreML)

---

## 🙏 Acknowledgments

- **MediaPipe**: Hand landmark detection
- **PyTorch**: Transformer model framework
- **scikit-learn**: Random Forest implementation
- **Flask**: Web application framework
- **Sign Language Detector Python Repo**: [computervisioneng](https://github.com/computervisioneng/sign-language-detector-python)

---

**Happy Signing!** 🤟
