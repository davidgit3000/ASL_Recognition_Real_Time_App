# ASL Sign Language Detection

A machine learning project for real-time American Sign Language (ASL) detection using computer vision and Random Forest classification.

## Project Structure

```
asl_sign_model/
├── config.py                      # Central configuration file
├── scripts/
│   ├── collect_images.py          # Collect training images
│   ├── create_dataset.py          # Extract hand landmarks
│   ├── train_classifier.py        # Train the model
│   └── inference_classifier.py    # Real-time detection
├── data/                          # Training images (A, B, C)
├── saved_predictions/             # Saved inference screenshots
├── data.pickle                    # Processed dataset
├── model.pickle                   # Trained model
└── confusion_matrix.png           # Model evaluation

```

## Configuration

All project parameters are centralized in `config.py`. Modify this file to customize:

### Data Collection

- `SIGNS_TO_COLLECT`: List of ASL signs to collect (e.g., `["A", "B", "C"]`)
- `DATASET_SIZE`: Number of images per sign (default: 100)

### Model Training

- `TEST_SIZE`: Train/test split ratio (default: 0.2)
- `N_ESTIMATORS`: Number of trees in Random Forest (default: 100)
- `MAX_DEPTH`: Maximum tree depth (default: 10)

### Camera & Detection

- `CAMERA_INDEX`: Camera to use (0 = default, 1 = external)
- `MIN_DETECTION_CONFIDENCE`: Hand detection threshold (default: 0.5)
- `CONFIDENCE_THRESHOLD_HIGH`: Green box threshold (default: 80%)
- `CONFIDENCE_THRESHOLD_MEDIUM`: Orange box threshold (default: 60%)

## Quick Start with Streamlit App 🚀

**NEW: Web-based interface with real-time detection and text-to-speech!**

```bash
# Install dependencies
pip install -r requirements.txt

# Launch the app
streamlit run streamlit_app.py
```

Features:

- 📹 **Real-time sign detection** with live camera
- 🔊 **Text-to-Speech** translation
- 📝 **Sentence builder** for combining signs
- ➕ **Add new signs** through the interface
- 📊 **Model statistics** and information

See [STREAMLIT_APP.md](STREAMLIT_APP.md) for detailed documentation.

---

## Command-Line Usage

### 1. Collect Training Data

```bash
python scripts/collect_images.py
```

- Press 'q' when ready to start capturing
- 100 images will be captured per sign
- Repeat for each sign in `SIGNS_TO_COLLECT`

### 2. Create Dataset

```bash
python scripts/create_dataset.py
```

- Extracts 63 hand landmark features (21 landmarks × x,y,z)
- Saves to `data.pickle`

### 3. Train Model

```bash
python scripts/train_classifier.py
```

- Trains Random Forest classifier
- Outputs accuracy, confusion matrix, feature importance
- Saves model to `model.pickle`

### 4. Real-Time Detection

```bash
python scripts/inference_classifier.py
```

- Press 's' to save current frame
- Press 'q' to quit
- Color coding:
  - 🟢 Green: High confidence (≥80%)
  - 🟠 Orange: Medium confidence (60-80%)
  - 🔴 Red: Low confidence (<60%)

## Requirements

```bash
pip install -r requirements.txt
```

## Model Performance

- **Accuracy**: 100% on test set (A, B, C signs)
- **Features**: 63 (21 hand landmarks × 3 coordinates)
- **Algorithm**: Random Forest Classifier

## Tips for Best Results

1. **Consistent lighting** during data collection and inference
2. **Same camera orientation** for training and testing
3. **Clear hand visibility** - avoid occlusion
4. **Steady hand position** - reduce motion blur
5. **Face palm toward camera** - match training pose

## Customization

To add more signs:

1. Update `SIGNS_TO_COLLECT` in `config.py`
2. Run `collect_images.py` to gather new data
3. Run `create_dataset.py` to process
4. Run `train_classifier.py` to retrain model

## Adding New Signs (Efficient Method)

To add new signs **without reprocessing existing data**:

1. **Update config**: Add new sign to `SIGNS_TO_COLLECT` in `config.py`

   ```python
   SIGNS_TO_COLLECT = ["A", "B", "C", "hello"]  # Added "hello"
   ```

2. **Collect new sign data only**:

   ```bash
   python scripts/collect_images.py
   # Only collects data for signs not in existing dataset
   ```

3. **Add to existing dataset** (no reprocessing):

   ```bash
   python scripts/add_new_sign.py
   # Automatically detects new signs and adds them
   # Or specify manually: python scripts/add_new_sign.py hello
   ```

4. **Retrain model** (fast with Random Forest):
   ```bash
   python scripts/train_classifier.py
   ```

### Alternative: Full Rebuild

To rebuild everything from scratch:

```bash
python scripts/create_dataset.py  # Reprocess all images
python scripts/train_classifier.py
```
