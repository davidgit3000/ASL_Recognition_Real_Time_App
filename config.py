"""
Configuration file for ASL Sign Language Detection Project
Modify these parameters to customize the behavior of all scripts
"""

# Data Collection Settings
SIGNS_TO_COLLECT = ["A", "B", "C", "hello", "good morning", "nice to meet you"]  # List of ASL signs to collect
DATASET_SIZE = 200  # Number of images per sign

# Data Directory
DATA_DIR = "./data"

# Model Settings
TEST_SIZE = 0.2  # Proportion of dataset to use for testing (0.2 = 20%)
RANDOM_STATE = 42  # Random seed for reproducibility

# Random Forest Hyperparameters
N_ESTIMATORS = 100  # Number of trees in the forest
MAX_DEPTH = 10  # Maximum depth of each tree
MIN_SAMPLES_SPLIT = 5  # Minimum samples required to split a node
MIN_SAMPLES_LEAF = 2  # Minimum samples required at a leaf node

# MediaPipe Settings
MIN_DETECTION_CONFIDENCE = 0.5  # Minimum confidence for hand detection (0.0 - 1.0)
MAX_NUM_HANDS = 2  # Maximum number of hands to detect

# Camera Settings
CAMERA_INDEX = 0  # Default camera index (0 = default camera, 1 = external camera)

# Inference Settings
CONFIDENCE_THRESHOLD_HIGH = 80  # Threshold for high confidence (green)
CONFIDENCE_THRESHOLD_MEDIUM = 60  # Threshold for medium confidence (orange)
CONFIDENCE_THRESHOLD_UNKNOWN = 50  # Below this threshold, classify as "Unknown sign"

# Output Directories
SAVE_DIR = "./saved_predictions"
MODEL_PATH = "./model.pickle"
DATA_PICKLE_PATH = "./data.pickle"
CONFUSION_MATRIX_PATH = "./confusion_matrix.png"
