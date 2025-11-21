import os
import sys
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (DATA_PICKLE_PATH, MODEL_PATH, CONFUSION_MATRIX_PATH,
                    TEST_SIZE, RANDOM_STATE, N_ESTIMATORS, MAX_DEPTH,
                    MIN_SAMPLES_SPLIT, MIN_SAMPLES_LEAF)

# Load the dataset
print("Loading dataset...")
with open(DATA_PICKLE_PATH, 'rb') as f:
    data_dict = pickle.load(f)

data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

print(f"Dataset loaded: {len(data)} samples")
print(f"Feature dimensions: {data.shape}")
print(f"Labels: {set(labels)}")

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    data, labels, test_size=TEST_SIZE, shuffle=True, stratify=labels, random_state=RANDOM_STATE
)

print(f"\nTraining set: {len(X_train)} samples")
print(f"Testing set: {len(X_test)} samples")

# Train the Random Forest Classifier
print("\nTraining Random Forest Classifier...")
model = RandomForestClassifier(
    n_estimators=N_ESTIMATORS,
    max_depth=MAX_DEPTH,
    min_samples_split=MIN_SAMPLES_SPLIT,
    min_samples_leaf=MIN_SAMPLES_LEAF,
    random_state=RANDOM_STATE,
    n_jobs=-1
)

model.fit(X_train, y_train)
print("Training complete!")

# Make predictions
print("\nEvaluating model...")
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\nTest Accuracy: {accuracy * 100:.2f}%")

# Print detailed classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=sorted(set(labels)), 
            yticklabels=sorted(set(labels)))
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig(CONFUSION_MATRIX_PATH)
print(f"\nConfusion matrix saved as '{CONFUSION_MATRIX_PATH}'")

# Feature importance (top 10)
feature_importance = model.feature_importances_
top_features_idx = np.argsort(feature_importance)[-10:][::-1]
print("\nTop 10 Most Important Features:")
for idx in top_features_idx:
    landmark_num = idx // 3
    coord = ['x', 'y', 'z'][idx % 3]
    print(f"  Feature {idx}: Landmark {landmark_num} ({coord}) - Importance: {feature_importance[idx]:.4f}")

# Save the trained model
with open(MODEL_PATH, 'wb') as f:
    pickle.dump({'model': model}, f)

print(f"\nModel saved as '{MODEL_PATH}'")

