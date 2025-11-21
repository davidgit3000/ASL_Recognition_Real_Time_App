import os
import sys
import cv2
import argparse

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_DIR, SIGNS_TO_COLLECT, DATASET_SIZE, CAMERA_INDEX

# Parse command line arguments
parser = argparse.ArgumentParser(description='Collect ASL sign images')
parser.add_argument('--recollect', action='store_true', 
                    help='Recollect all signs from scratch (ignores existing data)')
parser.add_argument('--force', action='store_true',
                    help='Alias for --recollect')
args = parser.parse_args()

recollect_all = args.recollect or args.force

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Check which signs already have data
signs_to_collect = []

if recollect_all:
    print("🔄 Recollect mode: Will collect all signs from scratch")
    signs_to_collect = list(SIGNS_TO_COLLECT)
else:
    print("➕ Incremental mode: Will only collect new/incomplete signs")
    for sign in SIGNS_TO_COLLECT:
        sign_dir = os.path.join(DATA_DIR, str(sign))
        if os.path.exists(sign_dir):
            existing_images = len([f for f in os.listdir(sign_dir) if not f.startswith('.')])
            if existing_images >= DATASET_SIZE:
                print(f"Sign '{sign}' already has {existing_images} images. Skipping...")
                continue
            else:
                print(f"Sign '{sign}' has {existing_images}/{DATASET_SIZE} images. Will collect remaining...")
        signs_to_collect.append(sign)

if not signs_to_collect:
    print("\n✅ All signs already have sufficient data!")
    print(f"Existing signs: {SIGNS_TO_COLLECT}")
    print(f"\nTo recollect all signs, run: python scripts/collect_images.py --recollect")
    exit(0)

print(f"\nSigns to collect: {signs_to_collect}")
print(f"Dataset size per sign: {DATASET_SIZE}")

cap = cv2.VideoCapture(CAMERA_INDEX)  # Use camera from config
if not cap.isOpened():
    print("Error: Could not open camera. Trying camera index 1...")
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("Error: No camera available. Please check your camera connection and permissions.")
        exit(1)

for sign in signs_to_collect:
    if not os.path.exists(os.path.join(DATA_DIR, str(sign))):
        os.makedirs(os.path.join(DATA_DIR, str(sign)))
    
    print(f"Collecting data for sign {sign}")

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("Error: Failed to capture frame from camera.")
            cap.release()
            cv2.destroyAllWindows()
            exit(1)
        cv2.putText(frame, 'Press q to quit', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    counter = 0
    while counter < DATASET_SIZE:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("Error: Failed to capture frame from camera.")
            cap.release()
            cv2.destroyAllWindows()
            exit(1)
        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        cv2.imwrite(os.path.join(DATA_DIR, str(sign), f"{counter}.jpg"), frame)
        print(f"Saved {counter + 1} images for sign {sign}")
        counter += 1

cap.release()
cv2.destroyAllWindows()

