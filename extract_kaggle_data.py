import os
import cv2
import csv
import zipfile
import mediapipe as mp
from kaggle.api.kaggle_api_extended import KaggleApi
from tqdm import tqdm

# ========================
# Kaggle Dataset Settings
# ========================
KAGGLE_DATASET = "grassknoted/asl-alphabet"  # change if using another ASL dataset
DATASET_ZIP = "asl_dataset.zip"
RAW_DIR = "asl_alphabet_train/asl_alphabet_train"
OUTPUT_DIR = "data/kaggle_landmarks"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ========================
# Step 1: Download Dataset
# ========================
def download_dataset():
    api = KaggleApi()
    api.authenticate()
    print("📥 Downloading dataset from Kaggle...")
    api.dataset_download_files(KAGGLE_DATASET, path=".", unzip=False)

    print("📦 Extracting dataset...")
    for file in os.listdir("."):
        if file.endswith(".zip"):
            with zipfile.ZipFile(file, "r") as zip_ref:
                zip_ref.extractall(".")
            print(f"✅ Extracted {file}")

# ========================
# Step 2: Extract Landmarks
# ========================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

def extract_landmarks(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    if results.multi_hand_landmarks:
        lm = results.multi_hand_landmarks[0]
        return [coord for point in lm.landmark for coord in (point.x, point.y, point.z)]
    return None

def process_dataset():
    for label in os.listdir(RAW_DIR):
        label_path = os.path.join(RAW_DIR, label)
        if not os.path.isdir(label_path):
            continue

        out_file = os.path.join(OUTPUT_DIR, f"{label}.csv")
        with open(out_file, "w", newline="") as f:
            writer = csv.writer(f)

            for img_name in tqdm(os.listdir(label_path), desc=f"Processing {label}"):
                img_path = os.path.join(label_path, img_name)
                img = cv2.imread(img_path)
                if img is None:
                    continue

                landmarks = extract_landmarks(img)
                if landmarks:
                    writer.writerow(landmarks)

        print(f"✅ Saved landmarks for {label} → {out_file}")

# ========================
# Main
# ========================
if __name__ == "__main__":
    if not os.path.exists(RAW_DIR):
        download_dataset()
    process_dataset()
