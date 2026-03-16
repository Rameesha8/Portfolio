import cv2
import torch
from torchvision import models, transforms
import torch.nn as nn
import os
import pandas as pd
from tqdm import tqdm

# ---------------- Paths ----------------
video_dir = "datasets/videos"
csv_path = "datasets/video_labels.csv"

# ---------------- Device ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- Load Image Model ----------------
image_model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
image_model.classifier[1] = nn.Linear(image_model.classifier[1].in_features, 2)
image_model.load_state_dict(torch.load("models/image_model.pth", map_location=device))
image_model.eval().to(device)

# ---------------- Transform ----------------
preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# ---------------- Parameters ----------------
frame_skip = 5  # Take every 5th frame
video_splits = ["train", "val"]
video_labels = []

# ---------------- Function to predict video label ----------------
def predict_video_label(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"⚠ Skipping corrupted video: {video_path}")
        return None

    frame_count = 0
    preds = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Sample every Nth frame
        if frame_count % frame_skip == 0:
            img = preprocess(frame).unsqueeze(0).to(device)
            with torch.no_grad():
                out = image_model(img)
                pred = torch.argmax(out, 1).item()
                preds.append(pred)

        frame_count += 1

    cap.release()
    if len(preds) == 0:
        return None

    # Majority vote
    video_label = max(set(preds), key=preds.count)
    return video_label

# ---------------- Process Videos ----------------
for split in video_splits:
    split_dir = os.path.join(video_dir, split)
    if not os.path.exists(split_dir):
        continue

    for video_file in tqdm(os.listdir(split_dir), desc=f"Processing {split} videos"):
        video_path = os.path.join(split_dir, video_file)
        label = predict_video_label(video_path)
        if label is not None:
            video_labels.append({
                "video": video_file,
                "split": split,
                "label": label  # 0 = Non-Smoker, 1 = Smoker
            })

# ---------------- Save CSV ----------------
df = pd.DataFrame(video_labels)
df.to_csv(csv_path, index=False)
print(f"\n✅ Video labels saved to {csv_path}")
