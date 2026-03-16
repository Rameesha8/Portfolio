import os
import cv2
import torch
from torchvision import models, transforms
import torch.nn as nn
import pandas as pd
from tqdm import tqdm
from PIL import Image

# ---------------- Paths ----------------
video_dir = "datasets/videos"
labels_csv_path = "datasets/video_labels.csv"

# ---------------- Resume Support ----------------
processed_videos = set()
if os.path.exists(labels_csv_path):
    existing_df = pd.read_csv(labels_csv_path)
    processed_videos = set(existing_df["filename"].tolist())

# ---------------- Load Image Model ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

image_model = models.efficientnet_b0(
    weights=models.EfficientNet_B0_Weights.DEFAULT
)
image_model.classifier[1] = nn.Linear(
    image_model.classifier[1].in_features, 2
)

image_model.load_state_dict(
    torch.load("models/image_model.pth", map_location=device)
)
image_model.eval().to(device)

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# ---------------- Predict Label for a Video ----------------
def predict_video_label(video_path, frame_step=30):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"⚠ Skipping corrupted video: {video_path}")
        return None

    frame_embeddings = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # ⏩ Skip frames to speed up
        if frame_idx % frame_step != 0:
            frame_idx += 1
            continue

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img = preprocess(img).unsqueeze(0).to(device)

        with torch.no_grad():
            feat = image_model.features(img)
            feat = nn.functional.adaptive_avg_pool2d(feat, 1).flatten(1)
            frame_embeddings.append(feat)

        frame_idx += 1

    cap.release()

    if not frame_embeddings:
        return None

    video_embedding = torch.stack(frame_embeddings).mean(dim=0)

    with torch.no_grad():
        out = image_model.classifier(video_embedding)
        pred = torch.argmax(out, dim=1).item()

    return pred  # 0 = non-smoker, 1 = smoker

# ---------------- Main Loop ----------------
new_rows = []

for split in ["train", "val"]:
    split_dir = os.path.join(video_dir, split)
    if not os.path.exists(split_dir):
        continue

    for video_file in tqdm(
        os.listdir(split_dir),
        desc=f"Processing {split} videos"
    ):
        if video_file in processed_videos:
            continue  # ✅ Resume skip

        video_path = os.path.join(split_dir, video_file)
        label = predict_video_label(video_path)

        if label is not None:
            new_rows.append({
                "filename": video_file,
                "split": split,
                "label": label
            })

# ---------------- Save CSV (Append Mode) ----------------
if new_rows:
    df_new = pd.DataFrame(new_rows)
    df_new.to_csv(
        labels_csv_path,
        mode="a",
        header=not os.path.exists(labels_csv_path),
        index=False
    )

print("\n✅ Label generation complete (resumable)")
print(f"CSV path: {labels_csv_path}")
