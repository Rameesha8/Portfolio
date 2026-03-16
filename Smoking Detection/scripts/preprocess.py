import cv2
import os
import torch
from torchvision import models, transforms
import torch.nn as nn
from tqdm import tqdm

# ---------------- CONFIG ----------------
FRAME_INTERVAL = 30      # 1 frame per second (30fps)
MAX_FRAMES = 300         # cap frames per video
IMAGE_SIZE = (224, 224)

# ---------------- Paths ----------------
video_dir = "datasets/videos"
frames_dir = "datasets/frames"
embeddings_dir = "datasets/embeddings"

os.makedirs(frames_dir, exist_ok=True)
os.makedirs(embeddings_dir, exist_ok=True)

# ---------------- Load Image Classifier ----------------
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

# ---------------- Preprocess ----------------
preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor()
])

# ---------------- PROCESS VIDEOS ----------------
print("🔹 Extracting frames & generating embeddings...")

for split in ["train", "val"]:
    split_video_dir = os.path.join(video_dir, split)
    if not os.path.exists(split_video_dir):
        continue

    split_frames_dir = os.path.join(frames_dir, split)
    os.makedirs(split_frames_dir, exist_ok=True)

    for video_file in tqdm(os.listdir(split_video_dir), desc=f"{split} videos"):
        video_path = os.path.join(split_video_dir, video_file)
        video_name = os.path.splitext(video_file)[0]

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"⚠ Skipping corrupted video: {video_file}")
            continue

        video_frame_dir = os.path.join(split_frames_dir, video_name)
        os.makedirs(video_frame_dir, exist_ok=True)

        frame_count = 0
        saved_frames = 0
        video_embeddings = []

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_count % FRAME_INTERVAL == 0:
                    if saved_frames >= MAX_FRAMES:
                        break

                    # Save frame (optional but kept for debugging)
                    frame_path = os.path.join(
                        video_frame_dir, f"{saved_frames:04d}.jpg"
                    )
                    cv2.imwrite(frame_path, frame)

                    # Generate embedding
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    img = preprocess(frame_rgb).unsqueeze(0).to(device)

                    with torch.no_grad():
                        feat = image_model.features(img)
                        feat = nn.functional.adaptive_avg_pool2d(feat, 1)
                        feat = feat.flatten(1)
                        video_embeddings.append(feat.cpu())

                    saved_frames += 1

                frame_count += 1

        except Exception as e:
            print(f"⚠ Error processing {video_file}: {e}")
            cap.release()
            continue

        cap.release()

        if len(video_embeddings) == 0:
            print(f"⚠ No valid frames in {video_file}")
            continue

        video_embeddings = torch.stack(video_embeddings)

        torch.save(
            video_embeddings,
            os.path.join(embeddings_dir, f"{video_name}.pt")
        )

        print(f"✅ {video_file}: {saved_frames} frames")

print("🎉 Preprocessing completed safely!")
print(f"📂 Frames: {frames_dir}")
print(f"📂 Embeddings: {embeddings_dir}")
