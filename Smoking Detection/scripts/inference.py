
import torch
import cv2
from ultralytics import YOLO
from torchvision import models, transforms
import torch.nn as nn
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- Image Model ----------------
image_model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
image_model.classifier[1] = nn.Linear(image_model.classifier[1].in_features, 2)
image_model.load_state_dict(torch.load("models/image_model.pth", map_location=device))
image_model.eval().to(device)

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

# ---------------- YOLO ----------------
detector = YOLO("runs/detect/smoking_detector2/weights/best.pt")

# ---------------- Temporal Model ----------------
class TemporalModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(1280, 256, 2, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(512, 2)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

temporal = TemporalModel().to(device)
temporal.load_state_dict(torch.load("models/temporal_model.pth", map_location=device))
temporal.eval()

# ---------------- Inference ----------------
def detect_smoking(video_path, stride=5):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30

    frames, embeddings, times = [], [], []
    frame_id = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_id % stride == 0:
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            img = transform(img).unsqueeze(0).to(device)

            with torch.no_grad():
                feat = image_model.features(img)
                feat = nn.functional.adaptive_avg_pool2d(feat,1).flatten(1)
                embeddings.append(feat.cpu())

            frames.append(frame)
            times.append(frame_id / fps)

        frame_id += 1

    cap.release()

    if len(embeddings) == 0:
        return {"smoking_probability": 0.0, "smoking_timestamps_sec": []}

    embeddings = torch.cat(embeddings, dim=0).unsqueeze(0).to(device)

    with torch.no_grad():
        out = temporal(embeddings)
        prob = torch.softmax(out, 1)[0,1].item()

    yolo_hits = []
    for t, frame in zip(times[::2], frames[::2]):
        results = detector(frame)
        for box in results[0].boxes:
            if int(box.cls[0]) == 0:  # smoking class
                yolo_hits.append(round(t,2))
                break

    return {
        "smoking_probability": round(prob, 3),
        "smoking_timestamps_sec": yolo_hits
    }

if __name__ == "__main__":
    print(detect_smoking("datasets/videos/test_video.mp4"))
