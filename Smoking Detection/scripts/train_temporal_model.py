import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------- Paths ----------------
EMB_DIR = "datasets/embeddings"
LABELS_CSV = "datasets/video_labels.csv"

os.makedirs("models", exist_ok=True)
os.makedirs("logs", exist_ok=True)

MODEL_PATH = "models/temporal_model.pth"
METRICS_PATH = "logs/temporal_metrics.csv"
CM_PATH = "logs/temporal_confusion_matrix.png"
LOSS_PATH = "logs/temporal_loss_curve.png"

# ---------------- Load CSV ----------------
labels_df = pd.read_csv(LABELS_CSV)

# ---------------- Dataset ----------------
class EmbeddingDataset(Dataset):
    def __init__(self, split):
        self.samples = []

        for _, row in labels_df[labels_df["split"] == split].iterrows():
            video_name = row["filename"].replace(".mp4", ".pt")
            emb_path = os.path.join(EMB_DIR, video_name)

            if not os.path.exists(emb_path):
                continue

            embedding = torch.load(emb_path)      # [seq_len, 1, 1280]
            embedding = embedding.squeeze(1)      # [seq_len, 1280]
            label = int(row["label"])

            self.samples.append((embedding, label))

        print(f"Loaded {len(self.samples)} {split} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

# ---------------- Collate Function ----------------
def collate_fn(batch):
    sequences, labels = zip(*batch)
    sequences = pad_sequence(sequences, batch_first=True)
    labels = torch.tensor(labels)
    return sequences, labels

# ---------------- DataLoaders ----------------
train_ds = EmbeddingDataset("train")
val_ds   = EmbeddingDataset("val")

train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, collate_fn=collate_fn)
val_loader   = DataLoader(val_ds, batch_size=2, shuffle=False, collate_fn=collate_fn)

# ---------------- Model ----------------
class TemporalModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=1280,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )
        self.fc = nn.Linear(512, 2)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TemporalModel().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# ---------------- Training ----------------
EPOCHS = 10
losses = []

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for x, y in train_loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    losses.append(avg_loss)
    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f}")

# ---------------- Validation ----------------
model.eval()
y_true, y_pred = [], []

with torch.no_grad():
    for x, y in val_loader:
        x = x.to(device)
        preds = torch.argmax(model(x), dim=1)
        y_true.extend(y.numpy())
        y_pred.extend(preds.cpu().numpy())

# ---------------- Metrics ----------------
acc = np.mean(np.array(y_true) == np.array(y_pred))
f1 = f1_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
cm = confusion_matrix(y_true, y_pred)

# ---------------- Save Metrics ----------------
pd.DataFrame([{
    "accuracy": acc,
    "f1_score": f1,
    "precision": precision,
    "recall": recall
}]).to_csv(METRICS_PATH, index=False)

# ---------------- Save Confusion Matrix ----------------
plt.figure(figsize=(6, 5))
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues",
    xticklabels=["Non-Smoker", "Smoker"],
    yticklabels=["Non-Smoker", "Smoker"]
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Temporal Model Confusion Matrix")
plt.savefig(CM_PATH)
plt.close()

# ---------------- Save Loss Curve ----------------
plt.plot(losses)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.savefig(LOSS_PATH)
plt.close()

# ---------------- Save Model ----------------
torch.save(model.state_dict(), MODEL_PATH)

print("\n✅ Temporal model training complete")
print("Model:", MODEL_PATH)
print("Metrics:", METRICS_PATH)
print("Confusion Matrix:", CM_PATH)
print("Loss Curve:", LOSS_PATH)
