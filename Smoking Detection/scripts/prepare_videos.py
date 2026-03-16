import os
import json
import requests
from tqdm import tqdm
import sys

# ----------------- Get split argument -----------------
split = sys.argv[1] if len(sys.argv) > 1 else "train"
if split not in ["train", "val"]:
    print("Invalid split. Use 'train' or 'val'.")
    sys.exit(1)

annotations_dir = f"datasets/PHAD/{split}_annotations"
output_dir = f"datasets/videos/{split}"
os.makedirs(output_dir, exist_ok=True)

# ----------------- Function to download videos -----------------
def download_videos_from_annotations(annotations_dir, output_dir):
    json_files = [f for f in os.listdir(annotations_dir) if f.endswith(".json")]

    for jf in tqdm(json_files, desc=f"Processing {split} annotation files"):
        path = os.path.join(annotations_dir, jf)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)  # list of dicts
        for item in data:
            video_url = item.get("mediaUrls/0")
            if video_url:
                video_name = video_url.split("/")[-1] + ".mp4"
                video_path = os.path.join(output_dir, video_name)
                
                if os.path.exists(video_path):
                    continue  # skip already downloaded

                try:
                    r = requests.get(video_url, stream=True)
                    with open(video_path, "wb") as f:
                        for chunk in r.iter_content(chunk_size=1024*1024):
                            if chunk:
                                f.write(chunk)
                except Exception as e:
                    print(f"Failed to download {video_url}: {e}")

# ----------------- Run -----------------
download_videos_from_annotations(annotations_dir, output_dir)
print(f"All {split} videos downloaded successfully!")
