import os
import json
import argparse
from pathlib import Path
import numpy as np
from collections import defaultdict
import uuid

# Attempt to import required libraries. 
# (In a real environment, you'd need ultralytics, decord, paddleocr, torch, etc.)
try:
    from decord import VideoReader, cpu
    import torch
    from ultralytics import YOLO
    from paddleocr import PaddleOCR
except ImportError as e:
    print(f"Warning: Missing dependencies for Module 2A: {e}")
    print("Please install decord, torch, ultralytics, paddlepaddle, paddleocr")

from config_v5 import config

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

class ReIDMemoryPool:
    def __init__(self, ttl=60):
        self.pool = {} # track_id -> {"embedding": vector, "timestamp": last_seen_ms}
        self.ttl = ttl
        
    def add(self, track_id, embedding, timestamp):
        self.pool[track_id] = {"embedding": embedding, "timestamp": timestamp}
        
    def clean(self, current_timestamp):
        # Remove items older than ttl
        keys_to_remove = []
        for tid, data in self.pool.items():
            if (current_timestamp - data["timestamp"]) / 1000.0 > self.ttl:
                keys_to_remove.append(tid)
        for k in keys_to_remove:
            del self.pool[k]

    def match(self, new_embedding, threshold=0.75):
        # Find best match based on cosine similarity
        best_tid = None
        best_sim = -1
        new_emb = np.array(new_embedding)
        new_emb = new_emb / (np.linalg.norm(new_emb) + 1e-8)
        
        for tid, data in self.pool.items():
            pool_emb = np.array(data["embedding"])
            pool_emb = pool_emb / (np.linalg.norm(pool_emb) + 1e-8)
            sim = np.dot(new_emb, pool_emb)
            if sim > best_sim:
                best_sim = sim
                best_tid = tid
                
        if best_sim > threshold:
            return best_tid, best_sim
        return None, 0.0

def process_video(video_path: Path, shots_path: Path, out_dir: Path):
    ensure_dir(out_dir)
    
    print(f"[Module 2A] Loading video: {video_path}")
    # Load video with decord
    # vr = VideoReader(str(video_path), ctx=cpu(0))
    # fps = vr.get_avg_fps()
    # sample_interval = max(1, int(fps / config.shot_fps))
    
    # Load models
    print(f"[Module 2A] Loading YOLOv8 and OCR models...")
    # yolo_model = YOLO('yolov8n.pt') # Placeholder for object detection + tracking
    # ocr = PaddleOCR(use_angle_cls=False, lang='en')
    
    # Data structures for output
    tracklets_dict = defaultdict(lambda: {
        "class_labels": [], 
        "start_ms": float('inf'), 
        "end_ms": 0, 
        "bbox_trajectory": []
    })
    
    static_objects = []
    ocr_local = []
    
    # TODO: Implement full frame iteration, YOLO tracking, and ReID logic
    # For now, we simulate the output to demonstrate the architecture
    print("[Module 2A] Extracting tracklets and spatial variance (Mocked for architecture demo)...")
    
    # Mocking tracklets
    mock_track_id = str(uuid.uuid4())
    tracklets_dict[mock_track_id] = {
        "class_labels": ["person"],
        "start_ms": 1000,
        "end_ms": 5000,
        "bbox_trajectory": [
            {"frame_idx": 5, "bbox": [0.1, 0.1, 0.2, 0.4]},
            {"frame_idx": 10, "bbox": [0.15, 0.15, 0.2, 0.4]}
        ]
    }
    
    # Finalize tracklets
    final_tracklets = []
    for tid, data in tracklets_dict.items():
        # Majority voting for class
        from collections import Counter
        class_counts = Counter(data["class_labels"])
        majority_class = class_counts.most_common(1)[0][0] if class_counts else "unknown"
        
        final_tracklets.append({
            "track_id": tid,
            "unified_id": tid, # ReID unified
            "class_label": majority_class,
            "start_ms": data["start_ms"],
            "end_ms": data["end_ms"],
            "bbox_trajectory": data["bbox_trajectory"],
            "reid_confidence": 0.95
        })
        
    tracklets_json = {
        "metadata": {
            "video_id": video_path.stem,
            "fps": config.shot_fps,
            "resolution": {"width": 1920, "height": 1080, "unit": "pixel"},
            "duration_ms": 10000
        },
        "tracklets": final_tracklets,
        "static_objects": static_objects,
        "ocr_local": ocr_local
    }
    
    with open(out_dir / "tracklets.json", "w", encoding="utf-8") as f:
        json.dump(tracklets_json, f, indent=2)
        
    print(f"[Module 2A] Saved {len(final_tracklets)} tracklets to tracklets.json")

def main():
    parser = argparse.ArgumentParser(description="Module 2A – Dynamic Producer")
    parser.add_argument("--video", type=str, required=True, help="Path to input video file")
    parser.add_argument("--shots", type=str, required=True, help="Path to shots.json")
    parser.add_argument("--output-dir", type=str, default=config.intermediate_dir)
    args = parser.parse_args()

    video_path = Path(args.video)
    shots_path = Path(args.shots)
    out_dir = Path(args.output_dir)
    
    process_video(video_path, shots_path, out_dir)

if __name__ == "__main__":
    main()
