import os
import json
import argparse
from pathlib import Path
import numpy as np
from collections import defaultdict, Counter
import uuid
import gc

from decord import VideoReader, cpu
import torch
from ultralytics import YOLO
from paddleocr import PaddleOCR

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
        keys_to_remove = []
        for tid, data in self.pool.items():
            if (current_timestamp - data["timestamp"]) / 1000.0 > self.ttl:
                keys_to_remove.append(tid)
        for k in keys_to_remove:
            del self.pool[k]

    def match(self, new_embedding, threshold=0.75):
        best_tid = None
        best_sim = -1
        new_emb = np.array(new_embedding)
        norm_new = np.linalg.norm(new_emb)
        if norm_new < 1e-8:
            return None, 0.0
        new_emb = new_emb / norm_new
        
        for tid, data in self.pool.items():
            pool_emb = np.array(data["embedding"])
            norm_pool = np.linalg.norm(pool_emb)
            if norm_pool < 1e-8:
                continue
            pool_emb = pool_emb / norm_pool
            sim = np.dot(new_emb, pool_emb)
            if sim > best_sim:
                best_sim = sim
                best_tid = tid
                
        if best_sim > threshold:
            return best_tid, best_sim
        return None, 0.0

def process_video(video_path: Path, shots_path: Path, out_dir: Path):
    ensure_dir(out_dir)
    tracklets_file = out_dir / "tracklets.json"
    
    if tracklets_file.exists():
        print(f"[Module 2A] {tracklets_file} already exists. Skipping.")
        return

    print(f"[Module 2A] Loading video: {video_path}")
    vr = VideoReader(str(video_path), ctx=cpu(0))
    video_fps = vr.get_avg_fps()
    duration_ms = len(vr) / video_fps * 1000
    sample_interval = max(1, int(video_fps / config.shot_fps))
    
    # Load Models
    print(f"[Module 2A] Loading YOLOv8...")
    yolo_model = YOLO('yolov8n.pt') # You might want yolov8s.pt or others depending on VRAM
    yolo_model.to(config.device)
    
    # Placeholder for ReID embedding extractor (e.g., FastReID). 
    # For this script, we'll use a mocked embedding function or a lightweight feature extractor.
    def get_reid_embedding(frame, bbox):
        # In a real scenario, crop frame by bbox and run through FastReID.
        # Here we mock a random vector to keep it runnable without a FastReID model file.
        return np.random.rand(512).tolist()
    
    reid_pool = ReIDMemoryPool(ttl=config.reid_pool_ttl_sec)
    
    tracklets_dict = defaultdict(lambda: {
        "class_labels": [], 
        "start_ms": float('inf'), 
        "end_ms": 0, 
        "bbox_trajectory": [],
        "unified_id": None
    })
    
    print("[Module 2A] Running YOLO tracking...")
    
    # Read shots info if needed
    shots = []
    if shots_path and shots_path.exists():
        with open(shots_path, "r", encoding="utf-8") as f:
            shots = json.load(f)
            
    frame_indices = range(0, len(vr), sample_interval)
    
    # Batch processing frames to prevent OOM
    batch_size = config.batch_size_gpu
    for i in range(0, len(frame_indices), batch_size):
        batch_indices = frame_indices[i:i+batch_size]
        frames = vr.get_batch(batch_indices).asnumpy()
        
        # ByteTrack requires consecutive frames usually, but if sampling at 5FPS, it may struggle.
        # Ultralytics track function handles this automatically.
        results = yolo_model.track(frames, persist=True, tracker="bytetrack.yaml", verbose=False)
        
        for j, res in enumerate(results):
            actual_idx = batch_indices[j]
            current_ms = (actual_idx / video_fps) * 1000
            reid_pool.clean(current_ms)
            
            if res.boxes is None or res.boxes.id is None:
                continue
                
            boxes = res.boxes.xyxyn.cpu().numpy() # Normalized bbox
            track_ids = res.boxes.id.int().cpu().numpy()
            classes = res.boxes.cls.int().cpu().numpy()
            names = res.names
            
            for box, tid, cls_idx in zip(boxes, track_ids, classes):
                str_tid = str(tid)
                label = names[cls_idx]
                
                # Get ReID embedding
                emb = get_reid_embedding(frames[j], box)
                
                # If first time seeing this track_id, try to match in ReID pool
                if tracklets_dict[str_tid]["unified_id"] is None:
                    matched_tid, sim = reid_pool.match(emb, threshold=config.reid_similarity_thresh)
                    tracklets_dict[str_tid]["unified_id"] = matched_tid if matched_tid else str_tid
                
                unified_tid = tracklets_dict[str_tid]["unified_id"]
                
                tracklets_dict[unified_tid]["class_labels"].append(label)
                tracklets_dict[unified_tid]["start_ms"] = min(tracklets_dict[unified_tid]["start_ms"], current_ms)
                tracklets_dict[unified_tid]["end_ms"] = max(tracklets_dict[unified_tid]["end_ms"], current_ms)
                tracklets_dict[unified_tid]["bbox_trajectory"].append({
                    "frame_idx": int(actual_idx),
                    "bbox": box.tolist()
                })
                
                # Update ReID pool with latest embedding
                reid_pool.add(unified_tid, emb, current_ms)
                
        # Prevent OOM
        del frames
        del results
        gc.collect()
        if config.device == "cuda":
            torch.cuda.empty_cache()
            
    print("[Module 2A] Finalizing tracklets...")
    final_tracklets = []
    static_objects = []
    
    for tid, data in tracklets_dict.items():
        if not data["bbox_trajectory"]:
            continue
            
        class_counts = Counter(data["class_labels"])
        majority_class = class_counts.most_common(1)[0][0] if class_counts else "unknown"
        
        # Spatial Variance for Static/Dynamic Split
        bboxes = np.array([b["bbox"] for b in data["bbox_trajectory"]])
        # bbox is [x1, y1, x2, y2] normalized
        centers_x = (bboxes[:, 0] + bboxes[:, 2]) / 2.0
        centers_y = (bboxes[:, 1] + bboxes[:, 3]) / 2.0
        std_x = np.std(centers_x)
        std_y = np.std(centers_y)
        
        if std_x < 0.02 and std_y < 0.02: # Threshold for static
            static_objects.append({
                "object_id": tid,
                "shot_id": "unknown", # To be mapped by temporal overlap later
                "class_label": majority_class,
                "bbox": bboxes[0].tolist(),
                "siglip_vector": [] # Can be extracted later if needed
            })
        else:
            final_tracklets.append({
                "track_id": tid,
                "unified_id": data["unified_id"],
                "class_label": majority_class,
                "start_ms": data["start_ms"],
                "end_ms": data["end_ms"],
                "bbox_trajectory": data["bbox_trajectory"],
                "reid_confidence": 1.0
            })
        
    tracklets_json = {
        "metadata": {
            "video_id": video_path.stem,
            "fps": config.shot_fps,
            "resolution": {"width": vr[0].shape[1], "height": vr[0].shape[0], "unit": "pixel"},
            "duration_ms": duration_ms
        },
        "tracklets": final_tracklets,
        "static_objects": static_objects,
        "ocr_local": [] # OCR local can be added similarly by cropping frame and calling PaddleOCR
    }
    
    with open(tracklets_file, "w", encoding="utf-8") as f:
        json.dump(tracklets_json, f, indent=2)
        
    print(f"[Module 2A] Saved {len(final_tracklets)} dynamic tracklets and {len(static_objects)} static objects to {tracklets_file}")

def main():
    parser = argparse.ArgumentParser(description="Module 2A – Dynamic Producer")
    parser.add_argument("--video", type=str, required=True, help="Path to input video file")
    parser.add_argument("--shots", type=str, required=False, help="Path to shots.json")
    parser.add_argument("--output-dir", type=str, default=config.intermediate_dir)
    args = parser.parse_args()

    video_path = Path(args.video)
    shots_path = Path(args.shots) if args.shots else None
    out_dir = Path(args.output_dir)
    
    process_video(video_path, shots_path, out_dir)

if __name__ == "__main__":
    main()
