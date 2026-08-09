import os
import json
import argparse
from pathlib import Path
import numpy as np
from collections import defaultdict, Counter
import gc

from decord import VideoReader, cpu
import torch
from PIL import Image
from ultralytics import YOLO
from paddleocr import PaddleOCR
from transformers import AutoModel, AutoImageProcessor

from config_v5 import config

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

class ReIDMemoryPool:
    def __init__(self, ttl=60):
        self.pool = {}
        self.ttl = ttl

    def add(self, track_id, embedding, timestamp):
        self.pool[track_id] = {"embedding": embedding, "timestamp": timestamp}

    def clean(self, current_timestamp):
        keys_to_remove = [tid for tid, data in self.pool.items()
                          if (current_timestamp - data["timestamp"]) / 1000.0 > self.ttl]
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
            sim = float(np.dot(new_emb, pool_emb))
            if sim > best_sim:
                best_sim = sim
                best_tid = tid
        if best_sim > threshold:
            return best_tid, best_sim
        return None, 0.0


def get_vision_embedding(model, pixel_values):
    """Extract image embedding from SigLIP model with robust fallback."""
    with torch.no_grad():
        if hasattr(model, 'get_image_features'):
            out = model.get_image_features(pixel_values=pixel_values)
            if isinstance(out, torch.Tensor):
                return out
            if hasattr(out, 'pooler_output'):
                return out.pooler_output
            return out[0] if isinstance(out, tuple) else out
        if hasattr(model, 'vision_model'):
            out = model.vision_model(pixel_values=pixel_values)
            if hasattr(out, 'pooler_output') and out.pooler_output is not None:
                return out.pooler_output
            return out.last_hidden_state[:, 0]
        out = model(pixel_values=pixel_values)
        if hasattr(out, 'pooler_output') and out.pooler_output is not None:
            return out.pooler_output
        return out.last_hidden_state[:, 0]


def crop_and_embed(frame_np, bbox, processor, model):
    """Crop bounding box from frame and compute SigLIP embedding."""
    h, w = frame_np.shape[:2]
    x1, y1, x2, y2 = bbox
    x1, y1, x2, y2 = int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)
    x1, y1 = max(0, x1 - 5), max(0, y1 - 5)
    x2, y2 = min(w, x2 + 5), min(h, y2 + 5)
    if x2 <= x1 or y2 <= y1:
        return [0.0] * 768

    crop = frame_np[y1:y2, x1:x2]
    img = Image.fromarray(crop).convert("RGB")
    inputs = processor(images=img, return_tensors="pt").to(config.device)
    if config.fp16 and config.device == "cuda":
        inputs['pixel_values'] = inputs['pixel_values'].half()

    features = get_vision_embedding(model, inputs['pixel_values'])
    return features.squeeze().float().cpu().tolist()


def run_local_ocr(frame_np, bbox, ocr):
    h, w = frame_np.shape[:2]
    x1, y1, x2, y2 = bbox
    x1, y1, x2, y2 = int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)
    if x2 <= x1 or y2 <= y1:
        return ""
    crop = frame_np[y1:y2, x1:x2]
    try:
        result = ocr.ocr(crop, cls=config.ocr_use_angle_cls)
        if not result or not result[0]:
            return ""
        texts = [line[1][0] for line in result[0] if line and len(line) > 1 and line[1]]
        return " ".join(texts)
    except Exception as e:
        return ""


def process_video(video_path: Path, shots_path: Path, out_dir: Path):
    ensure_dir(out_dir)
    tracklets_file = out_dir / "tracklets.json"

    if tracklets_file.exists():
        print(f"[Module 2A] {tracklets_file} already exists. Skipping.")
        return

    print(f"[Module 2A] Loading video: {video_path}")
    vr = VideoReader(str(video_path), ctx=cpu(0))
    video_fps = vr.get_avg_fps()
    total_video_frames = len(vr)
    duration_ms = total_video_frames / video_fps * 1000
    sample_interval = max(1, int(video_fps / config.shot_fps))

    # Load Models
    print(f"[Module 2A] Loading Models...")
    yolo_model = YOLO(config.yolo_model)
    yolo_model.to(config.device)

    siglip_processor = AutoImageProcessor.from_pretrained(config.siglip_model)
    siglip_model = AutoModel.from_pretrained(config.siglip_model).to(config.device)
    siglip_model.eval()
    if config.fp16 and config.device == "cuda":
        siglip_model.half()

    ocr = PaddleOCR(use_angle_cls=config.ocr_use_angle_cls, lang=config.ocr_lang)

    reid_pool = ReIDMemoryPool(ttl=config.reid_pool_ttl_sec)

    tracklets_dict = defaultdict(lambda: {
        "class_labels": [],
        "start_ms": float('inf'),
        "end_ms": 0,
        "bbox_trajectory": [],
        "unified_id": None
    })

    static_objects = []
    ocr_local = []

    print("[Module 2A] Running YOLO tracking + Real SigLIP ReID...")

    shots = []
    if shots_path and shots_path.exists():
        with open(shots_path, "r", encoding="utf-8") as f:
            shots = json.load(f)

    def get_shot_id_from_ms(ms):
        for s in shots:
            if s["start_ms"] <= ms <= s["end_ms"]:
                return s["shot_id"]
        return "unknown"

    # Convert range to list for proper slicing
    frame_indices = list(range(0, total_video_frames, sample_interval))

    batch_size = config.batch_size_gpu
    total_frames = len(frame_indices)
    processed = 0

    for i in range(0, total_frames, batch_size):
        batch_idx = frame_indices[i:i+batch_size]

        try:
            frames = vr.get_batch(batch_idx).asnumpy()
        except Exception as e:
            print(f"[Module 2A] Warning: Failed to read batch at index {i}: {e}. Skipping batch.")
            continue

        try:
            results = yolo_model.track(frames, persist=True, tracker="bytetrack.yaml", verbose=False)
        except Exception as e:
            print(f"[Module 2A] Warning: Batch YOLO tracking failed at batch {i//batch_size}: {e}")
            print(f"[Module 2A] Falling back to frame-by-frame tracking...")
            results = []
            for single_frame in frames:
                try:
                    single_result = yolo_model.track(single_frame, persist=True, tracker="bytetrack.yaml", verbose=False)
                    results.extend(single_result)
                except Exception:
                    results.append(None)

        for j, res in enumerate(results):
            if res is None:
                continue
            if j >= len(batch_idx):
                break
            actual_idx = batch_idx[j]
            current_ms = (actual_idx / video_fps) * 1000
            reid_pool.clean(current_ms)

            if res.boxes is None or res.boxes.id is None:
                continue

            boxes = res.boxes.xyxyn.cpu().numpy()
            track_ids = res.boxes.id.int().cpu().numpy()
            classes = res.boxes.cls.int().cpu().numpy()
            names = res.names

            for box, tid, cls_idx in zip(boxes, track_ids, classes):
                str_tid = str(int(tid))
                label = names[int(cls_idx)]

                emb = crop_and_embed(frames[j], box, siglip_processor, siglip_model)

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

                if label in config.ocr_trigger_classes:
                    text = run_local_ocr(frames[j], box, ocr)
                    if text:
                        ocr_local.append({
                            "track_id": unified_tid,
                            "frame_idx": int(actual_idx),
                            "recognized_text": text
                        })

                reid_pool.add(unified_tid, emb, current_ms)

        # Prevent OOM
        del frames, results
        gc.collect()
        if config.device == "cuda":
            torch.cuda.empty_cache()

        processed += len(batch_idx)
        if processed % (batch_size * 5) == 0 or processed >= total_frames:
            print(f"[Module 2A] Progress: {processed}/{total_frames} frames ({100*processed/total_frames:.1f}%)")

    print("[Module 2A] Finalizing tracklets...")
    final_tracklets = []

    for tid, data in tracklets_dict.items():
        if not data["bbox_trajectory"]:
            continue

        class_counts = Counter(data["class_labels"])
        majority_class = class_counts.most_common(1)[0][0] if class_counts else "unknown"

        bboxes = np.array([b["bbox"] for b in data["bbox_trajectory"]])
        centers_x = (bboxes[:, 0] + bboxes[:, 2]) / 2.0
        centers_y = (bboxes[:, 1] + bboxes[:, 3]) / 2.0
        std_x = float(np.std(centers_x))
        std_y = float(np.std(centers_y))

        if std_x < 0.02 and std_y < 0.02:
            mid_entry = data["bbox_trajectory"][len(data["bbox_trajectory"])//2]
            mid_idx = mid_entry["frame_idx"]
            mid_bbox = mid_entry["bbox"]

            try:
                mid_frame = vr[mid_idx].asnumpy()
                static_vec = crop_and_embed(mid_frame, mid_bbox, siglip_processor, siglip_model)
            except Exception:
                static_vec = [0.0] * 768

            shot_id = get_shot_id_from_ms(data["start_ms"])
            obj_ocr = [o["recognized_text"] for o in ocr_local if o["track_id"] == tid]
            obj_ocr_text = obj_ocr[0] if obj_ocr else ""

            static_objects.append({
                "object_id": f"static_{tid}",
                "shot_id": shot_id,
                "class_label": majority_class,
                "bbox": mid_bbox,
                "siglip_vector": static_vec,
                "ocr_text": obj_ocr_text
            })
        else:
            uid = data["unified_id"] if data["unified_id"] else tid
            final_tracklets.append({
                "track_id": uid,
                "unified_id": uid,
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
            "resolution": {"width": int(vr[0].shape[1]), "height": int(vr[0].shape[0]), "unit": "pixel"},
            "duration_ms": duration_ms
        },
        "tracklets": final_tracklets,
        "static_objects": static_objects,
        "ocr_local": ocr_local
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
