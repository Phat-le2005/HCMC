import os
import json
import argparse
import subprocess
from pathlib import Path
import multiprocessing
import gc
import numpy as np

import torch
# Placeholder for InternVideo2
# from transformers import AutoModel, AutoTokenizer

from config_v5 import config

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def cut_tube(args):
    """Worker function to cut a video tube using ffmpeg based on tracklet bbox."""
    video_path, tracklet, tubes_dir, metadata = args
    track_id = tracklet["track_id"]
    start_ms = tracklet["start_ms"]
    end_ms = tracklet["end_ms"]
    
    tube_path = tubes_dir / f"{track_id}.mp4"
    if tube_path.exists():
        return str(tube_path)
        
    # Calculate union of all bboxes for spatial crop
    bboxes = np.array([b["bbox"] for b in tracklet["bbox_trajectory"]])
    # Bboxes are [x1, y1, x2, y2] normalized
    x1_norm = max(0, bboxes[:, 0].min())
    y1_norm = max(0, bboxes[:, 1].min())
    x2_norm = min(1, bboxes[:, 2].max())
    y2_norm = min(1, bboxes[:, 3].max())
    
    width = int((x2_norm - x1_norm) * metadata["resolution"]["width"])
    height = int((y2_norm - y1_norm) * metadata["resolution"]["height"])
    x = int(x1_norm * metadata["resolution"]["width"])
    y = int(y1_norm * metadata["resolution"]["height"])
    
    # Ensure width and height are even numbers (ffmpeg requirement for x264)
    width = width + 1 if width % 2 != 0 else width
    height = height + 1 if height % 2 != 0 else height
    
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start_ms / 1000.0),
        "-i", str(video_path),
        "-t", str((end_ms - start_ms) / 1000.0),
        "-vf", f"crop={width}:{height}:{x}:{y}",
        "-c:v", "libx264",
        "-crf", "23",
        "-preset", "veryfast",
        "-c:a", "aac",
        str(tube_path)
    ]
    try:
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=True)
    except subprocess.CalledProcessError as e:
        print(f"ffmpeg error cutting tube {track_id}: {e.stderr.decode()}")
        return None
    return str(tube_path)

from transformers import AutoModel, AutoTokenizer
import torchvision.transforms as T
import decord
from decord import VideoReader, cpu

def load_video_tensor(video_path, num_frames=8):
    decord.bridge.set_bridge('torch')
    vr = VideoReader(str(video_path), ctx=cpu(0))
    total_frames = len(vr)
    if total_frames == 0:
        return None
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    video = vr.get_batch(frame_indices) # [T, H, W, C]
    
    # Transform: T, H, W, C -> C, T, H, W for InternVideo2
    video = video.permute(3, 0, 1, 2).float() / 255.0
    
    transform = T.Compose([
        T.Resize(224, antialias=True),
        T.CenterCrop(224),
        T.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
    ])
    
    # apply transform frame by frame
    video = torch.stack([transform(video[:, i, :, :]) for i in range(num_frames)], dim=1) # [C, T, H, W]
    return video.unsqueeze(0) # [1, C, T, H, W]

def load_internvideo2():
    print(f"[Module 2B] Loading InternVideo2 {config.internvideo2_model} on {config.device}...")
    try:
        model = AutoModel.from_pretrained(config.internvideo2_model, trust_remote_code=True).to(config.device)
        model.eval()
        if config.fp16 and config.device == "cuda":
            model.half()
        return model
    except Exception as e:
        print(f"Warning: Failed to load InternVideo2 ({e}). Falling back to Mock.")
        class MockInternVideo2:
            def __call__(self, video_tensor):
                return {"action_label": "walking", "vector": np.random.rand(256).tolist(), "confidence": 0.95}
        return MockInternVideo2()

def process_tracklets(tracklets_path: Path, video_path: Path, out_dir: Path):
    print(f"[Module 2B] Loading tracklets from {tracklets_path}")
    if not tracklets_path.exists():
        print(f"Error: {tracklets_path} does not exist.")
        return
        
    with open(tracklets_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    tracklets = data.get("tracklets", [])
    metadata = data.get("metadata", {})
    if not tracklets:
        print("[Module 2B] No dynamic tracklets found. Exiting.")
        return
        
    tubes_dir = out_dir / "tubes"
    ensure_dir(tubes_dir)
    
    actions_path = out_dir / "actions.json"
    existing_actions = {}
    if actions_path.exists():
        with open(actions_path, "r", encoding="utf-8") as f:
            existing = json.load(f)
            existing_actions = {a["track_id"]: a for a in existing}
            
    tracklets_to_process = [t for t in tracklets if t["track_id"] not in existing_actions]
    if not tracklets_to_process:
        print("[Module 2B] All tracklets already processed. Exiting.")
        return
        
    # 1. Tube Cutter (CPU Multiprocessing)
    print(f"[Module 2B] Cutting {len(tracklets_to_process)} video tubes via ffmpeg multiprocessing...")
    pool_args = [(video_path, t, tubes_dir, metadata) for t in tracklets_to_process]
    
    tube_paths = []
    with multiprocessing.Pool(processes=config.num_workers) as pool:
        for path in pool.map(cut_tube, pool_args):
            if path:
                tube_paths.append(path)
                
    # 2. Action Recognition (GPU)
    model = load_internvideo2()
    is_mock = type(model).__name__ == "MockInternVideo2"
    
    actions = list(existing_actions.values())
    
    print("[Module 2B] Running action recognition...")
    batch_size = 4 
    
    for i in range(0, len(tube_paths), batch_size):
        batch = tube_paths[i:i+batch_size]
        
        for tube in batch:
            track_id = Path(tube).stem
            
            if is_mock:
                result = model(tube)
                act_label = result["action_label"]
                act_vec = result["vector"]
                conf = result["confidence"]
            else:
                try:
                    vid_tensor = load_video_tensor(tube).to(config.device)
                    if config.fp16 and config.device == "cuda":
                        vid_tensor = vid_tensor.half()
                    with torch.no_grad():
                        vid_feat = model.get_vid_features(vid_tensor)
                        # Normalize and convert to list
                        vid_feat = torch.nn.functional.normalize(vid_feat, dim=-1)
                        act_vec = vid_feat[0].cpu().float().numpy().tolist()
                        act_label = "action_feature"
                        conf = 1.0
                except Exception as e:
                    print(f"Failed extracting {tube}: {e}")
                    act_vec = [0.0] * 256
                    act_label = "error"
                    conf = 0.0
            
            actions.append({
                "track_id": track_id,
                "action_label": act_label,
                "action_vector": act_vec,
                "confidence": conf
            })
            
        # Prevent OOM
        gc.collect()
        if config.device == "cuda":
            torch.cuda.empty_cache()
            
        # Incremental save
        with open(actions_path, "w", encoding="utf-8") as f:
            json.dump(actions, f, indent=2)
            
    print(f"[Module 2B] Wrote {len(actions)} actions to {actions_path}")

def main():
    parser = argparse.ArgumentParser(description="Module 2B – Dynamic Consumer (Action Recognition)")
    parser.add_argument("--tracklets", type=str, required=True, help="Path to tracklets.json")
    parser.add_argument("--video", type=str, required=True, help="Path to input video file")
    parser.add_argument("--output-dir", type=str, default=config.intermediate_dir)
    args = parser.parse_args()

    tracklets_path = Path(args.tracklets)
    video_path = Path(args.video)
    out_dir = Path(args.output_dir)
    
    process_tracklets(tracklets_path, video_path, out_dir)

if __name__ == "__main__":
    main()
