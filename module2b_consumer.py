import os
import json
import argparse
import subprocess
from pathlib import Path
import multiprocessing
import gc

try:
    import torch
    # from transformers import AutoModel, AutoTokenizer
    # (InternVideo2 logic goes here in a real environment)
except ImportError:
    pass

from config_v5 import config

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def cut_tube(args):
    """Worker function to cut a video tube using ffmpeg based on tracklet bbox."""
    video_path, tracklet, tubes_dir = args
    track_id = tracklet["track_id"]
    start_ms = tracklet["start_ms"]
    end_ms = tracklet["end_ms"]
    
    # Calculate crop logic based on union of bboxes or a dynamic crop filter
    # For simplicity, we just cut the time segment
    tube_path = tubes_dir / f"{track_id}.mp4"
    
    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(video_path),
        "-ss", str(start_ms / 1000.0),
        "-to", str(end_ms / 1000.0),
        # You can add a -vf crop=... here if you have average bbox dimensions
        "-c:v", "libx264",
        "-crf", "23",
        "-preset", "veryfast",
        "-c:a", "aac",
        str(tube_path)
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return str(tube_path)

def process_tracklets(tracklets_path: Path, video_path: Path, out_dir: Path):
    print(f"[Module 2B] Loading tracklets from {tracklets_path}")
    with open(tracklets_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    tracklets = data.get("tracklets", [])
    if not tracklets:
        print("[Module 2B] No dynamic tracklets found. Exiting.")
        return
        
    tubes_dir = out_dir / "tubes"
    ensure_dir(tubes_dir)
    
    # 1. Tube Cutter (CPU Multiprocessing)
    print(f"[Module 2B] Cutting {len(tracklets)} video tubes via ffmpeg multiprocessing...")
    pool_args = [(video_path, t, tubes_dir) for t in tracklets]
    with multiprocessing.Pool(processes=config.num_workers) as pool:
        tube_paths = pool.map(cut_tube, pool_args)
        
    # 2. Action Recognition (GPU)
    print("[Module 2B] Loading InternVideo2 for Action Recognition...")
    # model = AutoModel.from_pretrained(config.internvideo2_model)
    # model.eval().cuda()
    
    actions = []
    
    # Simulating batches
    for tube in tube_paths:
        track_id = Path(tube).stem
        # mock feature extraction
        action_vector = [0.1] * 256 # mock 256-d vector
        
        actions.append({
            "track_id": track_id,
            "action_label": "walking", # mock
            "action_vector": action_vector,
            "confidence": 0.88
        })
        
        # Prevent OOM
        # gc.collect()
        # torch.cuda.empty_cache()

    actions_path = out_dir / "actions.json"
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
