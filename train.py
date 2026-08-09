import argparse
import logging
from pathlib import Path
import subprocess
import sys

from config_v5 import config

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)-8s %(name)s » %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("Train")

def run_module(module_script: str, args: list):
    cmd = [sys.executable, module_script] + args
    log.info(f"Running: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        log.error(f"Module {module_script} failed with exit code {e.returncode}")
        raise e

def process_video(video_path: Path):
    log.info(f"--- Processing Video: {video_path.name} ---")
    
    # 1. Static Pipeline
    run_module("module1_static.py", ["--video", str(video_path)])
    
    # 2. Dynamic Producer
    shots_file = Path(config.intermediate_dir) / "shots.json"
    run_module("module2a_producer.py", [
        "--video", str(video_path), 
        "--shots", str(shots_file)
    ])
    
    # 3. Dynamic Consumer
    tracklets_file = Path(config.intermediate_dir) / "tracklets.json"
    run_module("module2b_consumer.py", [
        "--video", str(video_path), 
        "--tracklets", str(tracklets_file)
    ])
    
    # 4. Graph Builder (dry-run by default — no DB on Kaggle)
    run_module("module3_graph_builder.py", [
        "--input-dir", config.intermediate_dir,
        "--output-dir", config.output_dir,
        "--dry-run"
    ])
    
    log.info(f"--- Completed Video: {video_path.name} ---")

def main():
    parser = argparse.ArgumentParser(description="Run ATSME v5.2 Extraction Pipeline")
    parser.add_argument("--video_dir", type=str, help="Override video directory in config")
    parser.add_argument("--output_dir", type=str, help="Override output directory in config")
    args = parser.parse_args()

    # Khởi tạo cấu hình
    if args.video_dir:
        config.video_dir = args.video_dir
    if args.output_dir:
        config.output_dir = args.output_dir

    config.ensure_dirs()

    log.info("Loaded Configuration:")
    for k, v in config.__dict__.items():
        if not k.startswith("__"):
            log.info(f"  {k}: {v}")

    video_dir = Path(config.video_dir)
    if not video_dir.exists():
        log.error(f"Video directory or file not found: {video_dir}")
        return

    if video_dir.is_file() and video_dir.suffix == '.mp4':
        video_paths = [video_dir]
    else:
        video_paths = sorted(video_dir.glob("*.mp4"))
        
    log.info(f"Found {len(video_paths)} videos for {video_dir}")

    if not video_paths:
        return

    # Chạy xử lý batch tuần tự cho toàn bộ videos
    for video_path in video_paths:
        try:
            process_video(video_path)
        except Exception as e:
            log.error(f"Failed processing {video_path.name}: {e}")
            continue

    log.info("Batch processing complete.")

if __name__ == "__main__":
    main()
