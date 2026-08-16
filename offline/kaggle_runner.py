import argparse
import logging
import os
import shutil
from pathlib import Path
import subprocess
import sys
import time
from config_v5 import config

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)-8s %(name)s » %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("KaggleRunner")

def run_module(module_script: str, args: list):
    cmd = [sys.executable, module_script] + args
    log.info(f"Running: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        log.error(f"Module {module_script} failed with exit code {e.returncode}")
        raise e

def cleanup_temp_files():
    """Wipes intermediate directories to prevent Kaggle disk from filling up (73GB limit)."""
    log.info("Cleaning up temporary files...")
    
    # Clean intermediate dir
    intermediate = Path(config.intermediate_dir)
    if intermediate.exists():
        for item in intermediate.iterdir():
            if item.is_file():
                item.unlink()
            elif item.is_dir():
                shutil.rmtree(item)
                
    # Clean output dir
    output = Path(config.output_dir)
    if output.exists():
        for item in output.iterdir():
            if item.is_file():
                item.unlink()
            elif item.is_dir():
                shutil.rmtree(item)
                
    # Recreate dirs
    config.ensure_dirs()
    
    # Create export dirs
    export_dir = Path("/kaggle/working/export")
    export_jsons = export_dir / "jsons"
    export_jsons.mkdir(parents=True, exist_ok=True)
    
    log.info("Cleanup complete.")

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
    
    # 4. Generate 360p Proxy
    run_module("generate_proxy.py", [
        "--video", str(video_path),
        "--output_dir", "/kaggle/working/export"
    ])
    
    # 5. Export JSONs (Rename and move)
    log.info("Exporting JSON files...")
    export_jsons = Path("/kaggle/working/export/jsons")
    json_files = ["shots.json", "scenes.json", "tracklets.json", "actions.json", "lexical_global.json"]
    
    for jf in json_files:
        src = Path(config.intermediate_dir) / jf
        if src.exists():
            dst = export_jsons / f"{video_path.stem}_{jf}"
            shutil.copy2(src, dst)
            
    log.info(f"--- Completed Video: {video_path.name} ---")

def main():
    parser = argparse.ArgumentParser(description="Kaggle Resumable Runner (12-hour limit safe)")
    parser.add_argument("--video_dir", type=str, help="Directory containing videos")
    parser.add_argument("--progress_file", type=str, default="/kaggle/working/processed_videos.txt", help="Path to progress log file")
    args = parser.parse_args()

    if args.video_dir:
        config.video_dir = args.video_dir
        
    config.ensure_dirs()
    video_dir = Path(config.video_dir)
    progress_file = Path(args.progress_file)
    
    log.info(f"Reading videos from: {video_dir}")
    log.info(f"Tracking progress in: {progress_file}")

    # Load processed list
    processed_vids = set()
    if progress_file.exists():
        with open(progress_file, "r", encoding="utf-8") as f:
            for line in f:
                processed_vids.add(line.strip())
                
    log.info(f"Found {len(processed_vids)} already processed videos.")

    # Find videos recursively (support nested Dataset_AIC2026/Videos_L21_a/video/)
    video_paths = sorted(video_dir.rglob("*.mp4"))
    log.info(f"Found {len(video_paths)} total videos.")

    # Filter out processed
    pending_videos = [v for v in video_paths if v.stem not in processed_vids]
    log.info(f"{len(pending_videos)} videos remaining to process.")

    if not pending_videos:
        log.info("All videos processed! Exiting.")
        return

    # Kaggle 12-hour limit survival: Auto-stop after 11.5 hours
    MAX_RUNTIME_SEC = 11.5 * 3600
    start_time = time.time()

    # Process sequentially
    for video_path in pending_videos:
        elapsed_time = time.time() - start_time
        if elapsed_time > MAX_RUNTIME_SEC:
            log.warning(f"Reached {elapsed_time/3600:.2f} hours. Auto-stopping to let Kaggle save outputs!")
            break
            
        try:
            process_video(video_path)
            
            # Log success
            with open(progress_file, "a", encoding="utf-8") as f:
                f.write(f"{video_path.stem}\n")
            log.info(f"Marked {video_path.stem} as processed.")
            
            # CRITICAL: Clean up disk space
            cleanup_temp_files()
            
        except Exception as e:
            log.error(f"Failed processing {video_path.name}: {e}")
            log.warning("Skipping to next video. Logging to failed_videos.txt.")
            
            # Log failure so user can inspect later
            failed_file = progress_file.parent / "failed_videos.txt"
            with open(failed_file, "a", encoding="utf-8") as f:
                f.write(f"{video_path.stem} - ERROR: {str(e)}\n")
                
            # Clean up partial run to save space
            cleanup_temp_files()
            continue

    log.info("Batch processing complete.")

if __name__ == "__main__":
    main()
