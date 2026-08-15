import argparse
import subprocess
from pathlib import Path
from config_v5 import config

def ensure_dir(d: Path):
    if not d.exists():
        d.mkdir(parents=True)

def generate_proxy(video_path: Path, output_dir: Path):
    proxy_dir = output_dir / "proxies"
    ensure_dir(proxy_dir)
    
    proxy_path = proxy_dir / f"{video_path.stem}_proxy.mp4"
    if proxy_path.exists():
        print(f"Proxy already exists: {proxy_path.name}")
        return
        
    print(f"Generating proxy for {video_path.name}...")
    
    # Scale to 360p height, keep aspect ratio. fast preset for speed.
    cmd = [
        "ffmpeg", "-y",
        "-vsync", "0", # CRITICAL: Prevent dropping or duplicating frames to ensure Frame_ID matches 1:1 with original MP4
        "-i", str(video_path),
        "-vf", "scale=-2:360",
        "-c:v", "libx264",
        "-crf", "28",
        "-preset", "veryfast",
        "-c:a", "aac", # keep audio but compress
        "-b:a", "96k",
        str(proxy_path)
    ]
    
    try:
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=True)
        print(f" -> Saved to {proxy_path.name}")
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] FFmpeg failed for {video_path.name}: {e.stderr.decode()}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, default="", help="Path to single video")
    parser.add_argument("--video_dir", type=str, default=config.video_dir)
    parser.add_argument("--output_dir", type=str, default=config.output_dir)
    args = parser.parse_args()
    
    out_dir = Path(args.output_dir)
    
    if args.video:
        video_path = Path(args.video)
        if not video_path.exists():
            print(f"Video not found: {video_path}")
            return
        generate_proxy(video_path, out_dir)
        print("Done generating proxy.")
        return
        
    video_dir = Path(args.video_dir)
    if not video_dir.exists():
        print(f"Video directory not found: {video_dir}")
        return
        
    videos = sorted(video_dir.rglob("*.mp4"))
    print(f"Found {len(videos)} videos. Generating 360p proxies...")
    
    for v in videos:
        generate_proxy(v, out_dir)
        
    print("Done generating proxies.")

if __name__ == "__main__":
    main()
