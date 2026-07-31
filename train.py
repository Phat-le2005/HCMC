import argparse
import logging
from pathlib import Path

from config import ATSMEConfig
from pipeline import PipelineManager

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)-8s %(name)s » %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("Train")

def main():
    parser = argparse.ArgumentParser(description="Run ATSME Extraction Pipeline")
    parser.add_argument("--video_dir", type=str, help="Override video directory in config")
    parser.add_argument("--mapping_csv", type=str, help="Override mapping CSV path in config")
    parser.add_argument("--output_dir", type=str, help="Override output directory in config")
    args = parser.parse_args()

    # Khởi tạo cấu hình mặc định từ class
    config = ATSMEConfig()
    
    # Ghi đè cấu hình nếu có truyền qua tham số dòng lệnh
    if args.video_dir:
        config.video_dir = args.video_dir
    if args.mapping_csv:
        config.mapping_csv = args.mapping_csv
    if args.output_dir:
        config.output_dir = args.output_dir

    log.info("Loaded Configuration:")
    for k, v in config.__dict__.items():
        log.info(f"  {k}: {v}")

    # Khởi tạo Pipeline Manager với file cấu hình
    pipeline = PipelineManager(config)

    video_dir = Path(config.video_dir)
    if not video_dir.exists():
        log.error(f"Video directory not found: {video_dir}")
        return

    video_paths = sorted(video_dir.glob("*.mp4"))
    log.info(f"Found {len(video_paths)} videos in {video_dir}")

    if not video_paths:
        return

    # Chạy xử lý batch cho toàn bộ videos
    pipeline.process_batch(video_paths)
    log.info("Batch processing complete.")

if __name__ == "__main__":
    main()
