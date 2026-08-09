import os
from dataclasses import dataclass, field
from typing import List

@dataclass
class ATSMEConfig:
    # -------------------- I/O Paths --------------------
    video_dir: str = os.getenv("ATSME_VIDEO_DIR", str(Path(__file__).parent.parent / "videos"))
    intermediate_dir: str = os.getenv("ATSME_INTERMEDIATE_DIR", str(Path(__file__).parent / "intermediate"))
    output_dir: str = os.getenv("ATSME_OUTPUT_DIR", str(Path(__file__).parent / "output"))
    # -------------------- Model IDs --------------------
    siglip_model: str = "google/siglip2-so400m-patch14-384"
    internvideo2_model: str = "OpenGVLab/InternVideo2-Stage2_1B-224p-f4"
    fastreid_model: str = "market_bot_R50"
    audio_embedding: str = "microsoft/wavlm-base"
    whisper_model: str = "medium"
    # -------------------- Hyper‑parameters --------------------
    shot_detector_content_threshold: float = 30.0  # PySceneDetect ContentDetector
    shot_fps: int = 5
    keyframe_resize: tuple = (640, 640)
    batch_size_gpu: int = 32
    num_workers: int = 8
    fp16: bool = True
    device: str = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    ocr_lang: str = "en"
    # -------------------- Tracking --------------------
    iou_threshold: float = 0.3
    reid_similarity_thresh: float = 0.75
    reid_pool_ttl_sec: int = 60
    # -------------------- Clustering --------------------
    scene_cluster_alpha: float = 0.005  # penalty for temporal distance
    scene_min_shots: int = 3
    overlap_threshold: float = 0.5  # seconds
    # -------------------- Service Endpoints --------------------
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "password"
    milvus_host: str = "localhost"
    milvus_port: int = 19530
    elasticsearch_host: str = "http://localhost:9200"
    # -------------------- Misc --------------------
    log_level: str = "INFO"

    def ensure_dirs(self):
        for d in [self.intermediate_dir, self.output_dir]:
            try:
                os.makedirs(d, exist_ok=True)
            except Exception as e:
                print(f"Warning: Could not create directory {d}: {e}")

# Global config instance
config = ATSMEConfig()
