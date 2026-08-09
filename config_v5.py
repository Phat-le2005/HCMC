import os
from dataclasses import dataclass, field
from typing import List

@dataclass
class ATSMEConfig:
    # -------------------- I/O Paths --------------------
    video_dir: str = r"d:/HCMC/videos"
    intermediate_dir: str = r"d:/HCMC/ATSME/intermediate"
    output_dir: str = r"d:/HCMC/ATSME/output"
    # -------------------- Model IDs --------------------
    siglip_model: str = "google/siglip2-so400m-patch14-384"
    internvideo2_model: str = "OpenGVLab/InternVideo2-Stage2_1B-224p-f4"
    fastreid_model: str = "market_bot_R50"
    audio_embedding: str = "WavLM-Base"
    whisper_model: str = "medium"
    # -------------------- Hyper‑parameters --------------------
    shot_detector_content_threshold: float = 30.0  # PySceneDetect ContentDetector
    shot_fps: int = 5
    keyframe_resize: tuple = (640, 640)
    batch_size_gpu: int = 32
    num_workers: int = 8
    fp16: bool = True
    # -------------------- Tracking --------------------
    iou_threshold: float = 0.3
    reid_similarity_thresh: float = 0.75
    reid_pool_ttl_sec: int = 60
    # -------------------- Clustering --------------------
    scene_cluster_alpha: float = 0.005  # penalty for temporal distance
    scene_min_shots: int = 3
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
            os.makedirs(d, exist_ok=True)

# Global config instance
config = ATSMEConfig()
config.ensure_dirs()
