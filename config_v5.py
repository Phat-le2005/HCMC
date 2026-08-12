import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict

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
    whisper_model: str = "large-v3"
    yolo_model: str = "yolov8n.pt"
    # -------------------- Hyper‑parameters --------------------
    use_hybrid_sbd: bool = True
    shot_detector_content_threshold: float = 30.0  # PySceneDetect ContentDetector
    shot_merge_threshold_ms: float = 500.0
    shot_fps: int = 5
    keyframe_resize: tuple = (640, 640)
    batch_size_gpu: int = 32
    num_workers: int = 8
    fp16: bool = True
    device: str = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    ocr_lang: str = "vi"
    ocr_use_angle_cls: bool = True
    ocr_keyframes_per_shot: int = 3
    asr_language: str = "vi"
    # -------------------- Tracking --------------------
    iou_threshold: float = 0.3
    reid_similarity_thresh: float = 0.75
    reid_pool_ttl_sec: int = 60
    reid_method: str = "siglip"
    ocr_trigger_classes: List[str] = field(default_factory=lambda: ["book", "signboard", "tv", "car"])
    # -------------------- Clustering --------------------
    scene_cluster_alpha: float = 0.005  # penalty for temporal distance
    scene_min_shots: int = 3
    overlap_threshold: float = 0.5  # seconds
    # -------------------- News Classification --------------------
    news_prompts: Dict[str, str] = field(default_factory=lambda: {
        "anchor": "a single news anchor sitting behind a desk in a TV studio reading news, with studio lights and a screen behind",
        "field_report": "a journalist or reporter standing outdoors on location holding a microphone, reporting live from a scene",
        "interview": "a close-up shot of a person being interviewed, with a name tag or title overlay on screen",
        "infographic": "a full-screen infographic, chart, map, statistics table, or animated data visualization on a news broadcast",
        "b_roll": "outdoor footage of streets, buildings, landscapes, crowds, vehicles, or nature without any person talking directly to camera",
        "transition": "a TV channel logo, program title card, opening sequence, or animated transition graphic between news segments"
    })
    news_classify_threshold: float = 0.35
    # -------------------- Service Endpoints --------------------
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "password"
    milvus_host: str = "localhost"
    milvus_port: int = 19530
    elasticsearch_host: str = "http://localhost:9200"
    
    # -------------------- DB Collections/Indices --------------------
    milvus_scene_collection: str = "atsme_scene_vectors"
    milvus_shot_collection: str = "atsme_shot_vectors"
    milvus_object_collection: str = "atsme_object_vectors"
    milvus_event_collection: str = "atsme_event_vectors"
    es_global_index: str = "atsme_global_text"
    es_local_index: str = "atsme_local_text"
    
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
