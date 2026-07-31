from dataclasses import dataclass

@dataclass
class ATSMEConfig:
    """
    Tất cả cấu hình siêu tham số (Hyperparameters) và đường dẫn (Paths)
    cho hệ thống ATSME.
    """
    
    # ── I/O Paths ─────────────────────────────────────────────────────────────
    video_dir: str = "/kaggle/input/datasets/pha1t2/video1/video/L21_V001.mp4"
    mapping_csv: str = "/kaggle/input/datasets/pha1t2/mapkey/map-keyframes"
    output_dir: str = "/kaggle/working/outputs"
    
    # ── Models ────────────────────────────────────────────────────────────────
    yolo_model_path: str = "yolov9c.pt"
    siglip_model_id: str = "google/siglip-so400m-patch14-384"
    qwen_model_id: str = "Qwen/Qwen2.5-VL-2B-Instruct"
    whisper_size: str = "base"
    use_vllm: bool = False
    
    # ── Shot Segmentation Params ──────────────────────────────────────────────
    shot_threshold: float = 27.0
    shot_min_scene_len: int = 15
    
    # ── Tracking & Heuristics Params ──────────────────────────────────────────
    track_conf_thresh: float = 0.30
    track_iou_thresh: float = 0.45
    track_blur_thresh: float = 100.0
    track_min_bbox_area_ratio: float = 0.05
    track_n_keyframes: int = 3
    
    # ── Deep Semantic Params ──────────────────────────────────────────────────
    semantic_batch_size: int = 8
