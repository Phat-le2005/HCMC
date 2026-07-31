from __future__ import annotations

import gc
import json
import logging
import os
import subprocess
import tempfile
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from config import ATSMEConfig

warnings.filterwarnings("ignore")

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)-8s %(name)s » %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("ATSME")

# ── Device helpers ────────────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def free_vram(label: str = "") -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    gc.collect()
    if label:
        log.debug("VRAM freed after: %s", label)


# ─────────────────────────────────────────────────────────────────────────────
# 1.  SHOT SEGMENTATION
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class Shot:
    shot_id: int
    start_frame: int          
    end_frame: int            
    video_id: str = ""

    def __repr__(self) -> str:
        return f"Shot(id={self.shot_id}, frames={self.start_frame}-{self.end_frame})"


class ShotDetector:
    def __init__(self, config: ATSMEConfig) -> None:
        from scenedetect import open_video, SceneManager
        from scenedetect.detectors import ContentDetector

        self.threshold = config.shot_threshold
        self.min_scene_len = config.shot_min_scene_len
        self._open_video = open_video
        self._SceneManager = SceneManager
        self._ContentDetector = ContentDetector

    def detect(self, video_path: str | Path) -> list[Shot]:
        video_path = Path(video_path)
        video_id = video_path.stem

        log.info("Shot detection → %s", video_path.name)
        video = self._open_video(str(video_path))
        scene_manager = self._SceneManager()
        scene_manager.add_detector(
            self._ContentDetector(
                threshold=self.threshold,
                min_scene_len=self.min_scene_len,
            )
        )
        scene_manager.detect_scenes(video, show_progress=False)
        scene_list = scene_manager.get_scene_list()

        shots: list[Shot] = []
        MAX_FRAMES_PER_SHOT = 300  # Chia nhỏ để tránh tràn RAM hệ thống
        for start_tc, end_tc in scene_list:
            start_f = start_tc.get_frames()
            end_f = max(end_tc.get_frames() - 1, start_f)
            
            cur_start = start_f
            while cur_start <= end_f:
                cur_end = min(cur_start + MAX_FRAMES_PER_SHOT - 1, end_f)
                shots.append(
                    Shot(
                        shot_id=len(shots),
                        start_frame=cur_start,
                        end_frame=cur_end,
                        video_id=video_id,
                    )
                )
                cur_start = cur_end + 1

        if not shots:
            cap = cv2.VideoCapture(str(video_path))
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            start_f = 0
            end_f = max(total - 1, 0)
            
            cur_start = start_f
            while cur_start <= end_f:
                cur_end = min(cur_start + MAX_FRAMES_PER_SHOT - 1, end_f)
                shots.append(Shot(shot_id=len(shots), start_frame=cur_start, end_frame=cur_end, video_id=video_id))
                cur_start = cur_end + 1

        log.info("Detected %d shot(s) in '%s'", len(shots), video_path.name)
        return shots


# ─────────────────────────────────────────────────────────────────────────────
# 2.  TRACKING & HEURISTIC KEYFRAME SELECTION
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class Tracklet:
    tracklet_id: int
    shot_id: int
    video_id: str
    frames: list[int] = field(default_factory=list)
    bboxes: list[tuple[float, float, float, float]] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)
    keyframes: list[int] = field(default_factory=list)
    keyframe_images: list[np.ndarray] = field(default_factory=list)


class TrackletBuilder:
    def __init__(self, config: ATSMEConfig) -> None:
        from ultralytics import YOLO

        self.conf_thresh = config.track_conf_thresh
        self.iou_thresh = config.track_iou_thresh
        self.blur_thresh = config.track_blur_thresh
        self.min_bbox_area_ratio = config.track_min_bbox_area_ratio
        self.n_keyframes = config.track_n_keyframes

        log.info("Loading YOLO model: %s", config.yolo_model_path)
        self.model = YOLO(config.yolo_model_path)
        self.model.to(DEVICE)

    @staticmethod
    def _laplacian_variance(frame_bgr: np.ndarray) -> float:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    def _bbox_area_ratio(self, bbox: tuple[float, float, float, float], frame_h: int, frame_w: int) -> float:
        x1, y1, x2, y2 = bbox
        bbox_area = max(0.0, (x2 - x1) * (y2 - y1))
        return bbox_area / (frame_h * frame_w + 1e-9)

    def _heuristic_keyframe_selector(self, tracklet: Tracklet, frame_images: dict[int, np.ndarray], frame_h: int, frame_w: int) -> list[int]:
        candidates: list[tuple[int, float, float]] = []
        for frame_idx, bbox in zip(tracklet.frames, tracklet.bboxes):
            if frame_idx not in frame_images:
                continue
            img = frame_images[frame_idx]
            sharpness = self._laplacian_variance(img)
            if sharpness < self.blur_thresh:
                continue
            area_ratio = self._bbox_area_ratio(bbox, frame_h, frame_w)
            if area_ratio < self.min_bbox_area_ratio:
                continue
            candidates.append((frame_idx, sharpness, area_ratio))

        if not candidates:
            candidates = [(f, 0.0, 0.0) for f in tracklet.frames]

        candidates.sort(key=lambda x: x[0])
        candidate_frames = [c[0] for c in candidates]
        k = min(self.n_keyframes, len(candidate_frames))
        if k == 0: return []
        if k == 1: return [candidate_frames[0]]
        
        indices = np.linspace(0, len(candidate_frames) - 1, k, dtype=int)
        return [candidate_frames[i] for i in indices]

    def build_tracklets(self, video_path: str | Path, shots: list[Shot], video_id: str) -> list[Tracklet]:
        video_path = Path(video_path)
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        all_tracklets: list[Tracklet] = []
        global_tracklet_id = 0

        for shot in shots:
            cap.set(cv2.CAP_PROP_POS_FRAMES, shot.start_frame)
            shot_frames: dict[int, np.ndarray] = {}

            for fid in range(shot.start_frame, shot.end_frame + 1):
                ret, frame = cap.read()
                if not ret: break
                shot_frames[fid] = frame

            if not shot_frames:
                continue

            frame_list = [shot_frames[f] for f in sorted(shot_frames.keys())]
            frame_ids = sorted(shot_frames.keys())

            results = self.model.track(
                source=frame_list,
                conf=self.conf_thresh,
                iou=self.iou_thresh,
                tracker="bytetrack.yaml",
                persist=True,
                verbose=False,
                stream=True,
                device=DEVICE,
            )

            shot_tracklets: dict[int, Tracklet] = {}
            for local_i, result in enumerate(results):
                fid = frame_ids[local_i]
                if result.boxes is None or result.boxes.id is None:
                    continue

                boxes_xyxy = result.boxes.xyxy.cpu().numpy()
                track_ids = result.boxes.id.cpu().numpy().astype(int)
                confs = result.boxes.conf.cpu().numpy()

                for tid, bbox, conf in zip(track_ids, boxes_xyxy, confs):
                    if tid not in shot_tracklets:
                        shot_tracklets[tid] = Tracklet(
                            tracklet_id=global_tracklet_id,
                            shot_id=shot.shot_id,
                            video_id=video_id,
                        )
                        global_tracklet_id += 1
                    t = shot_tracklets[tid]
                    t.frames.append(fid)
                    t.bboxes.append(tuple(bbox.tolist()))
                    t.scores.append(float(conf))

            for t in shot_tracklets.values():
                t.keyframes = self._heuristic_keyframe_selector(t, shot_frames, frame_h, frame_w)
                t.keyframe_images = [shot_frames[f] for f in t.keyframes if f in shot_frames]
                all_tracklets.append(t)
                
            # Giải phóng RAM triệt để sau mỗi shot
            shot_frames.clear()
            if 'frame_list' in locals():
                del frame_list
            gc.collect()

        cap.release()
        free_vram("TrackletBuilder")
        return all_tracklets


# ─────────────────────────────────────────────────────────────────────────────
# 3.  DEEP SEMANTIC EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class SemanticResult:
    tracklet_id: int
    shot_id: int
    video_id: str
    siglip_vector: np.ndarray
    ocr_texts: list[str] = field(default_factory=list)
    event_evolution: list[dict] = field(default_factory=list)
    keyframe_global_ids: list[int] = field(default_factory=list)


class SemanticExtractor:
    QWEN_PROMPT = "Describe the action and state of the main object in 1 short sentence."

    def __init__(self, config: ATSMEConfig) -> None:
        self.config = config
        self._siglip_processor = None
        self._siglip_model = None
        self._qwen_processor = None
        self._qwen_model = None
        self._ocr_engine = None
        self._vllm_engine = None

    def _load_siglip(self) -> None:
        if self._siglip_model is not None: return
        from transformers import AutoProcessor, AutoModel
        self._siglip_processor = AutoProcessor.from_pretrained(self.config.siglip_model_id)
        self._siglip_model = AutoModel.from_pretrained(
            self.config.siglip_model_id, torch_dtype=torch.float16, device_map=DEVICE
        ).eval()

    def _load_qwen(self) -> None:
        if self._qwen_model is not None or self._vllm_engine is not None: return
        if self.config.use_vllm:
            try:
                from vllm import LLM
                self._vllm_engine = LLM(
                    model=self.config.qwen_model_id,
                    max_model_len=2048,
                    dtype="float16",
                    limit_mm_per_prompt={"image": 1},
                    gpu_memory_utilization=0.45,
                )
                return
            except Exception as e:
                log.warning("vLLM load failed (%s). Falling back to HuggingFace.", e)

        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
        self._qwen_processor = AutoProcessor.from_pretrained(self.config.qwen_model_id, trust_remote_code=True)
        self._qwen_model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.config.qwen_model_id, torch_dtype=torch.float16, device_map=DEVICE, trust_remote_code=True
        ).eval()

    def _load_ocr(self) -> None:
        if self._ocr_engine is not None: return
        from paddleocr import PaddleOCR # type: ignore
        self._ocr_engine = PaddleOCR(use_angle_cls=True, lang="en", show_log=False, use_gpu=(DEVICE == "cuda"))

    def _embed_images(self, images_bgr: list[np.ndarray]) -> np.ndarray:
        from PIL import Image
        self._load_siglip()
        pil_images = [Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) for img in images_bgr]
        vectors = []
        for i in range(0, len(pil_images), self.config.semantic_batch_size):
            batch = pil_images[i : i + self.config.semantic_batch_size]
            inputs = self._siglip_processor(images=batch, return_tensors="pt", padding=True).to(DEVICE)
            with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.float16):
                outputs = self._siglip_model.get_image_features(**inputs)
            vectors.append(outputs.cpu().to(torch.float16).numpy())
            free_vram("SigLIP batch")
        return np.concatenate(vectors, axis=0)

    def _ocr_image(self, image_bgr: np.ndarray) -> str:
        self._load_ocr()
        result = self._ocr_engine.ocr(image_bgr, cls=True)
        if not result or result[0] is None: return ""
        return " | ".join([line[1][0] for line in result[0] if line and line[1]])

    def _caption_image(self, image_bgr: np.ndarray) -> str:
        from PIL import Image
        pil_img = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        
        if self._vllm_engine is not None:
            from vllm import SamplingParams
            messages = [{"role": "user", "content": [{"type": "image", "image": pil_img}, {"type": "text", "text": self.QWEN_PROMPT}]}]
            sampling = SamplingParams(max_tokens=64, temperature=0.0)
            outputs = self._vllm_engine.chat(messages, sampling_params=sampling)
            return outputs[0].outputs[0].text.strip()

        self._load_qwen()
        messages = [{"role": "user", "content": [{"type": "image", "image": pil_img}, {"type": "text", "text": self.QWEN_PROMPT}]}]
        text_prompt = self._qwen_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        from qwen_vl_utils import process_vision_info
        image_inputs, video_inputs = process_vision_info(messages)
        
        inputs = self._qwen_processor(
            text=[text_prompt], images=image_inputs, videos=video_inputs, return_tensors="pt", padding=True
        ).to(DEVICE)

        with torch.no_grad():
            generated_ids = self._qwen_model.generate(**inputs, max_new_tokens=64)

        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        output_text = self._qwen_processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        free_vram("Qwen inference")
        return output_text[0].strip() if output_text else ""

    def extract(self, tracklets: list[Tracklet]) -> list[SemanticResult]:
        results: list[SemanticResult] = []
        self._load_siglip()
        self._load_qwen()
        self._load_ocr()

        for idx, tracklet in enumerate(tracklets):
            if not tracklet.keyframe_images: continue

            kf_vectors = self._embed_images(tracklet.keyframe_images)
            pooled_vec = kf_vectors.mean(axis=0).astype(np.float16)

            ocr_texts: list[str] = []
            event_evolution: list[dict[str, Any]] = []

            for kf_frame, kf_img in zip(tracklet.keyframes, tracklet.keyframe_images):
                ocr_texts.append(self._ocr_image(kf_img))
                event_evolution.append({"frame": kf_frame, "action": self._caption_image(kf_img)})

            if (idx + 1) % 10 == 0: free_vram(f"SemanticExtractor batch {idx + 1}")

            results.append(SemanticResult(
                tracklet_id=tracklet.tracklet_id,
                shot_id=tracklet.shot_id,
                video_id=tracklet.video_id,
                siglip_vector=pooled_vec,
                ocr_texts=ocr_texts,
                event_evolution=event_evolution,
                keyframe_global_ids=list(tracklet.keyframes),
            ))

        return results


# ─────────────────────────────────────────────────────────────────────────────
# 4.  AUDIO EXTRACTION (ASR)
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class ASRSegment:
    text: str
    start_sec: float
    end_sec: float
    local_frame_id: int


class AudioExtractor:
    def __init__(self, config: ATSMEConfig) -> None:
        self.model_size = config.whisper_size
        self._model = None

    def _load_model(self) -> None:
        if self._model is not None: return
        import whisper
        log.info("Loading Whisper model: %s", self.model_size)
        self._model = whisper.load_model(self.model_size, device=DEVICE)

    @staticmethod
    def _extract_audio_wav(video_path: str | Path, out_wav: str) -> bool:
        cmd = ["ffmpeg", "-y", "-i", str(video_path), "-ar", "16000", "-ac", "1", "-vn", out_wav]
        return subprocess.run(cmd, capture_output=True).returncode == 0

    def transcribe(self, video_path: str | Path, fps: float) -> list[ASRSegment]:
        self._load_model()
        video_path = Path(video_path)

        with tempfile.TemporaryDirectory() as tmpdir:
            wav_path = os.path.join(tmpdir, "audio.wav")
            if not self._extract_audio_wav(video_path, wav_path):
                return []

            import whisper
            result = self._model.transcribe(wav_path, verbose=False)

        segments: list[ASRSegment] = []
        for seg in result.get("segments", []):
            mid_sec = (seg["start"] + seg["end"]) / 2.0
            segments.append(ASRSegment(
                text=seg["text"].strip(),
                start_sec=seg["start"],
                end_sec=seg["end"],
                local_frame_id=int(round(mid_sec * fps)),
            ))

        free_vram("AudioExtractor")
        return segments


# ─────────────────────────────────────────────────────────────────────────────
# 5.  GLOBAL ALIGNMENT & EXPORT
# ─────────────────────────────────────────────────────────────────────────────
class FrameMapper:
    def __init__(self, config: ATSMEConfig) -> None:
        self.csv_path = Path(config.mapping_csv)
        self._df = pd.read_csv(self.csv_path)
        self._df.columns = [c.strip().lower() for c in self._df.columns]
        
        self._lookup: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for vid, grp in self._df.groupby("video_id"):
            grp_sorted = grp.sort_values("local_frame_id")
            self._lookup[str(vid)] = (
                grp_sorted["local_frame_id"].to_numpy(dtype=np.int64),
                grp_sorted["global_frame_id"].to_numpy(dtype=np.int64),
            )

    def to_global(self, video_id: str, local_frame_id: int) -> int:
        if video_id not in self._lookup: return -1
        local_arr, global_arr = self._lookup[video_id]
        idx = int(np.argmin(np.abs(local_arr - local_frame_id)))
        return int(global_arr[idx])


class DataExporter:
    VISUAL_SCHEMA = pa.schema([
        pa.field("video_id", pa.string()),
        pa.field("shot_id", pa.string()),
        pa.field("tracklet_id", pa.string()),
        pa.field("global_frame_id", pa.int64()),
        pa.field("start_frame", pa.int64()),
        pa.field("end_frame", pa.int64()),
        pa.field("siglip_vector", pa.list_(pa.float16())),
        pa.field("bbox_trajectory", pa.string()),
    ])

    def __init__(self, config: ATSMEConfig) -> None:
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export(self, tracklets: list[Tracklet], semantic_results: list[SemanticResult], asr_segments: list[ASRSegment], frame_mapper: FrameMapper, video_id: str, shots: list[Shot]) -> tuple[Path, Path]:
        shot_by_id = {s.shot_id: s for s in shots}
        tracklet_by_id = {t.tracklet_id: t for t in tracklets}

        for res in semantic_results:
            res.keyframe_global_ids = [frame_mapper.to_global(video_id, f) for f in res.keyframe_global_ids]
            for ev in res.event_evolution:
                ev["frame"] = frame_mapper.to_global(video_id, ev["frame"])

        asr_full_text = " ".join(seg.text for seg in asr_segments)

        visual_rows: list[dict] = []
        for res in semantic_results:
            tracklet = tracklet_by_id.get(res.tracklet_id)
            shot = shot_by_id.get(res.shot_id)
            
            trajectory = [{"frame": frame_mapper.to_global(video_id, f), "bbox": list(b)} for f, b in zip(tracklet.frames, tracklet.bboxes)] if tracklet else []
            rep_global = res.keyframe_global_ids[0] if res.keyframe_global_ids else -1

            visual_rows.append({
                "video_id": video_id,
                "shot_id": str(res.shot_id),
                "tracklet_id": str(res.tracklet_id),
                "global_frame_id": rep_global,
                "start_frame": shot.start_frame if shot else -1,
                "end_frame": shot.end_frame if shot else -1,
                "siglip_vector": res.siglip_vector.tolist(),
                "bbox_trajectory": json.dumps(trajectory),
            })

        parquet_path = self.output_dir / "features_visual.parquet"
        if visual_rows:
            pq.write_table(pa.Table.from_pylist(visual_rows, schema=self.VISUAL_SCHEMA), str(parquet_path), compression="snappy")

        jsonl_path = self.output_dir / "features_lexical.jsonl"
        with jsonl_path.open("w", encoding="utf-8") as fout:
            for res in semantic_results:
                fout.write(json.dumps({
                    "video_id": video_id,
                    "tracklet_id": str(res.tracklet_id),
                    "global_frame_keys": res.keyframe_global_ids,
                    "ocr_text": res.ocr_texts,
                    "asr_text": asr_full_text,
                    "event_evolution": res.event_evolution,
                }, ensure_ascii=False) + "\n")

        return parquet_path, jsonl_path


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE MANAGER
# ─────────────────────────────────────────────────────────────────────────────
class PipelineManager:
    def __init__(self, config: ATSMEConfig) -> None:
        self.config = config
        self._shot_detector = None
        self._tracklet_builder = None
        self._semantic_extractor = None
        self._audio_extractor = None
        self._frame_mapper = None
        self._data_exporter = None

    @property
    def shot_detector(self) -> ShotDetector:
        if not self._shot_detector: self._shot_detector = ShotDetector(self.config)
        return self._shot_detector

    @property
    def tracklet_builder(self) -> TrackletBuilder:
        if not self._tracklet_builder: self._tracklet_builder = TrackletBuilder(self.config)
        return self._tracklet_builder

    @property
    def semantic_extractor(self) -> SemanticExtractor:
        if not self._semantic_extractor: self._semantic_extractor = SemanticExtractor(self.config)
        return self._semantic_extractor

    @property
    def audio_extractor(self) -> AudioExtractor:
        if not self._audio_extractor: self._audio_extractor = AudioExtractor(self.config)
        return self._audio_extractor

    @property
    def frame_mapper(self) -> FrameMapper:
        if not self._frame_mapper: self._frame_mapper = FrameMapper(self.config)
        return self._frame_mapper

    @property
    def data_exporter(self) -> DataExporter:
        if not self._data_exporter: self._data_exporter = DataExporter(self.config)
        return self._data_exporter

    def process_video(self, video_path: str | Path) -> tuple[Path, Path]:
        video_path = Path(video_path)
        video_id = video_path.stem
        log.info("=" * 70)
        log.info("ATSME Pipeline START → %s", video_path.name)
        log.info("=" * 70)

        shots = self.shot_detector.detect(video_path)
        tracklets = self.tracklet_builder.build_tracklets(video_path, shots, video_id)
        semantic_results = self.semantic_extractor.extract(tracklets)
        
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        cap.release()
        
        asr_segments = self.audio_extractor.transcribe(video_path, fps=fps)

        parquet_path, jsonl_path = self.data_exporter.export(
            tracklets, semantic_results, asr_segments, self.frame_mapper, video_id, shots
        )

        return parquet_path, jsonl_path

    def process_batch(self, video_paths: list[str | Path]) -> list[tuple[Path, Path]]:
        results = []
        for vp in video_paths:
            try:
                results.append(self.process_video(vp))
            except Exception as exc:
                log.error("Failed to process %s: %s", vp, exc, exc_info=True)
        return results
