import os
import json
import argparse
from pathlib import Path
from typing import List, Dict

import torch
from torchvision import transforms
from PIL import Image

from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector

# Audio extraction using ffmpeg (subprocess)
import subprocess

# Model imports (lazy)
# SigLIP-2
from transformers import AutoModel, AutoImageProcessor

# Audio embedding (WavLM)
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor

# OCR and ASR
from paddleocr import PaddleOCR
from whisper import load_model as load_whisper

# Clustering
from sklearn.cluster import AgglomerativeClustering
import numpy as np

# Config
from config_v5 import config

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def detect_shots(video_path: Path) -> List[Dict]:
    video_manager = VideoManager([str(video_path)])
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=config.shot_detector_content_threshold))
    video_manager.start()
    scene_manager.detect_scenes(frame_source=video_manager)
    # List of (start_frame, end_frame)
    shot_list = []
    for i, scene in enumerate(scene_manager.get_scene_list()):
        start_frame, end_frame = scene
        start_ms = video_manager.get_frame_timecode(start_frame).get_milliseconds()
        end_ms = video_manager.get_frame_timecode(end_frame).get_milliseconds()
        shot_list.append({
            "shot_id": f"shot_{i}",
            "start_ms": start_ms,
            "end_ms": end_ms,
            "start_frame": start_frame.get_frames(),
            "end_frame": end_frame.get_frames(),
        })
    video_manager.release()
    return shot_list

def extract_keyframe(video_path: Path, start_frame: int, end_frame: int, shot_id: str, out_dir: Path) -> Path:
    # Choose middle frame
    mid_frame = (start_frame + end_frame) // 2
    # Use ffmpeg to extract frame as image
    out_path = out_dir / f"{shot_id}_keyframe.jpg"
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-vf",
        f"select=eq(n\,{mid_frame})",
        "-vframes",
        "1",
        str(out_path),
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return out_path

def extract_audio_segment(video_path: Path, start_ms: int, end_ms: int, shot_id: str, out_dir: Path) -> Path:
    out_path = out_dir / f"{shot_id}.wav"
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-ss",
        str(start_ms / 1000),
        "-to",
        str(end_ms / 1000),
        "-ac", "1",
        "-ar", "16000",
        str(out_path),
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return out_path

def load_siglip():
    processor = AutoImageProcessor.from_pretrained(config.siglip_model)
    model = AutoModel.from_pretrained(config.siglip_model)
    model.eval()
    if config.fp16:
        model.half()
    return processor, model

def compute_image_vector(image_path: Path, processor, model) -> List[float]:
    img = Image.open(image_path).convert("RGB")
    img = img.resize(config.keyframe_resize)
    inputs = processor(images=img, return_tensors="pt")
    with torch.no_grad():
        embeddings = model(**inputs).pooler_output
    if config.fp16:
        embeddings = embeddings.half()
    return embeddings.squeeze().cpu().tolist()

def load_wavlm():
    extractor = AutoFeatureExtractor.from_pretrained(config.audio_embedding)
    model = AutoModelForAudioClassification.from_pretrained(config.audio_embedding)
    model.eval()
    if config.fp16:
        model.half()
    return extractor, model

def compute_audio_vector(wav_path: Path, extractor, model) -> List[float]:
    # Load raw audio
    import soundfile as sf
    audio, sr = sf.read(str(wav_path))
    inputs = extractor(audio, sampling_rate=sr, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs).logits
    return outputs.squeeze().cpu().tolist()

def run_ocr(image_path: Path, ocr: PaddleOCR) -> str:
    result = ocr.ocr(str(image_path), cls=False)
    texts = [line[1][0] for line in result]
    return " ".join(texts)

def run_whisper(audio_path: Path, model):
    # Whisper expects path to audio file; we can use its transcribe method
    return model.transcribe(str(audio_path))['text']

def main():
    parser = argparse.ArgumentParser(description="Module 1 – Static Pipeline")
    parser.add_argument("--video", type=str, required=True, help="Path to input video file")
    parser.add_argument("--output-dir", type=str, default=config.intermediate_dir)
    args = parser.parse_args()

    video_path = Path(args.video)
    out_dir = Path(args.output_dir)
    ensure_dir(out_dir)
    keyframe_dir = out_dir / "keyframes"
    audio_dir = out_dir / "audio_segments"
    ensure_dir(keyframe_dir)
    ensure_dir(audio_dir)

    print("[1] Detecting shots...")
    shots = detect_shots(video_path)
    with open(out_dir / "shots.json", "w", encoding="utf-8") as f:
        json.dump(shots, f, ensure_ascii=False, indent=2)

    print("[2] Loading models (SigLIP, WavLM, OCR, Whisper)...")
    img_processor, img_model = load_siglip()
    audio_extractor, audio_model = load_wavlm()
    ocr = PaddleOCR(use_angle_cls=False, lang='en')
    whisper = load_whisper(config.whisper_model)

    fused_vectors = []
    for shot in shots:
        shot_id = shot["shot_id"]
        # Keyframe
        key_path = extract_keyframe(video_path, shot["start_frame"], shot["end_frame"], shot_id, keyframe_dir)
        img_vec = compute_image_vector(key_path, img_processor, img_model)
        # Audio
        audio_path = extract_audio_segment(video_path, shot["start_ms"], shot["end_ms"], shot_id, audio_dir)
        audio_vec = compute_audio_vector(audio_path, audio_extractor, audio_model)
        # Fuse (concatenate)
        fused = img_vec + audio_vec
        fused_vectors.append(fused)
        # Store per‑shot embeddings (optional)
        shot["image_vector_path"] = str(key_path)
        shot["audio_path"] = str(audio_path)
        shot["image_vector"] = img_vec
        shot["audio_vector"] = audio_vec
        # Global OCR (keyframe)
        shot["global_ocr"] = run_ocr(key_path, ocr)
        # Global ASR (audio)
        shot["global_asr"] = run_whisper(audio_path, whisper)

    # Clustering shots into scenes
    print("[3] Clustering shots into scenes...")
    X = np.array(fused_vectors)
    # Compute temporal penalty matrix
    times = np.array([s["start_ms"] for s in shots])
    time_penalty = config.scene_cluster_alpha * np.abs(times[:, None] - times[None, :]) / 1000.0
    # Cosine similarity matrix
    norm_X = X / np.linalg.norm(X, axis=1, keepdims=True)
    sim = np.dot(norm_X, norm_X.T)
    sim = sim - time_penalty
    # Agglomerative clustering (precomputed distance)
    distance = 1 - sim
    clustering = AgglomerativeClustering(n_clusters=None, affinity='precomputed', linkage='average', distance_threshold=0.5)
    labels = clustering.fit_predict(distance)
    scenes = {}
    for label, shot in zip(labels, shots):
        scenes.setdefault(label, []).append(shot)
    scene_list = []
    for scene_id, shot_group in scenes.items():
        if len(shot_group) < config.scene_min_shots:
            continue
        start_ms = min(s["start_ms"] for s in shot_group)
        end_ms = max(s["end_ms"] for s in shot_group)
        scene_list.append({
            "scene_id": f"scene_{scene_id}",
            "shot_ids": [s["shot_id"] for s in shot_group],
            "start_ms": start_ms,
            "end_ms": end_ms,
        })
    with open(out_dir / "scenes.json", "w", encoding="utf-8") as f:
        json.dump(scene_list, f, ensure_ascii=False, indent=2)

    # Global lexical JSON
    lexical = {
        "video_id": video_path.stem,
        "shots": [{
            "shot_id": s["shot_id"],
            "ocr": s.get("global_ocr", ""),
            "asr": s.get("global_asr", "")
        } for s in shots]
    }
    with open(out_dir / "lexical_global.json", "w", encoding="utf-8") as f:
        json.dump(lexical, f, ensure_ascii=False, indent=2)

    print("[DONE] Static pipeline completed. Outputs written to", out_dir)

if __name__ == "__main__":
    main()
