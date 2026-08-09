import os
import json
import argparse
from pathlib import Path
from typing import List, Dict

import torch
from torchvision import transforms
from PIL import Image
import soundfile as sf

from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector
import subprocess

from transformers import AutoModel, AutoImageProcessor
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor
from paddleocr import PaddleOCR
from whisper import load_model as load_whisper

from sklearn.cluster import AgglomerativeClustering
import numpy as np

from config_v5 import config

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def detect_shots(video_path: Path) -> List[Dict]:
    video_manager = VideoManager([str(video_path)])
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=config.shot_detector_content_threshold))
    video_manager.start()
    scene_manager.detect_scenes(frame_source=video_manager)
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
    mid_frame = (start_frame + end_frame) // 2
    out_path = out_dir / f"{shot_id}_keyframe.jpg"
    if out_path.exists():
        return out_path
    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-vf", f"select=eq(n\,{mid_frame})",
        "-vframes", "1",
        str(out_path)
    ]
    try:
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=True)
    except subprocess.CalledProcessError as e:
        print(f"ffmpeg error extracting keyframe for {shot_id}: {e.stderr.decode()}")
    return out_path

def extract_audio_segment(video_path: Path, start_ms: int, end_ms: int, shot_id: str, out_dir: Path) -> Path:
    out_path = out_dir / f"{shot_id}.wav"
    if out_path.exists():
        return out_path
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start_ms / 1000.0),
        "-i", str(video_path),
        "-t", str((end_ms - start_ms) / 1000.0),
        "-ac", "1", "-ar", "16000",
        str(out_path)
    ]
    try:
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=True)
    except subprocess.CalledProcessError as e:
        print(f"ffmpeg error extracting audio for {shot_id}: {e.stderr.decode()}")
    return out_path

def load_siglip():
    processor = AutoImageProcessor.from_pretrained(config.siglip_model)
    model = AutoModel.from_pretrained(config.siglip_model).to(config.device)
    model.eval()
    if config.fp16 and config.device == "cuda":
        model.half()
    return processor, model

def compute_image_vector(image_path: Path, processor, model) -> List[float]:
    try:
        img = Image.open(image_path).convert("RGB").resize(config.keyframe_resize)
    except Exception as e:
        print(f"Error reading image {image_path}: {e}")
        return [0.0] * 768
    
    inputs = processor(images=img, return_tensors="pt").to(config.device)
    if config.fp16 and config.device == "cuda":
        inputs['pixel_values'] = inputs['pixel_values'].half()
        
    with torch.no_grad():
        embeddings = model(**inputs).pooler_output
    return embeddings.squeeze().cpu().tolist()

def load_wavlm():
    extractor = AutoFeatureExtractor.from_pretrained(config.audio_embedding)
    model = AutoModelForAudioClassification.from_pretrained(config.audio_embedding).to(config.device)
    model.eval()
    if config.fp16 and config.device == "cuda":
        model.half()
    return extractor, model

def compute_audio_vector(wav_path: Path, extractor, model) -> List[float]:
    try:
        audio, sr = sf.read(str(wav_path))
    except Exception as e:
        print(f"Error reading audio {wav_path}: {e}")
        return [0.0] * 768
        
    inputs = extractor(audio, sampling_rate=sr, return_tensors="pt").to(config.device)
    if config.fp16 and config.device == "cuda":
        inputs['input_values'] = inputs['input_values'].half()
        
    with torch.no_grad():
        outputs = model(**inputs).logits
    return outputs.squeeze().cpu().tolist()

def run_ocr(image_path: Path, ocr: PaddleOCR) -> str:
    result = ocr.ocr(str(image_path), cls=False)
    if not result or not result[0]:
        return ""
    texts = [line[1][0] for line in result[0] if line and line[1]]
    return " ".join(texts)

def run_whisper(audio_path: Path, model):
    if not audio_path.exists():
        return ""
    result = model.transcribe(str(audio_path))
    return result.get('text', '')

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
    
    shots_file = out_dir / "shots.json"

    print("[1] Detecting shots...")
    if shots_file.exists():
        print(f"Loading existing shots from {shots_file}")
        with open(shots_file, "r", encoding="utf-8") as f:
            shots = json.load(f)
    else:
        shots = detect_shots(video_path)
        with open(shots_file, "w", encoding="utf-8") as f:
            json.dump(shots, f, ensure_ascii=False, indent=2)

    print(f"[2] Loading models on {config.device}...")
    img_processor, img_model = load_siglip()
    audio_extractor, audio_model = load_wavlm()
    ocr = PaddleOCR(use_angle_cls=False, lang=config.ocr_lang)
    whisper = load_whisper(config.whisper_model, device=config.device)

    fused_vectors = []
    
    # Process shots with checkpointing
    for shot in shots:
        shot_id = shot["shot_id"]
        
        # Check if already processed
        if "global_ocr" in shot and "global_asr" in shot:
            fused = shot.get("image_vector", []) + shot.get("audio_vector", [])
            if fused:
                fused_vectors.append(fused)
            continue
            
        key_path = extract_keyframe(video_path, shot["start_frame"], shot["end_frame"], shot_id, keyframe_dir)
        img_vec = compute_image_vector(key_path, img_processor, img_model)
        
        audio_path = extract_audio_segment(video_path, shot["start_ms"], shot["end_ms"], shot_id, audio_dir)
        audio_vec = compute_audio_vector(audio_path, audio_extractor, audio_model)
        
        fused = img_vec + audio_vec
        fused_vectors.append(fused)
        
        shot["image_vector_path"] = str(key_path)
        shot["audio_path"] = str(audio_path)
        shot["image_vector"] = img_vec
        shot["audio_vector"] = audio_vec
        shot["global_ocr"] = run_ocr(key_path, ocr)
        shot["global_asr"] = run_whisper(audio_path, whisper)
        
        # Incremental save
        with open(shots_file, "w", encoding="utf-8") as f:
            json.dump(shots, f, ensure_ascii=False, indent=2)

    print("[3] Clustering shots into scenes...")
    if len(shots) >= config.scene_min_shots and fused_vectors:
        X = np.array(fused_vectors)
        times = np.array([s["start_ms"] for s in shots])
        time_penalty = config.scene_cluster_alpha * np.abs(times[:, None] - times[None, :]) / 1000.0
        
        # Prevent division by zero
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms = np.clip(norms, 1e-8, None)
        norm_X = X / norms
        
        sim = np.dot(norm_X, norm_X.T)
        sim = sim - time_penalty
        distance = 1 - sim
        
        clustering = AgglomerativeClustering(n_clusters=None, metric='precomputed', linkage='average', distance_threshold=0.5)
        labels = clustering.fit_predict(distance)
        scenes = {}
        for label, shot in zip(labels, shots):
            scenes.setdefault(label, []).append(shot)
            
        scene_list = []
        for scene_id, shot_group in scenes.items():
            if len(shot_group) < config.scene_min_shots:
                continue
            scene_list.append({
                "scene_id": f"scene_{scene_id}",
                "shot_ids": [s["shot_id"] for s in shot_group],
                "start_ms": min(s["start_ms"] for s in shot_group),
                "end_ms": max(s["end_ms"] for s in shot_group),
            })
        with open(out_dir / "scenes.json", "w", encoding="utf-8") as f:
            json.dump(scene_list, f, ensure_ascii=False, indent=2)
    else:
        print("Not enough shots or valid vectors for scene clustering. Skipping.")

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
