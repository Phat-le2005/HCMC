import os
import json
import argparse
from pathlib import Path
from typing import List, Dict
import subprocess
import gc

import torch
from PIL import Image
import soundfile as sf
import numpy as np
from sklearn.cluster import AgglomerativeClustering

from scenedetect import detect, ContentDetector
try:
    from transnetv2 import TransNetV2
except ImportError:
    TransNetV2 = None

from transformers import AutoModel, AutoImageProcessor, AutoFeatureExtractor
from paddleocr import PaddleOCR
from whisper import load_model as load_whisper

from config_v5 import config
from news_classifier import NewsSceneClassifier

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def get_video_fps(video_path: Path) -> float:
    """Extract actual FPS from video file using OpenCV."""
    import cv2
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    if fps <= 0:
        print(f"Warning: Could not read FPS from {video_path}, defaulting to 25.0")
        return 25.0
    return fps

def detect_shots_pyscene(video_path: Path) -> List[tuple]:
    scene_list = detect(str(video_path), ContentDetector(threshold=config.shot_detector_content_threshold))
    shot_list = []
    for scene in scene_list:
        start_ms = scene[0].get_seconds() * 1000.0
        end_ms = scene[1].get_seconds() * 1000.0
        shot_list.append((start_ms, end_ms, scene[0].get_frames(), scene[1].get_frames()))
    return shot_list

def detect_shots_transnet(video_path: Path, fps: float) -> List[tuple]:
    if TransNetV2 is None:
        print("Warning: TransNetV2 not installed. Skipping hybrid SBD.")
        return []
    model = TransNetV2()
    video_frames, single_frame_predictions, all_frame_predictions = model.predict_video(str(video_path))
    scenes = model.predictions_to_scenes(single_frame_predictions)
    shot_list = []
    for start_frame, end_frame in scenes:
        start_ms = (start_frame / fps) * 1000
        end_ms = (end_frame / fps) * 1000
        shot_list.append((start_ms, end_ms, int(start_frame), int(end_frame)))
    return shot_list

def merge_shot_boundaries(pyscene_shots, transnet_shots, fps: float):
    """Merge shot boundaries from two detectors using actual video FPS."""
    if not transnet_shots:
        return [{"shot_id": f"shot_{i}", "start_ms": s[0], "end_ms": s[1], "start_frame": s[2], "end_frame": s[3]} for i, s in enumerate(pyscene_shots)]
    all_boundaries = set()
    for s in pyscene_shots + transnet_shots:
        all_boundaries.add(s[0])
        all_boundaries.add(s[1])
    sorted_bounds = sorted(list(all_boundaries))
    merged_bounds = []
    for b in sorted_bounds:
        if not merged_bounds:
            merged_bounds.append(b)
        else:
            if b - merged_bounds[-1] > config.shot_merge_threshold_ms:
                merged_bounds.append(b)
    final_shots = []
    for i in range(len(merged_bounds) - 1):
        start_ms = merged_bounds[i]
        end_ms = merged_bounds[i+1]
        final_shots.append({
            "shot_id": f"shot_{i}",
            "start_ms": start_ms,
            "end_ms": end_ms,
            "start_frame": int(start_ms * fps / 1000.0),
            "end_frame": int(end_ms * fps / 1000.0)
        })
    return final_shots

def extract_multi_keyframes(video_path: Path, start_ms: float, end_ms: float, shot_id: str, out_dir: Path) -> List[Path]:
    duration_ms = end_ms - start_ms
    if duration_ms <= 0:
        return []
    points = [0.25, 0.5, 0.75] if config.ocr_keyframes_per_shot == 3 else [0.5]
    paths = []
    for i, p in enumerate(points):
        target_ms = start_ms + (duration_ms * p)
        out_path = out_dir / f"{shot_id}_k{i}.jpg"
        if not out_path.exists():
            cmd = [
                "ffmpeg", "-y",
                "-ss", str(target_ms / 1000.0),
                "-i", str(video_path),
                "-vframes", "1",
                "-q:v", "2",
                str(out_path)
            ]
            try:
                subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=True)
            except subprocess.CalledProcessError:
                continue
        if out_path.exists():
            paths.append(out_path)
    return paths

def extract_audio_segment(video_path: Path, start_ms: float, end_ms: float, shot_id: str, out_dir: Path) -> Path:
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

def get_vision_embedding(model, pixel_values):
    """Extract image embedding from SigLIP model with robust fallback."""
    with torch.no_grad():
        if hasattr(model, 'get_image_features'):
            out = model.get_image_features(pixel_values=pixel_values)
            if isinstance(out, torch.Tensor):
                return out
            if hasattr(out, 'pooler_output'):
                return out.pooler_output
            return out[0] if isinstance(out, tuple) else out

        if hasattr(model, 'vision_model'):
            out = model.vision_model(pixel_values=pixel_values)
            if hasattr(out, 'pooler_output') and out.pooler_output is not None:
                return out.pooler_output
            return out.last_hidden_state[:, 0]

        out = model(pixel_values=pixel_values)
        if hasattr(out, 'pooler_output') and out.pooler_output is not None:
            return out.pooler_output
        return out.last_hidden_state[:, 0]

def compute_image_vector(image_path: Path, processor, model) -> List[float]:
    try:
        img = Image.open(image_path).convert("RGB").resize(config.keyframe_resize)
    except Exception as e:
        print(f"Error reading image {image_path}: {e}")
        return [0.0] * 768

    inputs = processor(images=img, return_tensors="pt").to(config.device)
    if config.fp16 and config.device == "cuda":
        inputs['pixel_values'] = inputs['pixel_values'].half()

    embeddings = get_vision_embedding(model, inputs['pixel_values'])
    vec = embeddings.squeeze().float().cpu().tolist()
    if isinstance(vec, float):
        vec = [vec]
    return vec

def load_wavlm():
    """Load WavLM as a feature extractor (NOT classifier)."""
    extractor = AutoFeatureExtractor.from_pretrained(config.audio_embedding)
    # Use AutoModel instead of AutoModelForAudioClassification
    # wavlm-base is a pretrained encoder without a classification head
    model = AutoModel.from_pretrained(config.audio_embedding).to(config.device)
    model.eval()
    if config.fp16 and config.device == "cuda":
        model.half()
    return extractor, model

def compute_audio_vector(wav_path: Path, extractor, model) -> List[float]:
    if not wav_path.exists():
        return [0.0] * 768
    try:
        audio, sr = sf.read(str(wav_path))
    except Exception as e:
        print(f"Error reading audio {wav_path}: {e}")
        return [0.0] * 768
    if len(audio) == 0:
        return [0.0] * 768

    inputs = extractor(audio, sampling_rate=sr, return_tensors="pt").to(config.device)
    if config.fp16 and config.device == "cuda":
        inputs['input_values'] = inputs['input_values'].half()

    with torch.no_grad():
        outputs = model(**inputs)
        # WavLM-base returns last_hidden_state, take mean pooling
        hidden = outputs.last_hidden_state  # (1, seq_len, 768)
        pooled = hidden.mean(dim=1)  # (1, 768)
    return pooled.squeeze().float().cpu().tolist()

def run_ocr_multi(image_paths: List[Path], ocr: PaddleOCR) -> str:
    all_texts = set()
    for img_path in image_paths:
        try:
            result = ocr.ocr(str(img_path), cls=config.ocr_use_angle_cls)
            if not result or not result[0]:
                continue
            texts = [line[1][0] for line in result[0] if line and len(line) > 1 and line[1]]
            all_texts.update(texts)
        except Exception as e:
            print(f"OCR error on {img_path}: {e}")
    return " ".join(all_texts)

def run_whisper(audio_path: Path, model):
    if not audio_path.exists():
        return ""
    try:
        result = model.transcribe(str(audio_path), language=config.asr_language)
        text = result.get('text', '').strip()
        if len(text) < 3 and not text.isalnum():
            return ""
        return text
    except Exception as e:
        print(f"Whisper error on {audio_path}: {e}")
        return ""

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
        with open(shots_file, "r", encoding="utf-8") as f:
            shots = json.load(f)
    else:
        video_fps = get_video_fps(video_path)
        print(f"    Detected video FPS: {video_fps}")
        pyscene_shots = detect_shots_pyscene(video_path)
        if config.use_hybrid_sbd:
            transnet_shots = detect_shots_transnet(video_path, video_fps)
            shots = merge_shot_boundaries(pyscene_shots, transnet_shots, video_fps)
        else:
            shots = [{"shot_id": f"shot_{i}", "start_ms": s[0], "end_ms": s[1], "start_frame": s[2], "end_frame": s[3]} for i, s in enumerate(pyscene_shots)]

        with open(shots_file, "w", encoding="utf-8") as f:
            json.dump(shots, f, ensure_ascii=False, indent=2)

    if not shots:
        print("[WARN] No shots detected. Exiting.")
        return

    print(f"    Found {len(shots)} shots.")
    print(f"[2] Loading models on {config.device}...")
    img_processor, img_model = load_siglip()
    audio_extractor, audio_model = load_wavlm()
    ocr = PaddleOCR(use_angle_cls=config.ocr_use_angle_cls, lang=config.ocr_lang)
    whisper_model = load_whisper(config.whisper_model, device=config.device)
    news_classifier = NewsSceneClassifier(config)

    fused_vectors = []

    # Process shots with checkpointing
    for idx, shot in enumerate(shots):
        shot_id = shot["shot_id"]

        # Check if already processed
        if "global_ocr" in shot and "global_asr" in shot and "news_type" in shot:
            fused = shot.get("image_vector", []) + shot.get("audio_vector", [])
            if fused:
                fused_vectors.append(fused)
            continue

        key_paths = extract_multi_keyframes(video_path, shot["start_ms"], shot["end_ms"], shot_id, keyframe_dir)

        if not key_paths:
            print(f"  [WARN] No keyframes extracted for {shot_id}, skipping.")
            shot["global_ocr"] = ""
            shot["global_asr"] = ""
            shot["news_type"] = "unknown"
            shot["image_vector"] = [0.0] * 768
            shot["audio_vector"] = [0.0] * 768
            continue

        primary_k_path = key_paths[len(key_paths)//2]

        img_vec = compute_image_vector(primary_k_path, img_processor, img_model)

        audio_path = extract_audio_segment(video_path, shot["start_ms"], shot["end_ms"], shot_id, audio_dir)
        audio_vec = compute_audio_vector(audio_path, audio_extractor, audio_model)

        fused = img_vec + audio_vec
        fused_vectors.append(fused)

        shot["image_vector_path"] = str(primary_k_path)
        shot["audio_path"] = str(audio_path)
        shot["image_vector"] = img_vec
        shot["audio_vector"] = audio_vec
        shot["global_ocr"] = run_ocr_multi(key_paths, ocr)
        shot["global_asr"] = run_whisper(audio_path, whisper_model)

        # News Classification (Zero-shot)
        news_result = news_classifier.classify_image(str(primary_k_path))
        shot["news_type"] = news_result["label"]

        # Incremental save every 5 shots
        if (idx + 1) % 5 == 0 or idx == len(shots) - 1:
            with open(shots_file, "w", encoding="utf-8") as f:
                json.dump(shots, f, ensure_ascii=False, indent=2)
            print(f"  [Progress] {idx+1}/{len(shots)} shots processed.")

        # Periodic VRAM cleanup
        if (idx + 1) % 10 == 0:
            gc.collect()
            if config.device == "cuda":
                torch.cuda.empty_cache()

    # Final save
    with open(shots_file, "w", encoding="utf-8") as f:
        json.dump(shots, f, ensure_ascii=False, indent=2)

    print("[3] Clustering shots into scenes...")
    if len(fused_vectors) >= config.scene_min_shots:
        X = np.array(fused_vectors)
        times = np.array([s["start_ms"] for s in shots if "image_vector" in s][:len(fused_vectors)])
        time_penalty = config.scene_cluster_alpha * np.abs(times[:, None] - times[None, :]) / 1000.0

        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms = np.clip(norms, 1e-8, None)
        norm_X = X / norms

        sim = np.dot(norm_X, norm_X.T)
        sim = sim - time_penalty
        distance = 1 - sim
        np.fill_diagonal(distance, 0)
        distance = np.clip(distance, 0, None)

        clustering = AgglomerativeClustering(n_clusters=None, metric='precomputed', linkage='average', distance_threshold=0.5)
        labels = clustering.fit_predict(distance)

        scenes_dict = {}
        processed_shots = [s for s in shots if "image_vector" in s][:len(fused_vectors)]
        for label, shot in zip(labels, processed_shots):
            scenes_dict.setdefault(int(label), []).append(shot)

        from collections import Counter
        scene_list = []
        for scene_id, shot_group in scenes_dict.items():
            news_types = [s.get("news_type", "unknown") for s in shot_group]
            majority_news = Counter(news_types).most_common(1)[0][0]

            scene_list.append({
                "scene_id": f"scene_{scene_id}",
                "shot_ids": [s["shot_id"] for s in shot_group],
                "start_ms": min(s["start_ms"] for s in shot_group),
                "end_ms": max(s["end_ms"] for s in shot_group),
                "news_type": majority_news
            })
        with open(out_dir / "scenes.json", "w", encoding="utf-8") as f:
            json.dump(scene_list, f, ensure_ascii=False, indent=2)
    else:
        print("Not enough shots/vectors for scene clustering. Skipping.")

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
