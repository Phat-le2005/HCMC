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
from whisper import load_model as load_whisper

from config_v5 import config
from news_classifier import NewsSceneClassifier
from ocr_engine import HybridOCREngine
from ocr_postprocess import OCRPostProcessor

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

def run_ocr_multi(image_paths: List[Path], ocr_engine: HybridOCREngine) -> str:
    """Run 3-step OCR (Paddle detect → Crop+Pad → VietOCR) on multiple keyframes."""
    return ocr_engine.recognize_multi([str(p) for p in image_paths])

def run_whisper(audio_path: Path, model):
    if not audio_path.exists():
        return ""
        
    # Blacklist of common Whisper hallucinations in Vietnamese for empty/noisy audio
    hallucinations = [
        "ghiền mì gõ", "subscribe", "đăng ký kênh", "cảm ơn các bạn", 
        "hẹn gặp lại", "theo dõi", "bản quyền thuộc về", "subtitles by",
        "âm nhạc", "music"
    ]
    
    try:
        # Tuning parameters for short audio clips to reduce hallucinations
        result = model.transcribe(
            str(audio_path), 
            language=config.asr_language,
            condition_on_previous_text=False,  # Prevent getting stuck in loops
            no_speech_threshold=0.6,           # Be stricter on classifying silence
            logprob_threshold=-1.0             # Reject low confidence predictions
        )
        
        text = result.get('text', '').strip()
        
        if not text:
            return ""
            
        text_lower = text.lower()
        
        # Check against blacklist
        for phrase in hallucinations:
            if phrase in text_lower:
                return ""
                
        # Remove too short or non-alphanumeric junk
        if len(text) < 3 and not text.isalnum():
            return ""
            
        return text
    except Exception as e:
        print(f"Whisper error on {audio_path}: {e}")
        return ""

def free_vram(*models):
    """Unload models from GPU and free VRAM."""
    for m in models:
        if m is not None:
            del m
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

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

    # ═══════════════════════════════════════════════════════════════════
    # Stage 0: Shot Boundary Detection (CPU only)
    # ═══════════════════════════════════════════════════════════════════
    print("[Stage 0] Detecting shots...")
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

    # Extract all keyframes and audio segments (CPU, ffmpeg)
    print("[Stage 0] Extracting keyframes & audio segments...")
    for shot in shots:
        shot_id = shot["shot_id"]
        if "image_vector" in shot:
            continue  # Already processed
        extract_multi_keyframes(video_path, shot["start_ms"], shot["end_ms"], shot_id, keyframe_dir)
        extract_audio_segment(video_path, shot["start_ms"], shot["end_ms"], shot_id, audio_dir)

    # ═══════════════════════════════════════════════════════════════════
    # Stage 1: Image Embeddings + News Classification (SigLIP ~4.5GB)
    # ═══════════════════════════════════════════════════════════════════
    print("[Stage 1] SigLIP: Image embeddings + News classification...")
    img_processor, img_model = load_siglip()
    news_classifier = NewsSceneClassifier(config)  # Shares SigLIP weights

    for idx, shot in enumerate(shots):
        if "image_vector" in shot:
            continue
        shot_id = shot["shot_id"]
        key_paths = list(keyframe_dir.glob(f"{shot_id}_k*.jpg"))
        key_paths.sort()

        if not key_paths:
            shot["image_vector"] = [0.0] * 768
            shot["news_type"] = "unknown"
            shot["image_vector_path"] = ""
            continue

        primary_k_path = key_paths[len(key_paths)//2]
        shot["image_vector"] = compute_image_vector(primary_k_path, img_processor, img_model)
        shot["image_vector_path"] = str(primary_k_path)

        news_result = news_classifier.classify_image(str(primary_k_path))
        shot["news_type"] = news_result["label"]

        if (idx + 1) % 20 == 0:
            print(f"  [Stage 1] {idx+1}/{len(shots)}")

    free_vram(img_model, news_classifier)
    del img_processor, img_model, news_classifier
    print("  [Stage 1] Done. SigLIP unloaded.")

    # ═══════════════════════════════════════════════════════════════════
    # Stage 2: Audio Embeddings (WavLM ~0.4GB)
    # ═══════════════════════════════════════════════════════════════════
    print("[Stage 2] WavLM: Audio embeddings...")
    audio_extractor, audio_model = load_wavlm()

    for idx, shot in enumerate(shots):
        if "audio_vector" in shot:
            continue
        shot_id = shot["shot_id"]
        audio_path = audio_dir / f"{shot_id}.wav"
        shot["audio_vector"] = compute_audio_vector(audio_path, audio_extractor, audio_model)
        shot["audio_path"] = str(audio_path)

    free_vram(audio_model)
    del audio_extractor, audio_model
    print("  [Stage 2] Done. WavLM unloaded.")

    # Save checkpoint
    with open(shots_file, "w", encoding="utf-8") as f:
        json.dump(shots, f, ensure_ascii=False, indent=2)
    print("  [Checkpoint] Embeddings saved.")

    # ═══════════════════════════════════════════════════════════════════
    # Stage 3: OCR — PaddleOCR detect + VietOCR recognize (CPU/light GPU)
    # ═══════════════════════════════════════════════════════════════════
    print("[Stage 3] OCR: PaddleOCR detect → VietOCR recognize...")
    ocr_engine = HybridOCREngine(device=config.device)

    for idx, shot in enumerate(shots):
        if "global_ocr_raw" in shot:
            continue
        shot_id = shot["shot_id"]
        key_paths = list(keyframe_dir.glob(f"{shot_id}_k*.jpg"))
        key_paths.sort()

        if not key_paths:
            shot["global_ocr_raw"] = ""
            shot["global_ocr"] = ""
            continue

        shot["global_ocr_raw"] = run_ocr_multi(key_paths, ocr_engine)

        if (idx + 1) % 20 == 0:
            print(f"  [Stage 3] {idx+1}/{len(shots)}")

    free_vram(ocr_engine)
    del ocr_engine
    print("  [Stage 3] Done. OCR engine unloaded.")

    # ═══════════════════════════════════════════════════════════════════
    # Stage 4: OCR Correction — Qwen2.5-1.5B (~3GB)
    # ═══════════════════════════════════════════════════════════════════
    print("[Stage 4] Qwen2.5-1.5B: OCR spelling correction...")
    ocr_corrector = OCRPostProcessor(device=config.device, fp16=config.fp16)

    for idx, shot in enumerate(shots):
        if "global_ocr" in shot:
            continue
        raw = shot.get("global_ocr_raw", "")
        shot["global_ocr"] = ocr_corrector.correct(raw) if raw else ""

        if (idx + 1) % 20 == 0:
            print(f"  [Stage 4] {idx+1}/{len(shots)}")

    free_vram(ocr_corrector)
    del ocr_corrector
    print("  [Stage 4] Done. Qwen unloaded.")

    # Save checkpoint
    with open(shots_file, "w", encoding="utf-8") as f:
        json.dump(shots, f, ensure_ascii=False, indent=2)
    print("  [Checkpoint] OCR saved.")

    # ═══════════════════════════════════════════════════════════════════
    # Stage 5: ASR — Whisper large-v3 (~2.9GB)
    # ═══════════════════════════════════════════════════════════════════
    print("[Stage 5] Whisper: ASR transcription...")
    whisper_model = load_whisper(config.whisper_model, device=config.device)

    for idx, shot in enumerate(shots):
        if "global_asr" in shot:
            continue
        shot_id = shot["shot_id"]
        audio_path = audio_dir / f"{shot_id}.wav"
        shot["global_asr"] = run_whisper(audio_path, whisper_model)

        if (idx + 1) % 20 == 0:
            print(f"  [Stage 5] {idx+1}/{len(shots)}")

    free_vram(whisper_model)
    del whisper_model
    print("  [Stage 5] Done. Whisper unloaded.")

    # Final save
    with open(shots_file, "w", encoding="utf-8") as f:
        json.dump(shots, f, ensure_ascii=False, indent=2)
    print("  [Checkpoint] ASR saved.")

    # ═══════════════════════════════════════════════════════════════════
    # Stage 6: Scene Clustering (CPU only, no GPU needed)
    # ═══════════════════════════════════════════════════════════════════
    print("[Stage 6] Clustering shots into scenes...")
    fused_vectors = []
    for shot in shots:
        iv = shot.get("image_vector", [])
        av = shot.get("audio_vector", [])
        if iv and av:
            fused_vectors.append(iv + av)

    if len(fused_vectors) >= config.scene_min_shots:
        X = np.array(fused_vectors)
        times = np.array([s["start_ms"] for s in shots if s.get("image_vector")][:len(fused_vectors)])
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
        processed_shots = [s for s in shots if s.get("image_vector")][:len(fused_vectors)]
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
