"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  Semantic Boundary Detection via SigLIP Embedding Similarity               ║
║  ─────────────────────────────────────────────────────────────────────────  ║
║  Model:   google/siglip-base-patch16-224                                   ║
║  Method:  1-FPS Sampling → SigLIP Vision Encoder (FP16)                    ║
║           → L2-Normalized Embeddings → Cosine Similarity                   ║
║           → Valley Detection (scipy.signal.find_peaks)                     ║
║  Target:  GPU (CUDA) with CPU fallback                                     ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from scipy.signal import find_peaks


# ─── Video Sampler ───────────────────────────────────────────────────────────

def sample_video_at_fps(
    video_path: str,
    target_fps: float = 1.0,
) -> Tuple[List[np.ndarray], float, int]:
    """
    Sample video ở tần suất target_fps (mặc định 1 FPS).

    Returns:
        frames_bgr: Danh sách các frame BGR (numpy array)
        original_fps: FPS gốc của video
        total_frames: Tổng số frame trong video
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Không thể mở video: {video_path}")

    original_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Tính bước nhảy: nếu video 25fps và target 1fps → mỗi 25 frame lấy 1
    step = max(1, int(round(original_fps / target_fps)))

    frames_bgr: List[np.ndarray] = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % step == 0:
            frames_bgr.append(frame)
        frame_idx += 1

    cap.release()
    return frames_bgr, original_fps, total_frames


# ─── SigLIP Feature Extractor ───────────────────────────────────────────────

class SigLIPExtractor:
    """
    Trích xuất embedding vectors từ SigLIP Vision Encoder.

    Pipeline:
        Frame BGR → Resize 224×224 → SigLIP Processor → Vision Encoder (FP16)
        → Pooled Output → L2 Normalization → Unit Vector (dim=768)
    """

    def __init__(self, model_id: str, device: torch.device, batch_size: int = 32):
        from transformers import AutoProcessor, AutoModel

        self.device = device
        self.batch_size = batch_size

        print(f"⏳ Đang tải SigLIP model: {model_id}...")
        t0 = time.perf_counter()

        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(
            model_id, torch_dtype=torch.float16, device_map=str(device)
        ).eval()

        t1 = time.perf_counter()
        print(f"   ✅ Đã tải xong trong {t1 - t0:.1f}s")

    @torch.no_grad()
    def extract_embeddings(self, frames_bgr: List[np.ndarray]) -> torch.Tensor:
        """
        Trích xuất và chuẩn hóa L2 toàn bộ embedding vectors.

        Args:
            frames_bgr: Danh sách frame BGR

        Returns:
            embeddings: Tensor [N, D] đã chuẩn hóa L2 (unit vectors), trên GPU
        """
        from PIL import Image

        pil_images = [
            Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in frames_bgr
        ]

        all_embeddings: List[torch.Tensor] = []
        n = len(pil_images)

        print(f"⏳ Đang trích xuất {n} embeddings (batch_size={self.batch_size})...")
        t0 = time.perf_counter()

        for i in range(0, n, self.batch_size):
            batch = pil_images[i : i + self.batch_size]

            inputs = self.processor(images=batch, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Forward pass qua Vision Encoder (FP16 autocast)
            with torch.amp.autocast(device_type=self.device.type, enabled=(self.device.type == "cuda")):
                outputs = self.model.get_image_features(**inputs)
                
            # Trích xuất tensor thực sự từ object trả về (Fix lỗi BaseModelOutputWithPooling)
            if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                embeds = outputs.pooler_output
            elif hasattr(outputs, "image_embeds") and outputs.image_embeds is not None:
                embeds = outputs.image_embeds
            elif isinstance(outputs, torch.Tensor):
                embeds = outputs
            else:
                embeds = outputs[0] # Fallback lấy item đầu tiên

            # L2 Normalization ngay trên GPU
            # ||v||₂ = 1 → cosine_sim(a, b) = a · b (dot product thuần túy)
            normalized = F.normalize(embeds, p=2, dim=-1)
            all_embeddings.append(normalized)

            if (i // self.batch_size + 1) % 5 == 0 or (i + self.batch_size) >= n:
                print(f"   Batch {i // self.batch_size + 1}/{(n + self.batch_size - 1) // self.batch_size} "
                      f"({min(i + self.batch_size, n)}/{n} frames)")

        embeddings = torch.cat(all_embeddings, dim=0)  # [N, D]

        t1 = time.perf_counter()
        print(f"   ✅ Trích xuất xong {embeddings.shape[0]} vectors "
              f"(dim={embeddings.shape[1]}) trong {t1 - t0:.1f}s")

        return embeddings


# ─── Similarity & Boundary Detection ────────────────────────────────────────

def compute_sliding_similarity(
    embeddings: torch.Tensor,
    window_offsets: List[int] = [1, 2],
) -> np.ndarray:
    """
    Tính Cosine Similarity trượt (sliding) giữa frame t và frame t+k.

    Vì embeddings đã được L2-normalize, cosine similarity chính là
    tích vô hướng (dot product):
        cos_sim(a, b) = a · b / (||a|| · ||b||) = a · b   (khi ||a|| = ||b|| = 1)

    Args:
        embeddings: Tensor [N, D] đã L2-normalized
        window_offsets: Danh sách các offset k để so sánh (mặc định [1, 2])

    Returns:
        similarity_curve: Mảng 1D [N-1] chứa giá trị similarity trung bình
    """
    n = embeddings.shape[0]
    if n < 2:
        return np.array([1.0])

    # Tính cosine similarity cho từng offset bằng torch.matmul (SIMD trên GPU)
    sim_curves = []
    for k in window_offsets:
        if k >= n:
            continue
        # e[:-k] · e[k:]^T → diagonal = similarity giữa frame t và t+k
        # Tối ưu: dùng element-wise multiply rồi sum thay vì matmul full matrix
        sim = (embeddings[:-k] * embeddings[k:]).sum(dim=-1)  # [N-k]
        # Pad về cùng chiều dài N-1 bằng cách thêm 1.0 ở cuối
        padded = torch.ones(n - 1, device=embeddings.device, dtype=embeddings.dtype)
        padded[: len(sim)] = sim
        sim_curves.append(padded)

    # Trung bình các offset → giảm nhiễu
    avg_sim = torch.stack(sim_curves, dim=0).mean(dim=0)
    return avg_sim.cpu().float().numpy()


def detect_semantic_boundaries(
    similarity_curve: np.ndarray,
    prominence: float = 0.08,
    distance: int = 5,
    height_ratio: float = 0.15,
) -> Tuple[np.ndarray, dict]:
    """
    Phát hiện Semantic Boundaries bằng Valley Detection.

    Ý tưởng cốt lõi:
    - Khi ngữ nghĩa THAY ĐỔI (chuyển cảnh), cosine similarity SỤT GIẢM
      tạo ra các "đáy" (valleys) trên đồ thị.
    - Nhân đường cong với -1 để biến "đáy" thành "đỉnh", rồi dùng
      scipy.signal.find_peaks để phát hiện.

    Chống nhiễu:
    - prominence: Độ nổi bật tối thiểu của đỉnh. Loại bỏ các dao động nhỏ
      do nhân vật cử động tay/đầu (thường prominence < 0.05).
    - distance: Khoảng cách tối thiểu giữa 2 đỉnh (tính bằng số sample).
      Tránh cắt quá sát nhau.
    - height_ratio: Chỉ giữ lại các đỉnh có chiều cao > height_ratio × max_peak.
      Loại bỏ các dao động nền (background fluctuation).

    Args:
        similarity_curve: Mảng 1D cosine similarity
        prominence: Ngưỡng prominence tối thiểu (mặc định 0.08)
        distance: Khoảng cách tối thiểu giữa 2 boundaries (mặc định 5 samples)
        height_ratio: Tỷ lệ chiều cao tối thiểu so với đỉnh cao nhất

    Returns:
        peak_indices: Mảng chỉ số các vị trí boundary
        properties: Dict chứa thông tin chi tiết từ find_peaks
    """
    # Đảo ngược: valley → peak
    inverted = -similarity_curve

    # Phát hiện đỉnh (= đáy gốc)
    peaks, properties = find_peaks(
        inverted,
        prominence=prominence,
        distance=distance,
        wlen=50,  # Chiều rộng cửa sổ để tính prominence cục bộ
    )

    if len(peaks) == 0:
        return peaks, properties

    # Lọc thêm theo chiều cao tương đối
    peak_heights = inverted[peaks]
    max_height = peak_heights.max()
    if max_height > 0:
        height_mask = peak_heights > (height_ratio * max_height)
        peaks = peaks[height_mask]
        # Cập nhật properties
        for key in properties:
            if isinstance(properties[key], np.ndarray) and len(properties[key]) == len(height_mask):
                properties[key] = properties[key][height_mask]

    return peaks, properties


# ─── Main Pipeline ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Semantic Boundary Detection via SigLIP Embedding Similarity",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ví dụ sử dụng:
  python semantic_sbd.py --video "news.mp4"
  python semantic_sbd.py --video "news.mp4" --sample-fps 2 --prominence 0.1
  python semantic_sbd.py --video "news.mp4" --batch-size 64 --offsets 1 2 3
        """,
    )

    # ── Video I/O ────────────────────────────────────────────────────────────
    parser.add_argument("--video", type=str, required=True,
                        help="Đường dẫn tới video cần phân tích")
    parser.add_argument("--output", type=str, default="semantic_shots.jsonl",
                        help="File JSONL đầu ra (mặc định: semantic_shots.jsonl)")

    # ── Sampling ─────────────────────────────────────────────────────────────
    parser.add_argument("--sample-fps", type=float, default=1.0,
                        help="Tần suất lấy mẫu (frames/giây). Mặc định: 1.0")

    # ── Model ────────────────────────────────────────────────────────────────
    parser.add_argument("--model-id", type=str, default="google/siglip-base-patch16-224",
                        help="HuggingFace model ID cho SigLIP")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Số ảnh xử lý cùng lúc qua SigLIP (mặc định: 32)")

    # ── Similarity ───────────────────────────────────────────────────────────
    parser.add_argument("--offsets", type=int, nargs="+", default=[1, 2],
                        help="Các offset k để so sánh frame t với t+k (mặc định: 1 2)")

    # ── Peak Detection ───────────────────────────────────────────────────────
    parser.add_argument("--prominence", type=float, default=0.08,
                        help="Ngưỡng prominence tối thiểu cho find_peaks (mặc định: 0.08)")
    parser.add_argument("--distance", type=int, default=5,
                        help="Khoảng cách tối thiểu giữa 2 boundaries (mặc định: 5 samples)")
    parser.add_argument("--height-ratio", type=float, default=0.15,
                        help="Tỷ lệ chiều cao tối thiểu so với đỉnh cao nhất (mặc định: 0.15)")

    args = parser.parse_args()

    if not os.path.exists(args.video):
        print(f"❌ Không tìm thấy video: {args.video}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    video_name = Path(args.video).name

    print("=" * 70)
    print(f"  Semantic Boundary Detection — {video_name}")
    print(f"  Device: {device} | Model: {args.model_id}")
    print("=" * 70)

    # ── Bước 1: Sample Video ─────────────────────────────────────────────
    print(f"\n📹 [1/4] Sampling video ở {args.sample_fps} FPS...")
    t_start = time.perf_counter()

    frames_bgr, original_fps, total_frames = sample_video_at_fps(args.video, args.sample_fps)
    sample_step = max(1, int(round(original_fps / args.sample_fps)))

    print(f"   Video gốc: {original_fps:.1f} FPS, {total_frames:,} frames, "
          f"{total_frames / original_fps:.1f}s")
    print(f"   Đã sample: {len(frames_bgr)} frames (mỗi {sample_step} frames lấy 1)")

    if len(frames_bgr) < 3:
        print("⚠️ Video quá ngắn để phân tích semantic boundaries.")
        return

    # ── Bước 2: Trích xuất SigLIP Embeddings ─────────────────────────────
    print(f"\n🧠 [2/4] Trích xuất SigLIP embeddings...")

    extractor = SigLIPExtractor(args.model_id, device, args.batch_size)
    embeddings = extractor.extract_embeddings(frames_bgr)

    # Giải phóng bộ nhớ model sau khi trích xuất xong
    del extractor
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # ── Bước 3: Tính Cosine Similarity ───────────────────────────────────
    print(f"\n📊 [3/4] Tính Cosine Similarity (offsets={args.offsets})...")

    sim_curve = compute_sliding_similarity(embeddings, args.offsets)

    print(f"   Similarity curve: min={sim_curve.min():.4f}, "
          f"max={sim_curve.max():.4f}, mean={sim_curve.mean():.4f}")

    # ── Bước 4: Phát hiện Boundaries ─────────────────────────────────────
    print(f"\n🔍 [4/4] Phát hiện Semantic Boundaries "
          f"(prominence={args.prominence}, distance={args.distance})...")

    boundary_indices, peak_props = detect_semantic_boundaries(
        sim_curve,
        prominence=args.prominence,
        distance=args.distance,
        height_ratio=args.height_ratio,
    )

    print(f"   ✅ Tìm thấy {len(boundary_indices)} semantic boundaries")

    # ── Chuyển đổi sang Shots & Xuất JSONL ───────────────────────────────
    # Mỗi boundary_index trong sim_curve tương ứng với vị trí giữa
    # sample[i] và sample[i+1]. Frame thực = boundary_index × sample_step.
    shots = []
    prev_frame = 0

    for i, bi in enumerate(boundary_indices):
        # Frame thực trong video gốc
        boundary_frame = int((bi + 1) * sample_step)

        shots.append({
            "shot_id": int(i),
            "start_frame": int(prev_frame),
            "end_frame": int(boundary_frame - 1),
            "start_timestamp": round(float(prev_frame) / original_fps, 3),
            "end_timestamp": round(float(boundary_frame - 1) / original_fps, 3),
            "type": "semantic",
            "similarity_drop": round(float(1.0 - sim_curve[bi]), 4),
        })
        prev_frame = boundary_frame

    # Shot cuối cùng
    shots.append({
        "shot_id": int(len(shots)),
        "start_frame": int(prev_frame),
        "end_frame": int(total_frames - 1),
        "start_timestamp": round(float(prev_frame) / original_fps, 3),
        "end_timestamp": round(float(total_frames - 1) / original_fps, 3),
        "type": "semantic",
        "similarity_drop": 0.0,
    })

    # Ghi JSONL
    print(f"\n📝 Đang ghi kết quả ra {args.output}...")
    with open(args.output, "w", encoding="utf-8") as f:
        for shot in shots:
            f.write(json.dumps(shot, ensure_ascii=False) + "\n")

    total_time = time.perf_counter() - t_start
    print(f"✅ Hoàn thành! {len(shots)} shots trong {total_time:.1f}s")

    # ── Bảng kết quả ─────────────────────────────────────────────────────
    print(f"\n{'─' * 85}")
    print(f"{'Shot':>5} │ {'Start':>8} │ {'End':>8} │ "
          f"{'Start(s)':>10} │ {'End(s)':>10} │ {'SimDrop':>8} │ {'Type':>10}")
    print(f"{'─' * 85}")
    for s in shots[:15]:
        print(f"{s['shot_id']:>5} │ {s['start_frame']:>8,} │ {s['end_frame']:>8,} │ "
              f"{s['start_timestamp']:>10.3f} │ {s['end_timestamp']:>10.3f} │ "
              f"{s['similarity_drop']:>8.4f} │ {s['type']:>10}")
    if len(shots) > 15:
        print(f"  ... (còn {len(shots) - 15} shots nữa)")
    print(f"{'─' * 85}")


if __name__ == "__main__":
    main()
