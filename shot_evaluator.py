"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  Self-Supervised Shot Boundary Detection Evaluator v1.0                    ║
║  ─────────────────────────────────────────────────────────────────────────  ║
║  Model:   google/siglip-base-patch16-224 (Vision Encoder)                  ║
║  Method:  Embedding-based Quality Metrics (SCS, ISS, BSS, IDS, KRS, SQS)  ║
║  Target:  GPU (CUDA) with CPU fallback                                     ║
║  ─────────────────────────────────────────────────────────────────────────  ║
║  Metrics:                                                                  ║
║    SCS — Semantic Consistency Score     (intra-shot coherence)              ║
║    ISS — Inter-Shot Separation Score    (cross-boundary contrast)           ║
║    BSS — Boundary Sharpness Score       (transition sharpness)              ║
║    IDS — Information Density Score      (embedding variance, inverted)      ║
║    KRS — Keyframe Representativeness    (centroid proximity)                ║
║    SQS — Shot Quality Score             (weighted aggregate)               ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F


# ─── Configuration ──────────────────────────────────────────────────────────

@dataclass
class EvalConfig:
    """Cấu hình cho ShotQualityEvaluator."""

    # ── Model ────────────────────────────────────────────────────────────────
    model_id: str = "google/siglip-base-patch16-224"
    batch_size: int = 32

    # ── Sampling ─────────────────────────────────────────────────────────────
    sample_fps: float = 3.0  # Downsample: 3 frame/giây thay vì toàn bộ

    # ── BSS ──────────────────────────────────────────────────────────────────
    bss_window_k: int = 5  # K frames trước/sau boundary cho BSS

    # ── SQS Weights ──────────────────────────────────────────────────────────
    weight_scs: float = 0.40
    weight_iss: float = 0.30
    weight_bss: float = 0.15
    weight_ids: float = 0.05
    weight_krs: float = 0.10


# ─── Per-Shot Result ────────────────────────────────────────────────────────

@dataclass
class ShotMetrics:
    """Kết quả đánh giá chi tiết cho một shot."""

    shot_id: int
    start_frame: int
    end_frame: int
    num_sampled_frames: int
    scs: float  # Semantic Consistency Score
    iss: float  # Inter-Shot Separation Score (NaN cho shot cuối cùng)
    bss: float  # Boundary Sharpness Score  (NaN nếu không đủ window)
    ids: float  # Information Density Score
    krs: float  # Keyframe Representativeness Score
    sqs: float  # Shot Quality Score (weighted aggregate)
    keyframe_index: int  # Index (trong video gốc) của keyframe được chọn

    def to_dict(self) -> Dict[str, Any]:
        return {
            "shot_id": self.shot_id,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "num_sampled_frames": self.num_sampled_frames,
            "scs": round(self.scs, 4),
            "iss": round(self.iss, 4),
            "bss": round(self.bss, 4),
            "ids": round(self.ids, 4),
            "krs": round(self.krs, 4),
            "sqs": round(self.sqs, 4),
            "keyframe_index": self.keyframe_index,
        }


# ─── Core Evaluator ────────────────────────────────────────────────────────

class ShotQualityEvaluator:
    """
    Framework đánh giá chất lượng Shot Boundary Detection tự giám sát
    dựa trên Vision-Language Model (SigLIP).

    Pipeline:
        1. Downsample video ở tần suất cố định (mặc định 3 FPS)
        2. Trích xuất embedding bằng SigLIP Vision Encoder (FP16, batch)
        3. L2-normalize tất cả embeddings → cache lại trên GPU
        4. Tính 5 metrics cho mỗi shot dựa trên cached embeddings
        5. Tổng hợp SQS (weighted aggregate) cho toàn bộ video

    Args:
        config: Đối tượng EvalConfig chứa các siêu tham số.

    Example:
        >>> evaluator = ShotQualityEvaluator()
        >>> results = evaluator.evaluate(
        ...     video_path="news.mp4",
        ...     shots=[(0, 100), (101, 250), (251, 500)]
        ... )
        >>> print(results["video_sqs"])
    """

    def __init__(self, config: Optional[EvalConfig] = None):
        self.cfg = config or EvalConfig()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Lazy-loaded model (chỉ tải khi cần)
        self._model = None
        self._processor = None

        # ── Embedding cache ──────────────────────────────────────────────────
        # Mỗi video sẽ cache: {video_path: {"embeddings": Tensor, "frame_map": list}}
        self._cache: Dict[str, Dict[str, Any]] = {}

    # ────────────────────────────────────────────────────────────────────────
    # Model Management
    # ────────────────────────────────────────────────────────────────────────

    def _ensure_model(self) -> None:
        """Lazy-load SigLIP model (chỉ tải 1 lần duy nhất)."""
        if self._model is not None:
            return

        from transformers import AutoModel, AutoProcessor

        print(f"⏳ Đang tải SigLIP: {self.cfg.model_id}...")
        t0 = time.perf_counter()

        self._processor = AutoProcessor.from_pretrained(self.cfg.model_id)
        self._model = AutoModel.from_pretrained(
            self.cfg.model_id,
            torch_dtype=torch.float16,
            device_map=str(self.device),
        ).eval()

        print(f"   ✅ Model sẵn sàng ({time.perf_counter() - t0:.1f}s) | Device: {self.device}")

    # ────────────────────────────────────────────────────────────────────────
    # Video Sampling & Embedding Extraction (with Cache)
    # ────────────────────────────────────────────────────────────────────────

    def _sample_and_embed(self, video_path: str) -> Tuple[torch.Tensor, List[int]]:
        """
        Đọc video, downsample, trích xuất và cache embeddings.

        Nếu video đã từng được xử lý, trả về kết quả từ cache ngay lập tức
        mà không cần đọc lại video hay chạy lại model.

        Args:
            video_path: Đường dẫn tới video.

        Returns:
            embeddings: Tensor [N, D] đã L2-normalize, trên GPU.
            frame_map: List[int] ánh xạ index → frame gốc trong video.
                       frame_map[i] = frame_index gốc của embedding thứ i.
        """
        # ── Kiểm tra cache ───────────────────────────────────────────────
        abs_path = os.path.abspath(video_path)
        if abs_path in self._cache:
            cached = self._cache[abs_path]
            print(f"   ♻️  Cache hit: {len(cached['frame_map'])} embeddings")
            return cached["embeddings"], cached["frame_map"]

        # ── Đọc video ────────────────────────────────────────────────────
        self._ensure_model()

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Không thể mở video: {video_path}")

        original_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        step = max(1, int(round(original_fps / self.cfg.sample_fps)))

        print(f"   📹 Video: {original_fps:.0f} FPS, {total_frames:,} frames, "
              f"step={step} → ~{total_frames // step:,} samples")

        frames_bgr: List[np.ndarray] = []
        frame_map: List[int] = []
        idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % step == 0:
                frames_bgr.append(frame)
                frame_map.append(idx)
            idx += 1
        cap.release()

        if len(frames_bgr) == 0:
            raise ValueError(f"Video rỗng hoặc không đọc được frame nào: {video_path}")

        # ── Batch embedding extraction ───────────────────────────────────
        from PIL import Image

        pil_images = [
            Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in frames_bgr
        ]
        del frames_bgr  # Giải phóng RAM

        all_embeddings: List[torch.Tensor] = []
        n = len(pil_images)

        print(f"   🧠 Trích xuất {n} embeddings (batch={self.cfg.batch_size})...")
        t0 = time.perf_counter()

        with torch.no_grad():
            for i in range(0, n, self.cfg.batch_size):
                batch = pil_images[i: i + self.cfg.batch_size]
                inputs = self._processor(images=batch, return_tensors="pt", padding=True)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                with torch.amp.autocast(
                    device_type=self.device.type,
                    enabled=(self.device.type == "cuda"),
                ):
                    outputs = self._model.get_image_features(**inputs)

                # Trích xuất tensor từ output object (tương thích đa phiên bản)
                if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                    embeds = outputs.pooler_output
                elif hasattr(outputs, "image_embeds") and outputs.image_embeds is not None:
                    embeds = outputs.image_embeds
                elif isinstance(outputs, torch.Tensor):
                    embeds = outputs
                else:
                    embeds = outputs[0]

                # L2-normalize ngay trên GPU
                normalized = F.normalize(embeds.float(), p=2, dim=-1)
                all_embeddings.append(normalized)

        embeddings = torch.cat(all_embeddings, dim=0)  # [N, D]
        del pil_images, all_embeddings

        t1 = time.perf_counter()
        print(f"   ✅ {embeddings.shape[0]} vectors (dim={embeddings.shape[1]}) "
              f"trong {t1 - t0:.1f}s")

        # ── Lưu cache ────────────────────────────────────────────────────
        self._cache[abs_path] = {
            "embeddings": embeddings,
            "frame_map": frame_map,
        }

        return embeddings, frame_map

    # ────────────────────────────────────────────────────────────────────────
    # Helper: Tìm embedding indices thuộc về một shot
    # ────────────────────────────────────────────────────────────────────────

    def _get_shot_indices(
        self, frame_map: List[int], start_frame: int, end_frame: int
    ) -> List[int]:
        """
        Tìm các embedding indices (trong mảng embeddings) mà frame gốc
        nằm trong khoảng [start_frame, end_frame].

        Args:
            frame_map: Ánh xạ embedding index → frame gốc.
            start_frame: Frame bắt đầu của shot.
            end_frame: Frame kết thúc của shot.

        Returns:
            Danh sách các indices trong mảng embeddings.
        """
        return [
            i for i, f in enumerate(frame_map)
            if start_frame <= f <= end_frame
        ]

    # ────────────────────────────────────────────────────────────────────────
    # Metric 1: Semantic Consistency Score (SCS)
    # ────────────────────────────────────────────────────────────────────────

    def compute_scs(
        self,
        embeddings: torch.Tensor,
        shot_indices: List[int],
    ) -> float:
        """
        Đo độ đồng nhất ngữ nghĩa bên trong 1 shot.

        Công thức:
            SCS = (1/N) × Σ cos_sim(e[i], e[i+1])  ∀ cặp liền kề trong shot.

        Ý nghĩa:
        - SCS ≈ 1.0: Shot rất đồng nhất (MC đọc bản tin, ít thay đổi).
        - SCS ≈ 0.5: Shot chứa nhiều chuyển cảnh phụ → nên chia nhỏ hơn.
        - SCS < 0.3: Rất có thể thuật toán SBD đã gộp sai 2+ cảnh vào 1 shot.

        Args:
            embeddings: Tensor [N_total, D] toàn bộ video (đã L2-normalized).
            shot_indices: Danh sách embedding indices thuộc shot này.

        Returns:
            Điểm SCS ∈ [0.0, 1.0].
        """
        if len(shot_indices) < 2:
            return 1.0  # Shot quá ngắn (≤1 sample) → mặc định đồng nhất hoàn hảo

        shot_embeds = embeddings[shot_indices]  # [K, D]

        # Cosine similarity giữa các cặp liền kề: e[i] · e[i+1]
        similarities = (shot_embeds[:-1] * shot_embeds[1:]).sum(dim=-1)  # [K-1]

        # Clamp về [0, 1] (cosine sim có thể âm nhẹ do floating point)
        similarities = similarities.clamp(0.0, 1.0)

        return float(similarities.mean().item())

    # ────────────────────────────────────────────────────────────────────────
    # Metric 2: Inter-Shot Separation Score (ISS)
    # ────────────────────────────────────────────────────────────────────────

    def compute_iss(
        self,
        embeddings: torch.Tensor,
        current_shot_indices: List[int],
        next_shot_indices: List[int],
    ) -> float:
        """
        Đo độ khác biệt ngữ nghĩa giữa 2 shot liên tiếp.

        Công thức:
            ISS = 1.0 - cos_sim(last_frame_shot_N, first_frame_shot_N+1)

        Ý nghĩa:
        - ISS ≈ 1.0: Hai shot rất khác nhau → boundary chuẩn xác.
        - ISS ≈ 0.0: Hai shot gần giống hệt nhau → có thể cắt sai
          (cắt giữa 2 frame liên tiếp của cùng 1 cảnh).

        Args:
            embeddings: Tensor [N_total, D].
            current_shot_indices: Indices của shot hiện tại.
            next_shot_indices: Indices của shot tiếp theo.

        Returns:
            Điểm ISS ∈ [0.0, 1.0].
        """
        if len(current_shot_indices) == 0 or len(next_shot_indices) == 0:
            return 0.5  # Không đủ dữ liệu → trả về giá trị trung tính

        last_embed = embeddings[current_shot_indices[-1]]   # [D]
        first_embed = embeddings[next_shot_indices[0]]      # [D]

        cos_sim = (last_embed * first_embed).sum().clamp(-1.0, 1.0)
        return float((1.0 - cos_sim).item())

    # ────────────────────────────────────────────────────────────────────────
    # Metric 3: Boundary Sharpness Score (BSS)
    # ────────────────────────────────────────────────────────────────────────

    def compute_bss(
        self,
        embeddings: torch.Tensor,
        current_shot_indices: List[int],
        next_shot_indices: List[int],
    ) -> float:
        """
        Đo độ sắc nét của điểm cắt (Boundary Sharpness).

        Công thức:
            intra_before = mean(cos_sim(e[i], e[i+1])) cho K frames cuối shot N
            intra_after  = mean(cos_sim(e[i], e[i+1])) cho K frames đầu shot N+1
            cross        = cos_sim(last_frame_N, first_frame_N+1)

            BSS = (intra_before + intra_after) / 2.0 - cross

        Ý nghĩa:
        - BSS cao: Các frame bên trong mỗi shot rất giống nhau, nhưng
          frames qua ranh giới rất khác → điểm cắt sắc nét (hard-cut).
        - BSS ≈ 0: Không có sự khác biệt rõ ràng → boundary mờ nhạt
          hoặc cắt sai vị trí.

        Args:
            embeddings: Tensor [N_total, D].
            current_shot_indices: Indices của shot hiện tại.
            next_shot_indices: Indices của shot tiếp theo.

        Returns:
            Điểm BSS ∈ [0.0, 1.0] (đã clamp).
        """
        k = self.cfg.bss_window_k

        # Lấy K frames cuối của shot hiện tại
        window_before = current_shot_indices[-k:] if len(current_shot_indices) >= k \
            else current_shot_indices
        # Lấy K frames đầu của shot tiếp theo
        window_after = next_shot_indices[:k] if len(next_shot_indices) >= k \
            else next_shot_indices

        if len(window_before) < 2 and len(window_after) < 2:
            return 0.5  # Không đủ dữ liệu

        # Intra-similarity bên trước
        if len(window_before) >= 2:
            before_embeds = embeddings[window_before]
            intra_before = (before_embeds[:-1] * before_embeds[1:]).sum(dim=-1).clamp(0, 1).mean()
        else:
            intra_before = torch.tensor(0.5, device=embeddings.device)

        # Intra-similarity bên sau
        if len(window_after) >= 2:
            after_embeds = embeddings[window_after]
            intra_after = (after_embeds[:-1] * after_embeds[1:]).sum(dim=-1).clamp(0, 1).mean()
        else:
            intra_after = torch.tensor(0.5, device=embeddings.device)

        # Cross-boundary similarity
        last_embed = embeddings[current_shot_indices[-1]]
        first_embed = embeddings[next_shot_indices[0]]
        cross_sim = (last_embed * first_embed).sum().clamp(-1.0, 1.0)

        bss = (intra_before + intra_after) / 2.0 - cross_sim
        return float(bss.clamp(0.0, 1.0).item())

    # ────────────────────────────────────────────────────────────────────────
    # Metric 4: Information Density Score (IDS)
    # ────────────────────────────────────────────────────────────────────────

    def compute_ids(
        self,
        embeddings: torch.Tensor,
        shot_indices: List[int],
    ) -> float:
        """
        Đo lượng thông tin (đa dạng ngữ nghĩa) trong một shot.

        Công thức:
            variance = mean(||e[i] - μ||²)   (μ = mean embedding)
            IDS = 1.0 - min(variance / scale, 1.0)

        Ý nghĩa:
        - IDS ≈ 1.0: Shot có variance thấp → nội dung đồng nhất, 
          không chứa quá nhiều cảnh khác nhau → tốt.
        - IDS < 0.5: Shot có variance cao → nghi ngờ chứa nhiều cảnh
          bị gộp sai → cần kiểm tra lại SBD.

        Lưu ý:
        - Scale factor (0.5) được chọn dựa trên phân tích thực nghiệm
          với L2-normalized embeddings. Variance trung bình của một shot
          đúng thường < 0.1, variance của shot sai (gộp 2+ cảnh) thường > 0.3.

        Args:
            embeddings: Tensor [N_total, D].
            shot_indices: Indices trong shot.

        Returns:
            Điểm IDS ∈ [0.0, 1.0].
        """
        if len(shot_indices) < 2:
            return 1.0  # Shot quá ngắn → variance = 0 → IDS = 1.0

        shot_embeds = embeddings[shot_indices]  # [K, D]
        centroid = shot_embeds.mean(dim=0, keepdim=True)  # [1, D]

        # Squared L2 distance tới centroid
        distances_sq = ((shot_embeds - centroid) ** 2).sum(dim=-1)  # [K]
        variance = distances_sq.mean()

        # Chuẩn hóa ngược: variance cao → IDS thấp
        scale = 0.5  # Hằng số co giãn (tuning parameter)
        ids = 1.0 - min(float(variance.item()) / scale, 1.0)

        return max(ids, 0.0)

    # ────────────────────────────────────────────────────────────────────────
    # Metric 5: Keyframe Representativeness Score (KRS)
    # ────────────────────────────────────────────────────────────────────────

    def compute_krs(
        self,
        embeddings: torch.Tensor,
        shot_indices: List[int],
    ) -> Tuple[float, int]:
        """
        Tìm Keyframe và đo mức độ đại diện của nó cho toàn bộ shot.

        Thuật toán:
            1. Tính mean embedding (centroid) của shot.
            2. Keyframe = frame có cosine similarity cao nhất với centroid.
            3. KRS = mean(cos_sim(e[i], keyframe)) cho mọi frame trong shot.

        Ý nghĩa:
        - KRS ≈ 1.0: Keyframe đại diện rất tốt cho toàn bộ shot.
        - KRS < 0.7: Shot quá đa dạng, 1 keyframe không đủ đại diện.

        Args:
            embeddings: Tensor [N_total, D].
            shot_indices: Indices trong shot.

        Returns:
            (krs_score, keyframe_emb_index): Điểm KRS và index của keyframe
            trong mảng embeddings.
        """
        if len(shot_indices) < 2:
            return 1.0, shot_indices[0] if shot_indices else 0

        shot_embeds = embeddings[shot_indices]  # [K, D]
        centroid = shot_embeds.mean(dim=0)       # [D]
        centroid = F.normalize(centroid, p=2, dim=-1)  # Re-normalize

        # Tìm frame gần centroid nhất
        sim_to_centroid = (shot_embeds * centroid.unsqueeze(0)).sum(dim=-1)  # [K]
        keyframe_local_idx = int(sim_to_centroid.argmax().item())
        keyframe_global_idx = shot_indices[keyframe_local_idx]

        # KRS = mean cosine similarity từ tất cả frames tới keyframe
        keyframe_embed = embeddings[keyframe_global_idx]  # [D]
        sim_to_keyframe = (shot_embeds * keyframe_embed.unsqueeze(0)).sum(dim=-1)  # [K]
        krs = float(sim_to_keyframe.clamp(0.0, 1.0).mean().item())

        return krs, keyframe_global_idx

    # ────────────────────────────────────────────────────────────────────────
    # Metric 6: Shot Quality Score (SQS) — Weighted Aggregate
    # ────────────────────────────────────────────────────────────────────────

    def compute_sqs(
        self,
        scs: float,
        iss: float,
        bss: float,
        ids: float,
        krs: float,
    ) -> float:
        """
        Tổng hợp điểm chất lượng Shot (Shot Quality Score).

        Công thức:
            SQS = w_scs×SCS + w_iss×ISS + w_bss×BSS + w_ids×IDS + w_krs×KRS

        Trọng số mặc định (tổng = 1.0):
            SCS: 0.40 (đồng nhất ngữ nghĩa — quan trọng nhất)
            ISS: 0.30 (phân tách giữa 2 shot — thước đo boundary)
            BSS: 0.15 (độ sắc nét — phân biệt hard/soft cut)
            IDS: 0.05 (mật độ thông tin — phụ trợ)
            KRS: 0.10 (đại diện keyframe — hữu ích cho downstream)

        Args:
            scs: Semantic Consistency Score.
            iss: Inter-Shot Separation Score.
            bss: Boundary Sharpness Score.
            ids: Information Density Score.
            krs: Keyframe Representativeness Score.

        Returns:
            Điểm SQS ∈ [0.0, 1.0].
        """
        sqs = (
            self.cfg.weight_scs * scs
            + self.cfg.weight_iss * iss
            + self.cfg.weight_bss * bss
            + self.cfg.weight_ids * ids
            + self.cfg.weight_krs * krs
        )
        return float(max(0.0, min(1.0, sqs)))

    # ────────────────────────────────────────────────────────────────────────
    # Main Evaluation Pipeline
    # ────────────────────────────────────────────────────────────────────────

    def evaluate(
        self,
        video_path: str,
        shots: List[Tuple[int, int]],
    ) -> Dict[str, Any]:
        """
        Chạy đánh giá toàn bộ danh sách shot cho một video.

        Pipeline:
            1. Sample video ở sample_fps → trích xuất embeddings (cached)
            2. Ánh xạ shot (start_frame, end_frame) → embedding indices
            3. Tính 5 metrics cho mỗi shot
            4. Tổng hợp SQS cho mỗi shot
            5. Tính SQS trung bình cho toàn bộ video

        Args:
            video_path: Đường dẫn tới video.
            shots: List[(start_frame, end_frame)] — dự đoán từ thuật toán SBD.

        Returns:
            Dict chứa:
                - "video_path": str
                - "total_shots": int
                - "per_shot_metrics": List[ShotMetrics.to_dict()]
                - "video_scs": float (trung bình SCS)
                - "video_iss": float (trung bình ISS)
                - "video_bss": float (trung bình BSS)
                - "video_ids": float (trung bình IDS)
                - "video_krs": float (trung bình KRS)
                - "video_sqs": float (trung bình SQS)
        """
        print(f"\n{'=' * 70}")
        print(f"  📊 Đánh giá Shot Quality: {Path(video_path).name}")
        print(f"  Số shots cần đánh giá: {len(shots)}")
        print(f"{'=' * 70}")

        t0 = time.perf_counter()

        # ── Bước 1: Trích xuất & cache embeddings ────────────────────────
        print(f"\n🔧 [1/2] Chuẩn bị embeddings...")
        embeddings, frame_map = self._sample_and_embed(video_path)

        # ── Bước 2: Tính metrics cho từng shot ──────────────────────────
        print(f"\n📐 [2/2] Đang tính metrics cho {len(shots)} shots...")

        all_shot_indices: List[List[int]] = []
        for start_f, end_f in shots:
            indices = self._get_shot_indices(frame_map, start_f, end_f)
            all_shot_indices.append(indices)

        per_shot_metrics: List[ShotMetrics] = []

        for i, (start_f, end_f) in enumerate(shots):
            indices = all_shot_indices[i]

            # ── SCS ──────────────────────────────────────────────────────
            scs = self.compute_scs(embeddings, indices)

            # ── ISS (cần shot tiếp theo) ─────────────────────────────────
            if i < len(shots) - 1:
                next_indices = all_shot_indices[i + 1]
                iss = self.compute_iss(embeddings, indices, next_indices)
            else:
                iss = 0.5  # Shot cuối cùng → giá trị trung tính

            # ── BSS (cần shot tiếp theo) ─────────────────────────────────
            if i < len(shots) - 1:
                next_indices = all_shot_indices[i + 1]
                bss = self.compute_bss(embeddings, indices, next_indices)
            else:
                bss = 0.5

            # ── IDS ──────────────────────────────────────────────────────
            ids = self.compute_ids(embeddings, indices)

            # ── KRS ──────────────────────────────────────────────────────
            krs, keyframe_emb_idx = self.compute_krs(embeddings, indices)
            keyframe_real = frame_map[keyframe_emb_idx] if keyframe_emb_idx < len(frame_map) else 0

            # ── SQS ──────────────────────────────────────────────────────
            sqs = self.compute_sqs(scs, iss, bss, ids, krs)

            metric = ShotMetrics(
                shot_id=i,
                start_frame=start_f,
                end_frame=end_f,
                num_sampled_frames=len(indices),
                scs=scs,
                iss=iss,
                bss=bss,
                ids=ids,
                krs=krs,
                sqs=sqs,
                keyframe_index=keyframe_real,
            )
            per_shot_metrics.append(metric)

        # ── Tổng hợp video-level metrics ─────────────────────────────────
        n = len(per_shot_metrics)
        video_scs = sum(m.scs for m in per_shot_metrics) / n if n > 0 else 0.0
        video_iss = sum(m.iss for m in per_shot_metrics) / n if n > 0 else 0.0
        video_bss = sum(m.bss for m in per_shot_metrics) / n if n > 0 else 0.0
        video_ids = sum(m.ids for m in per_shot_metrics) / n if n > 0 else 0.0
        video_krs = sum(m.krs for m in per_shot_metrics) / n if n > 0 else 0.0
        video_sqs = sum(m.sqs for m in per_shot_metrics) / n if n > 0 else 0.0

        total_time = time.perf_counter() - t0

        # ── In bảng tóm tắt ──────────────────────────────────────────────
        self._print_summary(per_shot_metrics, video_scs, video_iss,
                            video_bss, video_ids, video_krs, video_sqs, total_time)

        return {
            "video_path": video_path,
            "total_shots": len(shots),
            "processing_time_seconds": round(total_time, 2),
            "per_shot_metrics": [m.to_dict() for m in per_shot_metrics],
            "video_scs": round(video_scs, 4),
            "video_iss": round(video_iss, 4),
            "video_bss": round(video_bss, 4),
            "video_ids": round(video_ids, 4),
            "video_krs": round(video_krs, 4),
            "video_sqs": round(video_sqs, 4),
        }

    # ────────────────────────────────────────────────────────────────────────
    # Pretty Print
    # ────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _print_summary(
        metrics: List[ShotMetrics],
        v_scs: float, v_iss: float, v_bss: float,
        v_ids: float, v_krs: float, v_sqs: float,
        elapsed: float,
    ) -> None:
        """In bảng tóm tắt kết quả đánh giá."""
        hdr = (f"{'Shot':>5} │ {'Frames':>12} │ {'#Emb':>5} │ "
               f"{'SCS':>6} │ {'ISS':>6} │ {'BSS':>6} │ "
               f"{'IDS':>6} │ {'KRS':>6} │ {'SQS':>6} │ {'Keyframe':>9}")
        sep = "─" * len(hdr)

        print(f"\n{sep}")
        print(hdr)
        print(sep)

        display_count = min(len(metrics), 20)
        for m in metrics[:display_count]:
            print(
                f"{m.shot_id:>5} │ "
                f"{m.start_frame:>5}-{m.end_frame:<5} │ "
                f"{m.num_sampled_frames:>5} │ "
                f"{m.scs:>6.3f} │ {m.iss:>6.3f} │ {m.bss:>6.3f} │ "
                f"{m.ids:>6.3f} │ {m.krs:>6.3f} │ {m.sqs:>6.3f} │ "
                f"{m.keyframe_index:>9,}"
            )
        if len(metrics) > display_count:
            print(f"  ... (còn {len(metrics) - display_count} shots nữa)")

        print(sep)
        print(f"{'AVG':>5} │ {'':>12} │ {'':>5} │ "
              f"{v_scs:>6.3f} │ {v_iss:>6.3f} │ {v_bss:>6.3f} │ "
              f"{v_ids:>6.3f} │ {v_krs:>6.3f} │ {v_sqs:>6.3f} │")
        print(sep)

        # Phán định chất lượng
        if v_sqs >= 0.75:
            verdict = "🟢 XUẤT SẮC — Thuật toán SBD cắt rất chính xác!"
        elif v_sqs >= 0.60:
            verdict = "🟡 KHÁ — Phần lớn boundaries hợp lý, một số cần tinh chỉnh."
        elif v_sqs >= 0.45:
            verdict = "🟠 TRUNG BÌNH — Nhiều boundaries chưa chính xác."
        else:
            verdict = "🔴 YẾU — Thuật toán SBD cần cải thiện đáng kể."

        print(f"\n🎯 Video SQS = {v_sqs:.4f}  →  {verdict}")
        print(f"⏱️  Thời gian xử lý: {elapsed:.1f}s")


# ─── CLI & Demo ──────────────────────────────────────────────────────────────

def load_shots_from_jsonl(jsonl_path: str) -> List[Tuple[int, int]]:
    """
    Đọc danh sách shot từ file JSONL (output của các script SBD trước).

    Mỗi dòng JSONL cần có trường "start_frame" và "end_frame".

    Args:
        jsonl_path: Đường dẫn tới file JSONL.

    Returns:
        List[(start_frame, end_frame)]
    """
    shots: List[Tuple[int, int]] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            shots.append((int(record["start_frame"]), int(record["end_frame"])))
    return shots


def main():
    parser = argparse.ArgumentParser(
        description="Self-Supervised Shot Quality Evaluator (SigLIP-based)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ví dụ sử dụng:
  # Đánh giá từ file JSONL (output của thuật toán SBD):
  python shot_evaluator.py --video "news.mp4" --shots-jsonl "shots.jsonl"

  # Đánh giá với mock data (demo):
  python shot_evaluator.py --demo --video "news.mp4"

  # Tùy chỉnh tham số:
  python shot_evaluator.py --video "news.mp4" --shots-jsonl "shots.jsonl" \\
      --sample-fps 5 --batch-size 64 --bss-window 8
        """,
    )
    parser.add_argument("--video", type=str, required=True,
                        help="Đường dẫn tới video cần đánh giá")
    parser.add_argument("--shots-jsonl", type=str, default=None,
                        help="File JSONL chứa danh sách shot (start_frame, end_frame)")
    parser.add_argument("--output", type=str, default="evaluation_results.json",
                        help="File JSON lưu kết quả đánh giá")
    parser.add_argument("--demo", action="store_true",
                        help="Chạy với mock data (chia đều video thành 10 shots)")

    # ── Tuning parameters ────────────────────────────────────────────────
    parser.add_argument("--sample-fps", type=float, default=3.0,
                        help="Tần suất lấy mẫu (mặc định: 3.0)")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size cho SigLIP (mặc định: 32)")
    parser.add_argument("--bss-window", type=int, default=5,
                        help="K frames trước/sau boundary cho BSS (mặc định: 5)")

    args = parser.parse_args()

    if not os.path.exists(args.video):
        print(f"❌ Không tìm thấy video: {args.video}")
        return

    # ── Xác định danh sách shots ─────────────────────────────────────────
    if args.shots_jsonl and os.path.exists(args.shots_jsonl):
        print(f"📄 Đọc shots từ: {args.shots_jsonl}")
        shots = load_shots_from_jsonl(args.shots_jsonl)
    elif args.demo:
        # Mock data: chia đều video thành 10 shots
        cap = cv2.VideoCapture(args.video)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        n_shots = 10
        chunk = total_frames // n_shots
        shots = [(i * chunk, (i + 1) * chunk - 1) for i in range(n_shots)]
        print(f"🎭 Demo mode: Chia đều {total_frames:,} frames thành {n_shots} shots")
    else:
        print("❌ Cần cung cấp --shots-jsonl hoặc --demo")
        return

    if len(shots) == 0:
        print("⚠️ Danh sách shots rỗng!")
        return

    # ── Chạy đánh giá ────────────────────────────────────────────────────
    config = EvalConfig(
        sample_fps=args.sample_fps,
        batch_size=args.batch_size,
        bss_window_k=args.bss_window,
    )

    evaluator = ShotQualityEvaluator(config)
    results = evaluator.evaluate(args.video, shots)

    # ── Lưu kết quả ─────────────────────────────────────────────────────
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n📝 Đã lưu kết quả chi tiết vào: {args.output}")


if __name__ == "__main__":
    main()
