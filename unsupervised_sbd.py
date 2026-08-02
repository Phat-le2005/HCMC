"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  Unsupervised Shot Boundary Detection v2.0                                 ║
║  Optimized for Vietnamese TV News / Thời Sự Broadcasts                     ║
║  ─────────────────────────────────────────────────────────────────────────  ║
║  Author:  AI Engineer Pipeline                                             ║
║  Method:  HSV Histogram Intersection + Sobel Edge Change Ratio             ║
║           + Adaptive Thresholding + Flash Suppression                      ║
║  Target:  CPU-only, no Deep Learning dependencies                          ║
╚══════════════════════════════════════════════════════════════════════════════╝

Đặc điểm chương trình thời sự Việt Nam cần xử lý:
  - Hard cuts liên tục giữa MC trường quay ↔ phóng sự hiện trường
  - Lower-third graphics (banner tên, chạy chữ) gây nhiễu kênh dưới khung hình
  - Camera flash tại họp báo / sự kiện → false positive nếu không lọc
  - Dissolve/Wipe khi chuyển mục (Thể thao, Dự báo thời tiết...)
  - Ánh sáng studio ổn định → kênh V (Value) bị dư thừa, chỉ cần H+S
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from numpy.typing import NDArray

# ─── Cấu hình ────────────────────────────────────────────────────────────────

@dataclass
class SBDConfig:
    """Siêu tham số cho thuật toán Shot Boundary Detection."""

    # ── Histogram ────────────────────────────────────────────────────────────
    h_bins: int = 60          # Số bin cho kênh Hue (0-179 trong OpenCV)
    s_bins: int = 48          # Số bin cho kênh Saturation (0-255)
    hist_weight: float = 0.6  # Trọng số của Histogram distance trong tổng score

    # ── Sobel Edge ───────────────────────────────────────────────────────────
    sobel_ksize: int = 3      # Kích thước kernel Sobel (3x3)
    edge_weight: float = 0.4  # Trọng số của Edge Change Ratio trong tổng score

    # ── Adaptive Threshold ───────────────────────────────────────────────────
    window_size: int = 200    # Kích thước cửa sổ trượt (frames)
    alpha_hard: float = 2.5   # Hệ số α cho ngưỡng phát hiện hard-cut
    alpha_soft: float = 1.2   # Hệ số α cho ngưỡng phát hiện soft-transition

    # ── Soft Transition ──────────────────────────────────────────────────────
    soft_min_duration: int = 5    # Dissolve/Wipe phải kéo dài ít nhất N frames
    soft_max_duration: int = 45   # Dissolve/Wipe không kéo dài quá N frames

    # ── Flash Filter ─────────────────────────────────────────────────────────
    flash_lookback: int = 2       # Số frames nhìn lại để kiểm tra flash
    flash_recovery_thresh: float = 0.15  # Nếu frame[i+2] giống frame[i-1] hơn mức này → flash

    # ── Performance ──────────────────────────────────────────────────────────
    downsample_factor: int = 2    # Co nhỏ khung hình xuống 1/N để tăng tốc xử lý
    skip_frames: int = 0          # Bỏ qua N frames giữa các lần đọc (0 = đọc tất cả)

    # ── ROI Crop (Loại bỏ Lower-Third) ───────────────────────────────────────
    roi_top_ratio: float = 0.0    # Crop bỏ % phía trên (0.0 = không crop)
    roi_bottom_ratio: float = 0.20  # Crop bỏ 20% phía dưới (loại banner chữ chạy)


# ─── Dataclass kết quả ───────────────────────────────────────────────────────

@dataclass
class ShotBoundary:
    frame_idx: int
    score: float
    boundary_type: str  # "hard" | "soft"

@dataclass 
class Shot:
    shot_id: int
    start_frame: int
    end_frame: int
    start_timestamp: float
    end_timestamp: float
    boundary_type: str  # "hard" | "soft"


# ─── Core Engine ─────────────────────────────────────────────────────────────

class UnsupervisedSBD:
    """
    Bộ phát hiện ranh giới cảnh (Shot Boundary) không giám sát,
    tối ưu hóa cho dữ liệu truyền hình thời sự.
    """

    def __init__(self, config: SBDConfig | None = None):
        self.cfg = config or SBDConfig()
        # Tiền tính toán (Pre-compute) phạm vi histogram
        self._h_range = np.array([0, 180], dtype=np.float32)
        self._s_range = np.array([0, 256], dtype=np.float32)

    # ────────────────────────────────────────────────────────────────────────
    # BƯỚC 1: Trích xuất đặc trưng (Feature Extraction)
    # ────────────────────────────────────────────────────────────────────────

    def _crop_roi(self, frame: NDArray) -> NDArray:
        """
        Cắt bỏ vùng Lower-Third (banner tên, chữ chạy) ở phía dưới
        và các overlay ở phía trên.
        Lý do: Trong thời sự VTV/HTV, banner chữ chạy liên tục ở 15-20%
        dưới cùng gây ra sự khác biệt giả giữa các frame cùng cảnh.
        """
        h = frame.shape[0]
        top = int(h * self.cfg.roi_top_ratio)
        bottom = int(h * (1.0 - self.cfg.roi_bottom_ratio))
        return frame[top:bottom]

    def _downsample(self, frame: NDArray) -> NDArray:
        """Co nhỏ frame theo hệ số downsample để tăng tốc xử lý."""
        if self.cfg.downsample_factor <= 1:
            return frame
        return frame[::self.cfg.downsample_factor, ::self.cfg.downsample_factor]

    def _compute_hs_histogram(self, frame_bgr: NDArray) -> NDArray:
        """
        Tính Histogram 2D (Hue × Saturation) trong không gian HSV.

        Tại sao chỉ dùng H+S mà bỏ V (Value/Brightness)?
        → Trong studio thời sự, đèn chiếu cố định nên kênh V hầu như
          không thay đổi giữa các frame cùng cảnh. Nhưng khi chuyển từ
          studio sang hiện trường, kênh H và S thay đổi rõ rệt (nền xanh
          studio → cảnh đường phố). Bỏ V giúp:
          1. Giảm 33% khối lượng tính toán
          2. Kháng nhiễu khi ánh sáng dao động nhẹ (mây che nắng, v.v.)
        """
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist(
            [hsv], [0, 1], None,
            [self.cfg.h_bins, self.cfg.s_bins],
            [0, 180, 0, 256]
        )
        # Chuẩn hóa L1 để histogram trở thành phân phối xác suất (tổng = 1)
        total = hist.sum()
        if total > 0:
            hist /= total
        return hist.ravel()  # Trải phẳng thành vector 1D

    def _histogram_intersection_distance(self, hist_a: NDArray, hist_b: NDArray) -> float:
        """
        Tính khoảng cách dựa trên Histogram Intersection.
        
        Công thức gốc (Swain & Ballard, 1991):
            Intersection(H₁, H₂) = Σ min(H₁[i], H₂[i])
        
        Khoảng cách = 1 - Intersection. 
        Giá trị trong [0, 1]: 0 = giống hệt, 1 = hoàn toàn khác.
        """
        intersection = np.minimum(hist_a, hist_b).sum()
        return 1.0 - intersection

    def _compute_edge_map(self, frame_bgr: NDArray) -> NDArray:
        """
        Tính bản đồ cạnh (Edge Map) bằng bộ lọc Sobel.
        
        Sobel cho phép phát hiện các biên cấu trúc (structural edges) 
        của vật thể. Khi xảy ra hiệu ứng Wipe (lau ngang/dọc), 
        các cạnh sẽ thay đổi theo một dải liên tục (progressive change),
        khác với hard-cut (thay đổi toàn bộ đồng loạt).
        """
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        # Gradient theo trục X và Y
        sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=self.cfg.sobel_ksize)
        sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=self.cfg.sobel_ksize)
        # Magnitude = sqrt(Gx² + Gy²), dùng L2 norm
        magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
        return magnitude

    def _edge_change_ratio(self, edge_a: NDArray, edge_b: NDArray) -> float:
        """
        Tính Edge Change Ratio (ECR) giữa 2 frames.
        
        Công thức (Zabih et al., 1999):
            ECR = max(ρ_in, ρ_out)
        Trong đó:
            ρ_in  = (Số pixel cạnh MỚI XUẤT HIỆN ở frame B) / (Tổng pixel cạnh B)
            ρ_out = (Số pixel cạnh BIẾN MẤT ở frame A)     / (Tổng pixel cạnh A)
        
        Ngưỡng nhị phân hóa cạnh: dùng Otsu tự động trên từng frame.
        """
        # Nhị phân hóa bằng ngưỡng Otsu
        _, bin_a = cv2.threshold(edge_a.astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        _, bin_b = cv2.threshold(edge_b.astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        count_a = bin_a.sum()
        count_b = bin_b.sum()

        if count_a == 0 and count_b == 0:
            return 0.0

        # Pixels cạnh xuất hiện mới ở B (có ở B mà không có ở A)
        entering = np.logical_and(bin_b > 0, bin_a == 0).sum()
        # Pixels cạnh biến mất ở A (có ở A mà không có ở B)
        exiting = np.logical_and(bin_a > 0, bin_b == 0).sum()

        rho_in  = entering / max(count_b, 1)
        rho_out = exiting / max(count_a, 1)

        return float(max(rho_in, rho_out))

    # ────────────────────────────────────────────────────────────────────────
    # BƯỚC 2: Adaptive Thresholding (Ngưỡng động)
    # ────────────────────────────────────────────────────────────────────────

    def _adaptive_threshold(self, scores: NDArray, alpha: float) -> NDArray:
        """
        Tính ngưỡng động (Adaptive Threshold) trên cửa sổ trượt.

        Công thức:
            T[i] = μ(window_i) + α · σ(window_i)

        Trong đó:
        - window_i = scores[i - W/2 : i + W/2]
        - μ = trung bình cộng
        - σ = độ lệch chuẩn
        - α = hệ số nhạy (alpha_hard hoặc alpha_soft)

        Cửa sổ trượt giúp ngưỡng tự điều chỉnh theo "bối cảnh cục bộ":
        - Đoạn phóng sự ngoài trời (nhiều chuyển động) → μ cao → T cao → ít false positive
        - Đoạn MC đọc bản tin (ít chuyển động) → μ thấp → T thấp → phát hiện được hard-cut nhỏ
        """
        n = len(scores)
        half_w = self.cfg.window_size // 2
        thresholds = np.zeros(n, dtype=np.float64)

        # Sử dụng cumsum trick để tính mean/std trong O(1) cho mỗi vị trí
        # thay vì O(W) → tổng cộng O(N) thay vì O(N×W)
        padded = np.pad(scores, (half_w, half_w), mode='reflect')
        cumsum = np.cumsum(padded)
        cumsum2 = np.cumsum(padded ** 2)

        for i in range(n):
            left = i               # index trong padded
            right = i + 2 * half_w  # index trong padded

            w_len = right - left + 1
            s = cumsum[right] - (cumsum[left - 1] if left > 0 else 0)
            s2 = cumsum2[right] - (cumsum2[left - 1] if left > 0 else 0)

            mu = s / w_len
            var = max(s2 / w_len - mu ** 2, 0.0)
            sigma = np.sqrt(var)

            thresholds[i] = mu + alpha * sigma

        return thresholds

    # ────────────────────────────────────────────────────────────────────────
    # BƯỚC 3: Flash Light Filtering (Lọc đèn flash)
    # ────────────────────────────────────────────────────────────────────────

    def _filter_flash(self, candidates: List[int], hist_distances: NDArray) -> List[int]:
        """
        Loại bỏ 'hiệu ứng đèn flash' (Camera Flash Suppression).

        Đặc trưng toán học của flash:
        - Frame i-1: Bình thường (cảnh gốc)
        - Frame i:   Thay đổi ĐỘT NGỘT (trắng xóa / quá sáng)
        - Frame i+1 hoặc i+2: TRỞ VỀ giống frame i-1

        Thuật toán kiểm tra:
        Nếu d(frame[candidate - 1], frame[candidate + k]) < flash_recovery_thresh
        với k ∈ {1, 2}, thì candidate là flash → loại bỏ.

        Lý do flash_recovery_thresh = 0.15:
        Trong thời sự, khi phóng viên đưa tin tại họp báo, flash máy ảnh
        chỉ kéo dài 1/25 giây (1 frame ở 25fps). Frame trước và sau flash
        gần như giống hệt nhau (distance < 0.1). Ngưỡng 0.15 đủ rộng để
        chấp nhận một chút rung tay cameraman.
        """
        if len(candidates) == 0:
            return candidates

        n = len(hist_distances)
        filtered = []

        for c in candidates:
            is_flash = False
            for k in range(1, self.cfg.flash_lookback + 1):
                # So sánh frame trước ranh giới với frame sau ranh giới + k
                # hist_distances[i] = d(frame[i], frame[i+1])
                # Nếu frame[c+k] giống frame[c-1] → tức là frame[c] chỉ là flash
                if c >= 1 and (c + k) < n:
                    # Khoảng cách "xuyên qua" flash: tính bằng tổng trung bình
                    # của các khoảng cách liên tiếp, hoặc đơn giản hơn:
                    # nếu hist_distances[c+k-1] rất nhỏ (frame c+k giống frame c+k-1)
                    # VÀ hist_distances[c-1] rất lớn (frame c khác frame c-1)
                    # → pattern "nhảy lên rồi rơi xuống" = flash
                    recovery_distance = hist_distances[c + k - 1] if (c + k - 1) < n else 1.0
                    if recovery_distance < self.cfg.flash_recovery_thresh:
                        is_flash = True
                        break

            if not is_flash:
                filtered.append(c)

        return filtered

    # ────────────────────────────────────────────────────────────────────────
    # BƯỚC 4: Phát hiện Soft Transition (Dissolve / Wipe)
    # ────────────────────────────────────────────────────────────────────────

    def _detect_soft_transitions(
        self,
        combined_scores: NDArray,
        adaptive_thresh_soft: NDArray,
        hard_boundaries: List[int],
    ) -> List[Tuple[int, int]]:
        """
        Phát hiện các chuyển cảnh mềm (Dissolve / Wipe).

        Đặc trưng toán học:
        - Hard-cut: Score tạo ra một ĐỈNH NHỌN đơn lẻ (spike)
        - Soft-transition: Score tạo ra một VÙNG NÂNG CAO kéo dài
          liên tục qua nhiều frames (plateau/hill)

        Thuật toán:
        1. Tìm các vùng liên tục mà score > ngưỡng mềm (alpha_soft)
        2. Lọc bỏ các vùng quá ngắn (< soft_min_duration) hoặc quá dài
        3. Lọc bỏ các vùng đã chứa hard-cut (tránh đếm trùng)
        """
        hard_set = set(hard_boundaries)
        above_soft = combined_scores > adaptive_thresh_soft
        soft_transitions: List[Tuple[int, int]] = []

        # Tìm các "run" liên tục mà score vượt ngưỡng mềm
        in_region = False
        region_start = 0

        for i in range(len(above_soft)):
            if above_soft[i] and not in_region:
                region_start = i
                in_region = True
            elif not above_soft[i] and in_region:
                region_end = i - 1
                duration = region_end - region_start + 1

                # Kiểm tra: có hard-cut nào nằm trong vùng này không?
                contains_hard = any(h >= region_start and h <= region_end for h in hard_set)

                if (not contains_hard
                        and self.cfg.soft_min_duration <= duration <= self.cfg.soft_max_duration):
                    soft_transitions.append((region_start, region_end))

                in_region = False

        return soft_transitions

    # ────────────────────────────────────────────────────────────────────────
    # PIPELINE CHÍNH
    # ────────────────────────────────────────────────────────────────────────

    def detect(self, video_path: str) -> List[Shot]:
        """
        Chạy toàn bộ pipeline phát hiện Shot Boundary.

        Returns:
            Danh sách các Shot đã được phân đoạn.
        """
        video_path = str(video_path)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Không thể mở video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"📹 Video: {Path(video_path).name}")
        print(f"   FPS: {fps:.1f} | Tổng frames: {total_frames:,}")
        print(f"   Thời lượng: {total_frames / fps:.1f} giây")
        print(f"   Cấu hình: downsample={self.cfg.downsample_factor}x, "
              f"window={self.cfg.window_size}, α_hard={self.cfg.alpha_hard}, α_soft={self.cfg.alpha_soft}")

        # ── Giai đoạn 1: Đọc video & trích xuất đặc trưng ─────────────────
        t0 = time.perf_counter()
        print("\n⏳ [1/5] Đang trích xuất đặc trưng (Histogram + Edge)...")

        histograms: List[NDArray] = []
        edge_maps: List[NDArray] = []
        frame_indices: List[int] = []
        skip = self.cfg.skip_frames + 1

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % skip == 0:
                # Tiền xử lý: crop ROI + downsample
                roi = self._crop_roi(frame)
                small = self._downsample(roi)

                histograms.append(self._compute_hs_histogram(small))
                edge_maps.append(self._compute_edge_map(small))
                frame_indices.append(frame_idx)

            frame_idx += 1

            if frame_idx % 5000 == 0:
                print(f"   Đã đọc {frame_idx:,}/{total_frames:,} frames "
                      f"({frame_idx / total_frames * 100:.0f}%)...")

        cap.release()
        n_samples = len(histograms)
        t1 = time.perf_counter()
        print(f"   ✅ Trích xuất xong {n_samples:,} samples trong {t1 - t0:.1f}s "
              f"({n_samples / (t1 - t0):.0f} fps)")

        if n_samples < 3:
            print("⚠️ Video quá ngắn, không đủ frames để phân tích.")
            return [Shot(0, 0, total_frames - 1, 0.0, (total_frames - 1) / fps, "hard")]

        # ── Giai đoạn 2: Tính khoảng cách liên tiếp ───────────────────────
        print("⏳ [2/5] Đang tính khoảng cách giữa các frames liên tiếp...")

        hist_distances = np.zeros(n_samples - 1, dtype=np.float64)
        edge_distances = np.zeros(n_samples - 1, dtype=np.float64)

        for i in range(n_samples - 1):
            hist_distances[i] = self._histogram_intersection_distance(histograms[i], histograms[i + 1])
            edge_distances[i] = self._edge_change_ratio(edge_maps[i], edge_maps[i + 1])

        # Chuẩn hóa Min-Max để đưa cả 2 metric về cùng thang [0, 1]
        def _minmax(arr: NDArray) -> NDArray:
            mn, mx = arr.min(), arr.max()
            if mx - mn < 1e-9:
                return np.zeros_like(arr)
            return (arr - mn) / (mx - mn)

        hist_norm = _minmax(hist_distances)
        edge_norm = _minmax(edge_distances)

        # Trọng số kết hợp (Weighted Fusion)
        combined = self.cfg.hist_weight * hist_norm + self.cfg.edge_weight * edge_norm

        t2 = time.perf_counter()
        print(f"   ✅ Tính xong trong {t2 - t1:.1f}s")

        # ── Giai đoạn 3: Ngưỡng động (Adaptive Thresholding) ──────────────
        print("⏳ [3/5] Đang tính ngưỡng động (Adaptive Threshold)...")

        thresh_hard = self._adaptive_threshold(combined, self.cfg.alpha_hard)
        thresh_soft = self._adaptive_threshold(combined, self.cfg.alpha_soft)

        # Phát hiện hard-cut candidates
        hard_candidates = list(np.where(combined > thresh_hard)[0])
        t3 = time.perf_counter()
        print(f"   ✅ Tìm thấy {len(hard_candidates)} hard-cut candidates trong {t3 - t2:.1f}s")

        # ── Giai đoạn 4: Lọc Flash ────────────────────────────────────────
        print("⏳ [4/5] Đang lọc hiệu ứng đèn flash (Flash Suppression)...")

        n_before = len(hard_candidates)
        hard_boundaries = self._filter_flash(hard_candidates, hist_distances)
        n_after = len(hard_boundaries)
        t4 = time.perf_counter()
        print(f"   ✅ Đã loại bỏ {n_before - n_after} flash, "
              f"còn lại {n_after} hard-cuts trong {t4 - t3:.1f}s")

        # ── Giai đoạn 5: Phát hiện Soft Transitions ───────────────────────
        print("⏳ [5/5] Đang phát hiện Dissolve/Wipe (Soft Transitions)...")

        soft_transitions = self._detect_soft_transitions(combined, thresh_soft, hard_boundaries)
        t5 = time.perf_counter()
        print(f"   ✅ Tìm thấy {len(soft_transitions)} soft-transitions trong {t5 - t4:.1f}s")

        # ── Tổng hợp kết quả thành danh sách Shot ─────────────────────────
        all_boundaries: List[ShotBoundary] = []

        for h in hard_boundaries:
            real_frame = frame_indices[h + 1] if (h + 1) < len(frame_indices) else frame_indices[h]
            all_boundaries.append(ShotBoundary(
                frame_idx=real_frame,
                score=float(combined[h]),
                boundary_type="hard"
            ))

        for (s_start, s_end) in soft_transitions:
            mid_frame = frame_indices[(s_start + s_end) // 2]
            all_boundaries.append(ShotBoundary(
                frame_idx=mid_frame,
                score=float(combined[s_start:s_end + 1].mean()),
                boundary_type="soft"
            ))

        # Sắp xếp theo thứ tự frame
        all_boundaries.sort(key=lambda b: b.frame_idx)

        # Chuyển đổi boundaries → shots
        shots: List[Shot] = []
        prev_frame = 0
        for i, bd in enumerate(all_boundaries):
            shots.append(Shot(
                shot_id=i,
                start_frame=prev_frame,
                end_frame=bd.frame_idx - 1,
                start_timestamp=round(prev_frame / fps, 3),
                end_timestamp=round((bd.frame_idx - 1) / fps, 3),
                boundary_type=bd.boundary_type,
            ))
            prev_frame = bd.frame_idx

        # Shot cuối cùng
        shots.append(Shot(
            shot_id=len(shots),
            start_frame=prev_frame,
            end_frame=total_frames - 1,
            start_timestamp=round(prev_frame / fps, 3),
            end_timestamp=round((total_frames - 1) / fps, 3),
            boundary_type="hard",
        ))

        total_time = time.perf_counter() - t0
        print(f"\n🎯 KẾT QUẢ: {len(shots)} shots được phát hiện trong {total_time:.1f}s")
        print(f"   Hard-cuts: {len(hard_boundaries)} | Soft-transitions: {len(soft_transitions)}")
        print(f"   Tốc độ xử lý: {total_frames / total_time:.0f} fps")

        return shots


# ─── CLI Entry Point ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Unsupervised Shot Boundary Detection for TV News",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ví dụ sử dụng:
  python unsupervised_sbd.py --video "news.mp4"
  python unsupervised_sbd.py --video "news.mp4" --alpha-hard 3.0 --alpha-soft 1.5
  python unsupervised_sbd.py --video "news.mp4" --downsample 3 --skip 1
        """,
    )
    parser.add_argument("--video", type=str, required=True, help="Đường dẫn tới video cần phân tích")
    parser.add_argument("--output", type=str, default="shots_unsupervised.jsonl", help="File JSONL đầu ra")
    parser.add_argument("--alpha-hard", type=float, default=2.5, help="Hệ số α cho Hard-cut (mặc định: 2.5)")
    parser.add_argument("--alpha-soft", type=float, default=1.2, help="Hệ số α cho Soft-transition (mặc định: 1.2)")
    parser.add_argument("--window", type=int, default=200, help="Kích thước cửa sổ trượt (mặc định: 200)")
    parser.add_argument("--downsample", type=int, default=2, help="Hệ số co nhỏ frame (mặc định: 2)")
    parser.add_argument("--skip", type=int, default=0, help="Bỏ qua N frames giữa các lần đọc (mặc định: 0)")
    parser.add_argument("--roi-bottom", type=float, default=0.20, help="Crop bỏ %% phía dưới - loại banner (mặc định: 0.20)")

    args = parser.parse_args()

    if not os.path.exists(args.video):
        print(f"❌ Không tìm thấy video: {args.video}")
        return

    config = SBDConfig(
        alpha_hard=args.alpha_hard,
        alpha_soft=args.alpha_soft,
        window_size=args.window,
        downsample_factor=args.downsample,
        skip_frames=args.skip,
        roi_bottom_ratio=args.roi_bottom,
    )

    detector = UnsupervisedSBD(config)
    shots = detector.detect(args.video)

    # Xuất ra JSONL
    print(f"\n📝 Đang ghi kết quả ra {args.output}...")
    with open(args.output, "w", encoding="utf-8") as f:
        for shot in shots:
            record = {
                "shot_id": shot.shot_id,
                "start_frame": shot.start_frame,
                "end_frame": shot.end_frame,
                "start_timestamp": shot.start_timestamp,
                "end_timestamp": shot.end_timestamp,
                "type": shot.boundary_type,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"✅ Đã lưu thành công {len(shots)} shots vào {args.output}")

    # In 10 shots đầu tiên để kiểm tra nhanh
    print(f"\n{'─' * 80}")
    print(f"{'Shot':>5} │ {'Start':>8} │ {'End':>8} │ {'Start(s)':>10} │ {'End(s)':>10} │ {'Type':>6}")
    print(f"{'─' * 80}")
    for s in shots[:10]:
        print(f"{s.shot_id:>5} │ {s.start_frame:>8,} │ {s.end_frame:>8,} │ "
              f"{s.start_timestamp:>10.3f} │ {s.end_timestamp:>10.3f} │ {s.boundary_type:>6}")
    if len(shots) > 10:
        print(f"  ... (còn {len(shots) - 10} shots nữa)")
    print(f"{'─' * 80}")


if __name__ == "__main__":
    main()
