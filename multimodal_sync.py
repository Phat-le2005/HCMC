"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  Multi-Modal Synchronization Pipeline for Broadcast Video v1.0             ║
║  ─────────────────────────────────────────────────────────────────────────  ║
║  Audio Branch:  pyannote/speaker-diarization-3.1  →  RTTM Speaker Turns    ║
║  Visual Branch: YOLOv8-face (batch)  →  Anchor/Field Classification        ║
║  Fusion:        Late Fusion Timestamp Alignment (±0.5s tolerance)          ║
║  ─────────────────────────────────────────────────────────────────────────  ║
║  Designed for Vietnamese TV News / Thời Sự Broadcasts                      ║
╚══════════════════════════════════════════════════════════════════════════════╝

Luồng xử lý:
    Video  ──┬──► [Audio Extraction] ──► [Pyannote Diarization] ──► Speaker Turns
             │
             └──► [Frame Sampling]   ──► [YOLOv8-face Batch]    ──► Anchor/Field States
                                                                         │
                     ┌───────────────────────────────────────────────────┘
                     ▼
              [Late Fusion Engine]
                     │
                     ├── Rule 1: Speaker Turn ∩ Visual State Change (±0.5s)
                     ├── Rule 2: Anchor Segment > 10s bị ngắt quãng
                     │
                     ▼
              Confirmed Shot Boundaries → JSONL
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


# ═══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class SpeakerSegment:
    """Một đoạn phát biểu của một Speaker từ RTTM."""
    speaker_id: str
    start_time: float   # giây (ms precision)
    end_time: float     # giây
    duration: float     # giây

    @property
    def mid_time(self) -> float:
        return (self.start_time + self.end_time) / 2.0


@dataclass
class SpeakerTurn:
    """
    Điểm chuyển đổi người nói (Speaker Turn).
    Xảy ra khi Speaker A kết thúc và Speaker B bắt đầu.
    """
    timestamp: float     # Thời điểm chuyển đổi (giây)
    from_speaker: str    # Speaker trước
    to_speaker: str      # Speaker sau
    gap: float           # Khoảng cách thời gian giữa 2 speaker (giây)


@dataclass
class VisualState:
    """Trạng thái thị giác của một khung hình."""
    timestamp: float     # Thời điểm (giây)
    frame_idx: int       # Index frame gốc
    state: str           # "anchor" | "field"
    num_faces: int       # Số khuôn mặt phát hiện được
    max_face_area: float # Diện tích khuôn mặt lớn nhất (tỷ lệ % so với frame)
    face_center_x: float # Tọa độ X trung tâm khuôn mặt lớn nhất (0.0-1.0)


@dataclass
class VisualStateChange:
    """Điểm thay đổi trạng thái thị giác (Anchor ↔ Field)."""
    timestamp: float
    from_state: str      # "anchor" | "field"
    to_state: str        # "anchor" | "field"


@dataclass
class ConfirmedBoundary:
    """Ranh giới shot đã được xác nhận qua Late Fusion."""
    timestamp: float
    frame_idx: int
    rule: str            # "speaker_visual_sync" | "anchor_interrupt"
    confidence: float    # 0.0 - 1.0
    details: Dict[str, Any] = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════════════════════
# AUDIO BRANCH — Speaker Diarization
# ═══════════════════════════════════════════════════════════════════════════════

class AudioBranch:
    """
    Trích xuất Speaker Timeline từ audio bằng pyannote/speaker-diarization-3.1.

    Pipeline:
        Video → [FFmpeg extract WAV 16kHz mono] → [Pyannote Diarization]
              → List[SpeakerSegment] → List[SpeakerTurn]
    """

    def __init__(self, hf_token: Optional[str] = None):
        """
        Args:
            hf_token: HuggingFace token (bắt buộc cho pyannote, yêu cầu
                      accept license trên https://hf.co/pyannote/speaker-diarization-3.1).
        """
        self.hf_token = hf_token or os.environ.get("HF_TOKEN", "")
        self._pipeline = None

    def _ensure_pipeline(self) -> None:
        """Lazy-load pyannote pipeline."""
        if self._pipeline is not None:
            return

        from pyannote.audio import Pipeline
        import torch

        print("⏳ [Audio] Đang tải pyannote/speaker-diarization-3.1...")
        t0 = time.perf_counter()

        self._pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            token=self.hf_token,
        )

        # Đẩy lên GPU nếu có
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._pipeline.to(device)

        print(f"   ✅ Pyannote sẵn sàng trên {device} ({time.perf_counter() - t0:.1f}s)")

    @staticmethod
    def extract_audio(video_path: str, output_wav: str) -> str:
        """
        Trích xuất audio từ video bằng FFmpeg.

        Chuẩn hóa:
        - Sample rate: 16000 Hz (yêu cầu của pyannote)
        - Channels: mono (1 kênh)
        - Codec: PCM 16-bit
        """
        print(f"   🔊 Đang trích xuất audio từ video...")
        cmd = [
            "ffmpeg", "-y", "-i", video_path,
            "-vn",                    # Bỏ video
            "-acodec", "pcm_s16le",   # PCM 16-bit
            "-ar", "16000",           # 16kHz
            "-ac", "1",               # Mono
            output_wav,
        ]
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=300
        )
        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg lỗi: {result.stderr[:500]}")

        print(f"   ✅ Audio: {output_wav}")
        return output_wav

    def diarize(self, video_path: str) -> Tuple[List[SpeakerSegment], List[SpeakerTurn]]:
        """
        Chạy Speaker Diarization. Nếu gặp lỗi HuggingFace (403 GatedRepo),
        sẽ tự động chuyển sang phương pháp dự phòng (Audio Energy Fallback).
        """
        # Trích xuất audio tạm
        tmp_dir = tempfile.mkdtemp()
        wav_path = os.path.join(tmp_dir, "audio.wav")
        self.extract_audio(video_path, wav_path)

        segments = []
        turns = []

        try:
            self._ensure_pipeline()
            print("   🎙️ Đang chạy Speaker Diarization (có thể mất vài phút)...")
            t0 = time.perf_counter()

            diarization = self._pipeline(wav_path)

            t1 = time.perf_counter()
            print(f"   ✅ Diarization xong trong {t1 - t0:.1f}s")

            for turn, _, speaker in diarization.itertracks(yield_label=True):
                segments.append(SpeakerSegment(
                    speaker_id=str(speaker),
                    start_time=round(turn.start, 3),
                    end_time=round(turn.end, 3),
                    duration=round(turn.end - turn.start, 3),
                ))
            segments.sort(key=lambda s: s.start_time)
            turns = self._extract_speaker_turns(segments)

        except Exception as e:
            if "GatedRepo" in str(e) or "403" in str(e):
                print(f"\n   ⚠️ CẢNH BÁO: Lỗi xác thực HuggingFace (Chưa Accept License).")
                print(f"   🔄 TỰ ĐỘNG CHUYỂN SANG: Audio Energy Fallback (Không cần token).")
                segments, turns = self._fallback_energy_detection(wav_path)
            else:
                raise e
        finally:
            # Cleanup
            try:
                os.remove(wav_path)
                os.rmdir(tmp_dir)
            except OSError:
                pass

        print(f"   📊 {len(segments)} segments, {len(turns)} turns")
        return segments, turns

    def _fallback_energy_detection(self, wav_path: str) -> Tuple[List[SpeakerSegment], List[SpeakerTurn]]:
        """
        Dự phòng: Phát hiện điểm thay đổi âm thanh dựa trên RMS Energy (tìm khoảng lặng).
        Hoạt động hoàn toàn offline, không cần AI model hay HuggingFace token.
        """
        import wave
        import struct

        with wave.open(wav_path, 'rb') as wf:
            n_channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            framerate = wf.getframerate()
            n_frames = wf.getnframes()
            data = wf.readframes(n_frames)

        # Chuyển đổi byte data sang numpy array
        if sampwidth == 2:
            fmt = f"<{n_frames * n_channels}h"
        else:
            raise ValueError("Chỉ hỗ trợ âm thanh 16-bit")
        
        samples = np.array(struct.unpack(fmt, data), dtype=np.float32)
        
        # Tính RMS Energy theo cửa sổ 100ms
        window_size = int(framerate * 0.1)
        # Reshape to calculate RMS per window
        num_windows = len(samples) // window_size
        samples = samples[:num_windows * window_size]
        windows = samples.reshape(-1, window_size)
        
        # RMS = sqrt(mean(square))
        rms = np.sqrt(np.mean(windows**2, axis=1))
        
        # Threshold: 10% của max RMS
        threshold = np.max(rms) * 0.10
        
        is_speech = rms > threshold
        
        segments = []
        turns = []
        
        # State machine để tìm các đoạn có tiếng
        in_speech = False
        start_time = 0.0
        speaker_idx = 0
        
        for i, speech in enumerate(is_speech):
            time_sec = i * 0.1
            if speech and not in_speech:
                in_speech = True
                start_time = time_sec
            elif not speech and in_speech:
                in_speech = False
                duration = time_sec - start_time
                if duration >= 0.5: # Đoạn nói dài ít nhất 0.5s
                    segments.append(SpeakerSegment(
                        speaker_id=f"SPEAKER_{speaker_idx}",
                        start_time=round(start_time, 3),
                        end_time=round(time_sec, 3),
                        duration=round(duration, 3)
                    ))
                    speaker_idx += 1
                    
                    # Nếu có đoạn trước đó, tạo 1 turn
                    if len(segments) > 1:
                        prev_seg = segments[-2]
                        gap = start_time - prev_seg.end_time
                        if 0.2 < gap < 5.0: # Khoảng lặng từ 0.2s đến 5s
                            turns.append(SpeakerTurn(
                                timestamp=round(start_time, 3),
                                from_speaker=prev_seg.speaker_id,
                                to_speaker=f"SPEAKER_{speaker_idx-1}",
                                gap=round(gap, 3)
                            ))
                            
        return segments, turns

    @staticmethod
    def _extract_speaker_turns(segments: List[SpeakerSegment]) -> List[SpeakerTurn]:
        """
        Trích xuất các điểm chuyển đổi người nói (Speaker Turns).

        Speaker Turn = thời điểm Speaker A kết thúc và Speaker B (khác A) bắt đầu.
        Gap < 2.0s được coi là chuyển đổi liên tục (không phải im lặng dài).
        """
        turns: List[SpeakerTurn] = []

        for i in range(len(segments) - 1):
            curr = segments[i]
            next_seg = segments[i + 1]

            # Chỉ tính khi đổi speaker
            if curr.speaker_id != next_seg.speaker_id:
                gap = next_seg.start_time - curr.end_time
                # Bỏ qua khoảng im lặng quá dài (> 2s = có thể là break quảng cáo)
                if gap < 2.0:
                    turns.append(SpeakerTurn(
                        timestamp=round(curr.end_time, 3),
                        from_speaker=curr.speaker_id,
                        to_speaker=next_seg.speaker_id,
                        gap=round(gap, 3),
                    ))

        return turns

    @staticmethod
    def save_rttm(segments: List[SpeakerSegment], output_path: str) -> None:
        """Lưu kết quả dưới định dạng RTTM chuẩn NIST."""
        with open(output_path, "w") as f:
            for seg in segments:
                # RTTM format: SPEAKER <file> 1 <start> <dur> <NA> <NA> <spk> <NA> <NA>
                f.write(
                    f"SPEAKER file 1 {seg.start_time:.3f} {seg.duration:.3f} "
                    f"<NA> <NA> {seg.speaker_id} <NA> <NA>\n"
                )


# ═══════════════════════════════════════════════════════════════════════════════
# VISUAL BRANCH — YOLOv8-Face + Anchor/Field Classification
# ═══════════════════════════════════════════════════════════════════════════════

class VisualBranch:
    """
    Phân loại khung hình thành Anchor/Field dựa trên khuôn mặt.

    Anchor: 1-2 khuôn mặt LỚN ở trung tâm khung hình (MC đọc bản tin).
    Field:  Khuôn mặt nhỏ/nhiều/không có (cảnh phóng sự, bản đồ, đồ họa).

    Pipeline:
        Video → [Sample ở visual_fps] → [YOLOv8-face Batch Detection]
              → [Anchor/Field Classification] → List[VisualState]
    """

    # Ngưỡng phân loại Anchor vs Field
    ANCHOR_MIN_FACE_AREA: float = 0.03   # Khuôn mặt chiếm ≥ 3% diện tích frame
    ANCHOR_MAX_FACES: int = 2            # Tối đa 2 khuôn mặt
    ANCHOR_CENTER_TOLERANCE: float = 0.35 # Khuôn mặt nằm trong 35% giữa frame

    def __init__(self, model_path: str = "yolov8n-face.pt", visual_fps: float = 2.0):
        """
        Args:
            model_path: Đường dẫn tới model YOLOv8-face (tự tải nếu chưa có).
            visual_fps: Tần suất lấy mẫu frame cho phân tích visual.
        """
        self.model_path = model_path
        self.visual_fps = visual_fps
        self._model = None

    def _ensure_model(self) -> None:
        """Lazy-load YOLOv8-face."""
        if self._model is not None:
            return

        from ultralytics import YOLO

        print(f"⏳ [Visual] Đang tải YOLOv8-face: {self.model_path}...")
        t0 = time.perf_counter()

        self._model = YOLO(self.model_path)
        print(f"   ✅ YOLO sẵn sàng ({time.perf_counter() - t0:.1f}s)")

    def analyze(self, video_path: str) -> Tuple[List[VisualState], List[VisualStateChange]]:
        """
        Phân tích visual states cho toàn bộ video.

        Returns:
            states: Danh sách trạng thái Anchor/Field theo thời gian.
            changes: Danh sách các điểm thay đổi trạng thái.
        """
        self._ensure_model()

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Không thể mở video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        step = max(1, int(round(fps / self.visual_fps)))

        print(f"   📹 Video: {fps:.0f} FPS, {total_frames:,} frames, step={step}")

        # ── Thu thập frames cần xử lý ────────────────────────────────────
        frames_batch: List[np.ndarray] = []
        frame_indices: List[int] = []
        timestamps: List[float] = []

        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % step == 0:
                frames_batch.append(frame)
                frame_indices.append(idx)
                timestamps.append(idx / fps)
            idx += 1
        cap.release()

        print(f"   🔍 Đang phát hiện khuôn mặt trên {len(frames_batch)} frames...")
        t0 = time.perf_counter()

        # ── YOLOv8 Batch Inference ───────────────────────────────────────
        batch_size = 32
        all_results = []
        for i in range(0, len(frames_batch), batch_size):
            batch = frames_batch[i:i + batch_size]
            results = self._model(batch, verbose=False, conf=0.5)
            all_results.extend(results)

        t1 = time.perf_counter()
        print(f"   ✅ YOLO xong trong {t1 - t0:.1f}s")

        # ── Phân loại Anchor/Field ───────────────────────────────────────
        states: List[VisualState] = []

        for i, (result, frame) in enumerate(zip(all_results, frames_batch)):
            h, w = frame.shape[:2]
            frame_area = h * w

            boxes = result.boxes
            num_faces = len(boxes) if boxes is not None else 0

            max_face_area = 0.0
            face_center_x = 0.5

            if num_faces > 0:
                # Tính diện tích và vị trí cho mỗi khuôn mặt
                for box in boxes.xyxy:
                    x1, y1, x2, y2 = box[:4].cpu().numpy()
                    area = (x2 - x1) * (y2 - y1) / frame_area
                    cx = ((x1 + x2) / 2.0) / w

                    if area > max_face_area:
                        max_face_area = area
                        face_center_x = cx

            # Phân loại
            state = self._classify_state(num_faces, max_face_area, face_center_x)

            states.append(VisualState(
                timestamp=round(timestamps[i], 3),
                frame_idx=frame_indices[i],
                state=state,
                num_faces=num_faces,
                max_face_area=round(max_face_area, 4),
                face_center_x=round(face_center_x, 3),
            ))

        # ── Trích xuất State Changes ─────────────────────────────────────
        changes = self._extract_state_changes(states)

        anchor_count = sum(1 for s in states if s.state == "anchor")
        field_count = sum(1 for s in states if s.state == "field")
        print(f"   📊 Anchor: {anchor_count} frames, Field: {field_count} frames, "
              f"Changes: {len(changes)}")

        return states, changes

    def _classify_state(
        self, num_faces: int, max_face_area: float, face_center_x: float
    ) -> str:
        """
        Phân loại khung hình thành Anchor hoặc Field.

        Anchor (MC đọc bản tin):
        - Có 1-2 khuôn mặt
        - Khuôn mặt lớn nhất chiếm ≥ 3% diện tích frame
        - Khuôn mặt nằm ở vùng trung tâm (0.15 ≤ center_x ≤ 0.85)

        Field (phóng sự / đồ họa):
        - Mọi trường hợp khác
        """
        if num_faces < 1 or num_faces > self.ANCHOR_MAX_FACES:
            return "field"

        if max_face_area < self.ANCHOR_MIN_FACE_AREA:
            return "field"

        center_margin = (1.0 - self.ANCHOR_CENTER_TOLERANCE) / 2.0
        if face_center_x < center_margin or face_center_x > (1.0 - center_margin):
            return "field"

        return "anchor"

    @staticmethod
    def _extract_state_changes(states: List[VisualState]) -> List[VisualStateChange]:
        """Trích xuất các điểm chuyển đổi trạng thái Anchor ↔ Field."""
        changes: List[VisualStateChange] = []
        for i in range(len(states) - 1):
            if states[i].state != states[i + 1].state:
                changes.append(VisualStateChange(
                    timestamp=round(states[i + 1].timestamp, 3),
                    from_state=states[i].state,
                    to_state=states[i + 1].state,
                ))
        return changes


# ═══════════════════════════════════════════════════════════════════════════════
# LATE FUSION ENGINE — Timestamp Alignment
# ═══════════════════════════════════════════════════════════════════════════════

class LateFusionEngine:
    """
    Hợp nhất (Late Fusion) kết quả Audio và Visual theo thời gian.

    Hai nhánh Audio (Speaker Diarization) và Visual (Face Classification) có
    Sampling Rate hoàn toàn khác nhau:
    - Audio:  pyannote xử lý ở 16kHz, segment boundaries chính xác đến ms.
    - Visual: YOLOv8 chạy ở 2 FPS → state changes chính xác đến 0.5s.

    Late Fusion giải quyết bất đồng bộ này bằng cách:
    1. Chiếu (project) tất cả sự kiện lên trục thời gian chung (giây).
    2. So khớp (match) các sự kiện trong cửa sổ dung sai ±tolerance.
    3. Xác nhận (confirm) boundary khi thỏa mãn ít nhất 1 trong 2 rules.
    """

    def __init__(
        self,
        tolerance: float = 0.5,
        anchor_max_duration: float = 10.0,
        min_boundary_gap: float = 2.0,
    ):
        """
        Args:
            tolerance: Cửa sổ dung sai ±seconds cho Rule 1 (mặc định ±0.5s).
            anchor_max_duration: Ngưỡng thời gian Anchor liên tục cho Rule 2 (mặc định 10s).
            min_boundary_gap: Khoảng cách tối thiểu giữa 2 boundaries (giây).
        """
        self.tolerance = tolerance
        self.anchor_max_duration = anchor_max_duration
        self.min_boundary_gap = min_boundary_gap

    def fuse(
        self,
        speaker_turns: List[SpeakerTurn],
        visual_states: List[VisualState],
        visual_changes: List[VisualStateChange],
        fps: float,
    ) -> List[ConfirmedBoundary]:
        """
        Chạy Late Fusion để xác nhận Shot Boundaries.

        Args:
            speaker_turns: Danh sách điểm chuyển đổi người nói.
            visual_states: Danh sách trạng thái Anchor/Field.
            visual_changes: Danh sách điểm thay đổi trạng thái visual.
            fps: FPS gốc của video (để chuyển timestamp → frame index).

        Returns:
            Danh sách ConfirmedBoundary đã sắp xếp theo thời gian.
        """
        print(f"\n🔗 [Fusion] Late Fusion Engine")
        print(f"   Tolerance: ±{self.tolerance}s | Anchor max: {self.anchor_max_duration}s")
        print(f"   Speaker turns: {len(speaker_turns)} | Visual changes: {len(visual_changes)}")

        boundaries: List[ConfirmedBoundary] = []

        # ── Rule 1: Speaker Turn ∩ Visual State Change (±tolerance) ──────
        rule1_boundaries = self._apply_rule1(speaker_turns, visual_changes, fps)
        boundaries.extend(rule1_boundaries)
        print(f"   Rule 1 (Speaker+Visual sync): {len(rule1_boundaries)} boundaries")

        # ── Rule 2: Anchor Segment > anchor_max_duration bị ngắt quãng ──
        rule2_boundaries = self._apply_rule2(visual_states, fps)
        boundaries.extend(rule2_boundaries)
        print(f"   Rule 2 (Anchor interrupt): {len(rule2_boundaries)} boundaries")

        # ── Deduplicate & Sort ───────────────────────────────────────────
        boundaries = self._deduplicate(boundaries)
        print(f"   ✅ Tổng: {len(boundaries)} confirmed boundaries (sau deduplicate)")

        return boundaries

    def _apply_rule1(
        self,
        speaker_turns: List[SpeakerTurn],
        visual_changes: List[VisualStateChange],
        fps: float,
    ) -> List[ConfirmedBoundary]:
        """
        Rule 1: Speaker Turn XẢY RA CÙNG LÚC (±tolerance) với Visual State Change.

        Thuật toán:
        Với mỗi Speaker Turn tại thời điểm t_audio:
            1. Tìm Visual State Change gần nhất tại thời điểm t_visual.
            2. Nếu |t_audio - t_visual| ≤ tolerance → CONFIRM boundary.
            3. Timestamp cuối cùng = trung bình (t_audio + t_visual) / 2
               (lấy điểm giữa để giảm sai số của cả 2 nhánh).
            4. Confidence tỷ lệ nghịch với khoảng cách thời gian.

        Tại sao dùng trung bình?
        - Audio (pyannote) chính xác đến ms nhưng có thể bị trễ do silence detection.
        - Visual (YOLOv8) chính xác đến 0.5s (sampling 2 FPS).
        - Trung bình giúp triệt tiêu sai số hệ thống (systematic bias) của 2 nhánh.
        """
        boundaries: List[ConfirmedBoundary] = []

        if not speaker_turns or not visual_changes:
            return boundaries

        # Sắp xếp visual changes theo thời gian
        vc_times = np.array([vc.timestamp for vc in visual_changes])

        for turn in speaker_turns:
            t_audio = turn.timestamp

            # Tìm visual change gần nhất (binary search)
            insert_idx = np.searchsorted(vc_times, t_audio)
            candidates = []
            for offset in [-1, 0]:
                idx = insert_idx + offset
                if 0 <= idx < len(vc_times):
                    candidates.append(idx)

            if not candidates:
                continue

            # Chọn visual change gần nhất
            best_idx = min(candidates, key=lambda j: abs(vc_times[j] - t_audio))
            t_visual = vc_times[best_idx]
            time_diff = abs(t_audio - t_visual)

            # Kiểm tra tolerance
            if time_diff <= self.tolerance:
                # Timestamp = trung bình của audio và visual
                fused_timestamp = (t_audio + t_visual) / 2.0
                frame_idx = int(round(fused_timestamp * fps))

                # Confidence: 1.0 khi diff=0, giảm tuyến tính xuống 0.5 khi diff=tolerance
                confidence = 1.0 - 0.5 * (time_diff / self.tolerance)

                vc = visual_changes[best_idx]
                boundaries.append(ConfirmedBoundary(
                    timestamp=round(fused_timestamp, 3),
                    frame_idx=frame_idx,
                    rule="speaker_visual_sync",
                    confidence=round(confidence, 3),
                    details={
                        "audio_timestamp": round(t_audio, 3),
                        "visual_timestamp": round(t_visual, 3),
                        "time_diff": round(time_diff, 3),
                        "from_speaker": turn.from_speaker,
                        "to_speaker": turn.to_speaker,
                        "visual_from": vc.from_state,
                        "visual_to": vc.to_state,
                    },
                ))

        return boundaries

    def _apply_rule2(
        self,
        visual_states: List[VisualState],
        fps: float,
    ) -> List[ConfirmedBoundary]:
        """
        Rule 2: Trạng thái 'Anchor' kéo dài > anchor_max_duration bị ngắt quãng.

        Đặc trưng thời sự:
        MC đọc bản tin liên tục (Anchor) thường kéo dài 5-15 giây.
        Nếu Anchor kéo dài > 10 giây rồi đột ngột chuyển sang Field,
        đó là dấu hiệu mạnh của chuyển cảnh (MC → Phóng sự).

        Thuật toán:
        1. Quét qua timeline visual, tìm các chuỗi Anchor liên tục.
        2. Nếu chuỗi Anchor kéo dài > anchor_max_duration:
           - Khi chuỗi bị ngắt (chuyển sang Field): CONFIRM boundary.
           - Confidence cao hơn nếu chuỗi Anchor dài hơn.
        """
        boundaries: List[ConfirmedBoundary] = []

        if len(visual_states) < 2:
            return boundaries

        anchor_start: Optional[float] = None
        anchor_start_idx: int = 0

        for i in range(len(visual_states)):
            state = visual_states[i]

            if state.state == "anchor":
                if anchor_start is None:
                    anchor_start = state.timestamp
                    anchor_start_idx = i
            else:
                # Anchor vừa bị ngắt
                if anchor_start is not None:
                    anchor_duration = state.timestamp - anchor_start

                    if anchor_duration > self.anchor_max_duration:
                        # Ngắt sau khi Anchor kéo dài quá lâu → boundary
                        confidence = min(1.0, 0.6 + 0.04 * (anchor_duration - self.anchor_max_duration))
                        frame_idx = state.frame_idx

                        boundaries.append(ConfirmedBoundary(
                            timestamp=round(state.timestamp, 3),
                            frame_idx=frame_idx,
                            rule="anchor_interrupt",
                            confidence=round(confidence, 3),
                            details={
                                "anchor_start": round(anchor_start, 3),
                                "anchor_duration": round(anchor_duration, 3),
                                "transition_to": state.state,
                            },
                        ))

                anchor_start = None

        return boundaries

    def _deduplicate(
        self,
        boundaries: List[ConfirmedBoundary],
    ) -> List[ConfirmedBoundary]:
        """
        Loại bỏ boundaries trùng lặp (quá gần nhau).

        Nếu 2 boundaries cách nhau < min_boundary_gap:
        - Giữ lại boundary có confidence cao hơn.
        - Nếu bằng nhau, ưu tiên Rule 1 (có cả audio + visual).
        """
        if len(boundaries) <= 1:
            return boundaries

        # Sắp xếp theo timestamp
        boundaries.sort(key=lambda b: b.timestamp)

        filtered: List[ConfirmedBoundary] = [boundaries[0]]

        for b in boundaries[1:]:
            last = filtered[-1]
            gap = b.timestamp - last.timestamp

            if gap < self.min_boundary_gap:
                # Quá gần → giữ cái tốt hơn
                if b.confidence > last.confidence:
                    filtered[-1] = b
                elif (b.confidence == last.confidence
                      and b.rule == "speaker_visual_sync"
                      and last.rule != "speaker_visual_sync"):
                    filtered[-1] = b
                # Else: giữ nguyên last
            else:
                filtered.append(b)

        return filtered


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

class MultiModalSyncPipeline:
    """
    Pipeline chính: Kết nối Audio Branch + Visual Branch + Late Fusion.
    """

    def __init__(
        self,
        hf_token: str = "",
        yolo_model: str = "yolov8n-face.pt",
        visual_fps: float = 2.0,
        tolerance: float = 0.5,
        anchor_max_duration: float = 10.0,
    ):
        self.audio_branch = AudioBranch(hf_token=hf_token)
        self.visual_branch = VisualBranch(model_path=yolo_model, visual_fps=visual_fps)
        self.fusion = LateFusionEngine(
            tolerance=tolerance,
            anchor_max_duration=anchor_max_duration,
        )

    def run(self, video_path: str, output_path: str = "multimodal_shots.jsonl") -> Dict[str, Any]:
        """
        Chạy toàn bộ pipeline Multi-Modal Synchronization.

        Args:
            video_path: Đường dẫn video.
            output_path: File JSONL đầu ra.

        Returns:
            Dict chứa kết quả tổng hợp.
        """
        print("=" * 70)
        print(f"  🎬 Multi-Modal Synchronization Pipeline")
        print(f"  Video: {Path(video_path).name}")
        print("=" * 70)

        t_start = time.perf_counter()

        # Lấy FPS gốc
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        # ── Audio Branch ─────────────────────────────────────────────────
        print(f"\n{'─' * 70}")
        print("🔊 AUDIO BRANCH — Speaker Diarization")
        print(f"{'─' * 70}")

        segments, speaker_turns = self.audio_branch.diarize(video_path)
        self.audio_branch.save_rttm(segments, output_path.replace(".jsonl", ".rttm"))

        # ── Visual Branch ────────────────────────────────────────────────
        print(f"\n{'─' * 70}")
        print("👁️ VISUAL BRANCH — Face Detection & Classification")
        print(f"{'─' * 70}")

        visual_states, visual_changes = self.visual_branch.analyze(video_path)

        # ── Late Fusion ──────────────────────────────────────────────────
        print(f"\n{'─' * 70}")
        print("🔗 LATE FUSION — Timestamp Alignment")
        print(f"{'─' * 70}")

        boundaries = self.fusion.fuse(speaker_turns, visual_states, visual_changes, fps)

        # ── Chuyển thành Shots ───────────────────────────────────────────
        shots = self._boundaries_to_shots(boundaries, total_frames, fps)

        # ── Xuất JSONL ───────────────────────────────────────────────────
        with open(output_path, "w", encoding="utf-8") as f:
            for shot in shots:
                f.write(json.dumps(shot, ensure_ascii=False) + "\n")

        total_time = time.perf_counter() - t_start

        # ── Summary ──────────────────────────────────────────────────────
        self._print_summary(boundaries, shots, total_time)

        return {
            "video_path": video_path,
            "total_frames": total_frames,
            "fps": fps,
            "num_speakers": len(set(s.speaker_id for s in segments)),
            "num_speaker_turns": len(speaker_turns),
            "num_visual_changes": len(visual_changes),
            "num_boundaries": len(boundaries),
            "num_shots": len(shots),
            "processing_time": round(total_time, 2),
            "output_file": output_path,
        }

    @staticmethod
    def _boundaries_to_shots(
        boundaries: List[ConfirmedBoundary],
        total_frames: int,
        fps: float,
    ) -> List[Dict[str, Any]]:
        """Chuyển đổi boundaries thành danh sách shots."""
        shots: List[Dict[str, Any]] = []
        prev_frame = 0

        for i, bd in enumerate(boundaries):
            shots.append({
                "shot_id": int(i),
                "start_frame": int(prev_frame),
                "end_frame": int(bd.frame_idx - 1),
                "start_timestamp": round(prev_frame / fps, 3),
                "end_timestamp": round((bd.frame_idx - 1) / fps, 3),
                "boundary_rule": bd.rule,
                "boundary_confidence": bd.confidence,
                "details": bd.details,
            })
            prev_frame = bd.frame_idx

        # Shot cuối cùng
        shots.append({
            "shot_id": int(len(shots)),
            "start_frame": int(prev_frame),
            "end_frame": int(total_frames - 1),
            "start_timestamp": round(prev_frame / fps, 3),
            "end_timestamp": round((total_frames - 1) / fps, 3),
            "boundary_rule": "end",
            "boundary_confidence": 1.0,
            "details": {},
        })

        return shots

    @staticmethod
    def _print_summary(boundaries: List[ConfirmedBoundary], shots: list, elapsed: float) -> None:
        """In bảng tóm tắt kết quả."""
        rule1 = sum(1 for b in boundaries if b.rule == "speaker_visual_sync")
        rule2 = sum(1 for b in boundaries if b.rule == "anchor_interrupt")

        print(f"\n{'═' * 70}")
        print(f"  🎯 KẾT QUẢ MULTI-MODAL SYNCHRONIZATION")
        print(f"{'═' * 70}")
        print(f"  Tổng boundaries:      {len(boundaries)}")
        print(f"    ├── Rule 1 (Sync):   {rule1}")
        print(f"    └── Rule 2 (Anchor): {rule2}")
        print(f"  Tổng shots:           {len(shots)}")
        print(f"  Thời gian xử lý:     {elapsed:.1f}s")
        print(f"{'═' * 70}")

        if boundaries:
            print(f"\n{'Boundary':>10} │ {'Time(s)':>8} │ {'Frame':>8} │ "
                  f"{'Rule':>22} │ {'Conf':>6} │ Details")
            print(f"{'─' * 90}")
            for i, b in enumerate(boundaries[:15]):
                detail_str = ""
                if b.rule == "speaker_visual_sync":
                    d = b.details
                    detail_str = (f"{d.get('from_speaker','?')}→{d.get('to_speaker','?')} | "
                                  f"{d.get('visual_from','?')}→{d.get('visual_to','?')} | "
                                  f"Δ={d.get('time_diff',0):.3f}s")
                elif b.rule == "anchor_interrupt":
                    d = b.details
                    detail_str = f"anchor {d.get('anchor_duration',0):.1f}s→{d.get('transition_to','?')}"

                print(f"{i:>10} │ {b.timestamp:>8.3f} │ {b.frame_idx:>8,} │ "
                      f"{b.rule:>22} │ {b.confidence:>6.3f} │ {detail_str}")

            if len(boundaries) > 15:
                print(f"  ... (còn {len(boundaries) - 15} boundaries nữa)")
            print(f"{'─' * 90}")


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Multi-Modal Synchronization Pipeline for Broadcast Video",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ví dụ sử dụng:
  python multimodal_sync.py --video "news.mp4" --hf-token "hf_xxxxx"
  python multimodal_sync.py --video "news.mp4" --tolerance 0.8 --anchor-max 15
        """,
    )
    parser.add_argument("--video", type=str, required=True,
                        help="Đường dẫn tới video broadcast")
    parser.add_argument("--output", type=str, default="multimodal_shots.jsonl",
                        help="File JSONL đầu ra")
    parser.add_argument("--hf-token", type=str, default="",
                        help="HuggingFace token cho pyannote (hoặc set env HF_TOKEN)")
    parser.add_argument("--yolo-model", type=str, default="yolov8n-face.pt",
                        help="Đường dẫn model YOLOv8-face")
    parser.add_argument("--visual-fps", type=float, default=2.0,
                        help="Tần suất lấy mẫu visual (mặc định: 2.0)")
    parser.add_argument("--tolerance", type=float, default=0.5,
                        help="Cửa sổ dung sai ±seconds cho Rule 1 (mặc định: 0.5)")
    parser.add_argument("--anchor-max", type=float, default=10.0,
                        help="Ngưỡng Anchor liên tục cho Rule 2 (mặc định: 10.0s)")

    args = parser.parse_args()

    if not os.path.exists(args.video):
        print(f"❌ Không tìm thấy video: {args.video}")
        return

    pipeline = MultiModalSyncPipeline(
        hf_token=args.hf_token,
        yolo_model=args.yolo_model,
        visual_fps=args.visual_fps,
        tolerance=args.tolerance,
        anchor_max_duration=args.anchor_max,
    )

    results = pipeline.run(args.video, args.output)
    print(f"\n📝 Kết quả đã lưu vào: {args.output}")
    print(f"📝 RTTM đã lưu vào: {args.output.replace('.jsonl', '.rttm')}")


if __name__ == "__main__":
    main()
