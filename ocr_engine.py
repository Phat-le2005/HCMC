"""
OCR Engine – 3-Step Pipeline for Vietnamese Text Recognition
============================================================
Step 1: Text Detection   → PaddleOCR (rec=False, chỉ detect vùng chữ)
Step 2: Crop & Pad        → Cắt bbox + padding 5px trên/dưới cho dấu tiếng Việt
Step 3: Text Recognition  → VietOCR (vgg_transformer, batch inference)
"""

import numpy as np
from PIL import Image
from typing import List, Tuple, Optional

import torch


class HybridOCREngine:
    """Paddle Detection + VietOCR Recognition engine."""

    def __init__(self, device: str = "cpu", vietocr_model: str = "vgg_transformer",
                 pad_top: int = 5, pad_bottom: int = 5, pad_left: int = 3, pad_right: int = 3,
                 batch_size: int = 16):
        self.device = device
        self.pad_top = pad_top
        self.pad_bottom = pad_bottom
        self.pad_left = pad_left
        self.pad_right = pad_right
        self.batch_size = batch_size

        # Step 1: PaddleOCR for detection only
        from paddleocr import PaddleOCR
        self.detector = PaddleOCR(
            use_angle_cls=True,
            lang="vi",
            use_gpu=False,  # CPU to avoid CUDNN mismatch
            rec=False,       # Detection only — no recognition
            show_log=False
        )
        print("[OCR Engine] PaddleOCR text detector loaded (CPU, rec=False).")

        # Step 3: VietOCR for recognition
        from vietocr.tool.predictor import Predictor
        from vietocr.tool.config import Cfg

        vietocr_cfg = Cfg.load_config_from_name(vietocr_model)
        vietocr_cfg['cnn']['pretrained'] = False
        vietocr_cfg['device'] = device
        self.recognizer = Predictor(vietocr_cfg)
        print(f"[OCR Engine] VietOCR recognizer loaded ({vietocr_model}, {device}).")

    def _detect_boxes(self, image) -> List[List[List[int]]]:
        """
        Step 1: Run PaddleOCR detection to get text bounding boxes.
        Returns list of polygons (4 corner points each).
        """
        if isinstance(image, str):
            result = self.detector.ocr(image, rec=False)
        elif isinstance(image, np.ndarray):
            result = self.detector.ocr(image, rec=False)
        elif isinstance(image, Image.Image):
            img_np = np.array(image)
            result = self.detector.ocr(img_np, rec=False)
        else:
            return []

        if not result or not result[0]:
            return []

        return result[0]  # List of polygon boxes

    def _polygon_to_rect(self, polygon) -> Tuple[int, int, int, int]:
        """Convert 4-point polygon to axis-aligned rectangle (x1, y1, x2, y2)."""
        pts = np.array(polygon)
        x1 = int(pts[:, 0].min())
        y1 = int(pts[:, 1].min())
        x2 = int(pts[:, 0].max())
        y2 = int(pts[:, 1].max())
        return x1, y1, x2, y2

    def _crop_and_pad(self, image: Image.Image, boxes: List) -> List[Image.Image]:
        """
        Step 2: Crop detected regions with padding for Vietnamese diacritics.
        Padding top/bottom is critical for characters like Ể, Ố, g, y.
        """
        w, h = image.size
        crops = []

        for polygon in boxes:
            x1, y1, x2, y2 = self._polygon_to_rect(polygon)

            # Apply padding (critical for Vietnamese diacritics)
            x1 = max(0, x1 - self.pad_left)
            y1 = max(0, y1 - self.pad_top)
            x2 = min(w, x2 + self.pad_right)
            y2 = min(h, y2 + self.pad_bottom)

            if x2 <= x1 or y2 <= y1:
                continue

            crop = image.crop((x1, y1, x2, y2))
            crops.append(crop)

        return crops

    def _recognize_batch(self, crops: List[Image.Image]) -> List[str]:
        """
        Step 3: Run VietOCR recognition on cropped text regions.
        Processes in batches for efficiency.
        """
        if not crops:
            return []

        results = []
        for i in range(0, len(crops), self.batch_size):
            batch = crops[i:i + self.batch_size]
            for crop_img in batch:
                try:
                    text = self.recognizer.predict(crop_img)
                    text = text.strip()
                    if text:
                        results.append(text)
                except Exception as e:
                    # Skip problematic crops silently
                    continue
        return results

    def recognize(self, image, deduplicate: bool = True) -> str:
        """
        Full 3-step OCR pipeline.
        
        Args:
            image: str (file path), np.ndarray, or PIL.Image
            deduplicate: if True, remove duplicate text segments
            
        Returns:
            Recognized text as a single string.
        """
        # Convert to PIL if needed
        if isinstance(image, str):
            pil_image = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            pil_image = Image.fromarray(image).convert("RGB")
        elif isinstance(image, Image.Image):
            pil_image = image.convert("RGB")
        else:
            return ""

        # Step 1: Detect text regions
        boxes = self._detect_boxes(pil_image)
        if not boxes:
            return ""

        # Step 2: Crop with padding
        crops = self._crop_and_pad(pil_image, boxes)
        if not crops:
            return ""

        # Step 3: Recognize with VietOCR
        texts = self._recognize_batch(crops)

        if deduplicate:
            # Preserve order while removing duplicates
            seen = set()
            unique = []
            for t in texts:
                if t.lower() not in seen:
                    seen.add(t.lower())
                    unique.append(t)
            texts = unique

        return " ".join(texts)

    def recognize_multi(self, image_paths: List[str], deduplicate: bool = True) -> str:
        """
        Run OCR on multiple images and merge results with deduplication.
        Used for multi-keyframe OCR per shot.
        """
        all_texts = []
        seen = set()

        for path in image_paths:
            text = self.recognize(path, deduplicate=False)
            if not text:
                continue
            segments = text.split()
            for seg in segments:
                if seg.lower() not in seen:
                    seen.add(seg.lower())
                    all_texts.append(seg)

        return " ".join(all_texts) if all_texts else ""

    def recognize_crop(self, frame_np: np.ndarray, bbox: List[float]) -> str:
        """
        Run OCR on a specific bounding box region within a frame.
        bbox is [x1, y1, x2, y2] normalized to 0-1.
        Used for local OCR in module2a.
        """
        h, w = frame_np.shape[:2]
        x1 = int(bbox[0] * w)
        y1 = int(bbox[1] * h)
        x2 = int(bbox[2] * w)
        y2 = int(bbox[3] * h)

        if x2 <= x1 or y2 <= y1:
            return ""

        crop = frame_np[y1:y2, x1:x2]
        return self.recognize(crop, deduplicate=True)
