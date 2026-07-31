# ATSME — Autonomous Tracklet-based Semantic & Multimodal Engine

> AI Challenge 2025 — Master Feature Extraction Script  
> Optimised for **Kaggle T4×2 / P100** (≤ 16 GB VRAM per device)

---

## Architecture

```
Input MP4 + map-keyframes-aic25-b1.csv
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│ Step 1 │ ShotDetector       PySceneDetect ContentDetector        │
│ Step 2 │ TrackletBuilder    YOLOv9 + ByteTrack + Heuristics      │
│ Step 3 │ SemanticExtractor  SigLIP-2 + Qwen2.5-VL + PaddleOCR   │
│ Step 4 │ AudioExtractor     OpenAI Whisper + ffmpeg              │
│ Step 5 │ FrameMapper        CSV → global_frame_id remapping      │
│        │ DataExporter       Parquet + JSONL                      │
└─────────────────────────────────────────────────────────────────┘
         │
         ▼
features_visual.parquet   →  Milvus (vector search)
features_lexical.jsonl    →  ElasticSearch (keyword + BM25)
```

---

## Output Schemas

### `features_visual.parquet`

| Column | Type | Description |
|---|---|---|
| `video_id` | `string` | Video filename stem |
| `shot_id` | `string` | Shot index within video |
| `tracklet_id` | `string` | Unique tracklet ID |
| `global_frame_id` | `int64` | BTC-standard representative frame |
| `start_frame` | `int64` | Shot start (global) |
| `end_frame` | `int64` | Shot end (global) |
| `siglip_vector` | `list<float16>` | 768-dim mean-pooled embedding |
| `bbox_trajectory` | `string (JSON)` | `[{"frame": int, "bbox": [x1,y1,x2,y2]}]` |

### `features_lexical.jsonl`

```json
{
  "video_id":          "L01_V001",
  "tracklet_id":       "42",
  "global_frame_keys": [1024, 1152, 1280],
  "ocr_text":          ["Speed 120 km/h", "", "STOP"],
  "asr_text":          "The vehicle approaches the intersection at high speed.",
  "event_evolution": [
    {"frame": 1024, "action": "A car accelerates on the highway."},
    {"frame": 1152, "action": "The car brakes near a traffic sign."},
    {"frame": 1280, "action": "The car stops at a red light."}
  ]
}
```

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Single video smoke test
python atsme_pipeline.py \
    --video  path/to/video.mp4 \
    --mapping path/to/map-keyframes-aic25-b1.csv \
    --output ./outputs \
    --yolo   yolov9c.pt \
    --whisper base

# Enable vLLM backend (A100 / H100 only)
python atsme_pipeline.py ... --use-vllm
```

---

## VRAM Management Strategy

| Model | VRAM (fp16) | Loaded When |
|---|---|---|
| YOLOv9c | ~1.2 GB | Step 2 (stays loaded) |
| SigLIP-2 SO400M | ~3.5 GB | Step 3 (lazy load) |
| Qwen2.5-VL-2B | ~5.0 GB | Step 3 (lazy load) |
| PaddleOCR | ~0.5 GB | Step 3 (lazy load) |
| Whisper (base) | ~0.1 GB | Step 4 (lazy load) |

- All models use **lazy loading** — instantiated only on first use.
- `torch.cuda.empty_cache()` + `gc.collect()` called after every processing batch.
- Qwen is unloaded before Whisper when memory is tight.
- `batch_size=8` for SigLIP can be reduced to `4` on P100 if OOM errors occur.

---

## Files

| File | Purpose |
|---|---|
| `atsme_pipeline.py` | **Main script** — all 5 classes + `main()` |
| `bytetrack.yaml` | ByteTrack tracker configuration |
| `requirements.txt` | pip dependencies |
| `atsme_kaggle_notebook.ipynb` | Ready-to-run Kaggle notebook |

---

## Heuristic Keyframe Selection Details

```
For each Tracklet in a Shot:
  candidates ← all frames of this tracklet

  Filter 1 → Laplacian Variance < 100  ⟹  discard (motion blur)
  Filter 2 → BBox area / frame area < 5%  ⟹  discard (too small)

  If candidates empty → fallback to all frames (no filter)

  Sort candidates by frame index
  Select k=3 frames via np.linspace  (head · mid · tail)
```

---

## CSV Mapping Format

The `map-keyframes-aic25-b1.csv` must contain at least these columns  
(column names are case-insensitive):

```
video_id, local_frame_id, global_frame_id
```

Frame remapping uses **nearest-neighbour** search via `np.argmin(|local - target|)`
so no exact match is required.
