# Review Hard Examples

Interactive desktop tool to review domain videos from truck cameras, run a chosen Ultralytics YOLO model live (or from a precomputed on‑disk cache), and quickly capture hard examples (misses, false positives, uncertain predictions) as full‑resolution images for later annotation and finetuning.

## Features (MVP)
- Choose a snippets folder and pick a video (sorted by size, shows size and duration).
- Choose a YOLO `.pt` weights file. Device auto-selects CUDA if available, else CPU.
- Plays the video with live inference and overlaid boxes.
- Pause and save the current raw, full‑resolution frame with one hotkey.
- Writes a JSON sidecar with metadata: video path, frame index, timestamp, model path, predictions, and optional tag (no model hash, no thresholds).
- Adjust confidence and NMS (IoU) thresholds while playing.
- Frame stepping, seek, and playback speed controls.
- Perceptual‑hash dedup to avoid saving near‑duplicate frames.
- Optional: Skip N frames after each save to move ahead to new content.
- Optional: Precompute detections before playback for smooth review on slow machines, with a compact on‑disk cache you can move across machines. Existing caches are used automatically when present.

## Quick Start

```
poetry install  # from this tool directory, optional if using repo root env
python mine_frames_from_video.py
```

1. Select the snippets folder. The table lists videos sorted by size with size and duration.
2. Pick a YOLO weights `.pt` file (device auto-selects).
3. Choose an output folder for saved frames.
4. Optionally enable "Precompute detections (cache to disk)".
5. Optionally set "Skip frames after save" to jump ahead after saving.
6. Click Start to launch the player.

## Player Controls
- Space: pause/resume
- S: save current raw frame (+ JSON sidecar); deduplicated by perceptual hash; optionally jumps ahead by configured skip
- F/N/U/C: tag frame as false_positive / false_negative / uncertain / clear tag
- H: toggle on‑screen help
- O: toggle overlay on/off
- Left/Right: step -1/+1 frame when paused
- Home/End: seek to start/end
- +/-: decrease/increase playback speed
- 1/2/3: set speed to 0.5x/1x/2x
- Trackbars: `conf`, `iou`, `position` (seek)

## Cache Format
- Saved next to the video as two small files:
  - `<video_stem>__<weights_stem>.preds.npz` (compressed):
    - `boxes` float16, shape `(M,4)` normalized xyxy in [0,1]
    - `confs` float16, shape `(M,)`
    - `clss` uint16, shape `(M,)`
    - `indptr` int64, shape `(total_frames+1,)` cumulative indices per frame
  - `<video_stem>__<weights_stem>.meta.json` (tiny JSON): fps, total_frames, frame width/height, model weights basename, class names list.
- The cache is compact but keeps all information necessary to render overlays and save predictions with the image.
- When precompute is enabled and a cache exists, it loads from disk; otherwise it computes once and writes the cache for reuse.

## Batch Precompute (Headless)
You can generate compact detection caches on a GPU server (no GUI) and then copy the cache files back to your laptop for smooth reviewing.

```
python mine_frames_from_video.py \
  --batch-precompute \
  --input-dir /path/to/snippets \
  --pattern "**/*.mp4" \
  --weights /path/to/model.pt [--force]
```

This writes `.preds.npz` and `.meta.json` files next to each video. The player automatically uses an existing cache when present. Use `--force` to recompute existing caches.

## Notes
- Saved images are raw frames at source resolution (no overlays) with `.jpg` extension and quality=95.
- Dedup uses perceptual hashing (phash) with a small Hamming distance threshold to filter near-duplicates.
- Timestamps are computed from frame index and FPS when precise capture timestamps are unavailable.
- Requires: Python 3.11+, opencv-python, ultralytics, torch, pillow, imagehash, numpy, tqdm.
- When precompute is enabled, the conf trackbar filters cached predictions. IoU changes do not retroactively re-run NMS.
