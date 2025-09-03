# Interactive Video Review Tool

Desktop application for manual video review and targeted frame capture. This tool provides real-time YOLO inference with an intuitive interface for collecting hard examples, edge cases, and quality control samples.

## Quick Start

```bash
# From the main frame-mining directory
poetry run python interactive/mine_frames_from_video.py
```

1. Select videos folder (shows size/duration table)
2. Choose YOLO model weights file
3. Set output directory for captured frames
4. Optionally enable detection caching
5. Click Start to launch player

## Key Features

### **Interactive Player**
- **Live Inference**: Real-time YOLO detection with overlay
- **Flexible Playback**: Speed control, frame stepping, seeking
- **Quick Capture**: One-hotkey frame saving (S key)
- **Manual Tagging**: Mark frames as false_positive/false_negative/uncertain
- **Dynamic Thresholds**: Adjust confidence/IoU while playing

### **Smart Caching System**
- **Pre-computation**: Generate detection cache for smooth playback
- **Portable Cache**: Move cache files between machines
- **Efficient Storage**: Compressed format with all detection data
- **Auto-detection**: Uses existing cache when available

### **Metadata Capture**
Each saved frame includes JSON sidecar with:
- Video path and frame index
- Timestamp and model information  
- All predictions with confidence scores
- Optional manual tags

### **Deduplication**
- **Perceptual Hashing**: Avoids saving near-duplicate frames
- **Configurable Threshold**: Hamming distance-based filtering
- **Smart Skipping**: Optionally jump ahead after each save

## Player Controls

| Key/Control | Action |
|-------------|--------|
| **Space** | Pause/resume playback |
| **S** | Save current frame + JSON metadata |
| **F** | Tag frame as false_positive |
| **N** | Tag frame as false_negative |
| **U** | Tag frame as uncertain |
| **C** | Clear current tag |
| **H** | Toggle help overlay |
| **O** | Toggle detection overlay |
| **Left/Right** | Step ±1 frame (when paused) |
| **Home/End** | Seek to start/end |
| **+/-** | Adjust playback speed |
| **1/2/3** | Set speed to 0.5x/1x/2x |

### Trackbars
- **conf**: Detection confidence threshold
- **iou**: NMS IoU threshold  
- **position**: Video seek bar

## Cache Format

Detection cache consists of two files per video:

### `<video>__<model>.preds.npz`
Compressed numpy arrays:
- `boxes`: float16 normalized xyxy coordinates
- `confs`: float16 confidence scores
- `clss`: uint16 class indices
- `indptr`: int64 frame indices (cumulative)

### `<video>__<model>.meta.json`
Metadata:
- Video FPS, dimensions, frame count
- Model path and class names
- Cache creation timestamp

## Batch Processing

Generate caches on GPU server, review on laptop:

```bash
# Headless cache generation
poetry run python interactive/mine_frames_from_video.py \
    --batch-precompute \
    --input-dir /path/to/videos \
    --pattern "**/*.mp4" \
    --weights /path/to/model.pt
```

Options:
- `--force`: Overwrite existing caches
- `--pattern`: File glob pattern (default: `**/*.{mp4,avi,mov,mkv}`)

## Use Cases

### **Hard Example Mining**
Perfect for finding model failures:
- False positives: Model detects non-waste
- False negatives: Model misses waste objects  
- Uncertain cases: Low-confidence predictions
- Edge cases: Unusual lighting, angles, occlusion

### **Quality Control**
Manual validation of automated results:
- Review automated pipeline output
- Verify annotation quality
- Check model performance on new data
- Collect challenging examples for retraining

### **Dataset Analysis**
Understanding your data:
- Visualize detection patterns
- Identify common failure modes
- Assess model confidence distribution
- Spot annotation inconsistencies

## Output Structure

```
output_dir/
├── saved_frames/
│   ├── video1_f000123.jpg        # Raw frame (full resolution)
│   ├── video1_f000123.json       # Metadata + predictions
│   ├── video2_f004567.jpg
│   └── video2_f004567.json
└── cache/                        # Detection caches (if enabled)
    ├── video1__yolov8n.preds.npz
    ├── video1__yolov8n.meta.json
    └── ...
```

### Example Metadata JSON
```json
{
  "video_path": "/path/to/video1.mp4",
  "frame_index": 123,
  "timestamp": 4.1,
  "model_path": "/path/to/yolov8n.pt",
  "predictions": [
    {"class": 0, "confidence": 0.85, "box": [0.1, 0.2, 0.3, 0.4]},
    {"class": 0, "confidence": 0.72, "box": [0.5, 0.6, 0.7, 0.8]}
  ],
  "tag": "false_negative",
  "capture_time": "2025-09-03T14:30:45"
}
```

## Configuration

The tool auto-detects optimal settings but can be customized:

### GUI Settings
- **Skip frames after save**: Jump ahead to avoid duplicates
- **Precompute detections**: Enable/disable caching
- **Quality**: JPEG quality for saved frames (default: 95)

### Detection Settings
- **Confidence threshold**: Filter low-confidence detections
- **IoU threshold**: NMS overlap threshold
- **Model device**: Auto-selects CUDA if available

## Performance Tips

### For Smooth Playback
1. **Enable precompute**: Cache detections for fluid review
2. **Use smaller models**: yolov8n vs yolov8m for speed
3. **Reduce video resolution**: Resize large videos if needed
4. **Close other applications**: Free up GPU/CPU resources

### For Large Collections
1. **Batch precompute**: Generate caches on GPU server
2. **Copy cache files**: Transfer to review machine
3. **Use file patterns**: Process specific video types
4. **Monitor storage**: Cache files add up over time

## Integration

This tool integrates well with other workflows:

- **After automated mining**: Review and validate selected frames
- **Before annotation**: Quick quality check of new data
- **During training**: Collect additional hard examples
- **For model evaluation**: Manual assessment of performance

See the main [frame-mining README](../README.md) for workflow recommendations.
