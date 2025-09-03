# Automated Keyframe Mining

Fully automated, intelligent pipeline for mining high-quality training frames from waste detection videos. This tool uses advanced computer vision and machine learning techniques to intelligently select the most valuable frames while avoiding redundancy.

## Quick Start

```bash
# From the main frame-mining directory
poetry run python auto/mine_frames_auto.py \
    --videos_dir /path/to/videos \
    --out_dir /path/to/output \
    --model yolov8n.pt \
    --stride_k 3
```

## Key Features

### **FrameStore Architecture**
- **Efficient Caching**: Pre-processes videos once, reuses detection data
- **Configurable Sampling**: Independent store and runtime strides
- **Optional Thumbnails**: JPEG-compressed thumbnails for visual analysis
- **Incremental Processing**: Only processes new/missing videos

### **Intelligent Triage System**
- **AUTO**: High-confidence tracks (strict quality gates) → auto-included
- **LABEL**: Challenging candidates → scored and selected intelligently  
- **DROP**: Failed detections → optionally exported for analysis

### **Advanced Scoring Algorithm**
**S = αU + βD + γR + δN + ε s_peak_cap**

- **U** (Uncertainty): Prioritizes low-confidence detections
- **D** (Diversity): Ensures visual variety using HSV histograms
- **R** (Hardness): Favors small objects and jittery tracking
- **N** (Novelty): Compares against historical memory bank
- **s_peak_cap**: Controlled peak confidence contribution

### **Adaptive Quota System**
- **Smart Balancing**: Automatically adjusts quotas based on content
- **Floor Guarantees**: Ensures minimum representation per scenario
- **Scenario Types**: Tiny objects, night scenes, standard conditions

### **Persistent Memory Bank**
Maintains historical embeddings across runs to ensure continuous novelty and prevent redundant data collection.

### **CVAT Integration**
- **Automatic Export**: Creates CVAT-compatible dataset with `--create_cvat`
- **Class Loading**: Automatically loads class names from workspace `params.yaml`
- **Ready-to-Import**: Generates data.yaml, train.txt, and zipped dataset
- **Fallback Classes**: Defaults to ["waste", "cigarette"] if params.yaml not found

## Usage Examples

### Basic Usage
```bash
poetry run python auto/mine_frames_auto.py \
    --videos_dir /path/to/videos \
    --out_dir /path/to/output
```

### Performance Optimization
```bash
# Faster processing with less storage
poetry run python auto/mine_frames_auto.py \
    --videos_dir /path/to/videos \
    --out_dir /path/to/output \
    --stride_k 5 \
    --store_stride 3 \
    --no_thumbs
```

### Quality Control
```bash
# Lower confidence threshold + annotated export for review
poetry run python auto/mine_frames_auto.py \
    --videos_dir /path/to/videos \
    --out_dir /path/to/output \
    --conf_thresh 0.08 \
    --save_annotated
```

### CVAT Dataset Export
```bash
# Create CVAT-compatible dataset with data.yaml and train.txt
poetry run python auto/mine_frames_auto.py \
    --videos_dir /path/to/videos \
    --out_dir /path/to/output \
    --create_cvat
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--videos_dir` | *required* | Directory containing video files |
| `--out_dir` | *required* | Output directory for results |
| `--model` | `yolov8n.pt` | YOLO model path |
| `--stride_k` | `3` | Process every k-th frame for tracking |
| `--conf_thresh` | `0.10` | Detection confidence threshold |
| `--store_stride` | `stride_k` | Stride for FrameStore build |
| `--no_thumbs` | `False` | Skip thumbnail storage (saves space) |
| `--save_annotated` | `False` | Export annotated images with boxes |
| `--create_cvat` | `False` | Create CVAT-compatible dataset export |

## Output Structure

```
output_dir/
├── images/                    # Training images (up to 800)
│   ├── video1_f000123.jpg
│   └── video2_f004567.jpg
├── labels/                    # YOLO format labels
│   ├── video1_f000123.txt
│   └── video2_f004567.txt
├── annotated/                 # Annotated images (if enabled)
│   └── video1_f000123.jpg
├── data.yaml                  # CVAT dataset config (if --create_cvat)
├── train.txt                  # CVAT training file list (if --create_cvat)
├── output_dataset_YYYYMMDD_HHMMSS.zip  # CVAT-ready zip (if --create_cvat)
└── logs/
    ├── frame_store/           # Cached detection data
    │   ├── video1/
    │   │   ├── index.pkl      # Metadata + detections
    │   │   └── thumbs/        # Thumbnails (optional)
    ├── summary.json           # Run statistics
    ├── tracks_AUTO.json       # High-quality tracks
    ├── tracks_LABEL.json      # Candidate tracks
    ├── tracks_DROP.json       # Failed tracks
    ├── selected_frames.json   # Final selections
    └── memory_bank.npy        # Historical embeddings
```

## Configuration

Key parameters can be adjusted in the script header:

```python
# Budgets
B_TRACKS = 400    # Max LABEL tracks to consider
B_FRAMES = 800    # Max output frames

# Scoring weights
ALPHA = 0.25      # Uncertainty
BETA = 0.25       # Diversity  
GAMMA = 0.20      # Hardness
DELTA = 0.15      # Novelty
EPSIL = 0.15      # Peak confidence

# Quality thresholds
S_PEAK_CAP = 0.70           # Peak confidence cap
SMALL_AREA = 0.008          # Small object threshold
MIN_SHARP_LABEL = 10.0      # Minimum sharpness
```

## Performance

- **FrameStore Build**: ~30-50 FPS on GPU (one-time cost)
- **Selection Pipeline**: Very fast using cached data
- **Typical Output**: 800 frames from large video collections
- **Memory**: Scales with video count and detection density

## Troubleshooting

**Low output counts**: Check detection performance, adjust `--conf_thresh`
**Performance issues**: Use `--no_thumbs`, increase `--store_stride`
**Memory issues**: Reduce `TRACKER_MAX_TRACKS` in script configuration

## Algorithm Details

The pipeline consists of several phases:

1. **FrameStore Build**: Pre-compute detections, metadata, optional thumbnails
2. **Vectorized Tracking**: Efficient IoU-based association in detection space
3. **Track Triage**: Classify into AUTO/LABEL/DROP categories
4. **Intelligent Scoring**: Rank LABEL tracks using multi-factor scoring
5. **Adaptive Selection**: Apply quotas and diversity constraints
6. **Keyframe Extraction**: Export strategically chosen moments
7. **Deduplication**: Remove temporal and perceptual duplicates

See the main [frame-mining README](../README.md) for detailed algorithm descriptions.
