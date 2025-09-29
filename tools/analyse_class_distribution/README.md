# Class Distribution Analyzer

This tool provides comprehensive analysis of class distributions in YOLO format datasets, including per-class counts, object size distribution using COCO-style bins, and per-split statistics to inform oversampling, weighting, and data collection decisions.

## Features

- Parse YOLO/COCO annotations from dataset.yaml
- Report per-split (train/val/test) statistics
- Total instances per class
- Number of images containing each class  
- Percentage distributions
- Empty images count
- Object size distribution using COCO-style area bins (Small <32²px, Medium 32²-96²px, Large ≥96²px)
- Export results to JSON format
- Support for nested dataset structures (waste-detection pipeline format)

## Usage

### Basic Analysis

```bash
python class_distribution_analyzer.py --dataset ../../datasets/waste-detection
```

### With Exports

```bash
python class_distribution_analyzer.py --dataset ../../datasets/waste-detection --export-json --output analysis_results
```

### Quiet Mode (Errors Only)

```bash
python class_distribution_analyzer.py --dataset ../../datasets/waste-detection --quiet
```

## Parameters

- `--dataset`, `-d`: Path to dataset directory containing dataset.yaml (required)
- `--output`, `-o`: Output directory for exports (default: analysis_output)
- `--export-json`: Export results to JSON file
- `--quiet`, `-q`: Suppress console output (only show errors)

## Output Format

The tool provides a clean, structured output with:

1. **Overall Totals**: Dataset-wide statistics and class distribution
2. **Per-Split Breakdown**: Detailed statistics for train/val/test splits
3. **Quick Reference Table**: Summary table for easy comparison

Example output:
```
============================================================
DATASET CLASS DISTRIBUTION ANALYSIS
============================================================

OVERALL TOTALS
--------------------
Images: 2,865 total, 419 empty (14.6%)
Objects: 7,178 total
Classes: 2 (waste, cigarette)
  waste: 6,274 instances (87.4%)
  cigarette: 904 instances (12.6%)

PER-SPLIT BREAKDOWN
--------------------
Split    Images   Objects  Empty  Empty%
----------------------------------------
train    2,427    6,302    313    12.9
val      348      807      62     17.8
test     90       69       44     48.9

TRAIN:
  waste: 5,522 instances (87.6%) in 2,033 images (83.8%)
    Sizes: 2,135 small, 2,183 medium, 1,204 large
  cigarette: 780 instances (12.4%) in 338 images (13.9%)
    Sizes: 747 small, 30 medium, 3 large
...
```

## Dataset Structure Support

The tool supports both standard flat YOLO structure and the nested structure used by the waste-detection pipeline:

### Standard Structure
```
dataset/
├── dataset.yaml
├── train/
│   ├── images/
│   └── labels/
├── val/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

### Nested Structure (waste-detection pipeline)
```
datasets/waste-detection/
├── train/
│   ├── dataset.yaml
│   ├── train/
│   │   ├── images/
│   │   └── labels/
│   └── val/
│       ├── images/
│       └── labels/
└── test/
    ├── dataset.yaml
    └── val/
        ├── images/
        └── labels/
```

## Export Formats

### JSON Export
- `class_distribution.json`: Complete analysis data in structured format
