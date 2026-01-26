# Preannotate Subclasses Tool

Auto-subclassify existing YOLO bounding boxes (e.g. `"waste"`) into **5 coarse material classes** using a Vision Language Model (VLM): `paper`, `plastic`, `bottle`, `can`, `other`.

## Overview

This tool contains one script:

1. **`preannotate_subclasses.py`** - Reclassify selected classes into `paper|plastic|bottle|can|other`

**Key Features:**
- **Multi-view voting** (close crop, padded crop, context view, plus a full-image bbox view)
- **Tie-break voting** on the context view when votes split (internal default)
- **Caching** (resume without reprocessing)
- **Visualization**: writes montage images into `viz/` for quick review

## Installation

```bash
cd tools/preannotate_subclasses
poetry install
```
---

## Preannotate (5 classes)

### Basic Usage

Run on a raw-data folder that contains:

```
raw_data/.../your_dataset/
├── data.yaml                 # optional, used to auto-detect the "waste" class id by name
├── images/...
└── labels/...
```

### How it works (high level)

For each image, for each YOLO box whose class is `waste`:

1. Generate multiple **views** of the bbox:
   - **CLOSE**: small crop around bbox (with red bbox highlight)
   - **PADDED**: larger crop around bbox
   - **CONTEXT**: larger view that keeps surrounding scene visible (helps disambiguate sometimes)
   - **FULL**: the full image with bbox highlighted (pure context; can help when crops are ambiguous)
2. Ask the VLM to choose **exactly one** label from the 5-class closed set for each view.
3. Combine the labels via a simple vote to produce the final label for the box.
4. Write an output YOLO label file under the output folder with updated class ids.

The tool also optionally writes a montage image to `viz/` for each processed bbox (or only uncertain ones).

Reclassify all `waste` boxes into `paper|plastic|bottle|can|other`:

```bash
poetry run python preannotate_subclasses.py \
  --raw-data ../../raw_data/train/cw33
```

### Model selection

```bash
poetry run python preannotate_subclasses.py \
  --raw-data ../../raw_data/train/cw33 \
  --model 8b
```

### Visualizations

By default `--viz all` will save many montage images. For quick iteration, `--viz uncertain` is usually enough.

```bash
poetry run python preannotate_subclasses.py \
  --raw-data ../../raw_data/train/cw33 \
  --viz uncertain
```

### Optional gating (skip tiny bboxes)

Some bboxes are too small to classify reliably. You can gate these based on normalized bbox size and drop them from the output labels:

```bash
poetry run python preannotate_subclasses.py \
  --raw-data ../../raw_data/train/cw33 \
  --min-box-side-norm 0.08 \
  --drop-gated
```

---

## Output Structure

```
output/<raw_data_name>/
├── <same relative structure as images>/
│   └── *.txt                    # YOLO labels (waste subclasses + appended non-waste classes)
├── classes.txt                  # Output class names (0..4 are the 5 subclasses; others appended; collisions prefixed with orig_)
├── subclass_cache.json          # Cache with predictions & confidence
├── report.json                  # Statistics and summary
├── review_uncertain.csv         # Uncertain predictions list
└── viz/                         # Montages (disable with --viz none)
```

### Non-waste classes

Only the `waste` class is reclassified into the 5 subclasses. All other classes are preserved by **remapping** them to new ids **appended after** the 5 subclasses:

- Output class ids `0..4`: `paper, plastic, bottle, can, other` (waste subclasses)
- Output class ids `5..`: original non-waste classes from `data.yaml` (e.g. `cigarette`, `leaf_*`)

If an original class name collides with one of the 5 subclass names (e.g. an original `bottle` class), it is written as `orig_bottle` in `classes.txt`.

### Confidence / “uncertain”

`confidence` is based on vote agreement (e.g. 3/4 views agree → 0.75). It is **not** a calibrated probability.
`review_uncertain.csv` lists bboxes whose agreement falls below the internal threshold.

---