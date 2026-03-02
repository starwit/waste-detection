# Trainer Core — Multi-backend Training Pipeline

Trainer Core is a DVC-driven training and evaluation pipeline for object detection. It prepares YOLO-style datasets, trains a selected backend, and produces comparable metrics/reports across model families.

## Supported backends

- Ultralytics YOLO (`backend: yolo`)
- RF-DETR (`backend: rfdetr`)
- RTMDet via MMDetection (`backend: mmdet`)

## Repository layout

- Backends (`yolo`, `rfdetr`, `mmdet`): `trainer_core/backends/`
- Dataset preparation/import: `trainer_core/dataprep/`
- Pipeline stages: `trainer_core/pipeline/`
- Evaluation + reporting: `trainer_core/evaluation/`
- Model adapters (so non-YOLO models look like Ultralytics for eval): `trainer_core/wrappers/`
- Config schema + overrides: `trainer_core/config/`

## Pipeline stages

The pipeline is split into three explicit lifecycle stages:

1. **Prepare** (`--stage prepare`)
   - Reads raw images/labels under `raw_data/`
   - Builds a YOLO-style dataset under `datasets/<dataset_name>/`
   - Applies class mapping / class merging (if configured)
2. **Train** (`--stage train`)
   - Trains the selected backend (`yolo`, `rfdetr`, or `mmdet`)
   - Writes run artifacts under `runs/` (including `weights/best.pt`)
   - Persists `runs/.last_train_result.json` for the evaluate stage
3. **Evaluate** (`--stage evaluate`)
   - Loads the trained model (from the persisted pointer if needed)
   - Resolves a baseline model for comparison
   - Writes `metrics.json` and results under `results_comparison/`
   - Organizes plots/metadata into the run folder

## Running

Run the stages via the core CLI:

```bash
python -m trainer_core.cli --stage prepare --dataset-name <name>
python -m trainer_core.cli --stage train --dataset-name <name> --model <model-key>
python -m trainer_core.cli --stage evaluate --dataset-name <name> --model <model-key>
```

### Common CLI options

- `--workspace-root .`: root directory for pipeline I/O (`raw_data/`, `datasets/`, `runs/`, `results_comparison/`, `metrics.json`)
- `--config params.yaml`: config path (defaults to `params.yaml`)
- `--model <key>`: selects a key under `models.*` (overrides `train.model`)
- `--set key=value`: override config keys (supports dot paths)
- `--val-split`, `--test-split`: adjust dataset splits
- `--recreate-dataset`: rebuild `datasets/<dataset_name>/` from `raw_data/`
- `--augment-multiplier`: increase augmentation rate during preparation
- `--folder-subset <folder> <ratio>`: override `prepare.folder_subsets`

## Configuration (`params.yaml`)

Backends are selected via `train.model` (a key under `models.*`) and `models.<key>.backend`.

See `trainer_core/config/schema.py` for the validated shape and defaults.

Minimal example:

```yaml
data:
  dataset_name: waste-detection
  # Either set a class list...
  custom_classes: [waste, cigarette]
  use_coco_classes: false
  # Optional: merge multiple source classes into one during training.
  class_mapping: {}

prepare:
  val_split: 0.1
  test_split: 0.1
  augment_multiplier: 1
  folder_subsets: {}

train:
  model: yolo11m                # key under models.*
  image_size: 1280
  epochs: 100
  batch_size: 4
  finetune:
    enabled: false
    weights: models/current_best/best.pt

models:
  yolo11m:
    backend: yolo
    checkpoint: yolo11m.pt
  rfdetr-medium:
    backend: rfdetr
    variant: medium
    resolution: 1280
  rtmdet-m:
    backend: mmdet
    config_name: rtmdet_m_8xb32-300e_coco
    cache_dir: models/pretrained/mmdet
    allow_download: true

evaluation:
  baseline_weights_path: models/current_best/best.pt
```

### `data.custom_classes` vs `data.use_coco_classes`

- If `data.custom_classes` is non-empty, those names become class 0…n-1.
- If it’s empty and `data.use_coco_classes: true`, the pipeline falls back to the configured COCO subset.

### Baseline & fine-tune weights

- `evaluation.baseline_weights_path` is optional: if it is missing or an empty file, the pipeline falls back to the selected model’s checkpoint (for YOLO: an official Ultralytics checkpoint).
- Fine-tuning weights (`train.finetune.weights`) are required when `train.finetune.enabled: true` and must be a non-empty file.

## Backends

### Ultralytics YOLO (`backend: yolo`)

- Uses `ultralytics.YOLO`.
- `models.<key>.checkpoint` can be an official checkpoint name (downloaded by Ultralytics) or a local `.pt` file.
- Fine-tuning is supported via `train.finetune.*`.

### RF-DETR (`backend: rfdetr`)

- Uses the `rfdetr` Python package.
- `rfdetr` is a standard project dependency (installed via Poetry with the rest of the repo).
- The backend trains via RF-DETR’s Roboflow dataset loader; the pipeline creates a tiny bridge layout under `.tmp/` and cleans it up after training.
- RF-DETR resolution has divisibility constraints; the pipeline will auto-adjust and print a warning if needed.

### RTMDet / MMDetection (`backend: mmdet`)

- Requires `mmdet`, `mmengine`, and **full `mmcv` ops** (`mmcv`, not `mmcv-lite`).
- Uses a temporary COCO export under `.tmp/` for training and evaluation.
- You can predownload configs/checkpoints for reproducible offline runs:

```bash
python tools/mmdet/download_rtmdet_assets.py --variant m --dest models/pretrained/mmdet
```

## Raw data layouts (`raw_data/`)

Put raw data in:

- `raw_data/train/` for train+val
- `raw_data/test/` for the final hold-out set (optional)

The importer accepts any of the following layouts:

| # | Layout | What to do | Notes |
|---|--------|------------|-------|
| 1 | **CVAT YOLO export** | Drop the whole export folder (`data.yaml`, `images/`, `labels/`, `train.txt`). | `train.txt` is automatically parsed. |
| 2 | **Standard YOLO** | Inside a subfolder create `images/` & `labels/`. | Class IDs will be remapped if needed. |
| 3 | **Scene-based test sets** | One subfolder per scene, each with its own `images/` & `labels/`. | Scene name is appended to filenames so metrics stay separate. |
| 4 | **Any folder containing `data.yaml` / `dataset.yaml`** | Copy it in. | Class IDs will be remapped by name matching if needed. |

## Custom classes

Configure custom classes in `params.yaml`:

```yaml
data:
  custom_classes: [waste, cigarette]
  use_coco_classes: false
```

When importing datasets that include a `data.yaml`, the pipeline can remap class IDs based on name matching.

## Class mapping (merging classes)

The class mapping feature allows you to **merge multiple classes into one** during training/evaluation without modifying your raw data.

Example:

```yaml
data:
  custom_classes: [waste, cigarette]
  use_coco_classes: false
  class_mapping:
    waste: [waste, cigarette]
```

With this configuration:

- Raw data remains unchanged in `raw_data/`
- During dataset preparation, labels are remapped
- The trained model sees only the merged target classes

## Folder subsets (`prepare.folder_subsets`)

You can limit or oversample specific source folders during dataset preparation.

Example:

```yaml
prepare:
  folder_subsets:
    uavvaste: 0.5        # use 50% of images from this folder
    taco: 0.2            # use 20%
    cw32-08-07-train: 2  # 200% = oversample (training split only)
```

Behavior:

- `0 < ratio < 1.0`: subsample a folder proportionally
- `ratio > 1.0` (float): oversample a folder (applied to training split only)
- `ratio >= 2` (int): treat as an **absolute count** (“use exactly N images”)

CLI override (multiple allowed):

```bash
python -m trainer_core.cli \
  --stage prepare -d waste-detection \
  --folder-subset uavvaste 0.5 \
  --folder-subset cw32-08-07-train 2.0
```

## Fine-tuning (YOLO)

Fine-tuning is supported for YOLO backends:

```yaml
train:
  finetune:
    enabled: true
    weights: models/current_best/best.pt
    lr: 0.0001
    epochs: 60
    freeze_backbone: false
```

Notes:

- Fine-tuning rejects missing/empty placeholder weight files.
- When fine-tuning, evaluation compares against the configured finetune weights (if available) before falling back to the general baseline.

## Outputs

- Prepared datasets: `datasets/<dataset_name>/train` and `datasets/<dataset_name>/test`
- Training runs: `runs/` (backend-specific subfolders) with `weights/best.pt` + `metadata.yaml`
- Metrics JSON: `metrics.json`
- Results table: `results_comparison/results.csv` (and formatted `results.txt`)
- Persisted pointer for evaluate: `runs/.last_train_result.json`

## Testing

This repo includes:

- Unit tests + E2E pipeline smoke tests using stubs (fast, default)
- Opt-in heavy integration tests (real backend training)

Run:

```bash
poetry install --with dev
poetry run pytest -q
poetry run pytest -q --heavy
```
