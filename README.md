# Waste Detection — YOLOv8 + DVC

This repository is a focused fork of our YOLO retraining template, specialized for detecting waste on streets. It uses DVC (Data Version Control) to track datasets, experiments, and trained models for reproducible results and easy collaboration.

### Why DVC here?

- Large files: Keeps datasets and model weights out of Git history while versioning them alongside code.
- Reproducibility: The exact model used for a release is pinned by `dvc.lock` and can be restored with `dvc pull`.


> About this fork
> - Task: waste detection
> - Current classes (from params.yaml): `waste`, `cigarette` (to keep it focused for now and not introduce too many classes while we don't have much training data)
> - Pipeline: Ultralytics YOLOv8 with DVC-managed data and outputs

---

## Table of Contents
1. [Models & Releases](#models--releases)
2. [Initial Project Setup](#initial-project-setup)
3. [Managing Project Data](#managing-project-data)
4. [Experiment Workflow](#experiment-workflow)
5. [Custom Classes](#custom-classes)
6. [Raw Data Structure](#raw-data-structure)
7. [Fine-tuning](#fine-tuning)
8. [Folder Subsets](#folder-subsets)

---

## Models & Releases

This repo publishes trained models with each GitHub Release and also tracks the currently applied model via DVC.

- Latest model (main): The model currently applied on the main branch is the one referenced in `dvc.lock` under the `runs/` output. To fetch it locally, run `dvc pull` (requires access to the configured DVC remote).
- Release assets: Each release includes the trained weights and metadata so you don’t need DVC to use the model:
  - `weights/best.pt`: the promoted YOLO weights for inference
  - `test_metrics.json`: evaluation metrics of the promoted run
  - `metadata.yaml`: training metadata (experiment name, epochs, image size, etc.)


## Initial Project Setup

Follow these steps once after creating this project from the template.

1. Create the new repository from the template on GitHub
2. Clone your new repository locally
3. Install dependencies (Poetry)
   ```bash
   poetry install
   poetry shell
   ```
4. Run the interactive setup script
   ```bash
   python setup_project.py
   ```
   The script will:
   - prompt for a project & dataset name
   - optionally ask for a comma-separated list of class names
   - patch `.dvc/config` (remote URL) and `params.yaml`
   - create the `raw_data/train` and `raw_data/test` folders
5. Commit the initial configuration
   ```bash
   git add .dvc/config params.yaml
   git commit -m "Initialize project configuration"
   ```

---

## Managing Project Data

After the initial setup, follow this process whenever you add or update the raw dataset.

1. Add your data under `raw_data/train/` and/or `raw_data/test/`
2. Track data changes with DVC
   ```bash
   dvc add raw_data
   ```
3. Commit the pointer file with Git
   ```bash
   git add raw_data.dvc
   git commit -m "Add/Update raw_data"
   ```
4. Push both code and data
   ```bash
   dvc push
   git push
   ```

---

## Experiment Workflow

All training parameters live in `params.yaml`. Example:

```yaml
data:
  dataset_name: waste-detection
  experiment_name: waste-detection
  custom_classes:
    - waste
    - cigarette
  use_coco_classes: false

train:
  model_size: m
  image_size: 1280
  epochs: 50
  batch_size: 8
```

- If `custom_classes` is non-empty, those names become class IDs 0…n-1.
- If it’s empty and `use_coco_classes: true`, a predefined COCO subset is used.

Run and compare experiments locally with DVC:

```bash
# Run with params from params.yaml
dvc exp run -n "large-model-150-epochs"

# Override parameters without editing params.yaml
dvc exp run -n "smaller-batch" -S train.batch_size=4

# Compare experiments
dvc exp show --sort-by metrics/fitness --sort-order desc
```

Promote a winning experiment:

```bash
dvc exp apply <exp-name>
git add .
git commit -m "Promote new model: <short description>"
dvc push && git push
```

---

## Custom Classes

This fork is configured for waste detection. By default, the classes are defined in `params.yaml`:

```yaml
data:
  custom_classes:
    - waste
    - cigarette
  use_coco_classes: false
```

To change classes, edit `params.yaml` or re-run `python setup_project.py` and follow the prompts. The pipeline will remap labels of imported datasets where a data YAML is present.

---

## Raw Data Structure

Put your raw data in `raw_data/train/` (for train+val) and `raw_data/test/` (hold‑out set). The importer accepts:

| # | Layout | What to do | Notes |
|---|--------|------------|-------|
| 1 | CVAT YOLO export | Drop the whole export folder (`data.yaml`, `images/`, `labels/`, `train.txt`). | `train.txt` is parsed automatically. |
| 2 | Standard YOLO | Inside a subfolder create `images/` & `labels/`. | Class IDs get remapped if needed. |
| 3 | Scene‑based test sets | One subfolder per scene with `images/` & `labels/`. | Scene name is appended to filenames so metrics stay separate. |
| 4 | Folder with `data.yaml` / `dataset.yaml` | Copy it in. | Classes are mapped by name to your `params.yaml` classes. |

Example scene‑based layout:

```
raw_data/
└── test/
    ├── Wolfsburg/
    │   ├── images/
    │   └── labels/
    └── Carmel/
        ├── images/
        └── labels/
```

---

## Fine-tuning

You can fine-tune from a pre-trained checkpoint (e.g., a prior waste model) instead of starting from the base YOLOv8 weights.

Configure in `params.yaml` under `train`:

```yaml
train:
  finetune_mode: true              # enable fine-tuning
  pretrained_model_path: taco-uav-model.pt
  finetune_lr: 0.0001              # lower LR for fine-tuning
  finetune_epochs: 60              # optional override for epochs
  freeze_backbone: false           # optionally freeze early layers
```

Notes:
- When fine-tuning, the pipeline evaluates the chosen pretrained model as the baseline and compares it to the fine-tuned result.
- The DVC pipeline depends on `pretrained_model_path` so runs are reproducible. Make sure the file is available (via DVC or local path).

---

## Folder Subsets

To limit or oversample specific source folders during dataset preparation, use `prepare.folder_subsets` in `params.yaml` or the CLI override.

Example `params.yaml` configuration:

```yaml
prepare:
  val_split: 0.1
  test_split: 0.1
  augment_multiplier: 1
  folder_subsets:
    uavvaste: 0.5        # use 50% of images from this folder
    taco: 0.2            # use 20%
    cw32-08-07-train: 2  # 200% = oversample
```

CLI override (multiple allowed):

```bash
python yolov8_training/train_pipeline.py \
  --stage prepare -d waste-detection \
  --folder-subset uavvaste 0.5 \
  --folder-subset cw32-08-07-train 2.0
```

Behavior:
- Ratios between 0 and 1.0 subsample a folder.
- Ratios > 1.0 oversample by repeating images; cross-folder duplicates are removed so balancing doesn’t leak duplicates between sources.

