# Waste Detection — Multi-Backend Object Detection + DVC

This repository is a focused fork of our object-detection retraining template, specialized for detecting waste on streets. It supports **multiple detection backends** — all Ultralytics YOLO versions as well as RF-DETR — with a unified DVC-managed pipeline for reproducible experiments and easy collaboration.

### Supported Backends

| Backend | Config value (`models.<key>.backend`) | Models |
|---------|---------------------------------------|--------|
| **Ultralytics YOLO** | `yolo` | Any checkpoint supported by `ultralytics` |
| **RF-DETR** | `rfdetr` | Nano, Small, Medium, Base, Large |

Switching between backends is a one-line change in `params.yaml` (`train.model`). The evaluation pipeline, baseline comparisons, side-by-side visualisations, and DVC tracking work identically for all backends.

### Why DVC here?

- **Large files:** Keeps datasets and model weights out of Git history while versioning them alongside code.
- **Reproducibility:** The exact model used for a release is pinned by `dvc.lock` and can be restored with `dvc pull`.

> About this fork
> - Task: waste detection
> - Current classes (from `params.yaml`): configurable (e.g. `waste`, `cigarette`, `leaves_dense`, `leaves_sparse`)
> - Pipeline: Ultralytics YOLO *or* RF-DETR, selected via `train.model` in `params.yaml`

---

## Table of Contents
1. [Models & Releases](#models--releases)
2. [Selecting a Training Backend](#selecting-a-training-backend)
3. [Initial Project Setup](#initial-project-setup)
4. [Managing Project Data](#managing-project-data)
5. [Experiment Workflow](#experiment-workflow)
6. [Custom Classes](#custom-classes)
7. [Class Mapping](#class-mapping)
8. [Raw Data Structure](#raw-data-structure)
9. [Fine-tuning](#fine-tuning)
10. [Folder Subsets](#folder-subsets)
11. [Testing](#testing)

---

## Models & Releases

This repo publishes trained models with each GitHub Release and also tracks the currently applied model via DVC.

- Latest model (main): The model currently applied on the main branch is the one referenced in `dvc.lock` under the `runs/` output. To fetch it locally, run `dvc pull` (requires access to the configured DVC remote).
- Release assets: Each release includes the trained weights and metadata so you don't need DVC to use the model:
  - `weights/best.pt`: the promoted weights for inference (YOLO `.pt` or RF-DETR checkpoint copied to this path)
  - `test_metrics.json`: evaluation metrics of the promoted run
  - `metadata.yaml`: training metadata (experiment name, epochs, image size, etc.)

### Baseline comparisons

The training pipeline loads a baseline model for comparison during evaluation:

- **First tries:** `evaluation.baseline_weights_path` from params.yaml (defaults to `models/current_best/best.pt`)
- **Falls back to:** Official YOLO COCO checkpoint (always works, even on fresh clones)

When you want to make a freshly trained run the new comparison baseline:

1. Export the run's best weights (and optional metadata) into `models/current_best/`:
   ```bash
   python yolov8_training/utils/export_baseline.py --run-dir runs/<experiment_name>
   ```
   This works for both YOLO and RF-DETR runs — the RF-DETR path copies its best checkpoint into the same `weights/best.pt` layout.
2. Track the updated files with DVC so others can fetch them:
   ```bash
   dvc add models/current_best/best.pt models/current_best/metadata.yaml
   dvc push
   ```

After these steps, subsequent training runs will compare against the newly exported baseline automatically.

---

## Selecting a Training Backend

All models are defined in the `models` section of `params.yaml`. Set `train.model` to the key of the model you want to train. All other pipeline stages (data preparation, evaluation, baseline comparison, output organisation) are shared and work identically for both backends.

### YOLO

```yaml
train:
  model: yolo11m           # key from models.* below
  image_size: 1280
  epochs: 100
  batch_size: 8

models:
  yolo11m:
    backend: yolo
    checkpoint: yolo11m.pt  # any Ultralytics-compatible checkpoint
  yolov8m:
    backend: yolo
    checkpoint: yolov8m.pt
```

Any Ultralytics-compatible checkpoint works — just add a new entry under `models`.

### RF-DETR

```yaml
train:
  model: rfdetr-medium      # key from models.* below
  image_size: 1280
  epochs: 150
  batch_size: 2

models:
  rfdetr-medium:
    backend: rfdetr
    variant: medium           # nano | small | medium | base | large
    resolution: 1280          # must be divisible by 32 (nano/small/medium) or 56 (base/large)
    epochs: 150               # override shared default — set high, early_stopping decides
    batch_size: 2
    target_effective_batch: 16  # grad_accum_steps auto-computed: 16 // 2 = 8
    # grad_accum_steps: 8     # uncomment to override auto-computation
    lr: 0.0001                # decoder LR; lr_encoder is handled internally
    gradient_checkpointing: false
    extra_train_kwargs:
      num_workers: 4
      multi_scale: true       # RF-DETR default — critical for generalisation
      expanded_scales: true   # RF-DETR default — extends scale augmentation range
      lr_drop: 120
      warmup_epochs: 1.0
      early_stopping: true
      early_stopping_patience: 15
      early_stopping_min_delta: 0.001
```

Model-specific keys (`epochs`, `batch_size`, `resolution`, etc.) override the shared defaults in `train`.

When RF-DETR is selected the pipeline:
1. Creates a lightweight YOLO-format directory with symlinks (`train/`, `valid/`, `test/` + `data.yaml`) — instant, no file copying (RF-DETR 1.4+ auto-detects this format)
2. Trains the RF-DETR model with auto-computed gradient accumulation
3. Copies the best checkpoint to `weights/best.pt` (same layout as YOLO)
4. Wraps the model in an adapter (`RFDETRModelAdapter`) so that evaluation, side-by-side comparisons, and metrics export work identically
5. Cleans up the temporary symlink directory

### Quick switch via CLI

```bash
# Train with RF-DETR without editing params.yaml
dvc exp run -n "rfdetr-test" -S train.model=rfdetr-medium

# Train with a different YOLO family
dvc exp run -n "yolo11-test" -S train.model=yolo11m
```


## Initial Project Setup

Follow these steps once after creating this project from the template.

**1. Create the New Repository**
   - On the GitHub page for this template, click the **"Use this template"** button.
   - Assign a name to your new repository (e.g., `waste-detection-2025`) and confirm its creation.

**2. Clone Your New Repository**

**3. Install Dependencies**
   This project uses Poetry for dependency management.
   ```bash
   poetry install
   poetry shell
   ```

**4. Run the Interactive Setup Script**

A helper script configures the project's connection to remote storage and (optionally) custom classes.

```bash
python setup_project.py
```

The script will:

* prompt for a project & dataset name  
* optionally ask for a comma-separated list of class names  
* patch `.dvc/config` (remote URL) and `params.yaml` accordingly  
* create the `raw_data/train` and `raw_data/test` folders

**5. Commit the Initial Configuration**
   The bootstrap script modifies configuration files. Commit these changes to save the project setup.
   ```bash
   git add .dvc/config params.yaml
   git commit -m "Initialize project configuration"
   ```
The project is now fully configured and ready for data.

---

## Managing Project Data

After the initial setup, follow this process whenever you add or update the raw dataset.

**1. Add Your Data**
   - Place your new or updated data files into the `raw_data/train/` or `raw_data/test/` directories, following the structure guide below.

**2. Track Data with DVC**
   - Use the `dvc add` command to tell DVC to track the state of your data directory.
   ```bash
   dvc add raw_data
   ```
   This command creates/updates a small `raw_data.dvc` file. This file acts as a pointer to the actual data, which DVC manages.

**3. Commit the Pointer File with Git**
   - Add the `.dvc` file to Git. This records the "version" of your data that corresponds to your code.
   ```bash
   git add raw_data.dvc
   git commit -m "Add new batch of training images"
   ```

**4. Push Both Code and Data**
   Pushing is a two-step process: `dvc push` uploads your large data files to the shared Hetzner storage, and `git push` uploads your code and the small DVC pointer file.
   ```bash
   # Step 1: Upload the actual data files to remote storage
   dvc push

   # Step 2: Upload the code and data pointers
   git push
   ```

---

## Experiment Workflow

This project uses a structured workflow for training and promoting models. The key principle is to perform extensive experimentation locally and only commit significant, "winning" models to the main project history. The workflow is identical for all backends (YOLO, RF-DETR).

### 1  Configure Parameters

All training parameters live in **`params.yaml`**.  
Key fields:

```yaml
data:
  dataset_name: waste-detection
  experiment_name: waste-detection

  # ↓ Optional — override COCO classes
  custom_classes:
    - "waste"
    - "cigarette"
  use_coco_classes: false     # fallback to COCO if true/empty

train:
  model: yolo11m             # key from models.* section
  image_size: 1280
  epochs: 50
  batch_size: 8

models:
  yolo11m:
    backend: yolo
    checkpoint: yolo11m.pt
  rfdetr-medium:
    backend: rfdetr
    variant: medium
    resolution: 1280
```

* If `custom_classes` is non-empty, those names become class 0…n-1.  
* If it's empty **and** `use_coco_classes: true`, the predefined COCO subset is used.

### Step 2: Run Experiments
Execute experiments using the `dvc exp run` command. This process is entirely local and does not create any Git commits. Use the `-n` flag to assign a descriptive name to each run.

```bash
# Run an experiment with the settings from params.yaml
dvc exp run -n "large-model-150-epochs"
```
For quick iterations, you can override parameters from the command line with the `-S` flag:
```bash
# Test a different parameter without editing params.yaml
dvc exp run -n "test-smaller-batch-size" -S train.batch_size=4
```

### Step 3: Review and Compare Results
Use `dvc exp show` to display a leaderboard of all local experiments. This table includes the parameters and performance metrics for each run, allowing for easy comparison.

```bash
# Sort the table by a key metric to find the best performer
dvc exp show --sort-by metrics/fitness --sort-order desc
```

### Step 4: Promote a Winning Experiment
Once you identify a superior experiment, promote it to become the official version in the main project branch.

1.  **Apply the winner's results** to your workspace. This command updates your `params.yaml` and output files to match the state of the selected experiment.
    ```bash
    dvc exp apply <name-of-your-winning-experiment>
    ```

2.  **Commit this "Golden" version** to Git. This is the only time a commit is made after a series of experiments.
    ```bash
    git add .
    git commit -m "Promote new model with 150 epochs, achieves 0.92 fitness"
    ```

3.  **Push the final result** to the shared remotes.
    ```bash
    dvc push  # Uploads the winning model's data files
    git push  # Pushes the commit with the updated project state
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

To change classes, edit `params.yaml` or re-run `python setup_project.py` and follow the prompts. The pipeline will remap labels of imported datasets where a `data.yaml` is present.

---

## Class Mapping

The class mapping feature allows you to **merge multiple classes into one** during training and evaluation without modifying your raw data. This is useful for testing if combining similar classes improves detection performance.

### How to Use

Edit `params.yaml` to add a `class_mapping` configuration:

```yaml
data:
  custom_classes:
    - waste
    - cigarette
  use_coco_classes: false
  
  # Merge classes together during training
  class_mapping:
    waste: [waste, cigarette]  # Both will be treated as 'waste'
```

With this configuration:
- Your raw data remains unchanged in `raw_data/`
- During dataset preparation, labels are automatically remapped
- The model trains with only 1 class: `waste`
- All cigarette detections are treated as waste detections

### Example: Testing Class Merging

To test if merging `waste` and `cigarette` improves cigarette detection:

1. Add the mapping to `params.yaml` as shown above
2. Run the pipeline: `dvc repro` or `dvc exp run -n "merged-classes"`
3. Compare results with your baseline (separate classes)
4. Keep the mapping if results improve, otherwise revert it

### More Examples

**Multiple merge groups:**
```yaml
class_mapping:
  recyclable: [plastic_bottle, glass_bottle, aluminum_can]
  organic: [food_waste, paper]
```

**Partial mapping (keep some classes separate):**
```yaml
custom_classes: [waste, cigarette, person]
class_mapping:
  waste: [waste, cigarette]
  # person remains separate
```

For detailed documentation, see [`docs/CLASS_MAPPING.md`](docs/CLASS_MAPPING.md).

> **Note:** When importing datasets, the pipeline also automatically maps classes from source `data.yaml` files to your configured classes based on name matching.

---

## Testing

This project includes unit tests, a lightweight E2E smoke suite, and an opt-in heavy integration suite.

- Using unittest
  - Run all tests:
    ```bash
    poetry run python -m unittest -v
    ```

- Using pytest (optional)
  - Install dev dependencies once:
    ```bash
    poetry install --with dev
    ```
  - Run default tests:
    ```bash
    poetry run pytest -q
    ```
    - This skips tests marked `heavy`.
  - Run heavy integration tests:
    ```bash
    poetry run pytest -q --heavy
    ```
    - This includes real-backend training checks and DVC repro integration checks.

Notes
- The standard E2E smoke test creates a tiny synthetic dataset under a temporary directory and runs both the prepare and train/eval stages. It also checks that scene metrics are exported in `results_comparison/results.csv`.
- Smoke tests use stubs to keep results deterministic and CI-friendly.
- The RF-DETR backend path is covered by a dedicated E2E test (`test_pipeline_can_select_rfdetr_backend_via_params`) that verifies training, evaluation, weight saving, and metrics export using a lightweight stub.
- Heavy tests include real YOLO/RF-DETR one-epoch backend runs and DVC stage reproducibility checks.
- Heavy tests are opt-in because they are integration-level, more environment-sensitive, and may be slower.
- Real-backend heavy tests require local checkpoint files (`yolov8n.pt`, `rf-detr-nano.pth`).

---

## Raw Data Structure

Put your raw data in `raw_data/train/` (for train+val) and `raw_data/test/` (for the final hold‑out set).
The importer now accepts **any** of the following:

| # | Layout | What to do | Notes |
|---|--------|------------|-------|
| 1 | **CVAT YOLO export** | Drop the whole export folder (`data.yaml`, `images/`, `labels/`, `train.txt`). | `train.txt` is automatically parsed. |
| 2 | **Standard YOLO** | Inside a subfolder create `images/` & `labels/`. | Class IDs will be remapped if needed. |
| 3 | **Scene‑based test sets** | One subfolder per scene, each with its own `images/` & `labels/`. | Scene name is appended to filenames so metrics stay separate. |
| 4 | **Any folder that already contains a `data.yaml` / `dataset.yaml`** | Just copy it in. | Class IDs will be remapped if needed (names have to be the same as in params.yaml). |

> **Tip:** When you use option 4 you can bring in public datasets or prior labeling runs "as is".
> The importer detects the YAML, builds a temporary copy, remaps the labels, and keeps going—no manual edits required.

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

## Fine-tuning

You can fine-tune from a pre-trained checkpoint (e.g., a prior waste model) instead of starting from the base YOLO weights.

Configure in `params.yaml` under `train.finetune`:

```yaml
train:
  finetune:
    enabled: true
    weights: taco-uav-model.pt
    lr: 0.0001
    epochs: 60
    freeze_backbone: false
```

Notes:
- When fine-tuning, the pipeline evaluates the chosen pretrained model as the baseline and compares it to the fine-tuned result.
- The DVC pipeline depends on `train.finetune.weights` so runs are reproducible. Make sure the file is available (via DVC or local path).

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
- Ratios > 1.0 oversample by repeating images; cross-folder duplicates are removed so balancing doesn't leak duplicates between sources.
