# Waste Detection — YOLOv8 + DVC

This repository is a focused fork of our YOLO retraining template, specialized for detecting waste on streets. It uses DVC (Data Version Control) to track datasets, experiments, and trained models for reproducible results and easy collaboration.

### Why DVC here?

- **Large files:** Keeps datasets and model weights out of Git history while versioning them alongside code.
- **Reproducibility:** The exact model used for a release is pinned by `dvc.lock` and can be restored with `dvc pull`.


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

A helper script configures the project’s connection to remote storage and (optionally) custom classes.

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

This project uses a structured workflow for training and promoting models. The key principle is to perform extensive experimentation locally and only commit significant, "winning" models to the main project history.

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
  model_size: m
  image_size: 1280
  epochs: 50
  batch_size: 8
```

* If `custom_classes` is non-empty, those names become class 0…n-1.  
* If it’s empty **and** `use_coco_classes: true`, the predefined COCO subset is used.

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

## Raw Data Structure

Put your raw data in `raw_data/train/` (for train+val) and `raw_data/test/` (for the final hold‑out set).  
The importer now accepts **any** of the following:

| # | Layout | What to do | Notes |
|---|--------|------------|-------|
| 1 | **CVAT YOLO export** | Drop the whole export folder (`data.yaml`, `images/`, `labels/`, `train.txt`). | `train.txt` is automatically parsed. |
| 2 | **Standard YOLO** | Inside a subfolder create `images/` & `labels/`. | Class IDs will be remapped if needed. |
| 3 | **Scene‑based test sets** | One subfolder per scene, each with its own `images/` & `labels/`. | Scene name is appended to filenames so metrics stay separate. |
| 4 | **Any folder that already contains a `data.yaml` / `dataset.yaml`** | Just copy it in. | Class IDs will be remapped if needed (names have to be the same as in params.yaml). |

> **Tip:** When you use option 4 you can bring in public datasets or prior labeling runs “as is”.  
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
