# Waste Detection — YOLO + RF-DETR + RTMDet + DVC

This repository is the **waste-detection project/training repo**. It uses DVC (Data Version Control) to track datasets, experiments, and promoted baselines for reproducible model training.

Training/evaluation is implemented by the external Trainer Core package (`object-detector-trainer`) and invoked via the project entrypoint `train.py`.
DVC stages call `python -m train` plus `python -m object_detector_trainer.pipeline.check_optional_weight_deps` for preflight checks.
The project pipeline wrapper injects project-local defaults for `--workspace-root` and `--config` (`params.yaml`) so `object_detector_trainer` can be consumed as a dependency without relying on shell cwd. If no `--stage` is provided, it defaults to `train`.

Note on RTMDet dependencies:
- This project installs `object-detector-trainer` with RTMDet extras enabled.
- RTMDet/OpenMMLab currently works reliably with Python 3.11.
- If your machine defaults to Python 3.12 and install fails on `mmcv`, create/select a 3.11 Poetry env first (`poetry env use 3.11`).

Testing split:
- Project-level tests in this repo:
  - fast contract tests: `poetry run pytest`
  - heavy DVC integration tests: `poetry run pytest --heavy`
- Trainer backend-heavy tests live in `object-detector-trainer`.
- Ensure trainer deps are installed first (`poetry -C ../object-detector-trainer install`).
- From this project root, run trainer heavy tests via: `poetry -C ../object-detector-trainer run pytest -m heavy`.

### Why DVC here?

- **Large files:** Keeps datasets and model weights out of Git history while versioning them alongside code.
- **Reproducibility:** The exact artifacts used for a run are pinned by DVC metadata (`*.dvc` files and, for pipeline runs, `dvc.lock`) and can be restored with `dvc pull`.


> About this fork
> - Task: waste detection
> - Current training classes: `waste`, `cigarette`, `leaves_dense`, `leaves_sparse`
> - Pipeline: Ultralytics YOLO, RF-DETR, and RTMDet with DVC-managed data and outputs
---

## Table of Contents
1. [Models & Releases](#models--releases)
2. [Initial Project Setup](#initial-project-setup)
3. [Managing Project Data](#managing-project-data)
4. [Experiment Workflow](#experiment-workflow)
5. [Core Pipeline Docs](#core-pipeline-docs)

---

## Models & Releases

This repo publishes trained models with each GitHub Release and also tracks the currently applied model via DVC.

- Latest baseline (main): The model currently applied on the main branch is tracked at `models/current_best/best.pt` (via `models/current_best/best.pt.dvc`). To fetch it locally, run `dvc pull models/current_best/best.pt` (requires access to the configured DVC remote).
- Release assets: Each release includes the trained weights and metadata so you don’t need DVC to use the model:
  - `weights/best.pt`: the promoted weights for inference
  - `test_metrics.json`: evaluation metrics of the promoted run
  - `metadata.yaml`: training metadata (experiment name, epochs, image size, etc.)

### Baseline comparisons

The training pipeline loads a baseline model for comparison during evaluation:

- **First tries:** `evaluation.baseline_weights_path` from params.yaml (defaults to `models/current_best/best.pt`)
- **Then tries:** Fine-tune weights if in finetune mode
- **Falls back to:** Official YOLO COCO checkpoint (always works, even on fresh clones)

On fresh clones, `setup_project.py` creates a 0-byte placeholder at `models/current_best/best.pt` so DVC can track the baseline path as a direct file dependency. The pipeline ignores empty placeholders automatically; run `dvc pull` or export a real baseline to replace it.

To keep startup errors understandable on fresh clones, the DVC pipeline includes a small preflight step (`check_optional_weight_deps`) that verifies configured optional weight paths exist before `train_model` / `evaluate_model`. If only `best.pt.dvc` exists but the local `best.pt` is not present yet, this step fails early with a clear message and points users to run `python setup_project.py` (for placeholders) or `dvc pull` (for real weights).

When you want to make a freshly trained run the new comparison baseline:

1. Export the run’s best weights (and optional metadata) into `models/current_best/`:
   ```bash
   python tools/export_baseline.py --run-dir runs/<experiment_name>
   ```
   You can override the weights or metadata paths via CLI flags if needed.
2. Track the updated files with DVC so others can fetch them:
   ```bash
   dvc add models/current_best/best.pt models/current_best/metadata.yaml
   dvc push
   ```

After these steps, subsequent training runs will compare against the newly exported baseline automatically.


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
* create 0-byte placeholder weight files for DVC's optional model dependencies

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

### Step 1: Configure Parameters

All training parameters live in **`params.yaml`**. For the complete configuration reference (including backend-specific keys), see:

- https://github.com/starwit/object-detector-trainer

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

## Core Pipeline Docs

The training core lives in a separate repository and is imported as a dependency:

- https://github.com/starwit/object-detector-trainer
