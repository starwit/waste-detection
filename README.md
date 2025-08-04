
# YOLO Retraining Pipeline Template

This repository provides a standardized, reusable template for retraining YOLO models. It uses DVC (Data Version Control) to manage datasets, track experiments, and ensure reproducible results across the team.

### What is DVC?

DVC (Data Version Control) is a tool that works alongside Git to handle large files like datasets and machine learning models.

**Goal in this Project:**
- **Version Control for Data:** Git tracks our code, while DVC tracks our large data files.
- **Reproducibility:** Anyone on the team can get the exact code, data, and model for any version of the project, ensuring that all results are reproducible.
- **Centralized Storage:** DVC manages uploading and downloading data to a shared remote storage (Hetzner in our case), keeping our Git repository small and fast.

> **What’s new?**  
> The pipeline now supports **arbitrary, project-specific class names**.  
> Define them once in `params.yaml → data.custom_classes` and the code will:
> * rewrite `dataset.yaml` with the right label set  
> * remap any existing YOLO TXT labels  
> * train & evaluate only on the classes you specify  
---

## Table of Contents
1. [Initial Project Setup](#initial-project-setup)
2. [Managing Project Data](#managing-project-data)
3. [The Experiment Workflow](#the-experiment-workflow)
4. [Custom Class Configuration](#custom-class-configuration)
5. [Raw Data Structure Guide](#raw-data-structure-guide)

---

## Initial Project Setup

Follow these steps once to create a new project from this template.

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

A helper script configures the project’s connection to remote storage **and** asks whether you want to specify custom classes.

```bash
python scripts/setup_project.py
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

## The Experiment Workflow

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


## Raw Data Structure Guide

Put your raw data in `raw_data/train/` (for train+val) and `raw_data/test/` (for the final hold‑out set).  
The importer now accepts **any** of the following:

| # | Layout | What to do | Notes |
|---|--------|------------|-------|
| 1 | **CVAT YOLO export** | Drop the whole export folder (`data.yaml`, `images/`, `labels/`, `train.txt`). | `train.txt` is automatically parsed. |
| 2 | **Standard YOLO** | Inside a subfolder create `images/` & `labels/`. | Class IDs will be remapped if needed. |
| 3 | **Scene‑based test sets** | One subfolder per scene, each with its own `images/` & `labels/`. | Scene name is appended to filenames so metrics stay separate. |
| 4 | **Any folder that already contains a `data.yaml` / `dataset.yaml`** | Just copy it in. | Class IDs will be remapped if needed (names have to be the same as in params.yaml). |

> **Tip:** When you use option&nbsp;4 you can bring in public datasets or prior labeling runs “as is”.  
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
