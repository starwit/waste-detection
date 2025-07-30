
# YOLO Retraining Pipeline Template

This repository provides a standardized, reusable template for retraining YOLO models. It uses DVC (Data Version Control) to manage datasets, track experiments, and ensure reproducible results across the team.

### What is DVC?

DVC (Data Version Control) is a tool that works alongside Git to handle large files like datasets and machine learning models.

**Goal in this Project:**
- **Version Control for Data:** Git tracks our code, while DVC tracks our large data files.
- **Reproducibility:** Anyone on the team can get the exact code, data, and model for any version of the project, ensuring that all results are reproducible.
- **Centralized Storage:** DVC manages uploading and downloading data to a shared remote storage (Hetzner in our case), keeping our Git repository small and fast.

---

## Table of Contents
1. [Initial Project Setup](#initial-project-setup)
2. [Managing Project Data](#managing-project-data)
3. [The Experiment Workflow](#the-experiment-workflow)
4. [Advanced Analysis & Utilities](#advanced-analysis--utilities)
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
   A helper script will configure the project's connection to the remote storage. It will prompt for a unique "project name" to ensure data is stored in a dedicated folder.
   ```bash
   python scripts/setup_project.py
   ```
   This script will:
   - Prompt for a project name.
   - Configure the DVC remote path.
   - Create the `raw_data/train` and `raw_data/test` folders.

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

### Step 1: Configure Parameters
All training parameters are controlled in **`params.yaml`**. Modify this file to define the settings for your next experiment (e.g., model size, image size, epochs).

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

Data should be placed in `raw_data/train/` for training/validation sets and `raw_data/test/` for the final test set. The following structures are supported:

1.  **CVAT YOLO Export:** Place the entire exported folder (containing `data.yaml`, `images/`, `labels/`, `train.txt`) directly into `raw_data/train/` or `raw_data/test/`.

2.  **Standard YOLO Format:** Create a subfolder (e.g., `my_dataset`) inside `raw_data/train/`. Within that subfolder, place your images in an `images/` directory and corresponding labels in a `labels/` directory.

3.  **Scene-Based Structure (Test Set Only):** For scene-specific evaluation, structure your test data into subfolders where each subfolder name represents a scene (e.g., 'Wolfsburg', 'Carmel').
    *   **Example:**
        ```
        raw_data/
        └── test/
            ├── WOB-Testset/  # Scene 1
            │   ├── images/
            │   └── labels/
            └── Carmel-Testset/ # Scene 2
                ├── images/
                └── labels/
        ```
