# Waste Detection

This repository contains the `waste-detection` training project. It uses DVC to track datasets, experiments, and the promoted comparison baseline, while the reusable training backend lives in the external `object-detector-trainer` package.

Project workflow:

- `poetry run dvc exp run` is the documented project workflow and executes the stages defined in `dvc.yaml`.
- `train.py` is the thin project wrapper invoked by the DVC stages.

Supported backends in this project:

- Ultralytics YOLO
- RF-DETR
- RTMDet

RTMDet/OpenMMLab note:

This project installs `object-detector-trainer` with RTMDet extras enabled.

RTMDet requires full `mmcv` ops (`mmcv`, not `mmcv-lite`). Two supported install methods:

Method 1: OpenMMLab prebuilt wheels

```bash
poetry run mim install mmcv==2.1.0
poetry run python -c "import mmcv._ext"
```

Method 2: build from source (requires a working CUDA toolkit and a C++ compiler)

```bash
MMCV_WITH_OPS=1 poetry run pip install "mmcv==2.1.0" --no-binary=mmcv --no-build-isolation --no-cache-dir
poetry run python -c "import mmcv._ext"
```

Recommended Python for RTMDet/OpenMMLab: 3.11 (`poetry env use 3.11`).

## Testing

There are two testing layers and they intentionally cover different things.

Project-level tests in this repo:

- `poetry run pytest`
  Fast contract tests for config resolution, wrapper behavior, and offline pipeline contracts.
- `poetry run pytest --heavy`
  Heavy project integration tests for DVC experiment wiring, fresh-clone behavior, rerun invalidation, baseline semantics, and project outputs.
  These tests execute the project `dvc exp run` workflow inside temporary git-backed workspaces and use lightweight backend stubs so they stay deterministic and do not depend on downloading real model assets.

Trainer-level tests in `object-detector-trainer`:

- Real backend-specific one-epoch contracts live in the trainer repo, not in this project repo.
- Run those from an `object-detector-trainer` checkout when you change trainer behavior.
- Those heavy trainer tests call real bootstrap directly and run one canonical real-training model per supported backend.
- The trainer heavy suite fails if a new backend is added to the supported-backend registry without adding a representative heavy-test case for it.
- The first heavy trainer run may download backend assets; later runs reuse the shared cache under `models/pretrained/<backend>/`.
- The trainer repo uses Poetry for installs and test runs.

## Clone And Setup

Typical fresh-clone setup:

```bash
poetry install
```

DVC access note:

- The checked-in DVC remote for this repo uses SSH.
- If you have access, `poetry run dvc pull` works as documented below.
- If you do not have access, skip `poetry run dvc pull`, provide your own `raw_data/`, and use the local-only training path instead.

## Common States

Use the path that matches your repo state.

### 1. Fresh clone of the waste-detection project

Use this when you want the repo as promoted in Git and DVC:

```bash
poetry install
poetry run dvc pull raw_data
poetry run dvc pull models/current_best/best.pt
```

Then run:

```bash
poetry run dvc exp run
```

### 2. Fresh clone, but you want to train on your own local data

Use this when you want to replace the tracked dataset with new local inputs and train locally:

1. Add or replace files under `raw_data/train/` and optionally `raw_data/test/`.
2. Run the DVC training pipeline up to `train_model`:

```bash
poetry run dvc exp run train_model
```

3. If you want full evaluation or the full `poetry run dvc exp run` workflow, pull the promoted baseline first:

```bash
poetry run dvc pull models/current_best/best.pt
```

Then run the full experiment pipeline:

```bash
poetry run dvc exp run
```

Notes:

- `poetry run dvc add raw_data` is only needed if you want to version and publish your new dataset through DVC.
- It is not required for a local training run.

### 3. Fresh clone, but you only want to train and not evaluate yet

This is possible without pulling the promoted baseline as long as training is not trying to fine-tune from it:

```bash
poetry run dvc exp run train_model
```

This does not make the full project pipeline ready. `evaluate_model` and the full `poetry run dvc exp run` require the promoted baseline weights in this repo.

Why the baseline pull matters in this repo:

- This repo already contains promoted baseline metadata at `models/current_best/metadata.yaml`.
- Because that metadata exists, evaluation treats the baseline as promoted and requires the matching local weights file.
- A fresh clone without `models/current_best/best.pt` can run `train_model` when fine-tuning is disabled, but `evaluate_model` and therefore the full `poetry run dvc exp run` will fail until the baseline weights are pulled locally.

## Baseline Semantics

The project always keeps `evaluation.baseline_weights_path` configured so DVC can track the path explicitly.

State machine:

- Missing/empty baseline weights and no nearby `metadata.yaml`
  This means “no promoted baseline yet”.
  Evaluation skips baseline comparison.
- Missing/empty baseline weights and nearby `metadata.yaml`
  This means “a promoted baseline exists but is not present locally”.
  Evaluation fails loudly and tells you to fetch it explicitly.
- Non-empty baseline weights plus nearby `metadata.yaml`
  Normal baseline comparison runs.

Fresh-clone DVC behavior:

- The preflight stage `check_optional_weight_deps` creates 0-byte placeholders for optional configured weight paths such as `models/current_best/best.pt` and `models/finetune/best.pt`.
- That placeholder exists only so DVC direct dependencies are valid on fresh clones.
- Runtime code treats empty files as missing; the placeholder is never used as a real model.

## Data Workflow

When you update the dataset for this project:

```bash
poetry run dvc add raw_data
git add raw_data.dvc
git commit -m "Update training data"
poetry run dvc push
git push
```

For a local-only experiment, you can skip `poetry run dvc add`, `poetry run dvc push`, and the Git steps entirely.

Expected raw data layout:

- `raw_data/train/<source>/images/*`
- `raw_data/train/<source>/labels/*`
- `raw_data/test/<source>/images/*`
- `raw_data/test/<source>/labels/*`

## Training Workflow

Project parameters live in `params.yaml`.

Useful configuration rules:

- `train.image_size`, `train.epochs`, and `train.batch_size` are shared defaults.
- `train.finetune.weights` is an optional fine-tune input path and defaults to `models/finetune/best.pt`.
- The promoted comparison baseline stays at `models/current_best/best.pt` and is tracked separately via DVC.
- `models_defaults.<backend>` applies per-backend defaults to every model of that backend.
- `models.<key>` overrides both shared and backend defaults.
- `models.<key>.asset_id` identifies the pretrained asset to bootstrap for the selected backend.

If you want to fine-tune from the promoted baseline, set:

```bash
poetry run dvc exp run -S train.finetune.enabled=true -S train.finetune.weights=models/current_best/best.pt
```

Run the full DVC experiment pipeline:

```bash
poetry run dvc exp run -n "my-experiment"
```

Override parameters for a one-off experiment:

```bash
poetry run dvc exp run -n "test-smaller-batch" -S train.batch_size=4
```

Review local experiment results:

```bash
poetry run dvc exp show --sort-by metrics/fitness --sort-order desc
```

Promote an experiment into your workspace:

```bash
poetry run dvc exp apply <experiment-name>
```

## Promoting A New Baseline

To make a trained run the new comparison baseline:

```bash
poetry run python tools/export_baseline.py --run-dir runs/<experiment_name>
```

Then track the promoted artifacts explicitly:

```bash
poetry run dvc add models/current_best/best.pt
poetry run dvc push
git add models/current_best/best.pt.dvc
git add models/current_best/metadata.yaml
```

If the exported run is RTMDet-based, also commit the copied config:

```bash
git add models/current_best/model_config.py
```

This repo keeps `metadata.yaml` and `model_config.py` in Git, while `best.pt` is tracked via DVC.

## Current Project Context

This project trains the following merged output classes:

- `waste`
- `cigarette`
- `leaves_dense`
- `leaves_sparse`

The raw class mapping is configured in `params.yaml`.

## Training Backend

The reusable training core lives in:

- https://github.com/starwit/object-detector-trainer
