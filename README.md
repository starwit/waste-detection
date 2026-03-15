# Waste Detection

This repository contains the `waste-detection` training project. It uses DVC to track datasets, experiments, and the promoted comparison baseline, while the reusable training backend lives in the external `object-detector-trainer` package.

Project entrypoints:

- `python -m train` delegates to `object-detector-trainer` with project-local defaults for `--workspace-root` and `--config`.
- `dvc exp run` executes the project pipeline stages defined in `dvc.yaml`.

Supported backends in this project:

- Ultralytics YOLO
- RF-DETR
- RTMDet

RTMDet/OpenMMLab note:

- This project installs `object-detector-trainer` with RTMDet extras enabled.
- If your local OpenMMLab runtime fails to install or import cleanly, use Python 3.11: `poetry env use 3.11`.

## Testing

There are two testing layers and they intentionally cover different things.

Project-level tests in this repo:

- `poetry run pytest`
  Fast contract tests for config resolution, wrapper behavior, and offline pipeline contracts.
- `poetry run pytest --heavy`
  Heavy project integration tests for DVC stage wiring, fresh-clone behavior, rerun invalidation, baseline semantics, and project outputs.
  These tests run the project pipeline stages with lightweight backend stubs where appropriate so they stay deterministic and do not depend on downloading real model assets.

Trainer-level tests in `object-detector-trainer`:

- Real backend-specific one-epoch contracts live in the trainer repo, not in this project repo.
- Run those from an `object-detector-trainer` checkout when you change trainer behavior.
- Those heavy trainer tests call real bootstrap directly.
- The first heavy trainer run may download backend assets; later runs reuse the shared cache under `models/pretrained/`.
- The trainer repo uses editable `pip` installs, not Poetry commands.

## Clone And Setup

Typical fresh-clone setup:

```bash
poetry install
```

## Common States

Use the path that matches your repo state.

### 1. Fresh clone of the waste-detection project

Use this when you want the repo as promoted in Git and DVC:

```bash
poetry install
dvc pull raw_data
dvc pull models/current_best/best.pt
```

Then run:

```bash
dvc exp run
```

### 2. Fresh clone, but you want to train on your own local data

Use this when you want to replace the tracked dataset with new local inputs:

1. Add or replace files under `raw_data/train/` and optionally `raw_data/test/`.
2. Run `dvc add raw_data` after the dataset is in place.
3. Pull the promoted baseline before running the full pipeline:

```bash
dvc pull models/current_best/best.pt
```

Then run:

```bash
dvc exp run
```

### 3. Fresh clone, but you only want to train and not evaluate yet

This is possible without pulling the promoted baseline as long as training is not trying to fine-tune from it:

```bash
python -m train --stage bootstrap
python -m train --stage prepare
python -m train --stage train
```

This does not make the full project pipeline ready. `evaluate` and `dvc exp run` require the promoted baseline weights in this repo.

Why the baseline pull matters:

- This repo already contains promoted baseline metadata at `models/current_best/metadata.yaml`.
- Because that metadata exists, evaluation treats the baseline as promoted and requires the matching local weights file.
- A fresh clone without `models/current_best/best.pt` can run `train` when fine-tuning is disabled, but `evaluate` and therefore full `dvc exp run` will fail until the baseline weights are pulled locally.

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
dvc add raw_data
git add raw_data.dvc
git commit -m "Update training data"
dvc push
git push
```

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
dvc exp run -S train.finetune.enabled=true -S train.finetune.weights=models/current_best/best.pt
```

Run the full DVC experiment pipeline:

```bash
dvc exp run -n "my-experiment"
```

Override parameters for a one-off experiment:

```bash
dvc exp run -n "test-smaller-batch" -S train.batch_size=4
```

Manual stage execution:

```bash
python -m train --stage bootstrap
python -m train --stage prepare
python -m train --stage train
python -m train --stage evaluate
```

Or run the full flow directly:

```bash
python -m train --stage all
```

Review local experiment results:

```bash
dvc exp show --sort-by metrics/fitness --sort-order desc
```

Promote an experiment into your workspace:

```bash
dvc exp apply <experiment-name>
```

## Promoting A New Baseline

To make a trained run the new comparison baseline:

```bash
python tools/export_baseline.py --run-dir runs/<experiment_name>
```

Then track the promoted artifacts explicitly:

```bash
dvc add models/current_best/best.pt
dvc push
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
