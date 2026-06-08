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

Note: `dvc exp run` creates experiment commits. If Git complains about missing user name/email, configure them (globally or in this repo):

```bash
git config --global user.name "Your Name"
git config --global user.email "you@example.com"
```

Or only for this repo:

```bash
git config user.name "Your Name"
git config user.email "you@example.com"
```

DVC access note:

- The checked-in DVC remote for this repo uses SSH.
- If you have access, `poetry run dvc pull` works as documented below.
- If you do not have access, skip `poetry run dvc pull`, provide your own `raw_data/`, and use the local workflow below.

## Quick Start

Pick one workflow. This repo is designed to be reproducible via its DVC remote (dataset + promoted baseline). If you don't have remote access you can still run the pipeline locally on your own data, but you won't get baseline comparison against this repo's promoted baseline.

### Recommended: reproduce the promoted project (requires DVC remote access)

Use this when you want the repo as promoted in Git and DVC:

```bash
poetry install
poetry run dvc pull raw_data
poetry run dvc pull models/current_best/best.pt
poetry run dvc exp run
```

### Without DVC remote access (local data, evaluation only)

Use this if you do not have access to this repo's DVC remote but still want to run `dvc exp run` end-to-end on your own data.

1. Remove remote DVC pointers from your local checkout if they are present:

```bash
rm -f raw_data.dvc models/current_best/best.pt.dvc
poetry run dvc remote remove dvc-hetzner
```

2. Put your training data under `raw_data/train/`.
3. For evaluation, either provide `raw_data/test/` or set a non-zero test split (example below).

Run the full pipeline with a local baseline path (this disables baseline comparison):

```bash
poetry run dvc exp run -n local \
  -S evaluation.baseline_weights_path=models/local_only/best.pt \
  -S prepare.test_split=0.2
```

Notes:

- This does not require this repo's DVC remote, but some backends may still download upstream pretrained assets (see `models_defaults.<backend>.allow_download`).
- `raw_data.dvc` and `models/current_best/best.pt.dvc` belong to this promoted project. If they remain in a no-remote checkout, DVC may still try to restore those remote-tracked files during `dvc exp run` or `dvc exp apply`.
- `models/local_only/best.pt` is intentionally treated as “no baseline” (no nearby `metadata.yaml`), so evaluation runs on the trained model only.
- Don’t drop a real weights file into `models/local_only/best.pt` unless you also add matching `metadata.yaml` next to it.
- If `raw_data/test/` contains images, it is used as the test set and `prepare.test_split` is ignored. If `raw_data/test/` is absent, set `prepare.test_split > 0` for the full pipeline.
- Empty `raw_data/train/` or split/subset settings that leave no training images fail before training starts.

If you later want baseline comparison against the promoted baseline in this repo, pull it explicitly and rerun without the override:

```bash
poetry run dvc pull models/current_best/best.pt
poetry run dvc exp run
```

## Baseline Behavior (why `dvc pull models/current_best/best.pt` matters)

This repo has a promoted baseline at `models/current_best/`. The baseline metadata is committed to Git (so everyone agrees what the baseline *is*), while the weights file is tracked via DVC.

- If `models/current_best/metadata.yaml` exists, evaluation expects the corresponding weights file to be present locally (pull it via DVC). If it’s missing, evaluation fails loudly so you don’t accidentally compare against “nothing”.
- On a fresh clone, the pipeline creates 0-byte placeholders for configured weight paths so DVC stage dependencies are valid. Those placeholders are not usable weights; runtime code treats empty files as missing.

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
