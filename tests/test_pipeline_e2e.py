"""End-to-end pipeline scenarios mirroring a freshly cloned repository.

Each test builds a tiny synthetic dataset inside ``tmp_path``, runs the prepare
stage, and executes the training/evaluation stage. We assert that expected
artifacts (datasets, results CSV, metrics) are created and that fallbacks work
correctly.

Tests rely on a lightweight YOLO stub so we can exercise orchestration logic
without triggering real Ultralytics downloads or GPU-heavy training, keeping
the suite fast, deterministic, and CI-friendly.
"""

from __future__ import annotations

import copy
import csv
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import yaml

from yolov8_training.train_pipeline import run_prepare_stage, run_train_eval_stage


BASE_PARAMS = {
    "data": {
        "dataset_name": "waste-detection",
        "experiment_name": "waste-detection",
        "custom_classes": ["waste"],
        "use_coco_classes": False,
    },
    "prepare": {
        "val_split": 0.5,
        "test_split": 0.0,
        "augment_multiplier": 1,
        "folder_subsets": {},
    },
    "train": {
        "model_size": "n",
        "image_size": 320,
        "epochs": 1,
        "batch_size": 1,
        "finetune_mode": False,
        "pretrained_model_path": "models/current_best/best.pt",
        "finetune_lr": 0.0001,
        "finetune_epochs": 1,
        "freeze_backbone": False,
    },
    "evaluation": {
        "baseline_weights_path": "models/current_best/best.pt",
    },
}


class StubValMetrics:
    def __init__(self):
        self.speed = {"inference": 1.0}
        self.results_dict = {
            "metrics/precision(B)": 0.5,
            "metrics/recall(B)": 0.6,
            "metrics/mAP50(B)": 0.4,
            "metrics/mAP50-95(B)": 0.3,
        }
        self.fitness = 0.42


class StubYOLO:
    """Minimal YOLO stand-in used to avoid real downloads/training in tests."""

    recorded_models: list[str] = []
    raise_on_official: bool = False
    workspace: Path | None = None

    def __init__(self, model_path: str | Path):
        model_str = str(model_path)
        workspace = StubYOLO.workspace or Path.cwd()
        if StubYOLO.raise_on_official and model_str.startswith("yolov8"):
            raise RuntimeError(f"Simulated offline failure loading {model_str}")

        StubYOLO.recorded_models.append(model_str)
        self.model_source = model_str
        self.model_name = Path(model_str).stem or "stub"
        self._workspace = workspace
        self.trainer = None
        # Mimic internal model meta used by evaluate.mean_table fallback logic
        self.model = SimpleNamespace(yaml={"model_name": "yolov8n"})

    # ---- Training / evaluation API surface ----
    def train(self, **kwargs):
        project = kwargs.get("project", "runs")
        name = kwargs.get("name") or "train"
        save_dir = self._workspace / project / name
        save_dir.mkdir(parents=True, exist_ok=True)
        (save_dir / "weights").mkdir(parents=True, exist_ok=True)
        (save_dir / "weights" / "best.pt").touch()
        self.trainer = SimpleNamespace(save_dir=str(save_dir))
        return SimpleNamespace(save_dir=str(save_dir))

    def val(self, **kwargs):  # noqa: D401 - Tiny stub
        return StubValMetrics()

    def predict(self, *args, **kwargs):  # noqa: D401 - Tiny stub
        class _Result:
            boxes = []

            @staticmethod
            def plot():
                return np.zeros((8, 8, 3), dtype=np.uint8)

        return [_Result()]


def _merge_dict(target: dict, overrides: dict) -> dict:
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            target[key] = _merge_dict(target[key], value)
        else:
            target[key] = value
    return target


def _write_params_yaml(workspace: Path, overrides: dict | None = None) -> dict:
    params = _merge_dict(copy.deepcopy(BASE_PARAMS), overrides or {})
    with open(workspace / "params.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(params, f, sort_keys=False)
    return params


def _create_minimal_dataset(base_dir: Path) -> None:
    """Create a tiny dataset with two training folders and one held-out test scene."""

    # Training folder 1
    images_dir = base_dir / "raw_data" / "train" / "source1" / "images"
    labels_dir = base_dir / "raw_data" / "train" / "source1" / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    img1 = np.full((128, 128, 3), 100, dtype=np.uint8)
    cv2.rectangle(img1, (32, 32), (96, 96), (255, 255, 255), -1)
    cv2.imwrite(str(images_dir / "img1.jpg"), img1)
    with open(labels_dir / "img1.txt", "w", encoding="utf-8") as f:
        f.write("0 0.5 0.5 0.5 0.5\n")

    img2 = np.full((128, 128, 3), 150, dtype=np.uint8)
    cv2.imwrite(str(images_dir / "img2.jpg"), img2)
    with open(labels_dir / "img2.txt", "w", encoding="utf-8") as f:
        f.write("0 0.5 0.5 0.3 0.3\n")

    # Training folder 2
    images_dir2 = base_dir / "raw_data" / "train" / "source2" / "images"
    labels_dir2 = base_dir / "raw_data" / "train" / "source2" / "labels"
    images_dir2.mkdir(parents=True, exist_ok=True)
    labels_dir2.mkdir(parents=True, exist_ok=True)
    img3 = np.full((128, 128, 3), 60, dtype=np.uint8)
    cv2.line(img3, (0, 0), (127, 127), (255, 255, 255), 3)
    cv2.imwrite(str(images_dir2 / "img3.jpg"), img3)
    with open(labels_dir2 / "img3.txt", "w", encoding="utf-8") as f:
        f.write("0 0.5 0.5 0.2 0.2\n")

    # Held-out test scene
    test_images_dir = base_dir / "raw_data" / "test" / "sourceT" / "images"
    test_labels_dir = base_dir / "raw_data" / "test" / "sourceT" / "labels"
    test_images_dir.mkdir(parents=True, exist_ok=True)
    test_labels_dir.mkdir(parents=True, exist_ok=True)
    imgT = np.full((128, 128, 3), 80, dtype=np.uint8)
    cv2.circle(imgT, (64, 64), 20, (255, 255, 255), -1)
    cv2.imwrite(str(test_images_dir / "test1.jpg"), imgT)
    with open(test_labels_dir / "test1.txt", "w", encoding="utf-8") as f:
        f.write("0 0.5 0.5 0.25 0.25\n")


def _build_args(dataset_name: str, overrides: dict | None = None) -> SimpleNamespace:
    args = {
        "stage": None,
        "seed": 42,
        "dataset_name": dataset_name,
        "model_size": "n",
        "image_size": 320,
        "epochs": 1,
        "batch_size": 1,
        "val_split": 0.5,
        "test_split": 0.0,
        "recreate_dataset": True,
        "augment_multiplier": 1,
        "folder_subset": None,
    }
    if overrides:
        args.update(overrides)
    return SimpleNamespace(**args)


@pytest.fixture
def stubbed_pipeline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Prepare a clean workspace and replace Ultralytics YOLO with a deterministic stub."""

    monkeypatch.chdir(tmp_path)
    StubYOLO.workspace = tmp_path
    StubYOLO.recorded_models = []
    StubYOLO.raise_on_official = False

    # Replace YOLO constructors with the stub
    # Swap in the stub everywhere the pipeline imports YOLO so training/eval stays local.
    monkeypatch.setattr("yolov8_training.train_pipeline.YOLO", StubYOLO)
    monkeypatch.setattr("yolov8_training.utils.evaluate.YOLO", StubYOLO)

    # Silence heavy post-processing during tests
    # Skip expensive visualisations/scene scanning; return deterministic metrics instead.
    monkeypatch.setattr(
        "yolov8_training.train_pipeline.generate_side_by_side_comparisons",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "yolov8_training.utils.evaluate.generate_side_by_side_comparisons",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "yolov8_training.utils.evaluate.calculate_scene_metrics",
        lambda *args, **kwargs: {"scene_sourceT_fitness": 0.75},
    )

    return StubYOLO


def _assert_results_exist(base_dir: Path, dataset_name: str, scene_suffix: str = "sourceT") -> None:
    dataset_root = base_dir / "datasets" / dataset_name
    assert dataset_root.exists()

    results_csv = base_dir / "results_comparison" / "results.csv"
    assert results_csv.exists()

    with open(results_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        header = reader.fieldnames or []

    col_name = f"scene_{scene_suffix}_fitness"
    assert col_name in header
    assert rows, "results.csv should contain at least one row"

    for row in rows:
        val_str = (row.get(col_name) or "").strip()
        if val_str and val_str != "-":
            assert float(val_str) >= 0.0

    assert (base_dir / "metrics.json").exists()


def test_pipeline_fresh_clone_uses_fallback(stubbed_pipeline: StubYOLO):
    """Fresh clone with missing baseline weights should succeed via fallback checkpoints."""

    workspace = Path.cwd()
    dataset_name = "e2e_dataset"
    # Recreate the minimal data tree and defaults a new contributor would see.
    _create_minimal_dataset(workspace)
    _write_params_yaml(workspace, {"data": {"dataset_name": dataset_name}})

    args = _build_args(dataset_name)

    run_prepare_stage(args)
    run_train_eval_stage(args)

    _assert_results_exist(workspace, dataset_name)

    # Ensure fallback checkpoint was requested at least once
    assert any(model.startswith("yolov8") for model in StubYOLO.recorded_models)


def test_pipeline_offline_error_when_no_weights(stubbed_pipeline: StubYOLO):
    """If official checkpoints cannot be loaded, raise the explicit fallback error."""

    workspace = Path.cwd()
    dataset_name = "e2e_dataset"
    # Same fresh clone setup, but simulate a network-less environment via the stub.
    _create_minimal_dataset(workspace)
    _write_params_yaml(workspace, {"data": {"dataset_name": dataset_name}})

    args = _build_args(dataset_name)
    run_prepare_stage(args)

    StubYOLO.raise_on_official = True

    with pytest.raises(RuntimeError, match="Failed to initialize training without local weights"):
        run_train_eval_stage(args)


def test_pipeline_uses_local_baseline_when_available(stubbed_pipeline: StubYOLO):
    """When promoted baseline weights exist, they should be loaded instead of COCO."""

    workspace = Path.cwd()
    dataset_name = "e2e_dataset"
    baseline_dir = workspace / "models" / "current_best"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    (baseline_dir / "best.pt").touch()

    # Mirrors a repo where baseline weights have been tracked via DVC/export script.
    _create_minimal_dataset(workspace)
    _write_params_yaml(
        workspace,
        {
            "data": {"dataset_name": dataset_name},
            "train": {"pretrained_model_path": str(baseline_dir / "best.pt")},
        },
    )

    args = _build_args(dataset_name)

    run_prepare_stage(args)
    run_train_eval_stage(args)

    _assert_results_exist(workspace, dataset_name)

    assert str(baseline_dir / "best.pt") in StubYOLO.recorded_models


def test_pipeline_finetune_missing_weights_falls_back(stubbed_pipeline: StubYOLO, capsys):
    """Fine-tune mode with missing weights should warn and continue via fallback."""

    workspace = Path.cwd()
    dataset_name = "e2e_dataset"
    # Enable finetune mode but omit weights to ensure warning + fallback path is exercised.
    _create_minimal_dataset(workspace)
    _write_params_yaml(
        workspace,
        {
            "data": {"dataset_name": dataset_name},
            "train": {
                "finetune_mode": True,
                "pretrained_model_path": "models/current_best/missing.pt",
                "finetune_epochs": 1,
            },
        },
    )

    args = _build_args(dataset_name)

    run_prepare_stage(args)
    run_train_eval_stage(args)

    _assert_results_exist(workspace, dataset_name)

    stdout = capsys.readouterr().out
    assert "Falling back to training from scratch" in stdout
    assert any(model.startswith("yolov8") for model in StubYOLO.recorded_models)
