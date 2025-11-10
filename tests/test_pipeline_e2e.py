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

import csv
from pathlib import Path

import pytest

from tests.pipeline_test_utils import build_args, create_minimal_dataset, write_params_yaml
from tests.ultralytics_stub import StubYOLO
from yolov8_training.train_pipeline import run_prepare_stage, run_train_eval_stage




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
    create_minimal_dataset(workspace)
    write_params_yaml(workspace, {"data": {"dataset_name": dataset_name}})

    args = build_args(dataset_name)

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
    create_minimal_dataset(workspace)
    write_params_yaml(workspace, {"data": {"dataset_name": dataset_name}})

    args = build_args(dataset_name)
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
    (baseline_dir / "best.pt").write_bytes(b"stub-weights")

    # Mirrors a repo where baseline weights have been tracked via DVC/export script.
    create_minimal_dataset(workspace)
    write_params_yaml(
        workspace,
        {
            "data": {"dataset_name": dataset_name},
            "train": {"pretrained_model_path": str(baseline_dir / "best.pt")},
        },
    )

    args = build_args(dataset_name)

    run_prepare_stage(args)
    run_train_eval_stage(args)

    _assert_results_exist(workspace, dataset_name)

    assert str(baseline_dir / "best.pt") in StubYOLO.recorded_models


def test_pipeline_finetune_missing_weights_falls_back(stubbed_pipeline: StubYOLO, capsys):
    """Fine-tune mode with missing weights should warn and continue via fallback."""

    workspace = Path.cwd()
    dataset_name = "e2e_dataset"
    # Enable finetune mode but omit weights to ensure warning + fallback path is exercised.
    create_minimal_dataset(workspace)
    write_params_yaml(
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

    args = build_args(dataset_name)

    run_prepare_stage(args)
    run_train_eval_stage(args)

    _assert_results_exist(workspace, dataset_name)

    stdout = capsys.readouterr().out
    assert "Falling back to training from scratch" in stdout
    assert any(model.startswith("yolov8") for model in StubYOLO.recorded_models)
