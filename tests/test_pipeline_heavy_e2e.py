"""Heavy end-to-end tests that run real backend training.

These tests are opt-in via ``pytest --heavy`` (see ``tests/conftest.py``).
They validate full prepare + train/eval execution with real model backends.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from tests.pipeline_test_utils import build_args, create_minimal_dataset, write_params_yaml
from yolov8_training.train_pipeline import run_prepare_stage, run_train_eval_stage

pytestmark = pytest.mark.heavy


def _require_repo_weight(filename: str) -> Path:
    """Resolve a required local checkpoint from repo root, or skip with context."""
    repo_root = Path(__file__).resolve().parents[1]
    candidate = repo_root / filename
    if not candidate.exists() or candidate.stat().st_size == 0:
        pytest.skip(
            f"Heavy test prerequisite missing: {candidate}. "
            f"Provide local '{filename}' before running --heavy tests."
        )
    return candidate


def _assert_common_pipeline_artifacts(workspace: Path, dataset_name: str) -> None:
    dataset_root = workspace / "datasets" / dataset_name
    assert dataset_root.exists()

    results_csv = workspace / "results_comparison" / "results.csv"
    assert results_csv.exists()
    with open(results_csv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert rows, "results.csv should contain at least one row"

    metrics_path = workspace / "metrics.json"
    assert metrics_path.exists()
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    for key in ("precision", "recall", "map", "map50", "fitness", "f1_score"):
        assert key in metrics


def test_heavy_e2e_yolo_one_epoch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Run real YOLO training/evaluation for one epoch on a tiny synthetic dataset."""
    monkeypatch.chdir(tmp_path)

    yolo_ckpt = _require_repo_weight("yolov8n.pt")
    dataset_name = "heavy_e2e_yolo"

    create_minimal_dataset(tmp_path)
    write_params_yaml(
        tmp_path,
        {
            "data": {"dataset_name": dataset_name},
            "train": {
                "model": "yolov8n",
                "epochs": 1,
                "batch_size": 1,
                "image_size": 128,
            },
            "models": {
                "yolov8n": {
                    "backend": "yolo",
                    "checkpoint": str(yolo_ckpt),
                }
            },
            "evaluation": {
                "baseline_weights_path": str(yolo_ckpt),
            },
        },
    )

    args = build_args(dataset_name)
    run_prepare_stage(args)
    run_train_eval_stage(args)

    _assert_common_pipeline_artifacts(tmp_path, dataset_name)
    assert list((tmp_path / "runs").glob("**/weights/best.pt")), "Expected YOLO weights/best.pt"


def test_heavy_e2e_rfdetr_one_epoch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Run real RF-DETR training/evaluation for one epoch on a tiny synthetic dataset."""
    monkeypatch.chdir(tmp_path)

    yolo_ckpt = _require_repo_weight("yolov8n.pt")
    rfdetr_ckpt = _require_repo_weight("rf-detr-nano.pth")
    dataset_name = "heavy_e2e_rfdetr"

    create_minimal_dataset(tmp_path)

    # RF-DETR can fail to create checkpoint_best_regular/ema when mAP stays at 0.0
    # on tiny datasets, but still writes checkpoint.pth files. Patch the final
    # copy step to use a fallback checkpoint so this test validates our pipeline
    # behavior instead of failing on this upstream edge case.
    import rfdetr.main as rfdetr_main

    copy2_original = rfdetr_main.shutil.copy2

    def _copy2_with_best_checkpoint_fallback(src, dst, *args, **kwargs):
        src_path = Path(src)
        if src_path.name in {"checkpoint_best_regular.pth", "checkpoint_best_ema.pth"} and not src_path.exists():
            fallback_candidates = [src_path.parent / "checkpoint.pth"]
            fallback_candidates.extend(sorted(src_path.parent.glob("checkpoint*.pth"), reverse=True))
            for candidate in fallback_candidates:
                if candidate.exists() and candidate.stat().st_size > 0:
                    return copy2_original(candidate, dst, *args, **kwargs)
        return copy2_original(src, dst, *args, **kwargs)

    monkeypatch.setattr(rfdetr_main.shutil, "copy2", _copy2_with_best_checkpoint_fallback)

    write_params_yaml(
        tmp_path,
        {
            "data": {"dataset_name": dataset_name},
            "train": {
                "model": "rfdetr-nano",
                "image_size": 128,
                "epochs": 1,
                "batch_size": 1,
            },
            "models": {
                "rfdetr-nano": {
                    "backend": "rfdetr",
                    "variant": "nano",
                    "resolution": 128,
                    "epochs": 1,
                    "batch_size": 1,
                    "grad_accum_steps": 1,
                    "pretrain_weights": str(rfdetr_ckpt),
                    "extra_train_kwargs": {
                        "checkpoint_interval": 1,
                        "run_test": False,
                    },
                }
            },
            "evaluation": {
                "baseline_weights_path": str(yolo_ckpt),
            },
        },
    )

    args = build_args(dataset_name)
    run_prepare_stage(args)
    run_train_eval_stage(args)

    _assert_common_pipeline_artifacts(tmp_path, dataset_name)

    runs_rfdetr = tmp_path / "runs" / "rfdetr"
    assert runs_rfdetr.exists()
    assert list(runs_rfdetr.glob("**/weights/best.pt")), "Expected RF-DETR weights/best.pt"
