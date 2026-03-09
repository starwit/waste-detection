from __future__ import annotations

"""Project-level integration tests for the train wrapper + trainer-core API.

These tests intentionally avoid relying on the repo's real params.yaml, because
projects are expected to customize it freely. Instead, they build a small,
project-shaped params.yaml in a temp workspace and run the full pipeline with
monkeypatched backends.
"""

import csv
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from object_detector_trainer.cli import run_all_stages
from tests.pipeline_test_utils import (
    BASE_PARAMS,
    create_baseline_artifact,
    create_local_yolo_checkpoint,
    create_minimal_dataset,
    write_params_yaml,
)
from tests.ultralytics_stub import StubYOLO

_REQUIRED_METRIC_KEYS = (
    "precision",
    "recall",
    "map",
    "map50",
    "fitness",
    "f1_score",
    "ms_per_frame",
)

# Representative model per backend from BASE_PARAMS.
_BACKEND_CASES: list[tuple[str, str]] = [
    ("yolov8n", "yolo"),
    ("rtmdet-tiny", "rtmdet"),
    ("rfdetr-nano", "rfdetr"),
]


class _ProjectContractBox:
    def __init__(self) -> None:
        self.ap_class_index = np.array([0])
        self.p = np.array([0.5])
        self.r = np.array([0.6])
        self._ap50 = np.array([0.4])
        self._ap = np.array([0.3])

    @property
    def ap50(self):
        return self._ap50

    @property
    def ap(self):
        return self._ap


class _ProjectContractValMetrics:
    def __init__(self) -> None:
        self.speed = {"preprocess": 0.5, "inference": 1.0, "postprocess": 0.5}
        self.results_dict = {
            "metrics/precision(B)": 0.5,
            "metrics/recall(B)": 0.6,
            "metrics/mAP50(B)": 0.4,
            "metrics/mAP50-95(B)": 0.3,
            "metrics/f1(B)": 0.5454545,
        }
        self.fitness = 0.35
        self.box = _ProjectContractBox()


class _ProjectContractModel:
    """Stub satisfying the evaluate stage's model interface for all backends."""

    def __init__(self, *, model_name: str, model_backend: str, image_size: int = 320) -> None:
        self.model_name = model_name
        self.model_backend = model_backend
        self.model = SimpleNamespace(yaml={"model_name": model_name})
        self.trainer = None
        self.resolution = int(image_size)
        self.class_names = {0: "waste"}

    def val(self, **kwargs):
        return _ProjectContractValMetrics()

    def predict(self, *args, **kwargs):
        class _Result:
            boxes = []

            @staticmethod
            def plot():
                return np.zeros((8, 8, 3), dtype=np.uint8)

        return [_Result()]


def _patch_project_trainers(monkeypatch: pytest.MonkeyPatch, workspace: Path) -> None:
    def _mk_runner(backend: str):
        def _runner(*args, **kwargs):
            run_dir = workspace / "runs" / backend / f"{backend}-contract"
            (run_dir / "weights").mkdir(parents=True, exist_ok=True)
            (run_dir / "weights" / "best.pt").write_bytes(b"trained-stub")
            model = _ProjectContractModel(
                model_name=f"{backend}-contract", model_backend=backend, image_size=320
            )
            return model, run_dir, f"{backend}-contract", 320, 1

        return _runner

    monkeypatch.setattr("object_detector_trainer.backends.yolo.train_backend", _mk_runner("yolo"))
    monkeypatch.setattr("object_detector_trainer.backends.rfdetr.train_backend", _mk_runner("rfdetr"))
    monkeypatch.setattr("object_detector_trainer.backends.rtmdet.train_backend", _mk_runner("rtmdet"))
    monkeypatch.setattr("object_detector_trainer.pipeline.model_state._load_yolo_model", StubYOLO)
    monkeypatch.setattr(
        "object_detector_trainer.evaluation.visual_comparison.generate_side_by_side_comparisons",
        lambda *a, **k: None,
    )
    monkeypatch.setattr(
        "object_detector_trainer.evaluation.scene_metrics.calculate_scene_metrics",
        lambda *a, **k: {},
    )
    monkeypatch.setattr(
        "object_detector_trainer.plugins.replay.build_or_update_replay_set",
        lambda *a, **k: None,
    )


@pytest.fixture
def project_workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)
    StubYOLO.workspace = tmp_path
    StubYOLO.recorded_models = []
    StubYOLO.raise_on_official = False
    return tmp_path


def _project_contract_args(*, dataset_name: str, model: str) -> SimpleNamespace:
    return SimpleNamespace(
        stage=None,
        seed=42,
        dataset_name=dataset_name,
        model=model,
        val_split=0.5,
        test_split=0.0,
        recreate_dataset=True,
        augment_multiplier=1,
        folder_subset=None,
        set=[],
        config="params.yaml",
    )


def _write_project_contract_params(workspace: Path, *, dataset_name: str, model: str) -> None:
    baseline_path = create_baseline_artifact(workspace)
    model_cfg = dict(BASE_PARAMS["models"][model])

    common: dict = {
        "data": {
            "dataset_name": dataset_name,
            "custom_classes": ["waste"],
            "use_coco_classes": False,
            "class_mapping": {"waste": ["waste"]},
        },
        "train": {"model": model, "epochs": 1, "batch_size": 1, "image_size": 320},
        "evaluation": {"baseline_weights_path": str(baseline_path)},
        "models": {model: model_cfg},
    }

    write_params_yaml(workspace, common)
    if str(model_cfg["backend"]).strip().lower() == "yolo":
        create_local_yolo_checkpoint(workspace, checkpoint_path=str(model_cfg["checkpoint"]))


@pytest.mark.parametrize(
    ("model_key", "expected_backend"),
    _BACKEND_CASES,
)
def test_project_pipeline_contract_per_backend(
    project_workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
    model_key: str,
    expected_backend: str,
) -> None:
    """Full pipeline (prepare→train→evaluate) works with project-shaped params per backend."""
    dataset_name = f"project-contract-{model_key}"
    create_minimal_dataset(project_workspace)
    _write_project_contract_params(project_workspace, dataset_name=dataset_name, model=model_key)
    _patch_project_trainers(monkeypatch, project_workspace)

    args = _project_contract_args(dataset_name=dataset_name, model=model_key)
    run_all_stages(args)

    metrics_path = project_workspace / "metrics.json"
    assert metrics_path.exists(), "metrics.json must exist after pipeline run"
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    for key in _REQUIRED_METRIC_KEYS:
        assert key in payload, f"Missing metric key: {key!r}"
        assert math.isfinite(float(payload[key])), f"Metric {key!r} must be finite"

    results_csv = project_workspace / "results_comparison" / "results.csv"
    assert results_csv.exists(), "results_comparison/results.csv must exist"
    with open(results_csv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert rows, "results.csv must contain at least one row"

    last_result_path = project_workspace / "runs" / ".last_train_result.json"
    assert last_result_path.exists(), "runs/.last_train_result.json must exist"
    last_result = json.loads(last_result_path.read_text(encoding="utf-8"))
    actual_backend = last_result["reload_metadata"]["model_backend"]
    assert actual_backend == expected_backend, (
        f"Expected model_backend={expected_backend!r}, got {actual_backend!r}"
    )
