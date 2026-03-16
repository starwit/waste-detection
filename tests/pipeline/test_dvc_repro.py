from __future__ import annotations

import os
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from tests.pipeline_test_utils import (
    create_baseline_artifact,
    create_local_yolo_checkpoint,
    create_minimal_dataset,
    write_params_yaml,
)

pytestmark = pytest.mark.heavy

# These tests exercise project-level DVC experiment wiring and fresh-clone semantics.
# Backend training is stubbed where appropriate so the contracts stay deterministic.


def _install_ultralytics_stub(workspace: Path) -> None:
    """Provide a lightweight Ultralytics stub so DVC runs stay local inside tests."""

    stub_path = workspace / "ultralytics.py"
    stub_path.write_text(
        "from tests.ultralytics_stub import StubYOLO\n\n"
        "YOLO = StubYOLO\n"
    )


def _install_rtmdet_dvc_stub(workspace: Path) -> None:
    """Patch RTMDet bootstrap/train/reload inside DVC subprocesses via sitecustomize."""

    src_path = Path(__file__).resolve().parents[1] / "rtmdet_dvc_sitecustomize.py"
    shutil.copyfile(src_path, workspace / "sitecustomize.py")


def _copy_workspace(src: Path, dst: Path) -> None:
    """Shallow-copy the repo for test runs without heavyweight artifacts."""
    ignore = shutil.ignore_patterns(
        ".git",
        ".dvc",
        ".venv",
        "__pycache__",
        ".pytest_cache",
        "*.pyc",
        "*.pyo",
        "*.pyd",
        "datasets",
        "raw_data",
        "runs",
        "results_comparison",
        "feature_vs_main_diff.txt",
        "repo-to-text_*",
        "metrics.json",
        "WAS_videos",
        "analysis_output",
        "*.pt",
        "*.pth",
        "*.onnx",
        "*.engine",
        "*.mkv",
    )
    shutil.copytree(src, dst, dirs_exist_ok=True, ignore=ignore)


def _make_env(workspace: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = (
        f"{workspace}{os.pathsep}{env.get('PYTHONPATH', '')}".rstrip(os.pathsep)
    )
    env["DVC_NO_ANALYTICS"] = "1"
    python_dir = Path(sys.executable).parent
    env["PATH"] = f"{python_dir}{os.pathsep}{env['PATH']}"
    return env


def _run_dvc(
    workspace: Path,
    env: dict[str, str],
    *args: str,
    check: bool = True,
    capture: bool = False,
) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "dvc", *args],
        check=check,
        cwd=workspace,
        env=env,
        capture_output=capture,
        text=capture,
    )


def _init_dvc_workspace(workspace: Path, env: dict[str, str]) -> None:
    # `dvc exp run` requires a real git repo with at least one commit.
    subprocess.run(["git", "init", "-q"], check=True, cwd=workspace, env=env)
    subprocess.run(
        ["git", "config", "user.email", "wd-tests@example.com"],
        check=True,
        cwd=workspace,
        env=env,
    )
    subprocess.run(
        ["git", "config", "user.name", "Waste Detection Tests"],
        check=True,
        cwd=workspace,
        env=env,
    )
    _run_dvc(workspace, env, "init", check=True)
    subprocess.run(["git", "add", "-A"], check=True, cwd=workspace, env=env)
    subprocess.run(
        ["git", "commit", "-qm", "test workspace bootstrap"],
        check=True,
        cwd=workspace,
        env=env,
    )


def _write_image(path: Path, fill: int) -> None:
    image = np.full((96, 96, 3), fill, dtype=np.uint8)
    cv2.rectangle(image, (24, 24), (72, 72), (255, 255, 255), -1)
    cv2.imwrite(str(path), image)


def _create_class_mapping_dataset(workspace: Path) -> None:
    train_images = workspace / "raw_data" / "train" / "source1" / "images"
    train_labels = workspace / "raw_data" / "train" / "source1" / "labels"
    test_images = workspace / "raw_data" / "test" / "sourceT" / "images"
    test_labels = workspace / "raw_data" / "test" / "sourceT" / "labels"

    for path in (train_images, train_labels, test_images, test_labels):
        path.mkdir(parents=True, exist_ok=True)

    train_specs = [
        ("train_waste.jpg", "0 0.5 0.5 0.4 0.4\n", 70),
        ("train_cigarette.jpg", "1 0.5 0.5 0.3 0.3\n", 120),
        ("train_cigarette_2.jpg", "1 0.4 0.4 0.2 0.2\n", 160),
    ]
    for filename, label_text, fill in train_specs:
        _write_image(train_images / filename, fill)
        (train_labels / f"{Path(filename).stem}.txt").write_text(label_text, encoding="utf-8")

    _write_image(test_images / "test_cigarette.jpg", 210)
    (test_labels / "test_cigarette.txt").write_text(
        "1 0.5 0.5 0.25 0.25\n",
        encoding="utf-8",
    )


def _assert_eval_artifacts(workspace: Path) -> None:
    results_dir = workspace / "results_comparison"
    assert (results_dir / "results.csv").exists()
    assert (results_dir / "results.txt").exists()
    assert (workspace / "metrics.json").exists()
    assert (workspace / "runs" / ".last_train_result.json").exists()


def _load_results_models(workspace: Path) -> list[str]:
    csv_path = workspace / "results_comparison" / "results.csv"
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [str(row.get("MODEL", "")) for row in reader]


def test_dvc_train_model_reruns_when_finetune_weights_param_changes(tmp_path: Path) -> None:
    """Changing finetune weights path in params must rerun `train_model`."""
    repo_root = Path(__file__).resolve().parents[2]
    workspace = tmp_path / "repo"
    _copy_workspace(repo_root, workspace)
    _install_ultralytics_stub(workspace)

    finetune_dir = workspace / "models" / "finetune"
    finetune_dir.mkdir(parents=True, exist_ok=True)
    v1_weights = finetune_dir / "v1.pt"
    v2_weights = finetune_dir / "v2.pt"
    v1_weights.write_bytes(b"weights-v1")
    v2_weights.write_bytes(b"weights-v2")

    dataset_name = "dvc-finetune-rerun"
    create_minimal_dataset(workspace)
    write_params_yaml(
        workspace,
        {
            "data": {"dataset_name": dataset_name},
            "train": {
                "image_size": 320,
                "epochs": 1,
                "batch_size": 1,
                "model": "yolov8n",
                "finetune": {
                    "enabled": True,
                    "weights": str(v1_weights),
                    "epochs": 1,
                },
            },
            "models": {
                "yolov8n": {
                    "backend": "yolo",
                    "asset_id": "yolov8n.pt",
                }
            },
        },
    )
    create_local_yolo_checkpoint(workspace)

    env = _make_env(workspace)
    _init_dvc_workspace(workspace, env)

    _run_dvc(workspace, env, "exp", "run", "train_model", check=True)
    marker = workspace / "runs" / ".last_train_result.json"
    assert marker.exists()
    first_mtime = marker.stat().st_mtime

    write_params_yaml(
        workspace,
        {
            "data": {"dataset_name": dataset_name},
            "train": {
                "image_size": 320,
                "epochs": 1,
                "batch_size": 1,
                "model": "yolov8n",
                "finetune": {
                    "enabled": True,
                    "weights": str(v2_weights),
                    "epochs": 1,
                },
            },
            "models": {
                "yolov8n": {
                    "backend": "yolo",
                    "asset_id": "yolov8n.pt",
                }
            },
        },
    )
    create_local_yolo_checkpoint(workspace)

    _run_dvc(workspace, env, "exp", "run", "train_model", check=True)
    second_mtime = marker.stat().st_mtime
    assert second_mtime > first_mtime


def test_dvc_train_model_succeeds_without_local_baseline_file(tmp_path: Path) -> None:
    """Fresh clone contract: missing promoted baseline must not block `train_model`."""
    repo_root = Path(__file__).resolve().parents[2]
    workspace = tmp_path / "repo"
    _copy_workspace(repo_root, workspace)
    _install_ultralytics_stub(workspace)

    promoted_dir = workspace / "models" / "current_best"
    promoted_dir.mkdir(parents=True, exist_ok=True)
    for filename in ("best.pt", "metadata.yaml"):
        candidate = promoted_dir / filename
        if candidate.exists():
            candidate.unlink()

    create_minimal_dataset(workspace)
    write_params_yaml(
        workspace,
        {
            "data": {"dataset_name": "dvc-no-baseline"},
            "train": {
                "model": "yolov8n",
                "epochs": 1,
                "batch_size": 1,
                "finetune": {
                    "enabled": False,
                    "weights": "models/finetune/best.pt",
                },
            },
            "models": {
                "yolov8n": {
                    "backend": "yolo",
                    "asset_id": "yolov8n.pt",
                }
            },
            "evaluation": {
                "baseline_weights_path": "models/current_best/best.pt",
            },
        },
    )
    create_local_yolo_checkpoint(workspace)

    env = _make_env(workspace)
    _init_dvc_workspace(workspace, env)
    _run_dvc(workspace, env, "exp", "run", "train_model", check=True)

    run_marker = workspace / "runs" / ".last_train_result.json"
    assert run_marker.exists()


def test_dvc_train_model_finetune_requires_existing_weights(tmp_path: Path) -> None:
    """If finetune is enabled and weights are missing, `train_model` must fail clearly."""
    repo_root = Path(__file__).resolve().parents[2]
    workspace = tmp_path / "repo"
    _copy_workspace(repo_root, workspace)
    _install_ultralytics_stub(workspace)

    create_minimal_dataset(workspace)
    write_params_yaml(
        workspace,
        {
            "data": {"dataset_name": "dvc-finetune-missing"},
            "train": {
                "model": "yolov8n",
                "epochs": 1,
                "batch_size": 1,
                "finetune": {
                    "enabled": True,
                    "weights": "models/finetune/missing.pt",
                    "epochs": 1,
                },
            },
            "models": {
                "yolov8n": {
                    "backend": "yolo",
                    "asset_id": "yolov8n.pt",
                }
            },
        },
    )
    create_local_yolo_checkpoint(workspace)

    env = _make_env(workspace)
    _init_dvc_workspace(workspace, env)
    result = _run_dvc(
        workspace,
        env,
        "exp",
        "run",
        "train_model",
        check=False,
        capture=True,
    )
    assert result.returncode != 0
    combined = (result.stdout or "") + "\n" + (result.stderr or "")
    assert "Fine-tuning mode is enabled" in combined


def test_dvc_train_model_does_not_rerun_when_baseline_changes_if_finetune_disabled(
    tmp_path: Path,
) -> None:
    """Promoting or pulling the baseline must not invalidate `train_model` by itself."""
    repo_root = Path(__file__).resolve().parents[2]
    workspace = tmp_path / "repo"
    _copy_workspace(repo_root, workspace)
    _install_ultralytics_stub(workspace)

    baseline_weights = create_baseline_artifact(
        workspace,
        experiment_name="promoted-baseline",
    )

    create_minimal_dataset(workspace)
    write_params_yaml(
        workspace,
        {
            "data": {"dataset_name": "dvc-baseline-decoupled"},
            "train": {
                "model": "yolov8n",
                "epochs": 1,
                "batch_size": 1,
                "finetune": {
                    "enabled": False,
                    "weights": "models/finetune/best.pt",
                },
            },
            "models": {
                "yolov8n": {
                    "backend": "yolo",
                    "asset_id": "yolov8n.pt",
                }
            },
            "evaluation": {
                "baseline_weights_path": str(baseline_weights),
            },
        },
    )
    create_local_yolo_checkpoint(workspace)

    env = _make_env(workspace)
    _init_dvc_workspace(workspace, env)

    _run_dvc(workspace, env, "exp", "run", "train_model", check=True)
    marker = workspace / "runs" / ".last_train_result.json"
    assert marker.exists()
    first_mtime = marker.stat().st_mtime_ns

    baseline_weights.write_bytes(b"updated-baseline-weights")

    _run_dvc(workspace, env, "exp", "run", "train_model", check=True)
    second_mtime = marker.stat().st_mtime_ns
    assert second_mtime == first_mtime


def test_dvc_train_model_finetune_succeeds_when_weights_exist(tmp_path: Path) -> None:
    """If finetune is enabled and weights exist, `train_model` must succeed."""
    repo_root = Path(__file__).resolve().parents[2]
    workspace = tmp_path / "repo"
    _copy_workspace(repo_root, workspace)
    _install_ultralytics_stub(workspace)

    finetune_weights = workspace / "models" / "finetune" / "best.pt"
    finetune_weights.parent.mkdir(parents=True, exist_ok=True)
    finetune_weights.write_bytes(b"stub-finetune-weights")

    create_minimal_dataset(workspace)
    write_params_yaml(
        workspace,
        {
            "data": {"dataset_name": "dvc-finetune-ok"},
            "train": {
                "model": "yolov8n",
                "epochs": 1,
                "batch_size": 1,
                "finetune": {
                    "enabled": True,
                    "weights": str(finetune_weights),
                    "epochs": 1,
                },
            },
            "models": {
                "yolov8n": {
                    "backend": "yolo",
                    "asset_id": "yolov8n.pt",
                }
            },
        },
    )
    create_local_yolo_checkpoint(workspace)

    env = _make_env(workspace)
    _init_dvc_workspace(workspace, env)
    _run_dvc(workspace, env, "exp", "run", "train_model", check=True)

    run_marker = workspace / "runs" / ".last_train_result.json"
    assert run_marker.exists()


def test_dvc_evaluate_model_succeeds_without_local_baseline_file(tmp_path: Path) -> None:
    """Fresh clone contract: missing promoted baseline must not block `evaluate_model`."""
    repo_root = Path(__file__).resolve().parents[2]
    workspace = tmp_path / "repo"
    _copy_workspace(repo_root, workspace)
    _install_ultralytics_stub(workspace)

    promoted_dir = workspace / "models" / "current_best"
    promoted_dir.mkdir(parents=True, exist_ok=True)
    for filename in ("best.pt", "metadata.yaml"):
        candidate = promoted_dir / filename
        if candidate.exists():
            candidate.unlink()

    create_minimal_dataset(workspace)
    write_params_yaml(
        workspace,
        {
            "data": {
                "dataset_name": "dvc-fresh-full",
                "class_mapping": {},
            },
            "prepare": {
                "auto_replay": {},
            },
            "train": {
                "model": "yolov8n",
                "epochs": 1,
                "batch_size": 1,
                "finetune": {
                    "enabled": False,
                    "weights": "models/finetune/best.pt",
                },
            },
            "models": {
                "yolov8n": {
                    "backend": "yolo",
                    "asset_id": "yolov8n.pt",
                }
            },
            "evaluation": {
                "baseline_weights_path": "models/current_best/best.pt",
            },
        },
    )
    create_local_yolo_checkpoint(workspace)

    env = _make_env(workspace)
    _init_dvc_workspace(workspace, env)
    _run_dvc(workspace, env, "exp", "run", "evaluate_model", check=True)

    _assert_eval_artifacts(workspace)

    baseline_path = workspace / "models" / "current_best" / "best.pt"
    assert baseline_path.exists()
    assert baseline_path.stat().st_size == 0, "DVC preflight placeholder must stay empty until a baseline is promoted."

    models = _load_results_models(workspace)
    assert models, models
    assert len(models) == 1, models


def test_dvc_evaluate_model_fails_when_promoted_baseline_weights_missing(tmp_path: Path) -> None:
    """Existing project contract: promoted baseline metadata + missing weights must fail loudly."""

    repo_root = Path(__file__).resolve().parents[2]
    workspace = tmp_path / "repo"
    _copy_workspace(repo_root, workspace)
    _install_ultralytics_stub(workspace)

    promoted_dir = workspace / "models" / "current_best"
    promoted_dir.mkdir(parents=True, exist_ok=True)

    # Mark baseline as promoted via metadata, but ensure weights are not present locally.
    (promoted_dir / "metadata.yaml").write_text(
        "experiment_name: promoted-baseline\nmodel_backend: yolo\nimage_size: 320\n",
        encoding="utf-8",
    )
    weights_path = promoted_dir / "best.pt"
    if weights_path.exists():
        weights_path.unlink()

    create_minimal_dataset(workspace)
    write_params_yaml(
        workspace,
        {
            "data": {
                "dataset_name": "dvc-promoted-missing-weights",
                "class_mapping": {},
            },
            "prepare": {
                "auto_replay": {},
            },
            "train": {
                "model": "yolov8n",
                "epochs": 1,
                "batch_size": 1,
                "finetune": {
                    "enabled": False,
                    "weights": "models/finetune/best.pt",
                },
            },
            "models": {
                "yolov8n": {
                    "backend": "yolo",
                    "asset_id": "yolov8n.pt",
                }
            },
            "evaluation": {
                "baseline_weights_path": "models/current_best/best.pt",
            },
        },
    )
    create_local_yolo_checkpoint(workspace)

    env = _make_env(workspace)
    _init_dvc_workspace(workspace, env)
    result = _run_dvc(
        workspace,
        env,
        "exp",
        "run",
        "evaluate_model",
        check=False,
        capture=True,
    )
    assert result.returncode != 0
    combined = (result.stdout or "") + "\n" + (result.stderr or "")
    assert "Promoted baseline metadata exists" in combined
    assert "models/current_best/best.pt" in combined


def test_dvc_full_pipeline_existing_project_with_promoted_baseline(tmp_path: Path) -> None:
    """Existing project contract: full DVC pipeline uses promoted baseline cleanly."""
    repo_root = Path(__file__).resolve().parents[2]
    workspace = tmp_path / "repo"
    _copy_workspace(repo_root, workspace)
    _install_ultralytics_stub(workspace)

    baseline_weights = create_baseline_artifact(
        workspace,
        experiment_name="promoted-baseline",
    )

    create_minimal_dataset(workspace)
    write_params_yaml(
        workspace,
        {
            "data": {
                "dataset_name": "dvc-existing-full",
                "class_mapping": {},
            },
            "prepare": {
                "auto_replay": {},
            },
            "train": {
                "model": "yolov8n",
                "epochs": 1,
                "batch_size": 1,
                "finetune": {
                    "enabled": False,
                    "weights": "models/finetune/best.pt",
                },
            },
            "models": {
                "yolov8n": {
                    "backend": "yolo",
                    "asset_id": "yolov8n.pt",
                }
            },
            "evaluation": {
                "baseline_weights_path": str(baseline_weights),
            },
        },
    )
    create_local_yolo_checkpoint(workspace)

    env = _make_env(workspace)
    _init_dvc_workspace(workspace, env)
    _run_dvc(workspace, env, "exp", "run", "evaluate_model", check=True)

    _assert_eval_artifacts(workspace)
    models = _load_results_models(workspace)
    assert any("promoted-baseline" in model for model in models), models


def test_dvc_full_pipeline_rtmdet_contract_with_promoted_baseline(tmp_path: Path) -> None:
    """Project-level DVC smoke: non-YOLO backends must survive split DVC stages too."""
    repo_root = Path(__file__).resolve().parents[2]
    workspace = tmp_path / "repo"
    _copy_workspace(repo_root, workspace)
    _install_rtmdet_dvc_stub(workspace)

    baseline_dir = workspace / "models" / "current_best"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    baseline_weights = baseline_dir / "best.pt"
    baseline_weights.write_bytes(b"rtmdet-baseline-stub")
    (baseline_dir / "model_config.py").write_text("# promoted rtmdet config\n", encoding="utf-8")
    (baseline_dir / "metadata.yaml").write_text(
        yaml.safe_dump(
            {
                "experiment_name": "promoted-rtmdet-baseline",
                "model_backend": "rtmdet",
                "model_variant": "rtmdet_tiny_8xb32-300e_coco",
                "image_size": 320,
                "model_config_path": "model_config.py",
                "rtmdet_config_name": "rtmdet_tiny_8xb32-300e_coco",
                "rtmdet_cache_dir": "models/pretrained/rtmdet",
                "class_names": {0: "waste"},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    create_minimal_dataset(workspace)
    write_params_yaml(
        workspace,
        {
            "data": {
                "dataset_name": "dvc-rtmdet-full",
                "class_mapping": {},
            },
            "prepare": {
                "auto_replay": {},
            },
            "train": {
                "model": "rtmdet-tiny",
                "image_size": 320,
                "epochs": 1,
                "batch_size": 1,
                "finetune": {
                    "enabled": False,
                    "weights": "models/finetune/best.pt",
                },
            },
            "models": {
                "rtmdet-tiny": {
                    "backend": "rtmdet",
                    "asset_id": "rtmdet_tiny_8xb32-300e_coco",
                    "cache_dir": "models/pretrained/rtmdet",
                    "allow_download": False,
                    "epochs": 1,
                    "batch_size": 1,
                    "image_size": 320,
                }
            },
            "evaluation": {
                "baseline_weights_path": str(baseline_weights),
            },
        },
    )

    env = _make_env(workspace)
    env["WD_TEST_RTMDET_DVC_STUB"] = "1"
    _init_dvc_workspace(workspace, env)
    _run_dvc(workspace, env, "exp", "run", "evaluate_model", check=True)

    _assert_eval_artifacts(workspace)
    models = _load_results_models(workspace)
    assert any("promoted-rtmdet-baseline" in model for model in models), models
    assert any("dvc-rtmdet-contract" in model for model in models), models

    payload = json.loads((workspace / "runs" / ".last_train_result.json").read_text(encoding="utf-8"))
    assert payload["reload_metadata"]["model_backend"] == "rtmdet"
    assert str(payload["reload_metadata"]["model_config_path"]).endswith("model_config.py")


def test_dvc_evaluate_model_writes_merged_class_metrics_for_project_mapping(tmp_path: Path) -> None:
    """Project DVC contract: merged-class evaluation must survive the split pipeline."""
    repo_root = Path(__file__).resolve().parents[2]
    workspace = tmp_path / "repo"
    _copy_workspace(repo_root, workspace)
    _install_ultralytics_stub(workspace)

    promoted_dir = workspace / "models" / "current_best"
    promoted_dir.mkdir(parents=True, exist_ok=True)
    for filename in ("best.pt", "metadata.yaml"):
        candidate = promoted_dir / filename
        if candidate.exists():
            candidate.unlink()

    _create_class_mapping_dataset(workspace)
    write_params_yaml(
        workspace,
        {
            "data": {
                "dataset_name": "dvc-merged-class-mapping",
                "custom_classes": ["waste", "cigarette"],
                "use_coco_classes": False,
                "class_mapping": {
                    "waste": ["waste", "cigarette"],
                },
            },
            "prepare": {
                "auto_replay": {},
            },
            "train": {
                "model": "yolov8n",
                "epochs": 1,
                "batch_size": 1,
                "finetune": {
                    "enabled": False,
                    "weights": "models/finetune/best.pt",
                },
            },
            "models": {
                "yolov8n": {
                    "backend": "yolo",
                    "asset_id": "yolov8n.pt",
                }
            },
            "evaluation": {
                "baseline_weights_path": "models/current_best/best.pt",
            },
        },
    )
    create_local_yolo_checkpoint(workspace)

    env = _make_env(workspace)
    _init_dvc_workspace(workspace, env)
    _run_dvc(workspace, env, "exp", "run", "evaluate_model", check=True)

    _assert_eval_artifacts(workspace)

    raw_test_label = workspace / "raw_data" / "test" / "sourceT" / "labels" / "test_cigarette.txt"
    assert raw_test_label.read_text(encoding="utf-8").strip().split()[0] == "1"

    prepared_labels = workspace / "datasets" / "dvc-merged-class-mapping" / "test" / "val" / "labels"
    prepared_label = next(prepared_labels.glob("*.txt"))
    assert prepared_label.read_text(encoding="utf-8").strip().split()[0] == "0"

    metrics_payload = json.loads((workspace / "metrics.json").read_text(encoding="utf-8"))
    assert "cigarette_as_waste_ap50" in metrics_payload
    assert "cigarette_as_waste_n_objects" in metrics_payload

    payload = json.loads((workspace / "runs" / ".last_train_result.json").read_text(encoding="utf-8"))
    run_dir = Path(payload["train_output_dir"])
    merged_results = run_dir / "merged_class_results.csv"
    assert merged_results.exists()
    merged_rows = merged_results.read_text(encoding="utf-8")
    assert "cigarette" in merged_rows
    assert "waste" in merged_rows
