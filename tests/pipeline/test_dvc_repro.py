from __future__ import annotations

import os
import csv
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from tests.pipeline_test_utils import (
    create_baseline_artifact,
    create_local_yolo_checkpoint,
    create_minimal_dataset,
    write_params_yaml,
)

pytestmark = pytest.mark.heavy


def _install_ultralytics_stub(workspace: Path) -> None:
    """Provide a lightweight Ultralytics stub so DVC runs stay local inside tests."""

    stub_path = workspace / "ultralytics.py"
    stub_path.write_text(
        "from tests.ultralytics_stub import StubYOLO\n\n"
        "YOLO = StubYOLO\n"
    )


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
    _run_dvc(workspace, env, "init", "--no-scm", check=True)


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
                    "checkpoint": "yolov8n.pt",
                }
            },
        },
    )
    create_local_yolo_checkpoint(workspace)

    env = _make_env(workspace)
    _init_dvc_workspace(workspace, env)

    _run_dvc(workspace, env, "repro", "train_model", check=True)
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
                    "checkpoint": "yolov8n.pt",
                }
            },
        },
    )
    create_local_yolo_checkpoint(workspace)

    _run_dvc(workspace, env, "repro", "train_model", check=True)
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
                    "weights": "models/current_best/best.pt",
                },
            },
            "models": {
                "yolov8n": {
                    "backend": "yolo",
                    "checkpoint": "yolov8n.pt",
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
    _run_dvc(workspace, env, "repro", "train_model", check=True)

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
                    "checkpoint": "yolov8n.pt",
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
        "repro",
        "train_model",
        check=False,
        capture=True,
    )
    assert result.returncode != 0
    combined = (result.stdout or "") + "\n" + (result.stderr or "")
    assert "Fine-tuning mode is enabled" in combined


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
                    "checkpoint": "yolov8n.pt",
                }
            },
        },
    )
    create_local_yolo_checkpoint(workspace)

    env = _make_env(workspace)
    _init_dvc_workspace(workspace, env)
    _run_dvc(workspace, env, "repro", "train_model", check=True)

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
                    "weights": "models/current_best/best.pt",
                },
            },
            "models": {
                "yolov8n": {
                    "backend": "yolo",
                    "checkpoint": "yolov8n.pt",
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
    _run_dvc(workspace, env, "repro", "evaluate_model", check=True)

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
                    "weights": "models/current_best/best.pt",
                },
            },
            "models": {
                "yolov8n": {
                    "backend": "yolo",
                    "checkpoint": "yolov8n.pt",
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
        "repro",
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
                    "weights": str(baseline_weights),
                },
            },
            "models": {
                "yolov8n": {
                    "backend": "yolo",
                    "checkpoint": "yolov8n.pt",
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
    _run_dvc(workspace, env, "repro", "evaluate_model", check=True)

    _assert_eval_artifacts(workspace)
    models = _load_results_models(workspace)
    assert any("promoted-baseline" in model for model in models), models
