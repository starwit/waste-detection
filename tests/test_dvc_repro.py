from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

from dvc.repo import Repo

from tests.pipeline_test_utils import create_minimal_dataset, write_params_yaml


def _install_ultralytics_stub(workspace: Path) -> None:
    """Provide a lightweight Ultralytics stub so DVC runs stay local inside tests."""

    stub_path = workspace / "ultralytics.py"
    stub_path.write_text(
        "from tests.ultralytics_stub import StubYOLO\n\n"
        "YOLO = StubYOLO\n"
    )


def _copy_workspace(src: Path, dst: Path) -> None:
    """Shallow-copy the repo for test runs without bringing heavyweight artefacts."""
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


def _collect_changed_deps(status: dict) -> set[str]:
    changed: set[str] = set()
    for entries in status.values():
        for entry in entries:
            if "changed deps" in entry and isinstance(entry["changed deps"], dict):
                changed.update(entry["changed deps"].keys())
    return changed


def _get_stage(repo: Repo, name: str):
    for stage in repo.index.stages:
        addressing = getattr(stage, "addressing", None)
        stage_name = getattr(stage, "name", None)
        if addressing == name or stage_name == name:
            return stage
    available = [getattr(s, "addressing", getattr(s, "name", "unknown")) for s in repo.index.stages]
    raise RuntimeError(f"Stage '{name}' not found. Available: {available}")


def test_dvc_repro_invalidates_when_baseline_changes(tmp_path: Path):
    """Assert `train_and_evaluate` is re-run when the promoted baseline changes."""
    repo_root = Path(__file__).resolve().parents[1]
    workspace = tmp_path / "repo"
    _copy_workspace(repo_root, workspace)
    _install_ultralytics_stub(workspace)
    models_dir = workspace / "models" / "current_best"
    models_dir.mkdir(parents=True, exist_ok=True)
    baseline_weights = models_dir / "best.pt"

    # Ensure clean DVC metadata inside the temporary workspace
    lockfile = workspace / "dvc.lock"
    if lockfile.exists():
        lockfile.unlink()

    dataset_name = "dvc-integration"
    create_minimal_dataset(workspace)
    write_params_yaml(
        workspace,
        {
            "data": {"dataset_name": dataset_name},
            "train": {
                "model_size": "n",
                "image_size": 320,
                "epochs": 1,
                "batch_size": 1,
                "pretrained_model_path": str(baseline_weights),
            },
            "evaluation": {
                "baseline_weights_path": str(baseline_weights)
            },
        },
    )

    env = _make_env(workspace)

    subprocess.run(
        [sys.executable, "-m", "dvc", "init", "--no-scm"],
        check=True,
        cwd=workspace,
        env=env,
    )

    subprocess.run(
        [sys.executable, "-m", "dvc", "repro", "train_and_evaluate"],
        check=True,
        cwd=workspace,
        env=env,
    )

    with Repo(str(workspace)) as repo:  # pragma: no cover - exercised in integration
        with repo.lock:  # type: ignore[attr-defined]
            stage = _get_stage(repo, "train_and_evaluate")
            clean_status = stage.status()

    assert _collect_changed_deps(clean_status) == set()

    baseline_weights.write_bytes(b"baseline-v1")
    os.utime(baseline_weights, None)

    with Repo(str(workspace)) as repo:  # pragma: no cover - exercised in integration
        with repo.lock:  # type: ignore[attr-defined]
            stage = _get_stage(repo, "train_and_evaluate")
            promoted_status = stage.status()

    assert "models/current_best" in _collect_changed_deps(promoted_status)

    subprocess.run(
        [sys.executable, "-m", "dvc", "repro", "train_and_evaluate"],
        check=True,
        cwd=workspace,
        env=env,
    )

    with Repo(str(workspace)) as repo:  # pragma: no cover - exercised in integration
        with repo.lock:  # type: ignore[attr-defined]
            stage = _get_stage(repo, "train_and_evaluate")
            rerun_status = stage.status()

    assert _collect_changed_deps(rerun_status) == set()

    baseline_weights.write_bytes(b"baseline-v2")
    os.utime(baseline_weights, None)

    with Repo(str(workspace)) as repo:  # pragma: no cover - exercised in integration
        with repo.lock:  # type: ignore[attr-defined]
            stage = _get_stage(repo, "train_and_evaluate")
            changed_status = stage.status()

    assert "models/current_best" in _collect_changed_deps(changed_status)
