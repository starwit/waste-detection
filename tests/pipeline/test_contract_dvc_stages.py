from __future__ import annotations

from pathlib import Path

import yaml


def test_dvc_stage_contracts_are_split_for_bootstrap_prepare_train_evaluate() -> None:
    """
    Contract: DVC stages are explicit and separated by lifecycle stage.

    This test intentionally enforces stage names and command shape so refactors
    cannot silently collapse train/evaluate responsibilities back together.
    """
    dvc_yaml_path = Path("dvc.yaml")
    assert dvc_yaml_path.exists(), "dvc.yaml must exist"

    payload = yaml.safe_load(dvc_yaml_path.read_text(encoding="utf-8")) or {}
    stages = payload.get("stages", {})

    for stage_name in (
        "check_optional_weight_deps",
        "bootstrap_model_assets",
        "prepare_data",
        "train_model",
        "evaluate_model",
    ):
        assert stage_name in stages, f"Missing DVC stage: {stage_name}"
    bootstrap_cmd = str(stages["bootstrap_model_assets"].get("cmd", ""))
    prepare_cmd = str(stages["prepare_data"].get("cmd", ""))
    train_cmd = str(stages["train_model"].get("cmd", ""))
    evaluate_cmd = str(stages["evaluate_model"].get("cmd", ""))

    assert "python -m object_detector_trainer.pipeline.check_optional_weight_deps" in str(
        stages["check_optional_weight_deps"].get("cmd", "")
    )
    assert "python -m train" in bootstrap_cmd
    assert "python -m train" in prepare_cmd
    assert "python -m train" in train_cmd
    assert "python -m train" in evaluate_cmd

    assert "--stage bootstrap" in bootstrap_cmd
    assert "--stage prepare" in prepare_cmd
    assert "--stage train" in train_cmd
    assert "--stage evaluate" in evaluate_cmd

    train_deps = stages["train_model"].get("deps", [])
    evaluate_deps = stages["evaluate_model"].get("deps", [])
    train_outs = stages["train_model"].get("outs", [])
    evaluate_outs = stages["evaluate_model"].get("outs", [])
    bootstrap_params = stages["bootstrap_model_assets"].get("params", [])
    train_params = stages["train_model"].get("params", [])
    assert ".tmp/bootstrap_manifest.json" in train_deps
    assert ".tmp/bootstrap_manifest.json" in evaluate_deps
    assert ".dvc_artifacts/train_runs" in train_outs
    assert ".dvc_artifacts/last_train_result.json" in train_outs
    assert ".dvc_artifacts/train_runs" in evaluate_deps
    assert ".dvc_artifacts/last_train_result.json" in evaluate_deps
    assert "runs" not in train_outs
    assert "runs" in evaluate_outs
    assert "results_comparison/" in evaluate_outs
    assert not (set(train_outs) & set(evaluate_outs))
    assert "models_defaults" in bootstrap_params
    assert "models_defaults" in train_params
    assert "evaluation.baseline_weights_path" not in bootstrap_params
    assert "evaluation.baseline_weights_path" not in train_params
    assert "${evaluation.baseline_weights_path}" not in train_deps
