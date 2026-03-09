from __future__ import annotations

from pathlib import Path

import yaml


def test_dvc_stage_contracts_are_split_for_prepare_train_evaluate() -> None:
    """
    Contract: DVC stages are explicit and separated by lifecycle stage.

    This test intentionally enforces stage names and command shape so refactors
    cannot silently collapse train/evaluate responsibilities back together.
    """
    dvc_yaml_path = Path("dvc.yaml")
    assert dvc_yaml_path.exists(), "dvc.yaml must exist"

    payload = yaml.safe_load(dvc_yaml_path.read_text(encoding="utf-8")) or {}
    stages = payload.get("stages", {})

    for stage_name in ("prepare_data", "train_model", "evaluate_model"):
        assert stage_name in stages, f"Missing DVC stage: {stage_name}"
    prepare_cmd = str(stages["prepare_data"].get("cmd", ""))
    train_cmd = str(stages["train_model"].get("cmd", ""))
    evaluate_cmd = str(stages["evaluate_model"].get("cmd", ""))

    assert "python -m train" in prepare_cmd
    assert "python -m train" in train_cmd
    assert "python -m train" in evaluate_cmd

    assert "--stage prepare" in prepare_cmd
    assert "--stage train" in train_cmd
    assert "--stage evaluate" in evaluate_cmd
