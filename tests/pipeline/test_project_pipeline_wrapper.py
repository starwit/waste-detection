from __future__ import annotations

import sys

import train as project_train


def _value_after(argv: list[str], flag: str) -> str:
    idx = argv.index(flag)
    return argv[idx + 1]


def test_wrapper_injects_workspace_and_config_defaults(monkeypatch) -> None:
    captured: dict[str, list[str]] = {}

    def fake_core_main(argv=None):
        captured["argv"] = list(argv or [])
        return 0

    monkeypatch.setattr(project_train, "core_main", fake_core_main)

    exit_code = project_train.main(["--stage", "prepare"])
    assert exit_code == 0

    forwarded = captured["argv"]
    assert "--workspace-root" in forwarded
    assert "--config" in forwarded
    assert _value_after(forwarded, "--workspace-root") == str(project_train.PROJECT_ROOT)
    assert _value_after(forwarded, "--config") == str(project_train.PROJECT_ROOT / "params.yaml")


def test_wrapper_defaults_to_train_stage_when_missing(monkeypatch) -> None:
    captured: dict[str, list[str]] = {}

    def fake_core_main(argv=None):
        captured["argv"] = list(argv or [])
        return 0

    monkeypatch.setattr(project_train, "core_main", fake_core_main)
    project_train.main([])

    forwarded = captured["argv"]
    assert "--stage" in forwarded
    assert _value_after(forwarded, "--stage") == "train"


def test_wrapper_preserves_explicit_workspace_and_config(monkeypatch) -> None:
    captured: dict[str, list[str]] = {}

    def fake_core_main(argv=None):
        captured["argv"] = list(argv or [])
        return 0

    monkeypatch.setattr(project_train, "core_main", fake_core_main)

    explicit = [
        "--stage",
        "train",
        "--workspace-root",
        "/tmp/custom-workspace",
        "--config",
        "/tmp/custom-workspace/custom.yaml",
    ]
    project_train.main(explicit)

    forwarded = captured["argv"]
    assert forwarded.count("--workspace-root") == 1
    assert forwarded.count("--config") == 1
    assert _value_after(forwarded, "--workspace-root") == "/tmp/custom-workspace"
    assert _value_after(forwarded, "--config") == "/tmp/custom-workspace/custom.yaml"


def test_wrapper_uses_sys_argv_when_argv_is_none(monkeypatch) -> None:
    captured: dict[str, list[str]] = {}

    def fake_core_main(argv=None):
        captured["argv"] = list(argv or [])
        return 0

    monkeypatch.setattr(project_train, "core_main", fake_core_main)
    monkeypatch.setattr(sys, "argv", ["train.py", "--stage", "evaluate"])

    project_train.main()
    forwarded = captured["argv"]

    assert forwarded[:2] == ["--stage", "evaluate"]
    assert "--workspace-root" in forwarded
    assert "--config" in forwarded
