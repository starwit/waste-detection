from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import yaml

from tools.export_baseline import export_baseline


def test_export_baseline_copies_rtmdet_config_and_writes_portable_metadata(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.chdir(tmp_path)

    (tmp_path / "params.yaml").write_text(
        yaml.safe_dump(
            {
                "evaluation": {
                    "baseline_weights_path": "models/current_best/best.pt",
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    run_dir = tmp_path / "runs" / "rtmdet-export"
    weights_dir = run_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    (weights_dir / "best.pt").write_bytes(b"rtmdet-weights")
    (run_dir / "model_config.py").write_text("# rtmdet config\n", encoding="utf-8")
    (run_dir / "metadata.yaml").write_text(
        yaml.safe_dump(
            {
                "experiment_name": "rtmdet-export",
                "model_backend": "rtmdet",
                "model_variant": "rtmdet_tiny_8xb32-300e_coco",
                "image_size": 320,
                "model_config_path": str(run_dir / "model_config.py"),
                "rtmdet_config_name": "rtmdet_tiny_8xb32-300e_coco",
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    export_baseline(
        SimpleNamespace(
            run_dir=str(run_dir),
            weights=None,
            metadata=None,
            baseline_path=None,
            tag=None,
        )
    )

    baseline_dir = tmp_path / "models" / "current_best"
    exported_metadata = yaml.safe_load((baseline_dir / "metadata.yaml").read_text(encoding="utf-8"))

    assert (baseline_dir / "best.pt").read_bytes() == b"rtmdet-weights"
    assert (baseline_dir / "model_config.py").read_text(encoding="utf-8") == "# rtmdet config\n"
    assert exported_metadata["model_config_path"] == "model_config.py"
    assert not Path(str(exported_metadata["model_config_path"])).is_absolute()
