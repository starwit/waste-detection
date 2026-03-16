from __future__ import annotations

"""Sitecustomize hook used by DVC subprocess tests.

`dvc exp run` executes stages in a subprocess. By writing this file into the
temporary test workspace as `sitecustomize.py` and setting an env var, we can
patch the RTMDet backend only for that subprocess without affecting the main
pytest process.
"""

import os


if os.environ.get("WD_TEST_RTMDET_DVC_STUB") == "1":
    from pathlib import Path
    from types import SimpleNamespace

    import numpy as np

    import object_detector_trainer.backends.registry as _registry
    import object_detector_trainer.backends.rtmdet as _rtmdet
    import object_detector_trainer.evaluation.scene_metrics as _scene_metrics
    import object_detector_trainer.evaluation.visual_comparison as _visual_comparison
    import object_detector_trainer.plugins.replay as _replay

    _original_bootstrap_model_assets = _registry.bootstrap_model_assets

    class _StubValMetrics:
        def __init__(self) -> None:
            self.speed = {"preprocess": 0.2, "inference": 0.8, "postprocess": 0.2}
            self.results_dict = {
                "metrics/precision(B)": 0.5,
                "metrics/recall(B)": 0.6,
                "metrics/mAP50(B)": 0.4,
                "metrics/mAP50-95(B)": 0.3,
                "metrics/f1(B)": 0.5454545,
            }
            self.fitness = 0.35
            self.per_class = {
                "waste": {
                    "precision": 0.5,
                    "recall": 0.6,
                    "map50": 0.4,
                    "map": 0.3,
                    "f1_score": 0.5454545,
                }
            }

    class _StubModel:
        def __init__(
            self,
            *,
            model_name: str,
            model_variant: str,
            model_config_path: str,
            rtmdet_config_name: str,
            rtmdet_cache_dir: str,
        ) -> None:
            self.model_name = model_name
            self.model_backend = "rtmdet"
            self.model_variant = model_variant
            self.resolution = 320
            self.model = SimpleNamespace(yaml={"model_name": model_name})
            self.trainer = None
            self.class_names = {0: "waste"}
            self.model_config_path = model_config_path
            self.rtmdet_config_name = rtmdet_config_name
            self.rtmdet_cache_dir = rtmdet_cache_dir

        def val(self, **kwargs):
            return _StubValMetrics()

        def predict(self, *args, **kwargs):
            class _Result:
                boxes = []

                @staticmethod
                def plot():
                    return np.zeros((8, 8, 3), dtype=np.uint8)

            return [_Result()]

    def _bootstrap_model_assets(model_key: str, model_cfg: dict):
        backend = str(model_cfg.get("backend", "")).strip().lower()
        if backend != "rtmdet":
            return _original_bootstrap_model_assets(model_key, model_cfg)

        asset_id = str(model_cfg.get("asset_id") or "").strip()
        cache_dir = Path(str(model_cfg.get("cache_dir") or "models/pretrained/rtmdet")).expanduser()
        if not cache_dir.is_absolute():
            cache_dir = Path.cwd() / cache_dir
        cache_dir.mkdir(parents=True, exist_ok=True)

        config_path = cache_dir / f"{asset_id}.py"
        checkpoint_path = cache_dir / f"{asset_id}_stub.pth"
        config_path.write_text("# rtmdet dvc stub\n", encoding="utf-8")
        checkpoint_path.write_bytes(b"rtmdet-dvc-stub")
        return config_path

    def _train_backend(
        *,
        training_path: Path,
        test_path: Path,
        dataset_name: str,
        resolved_cfg: dict,
        experiment_name: str | None,
    ):
        run_dir = Path("runs") / "rtmdet" / "dvc-rtmdet-contract"
        (run_dir / "weights").mkdir(parents=True, exist_ok=True)
        (run_dir / "weights" / "best.pt").write_bytes(b"rtmdet-trained-stub")
        model_config = run_dir / "model_config.py"
        model_config.write_text("# rtmdet trained stub\n", encoding="utf-8")

        model = _StubModel(
            model_name="dvc-rtmdet-contract",
            model_variant=str(resolved_cfg.get("rtmdet_config_name") or "rtmdet_tiny_8xb32-300e_coco"),
            model_config_path=str(model_config),
            rtmdet_config_name=str(resolved_cfg.get("rtmdet_config_name") or "rtmdet_tiny_8xb32-300e_coco"),
            rtmdet_cache_dir=str(resolved_cfg.get("rtmdet_cache_dir") or "models/pretrained/rtmdet"),
        )
        return model, run_dir, "dvc-rtmdet-contract", 320, 1

    def _load_rtmdet_baseline(*, weights_path: Path, metadata: dict, display_name: str):
        config_path = weights_path.parent / "model_config.py"
        return _StubModel(
            model_name=str(display_name),
            model_variant=str(
                metadata.get("model_variant")
                or metadata.get("rtmdet_config_name")
                or "rtmdet_tiny_8xb32-300e_coco"
            ),
            model_config_path=str(config_path),
            rtmdet_config_name=str(metadata.get("rtmdet_config_name") or "rtmdet_tiny_8xb32-300e_coco"),
            rtmdet_cache_dir=str(metadata.get("rtmdet_cache_dir") or "models/pretrained/rtmdet"),
        )

    _registry.bootstrap_model_assets = _bootstrap_model_assets
    _rtmdet.train_backend = _train_backend
    _rtmdet.load_rtmdet_baseline = _load_rtmdet_baseline
    _visual_comparison.generate_side_by_side_comparisons = lambda *a, **k: None
    _scene_metrics.calculate_scene_metrics = lambda *a, **k: {}
    _replay.build_or_update_replay_set = lambda *a, **k: None

