from __future__ import annotations

import json
import os
from pathlib import Path

import torch
import yaml

from trainer_core.backends import mmdet, rfdetr, yolo
from trainer_core.backends.shared import resolve_training_config
from trainer_core.config.loader import load_config
from trainer_core.types import PersistedTrainResult, TrainResult
from trainer_core.wrappers.rfdetr import RFDETRModelAdapter


def YOLO(*args, **kwargs):
    # Lazy import so non-YOLO workflows don't import Ultralytics at module import time.
    from ultralytics import YOLO as _YOLO

    return _YOLO(*args, **kwargs)


def _persist_train_result(train_result: TrainResult) -> None:
    payload = {
        "train_output_dir": str(train_result.train_output_dir),
        "experiment_name": train_result.experiment_name,
        "image_size": int(train_result.image_size),
        "train_epochs": int(train_result.train_epochs),
        "training_path": str(train_result.training_path),
        "test_path": str(train_result.test_path),
        "baseline_weights_path": train_result.baseline_weights_path,
        "fallback_checkpoint": train_result.fallback_checkpoint,
        "finetune_weights_path": train_result.finetune_weights_path,
        "best_weights_path": str(train_result.train_output_dir / "weights" / "best.pt"),
        "reload_metadata": train_result.reload_metadata,
    }
    runs_dir = Path("runs")
    runs_dir.mkdir(parents=True, exist_ok=True)
    with open(runs_dir / ".last_train_result.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def load_persisted_train_result() -> PersistedTrainResult:
    path = Path("runs/.last_train_result.json")
    if not path.exists():
        raise FileNotFoundError(
            "No persisted train result found at runs/.last_train_result.json. "
            "Run the train stage first or provide train_result explicitly."
        )
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    raw_reload_metadata = payload.get("reload_metadata")
    reload_metadata = raw_reload_metadata if isinstance(raw_reload_metadata, dict) else {}
    return PersistedTrainResult(
        train_output_dir=Path(payload["train_output_dir"]),
        experiment_name=str(payload["experiment_name"]),
        image_size=int(payload["image_size"]),
        train_epochs=int(payload["train_epochs"]),
        training_path=Path(payload["training_path"]),
        test_path=Path(payload["test_path"]),
        baseline_weights_path=payload.get("baseline_weights_path"),
        fallback_checkpoint=str(payload["fallback_checkpoint"]),
        finetune_weights_path=payload.get("finetune_weights_path"),
        best_weights_path=Path(payload["best_weights_path"]),
        reload_metadata=reload_metadata,
    )


def load_model_from_weights(
    path_candidate: str | Path | None,
    metadata_override: dict[str, object] | None = None,
) -> tuple[object | None, str | None]:
    if not path_candidate:
        return None, None

    candidate_path = Path(path_candidate).expanduser()
    if not candidate_path.is_absolute():
        candidate_path = Path.cwd() / candidate_path
    if not candidate_path.exists():
        return None, None
    if candidate_path.stat().st_size == 0:
        return None, None

    meta: dict[str, object] = {}
    for meta_path in (
        candidate_path.parent / "metadata.yaml",
        candidate_path.parent.parent / "metadata.yaml",
    ):
        if not meta_path.exists():
            continue
        with open(meta_path, "r", encoding="utf-8") as mf:
            parsed = yaml.safe_load(mf) or {}
        if isinstance(parsed, dict):
            meta.update(parsed)
            break
    if isinstance(metadata_override, dict):
        meta.update(metadata_override)

    display_name = (
        meta.get("experiment_name")
        or meta.get("baseline_display_name")
        or meta.get("run_name")
        or candidate_path.stem
    )

    backend = str(meta.get("model_backend", "yolo")).strip().lower()
    if backend == "rfdetr":
        from trainer_core.backends import rfdetr as core_rfdetr

        model_variant = str(meta.get("model_variant", "base")).strip().lower() or "base"
        resolution = int(meta.get("image_size", 640) or 640)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        rfdetr_model = core_rfdetr._get_rfdetr_model(
            model_variant=model_variant,
            pretrain_weights=str(candidate_path),
            device=device,
            resolution=int(resolution),
        )
        adapter = RFDETRModelAdapter(
            rfdetr_model,
            model_name=str(display_name),
            resolution=int(resolution),
            model_variant=model_variant,
        )
        return adapter, str(display_name)

    if backend == "mmdet":
        from trainer_core.backends import mmdet as core_mmdet

        adapter = core_mmdet.load_mmdet_baseline(
            weights_path=candidate_path,
            metadata=meta,
            display_name=str(display_name),
        )
        return adapter, str(display_name)

    model_instance = YOLO(str(candidate_path))
    return model_instance, str(display_name)


def resolve_baseline_model(
    baseline_weights_path: str | None,
    fallback_checkpoint: str,
    finetune_weights_path: str | None = None,
) -> tuple[object, str]:
    baseline_model, baseline_display_name = load_model_from_weights(baseline_weights_path)
    if baseline_model is not None:
        return baseline_model, (baseline_display_name or "baseline")

    if finetune_weights_path:
        primary = Path(str(baseline_weights_path)).expanduser() if baseline_weights_path else None
        secondary = Path(str(finetune_weights_path)).expanduser()
        if primary is not None and not primary.is_absolute():
            primary = Path.cwd() / primary
        if not secondary.is_absolute():
            secondary = Path.cwd() / secondary
        if primary is None or str(primary) != str(secondary):
            baseline_model, baseline_display_name = load_model_from_weights(finetune_weights_path)
            if baseline_model is not None:
                return baseline_model, (baseline_display_name or Path(finetune_weights_path).stem)

    baseline_checkpoint = str(fallback_checkpoint)
    try:
        baseline_model = YOLO(baseline_checkpoint)
    except Exception as e:
        raise RuntimeError(
            "Failed to load a baseline model. No local baseline weights were found "
            "and loading the official YOLO checkpoint also failed."
        ) from e
    return baseline_model, f"{Path(baseline_checkpoint).stem}-coco"


def run_train_stage(args, config=None) -> TrainResult:
    cfg = config or load_config(getattr(args, "config", "params.yaml"), args=args)
    resolved_cfg = resolve_training_config(args, cfg)
    experiment_name = os.getenv("DVC_EXP_NAME")

    dataset_name = Path(getattr(args, "dataset_name", None) or cfg.data.dataset_name)
    dataset_path = Path("datasets") / dataset_name
    training_path = dataset_path / "train"
    test_path = dataset_path / "test"

    backend = resolved_cfg["backend"]
    trainer_by_backend = {
        "rfdetr": rfdetr.train_backend,
        "mmdet": mmdet.train_backend,
        "yolo": yolo.train_backend,
    }
    train_backend = trainer_by_backend.get(backend)
    if train_backend is None:
        raise ValueError(f"Unsupported backend: {backend!r}")

    model, train_output_dir, experiment_display_name, image_size, train_epochs = train_backend(
        training_path=training_path,
        test_path=test_path,
        dataset_name=str(dataset_name),
        resolved_cfg=resolved_cfg,
        experiment_name=experiment_name,
    )
    finetune_weights_path = (
        str(resolved_cfg.get("pretrained_model_path"))
        if backend == "yolo"
        and bool(resolved_cfg.get("finetune_mode", False))
        and resolved_cfg.get("pretrained_model_path")
        else None
    )

    reload_metadata: dict[str, object] = {
        "experiment_name": str(experiment_display_name),
        "model_backend": str(backend),
        "image_size": int(image_size),
    }
    fallback_variant_by_backend = {
        "rfdetr": resolved_cfg.get("rfdetr_variant"),
        "mmdet": resolved_cfg.get("mmdet_config_name"),
    }
    model_variant = getattr(model, "model_variant", None) or fallback_variant_by_backend.get(backend)
    if model_variant:
        reload_metadata["model_variant"] = str(model_variant)

    if backend == "mmdet":
        for key, attr_name, cfg_key in (
            ("model_config_path", "model_config_path", "mmdet_config_path"),
            ("mmdet_config_name", "mmdet_config_name", "mmdet_config_name"),
            ("mmdet_cache_dir", "mmdet_cache_dir", "mmdet_cache_dir"),
        ):
            value = getattr(model, attr_name, None) or resolved_cfg.get(cfg_key)
            if value:
                reload_metadata[key] = str(value)

        mmdet_allow_download = getattr(model, "mmdet_allow_download", None)
        if mmdet_allow_download is None:
            mmdet_allow_download = resolved_cfg.get("mmdet_allow_download")
        if mmdet_allow_download is not None:
            reload_metadata["mmdet_allow_download"] = bool(mmdet_allow_download)

    class_names = getattr(model, "class_names", None)
    if isinstance(class_names, dict) and class_names:
        reload_metadata["class_names"] = {int(k): str(v) for k, v in class_names.items()}

    result = TrainResult(
        model=model,
        train_output_dir=train_output_dir,
        experiment_name=experiment_display_name,
        image_size=int(image_size),
        train_epochs=int(train_epochs),
        training_path=training_path,
        test_path=test_path,
        baseline_weights_path=resolved_cfg.get("baseline_weights_path"),
        fallback_checkpoint=str(resolved_cfg["fallback_checkpoint"]),
        finetune_weights_path=finetune_weights_path,
        reload_metadata=reload_metadata,
        params=resolved_cfg.get("params", {}),
    )
    _persist_train_result(result)
    return result
