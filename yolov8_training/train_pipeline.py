import argparse
import os
import time
import shutil
import random
import json
from pathlib import Path

# Set matplotlib backend to non-GUI before any imports that might use it
# This prevents "Cannot load backend 'tkagg'" errors on headless systems
import matplotlib
matplotlib.use('Agg')

import numpy as np
import torch
import yaml
from ultralytics import YOLO
from yolov8_training.utils.data_utils import (
    check_for_test_images,
    create_dataset_yaml,
    process_single_images,
    reorganize_output,
)
from yolov8_training.utils.evaluate import (
    evaluate_and_log_model_results,
    evaluate_merged_class_subsets,
    generate_side_by_side_comparisons,
    mean_table,
    write_merged_class_results,
)
from yolov8_training.utils.replay import build_or_update_replay_set

from yolov8_training.utils.find_duplicates import DuplicateDetector


def _ensure_default_baseline_stub() -> None:
    """
    Ensure configured baseline/fine-tune paths exist so DVC deps are satisfiable.

    Missing files are created as 0-byte stubs. Runtime loading rejects empty files
    and falls back to an official checkpoint for baseline comparison.
    """
    try:
        params = _load_params_yaml()
    except Exception:
        params = {}

    finetune_cfg = (params.get("train", {}) or {}).get("finetune", {}) or {}
    eval_cfg = params.get("evaluation", {}) or {}

    paths_to_check = [
        ("train.finetune.weights", finetune_cfg.get("weights")),
        ("evaluation.baseline_weights_path", eval_cfg.get("baseline_weights_path")),
    ]

    for param_name, path_str in paths_to_check:
        if not path_str:
            continue

        candidate = Path(path_str)
        if not candidate.is_absolute():
            candidate = Path.cwd() / candidate
        if candidate.exists():
            continue

        candidate.parent.mkdir(parents=True, exist_ok=True)
        candidate.touch()
        print(f"Created stub file for {param_name}: {candidate}")


def _load_params_yaml():
    """Load and return params from params.yaml as a dict, or {} on failure."""
    try:
        with open("params.yaml", "r") as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        print(f"Warning: Could not load params.yaml: {e}")
        return {}

def load_class_config(params: dict | None = None):
    """Load class configuration from params.yaml including class mapping"""
    if params is None:
        params = _load_params_yaml()

    custom_classes = params.get("data", {}).get("custom_classes", [])
    use_coco_classes = params.get("data", {}).get("use_coco_classes", True)
    class_mapping_config = params.get("data", {}).get("class_mapping", {})

    # Filter out None/empty values from custom_classes
    if custom_classes:
        custom_classes = [cls for cls in custom_classes if cls]

    return custom_classes, use_coco_classes, class_mapping_config

def load_folder_subset_config(params: dict | None = None):
    """Load folder subset configuration from params.yaml"""
    if params is None:
        params = _load_params_yaml()
    return params.get("prepare", {}).get("folder_subsets", {})


def _resolve_save_dir(model, results, default: Path) -> Path:
    """Return Ultralytics' actual save_dir if available, else default.

    Keeps train_model concise while handling version differences
    (trainer.save_dir vs. results.save_dir).
    """
    try:
        trainer = getattr(model, "trainer", None)
        candidate = getattr(trainer, "save_dir", None) if trainer is not None else None
        if candidate:
            return Path(candidate)
    except Exception:
        pass

    try:
        candidate = getattr(results, "save_dir", None)
        if candidate:
            return Path(candidate)
    except Exception:
        pass

    return default


def _resolve_unique_run_dir(root: Path, run_name: str) -> Path:
    """Return a unique directory path under root."""
    candidate = root / run_name
    if not candidate.exists():
        return candidate
    suffix = 1
    while (root / f"{run_name}_{suffix}").exists():
        suffix += 1
    return root / f"{run_name}_{suffix}"


def _rfdetr_resolution_divisor(model_variant: str) -> int:
    """
    RF-DETR requires resolution divisible by (patch_size * num_windows).
    For the official variants:
      - nano/small/medium: 16 * 2 = 32
      - base/large: 14 * 4 = 56
    """
    variant = (model_variant or "").lower()
    if variant in {"nano", "small", "medium"}:
        return 32
    if variant in {"base", "large"}:
        return 56
    return 56


def _normalize_rfdetr_resolution(model_variant: str, resolution: int | None, fallback: int) -> int:
    """Adjust resolution to satisfy RF-DETR constraints."""
    if resolution is None:
        resolution = int(fallback)
    divisor = _rfdetr_resolution_divisor(model_variant)
    if resolution % divisor != 0:
        adjusted = (resolution // divisor) * divisor
        if adjusted < divisor:
            adjusted = divisor
        print(
            f"Warning: RF-DETR resolution {resolution} is not divisible by {divisor}. "
            f"Using {adjusted}."
        )
        resolution = adjusted
    return int(resolution)

def _normalize_model_type(model_type: str | None) -> str:
    """
    Normalize user/config-supplied model type strings.

    Accepts common separators so config can use values like:
    - "rfdetr", "rf-detr", "rf_detr", "rf detr"
    - "yolo"
    """
    raw = (model_type or "yolo")
    if not isinstance(raw, str):
        raw = str(raw)
    compact = "".join(ch for ch in raw.strip().lower() if ch.isalnum())
    if compact in {"yolo"}:
        return "yolo"
    if compact in {"rfdetr"}:
        return "rfdetr"
    raise ValueError(
        f"Unsupported model_type: {model_type!r}. Expected one of: yolo | rfdetr (rf-detr, rf_detr)."
    )


def _default_yolo_checkpoint(model_key: str) -> str:
    key = str(model_key or "").strip()
    if key.endswith(".pt"):
        return key
    return f"{key}.pt"


def _infer_rfdetr_variant(model_key: str) -> str:
    key = str(model_key or "").strip().lower().replace("_", "-")
    for prefix in ("rfdetr-", "rf-detr-"):
        if key.startswith(prefix) and len(key) > len(prefix):
            return key[len(prefix):]
    return "base"


def _first_yolo_fallback_checkpoint(models_cfg: dict) -> str | None:
    """Return the checkpoint of the first YOLO model in *models_cfg* (sorted).

    Used as the COCO-comparison fallback when the selected backend is not YOLO.
    Sorting by key makes the result deterministic regardless of YAML key order.
    """
    for model_key in sorted(models_cfg):
        model_cfg = models_cfg[model_key]
        if not isinstance(model_cfg, dict):
            continue
        backend = _normalize_model_type(model_cfg.get("backend", "yolo"))
        if backend == "yolo":
            checkpoint = model_cfg.get("checkpoint")
            return str(checkpoint) if checkpoint else _default_yolo_checkpoint(str(model_key))
    return None


def resolve_training_config(args, params: dict | None = None) -> dict:
    if params is None:
        params = _load_params_yaml()

    train_cfg = params.get("train", {}) if isinstance(params.get("train", {}), dict) else {}
    models_cfg = params.get("models", {}) if isinstance(params.get("models", {}), dict) else {}
    eval_cfg = params.get("evaluation", {}) if isinstance(params.get("evaluation", {}), dict) else {}

    selected_model = getattr(args, "model", None) or train_cfg.get("model")
    if not selected_model:
        raise ValueError("Missing train.model in params.yaml and no --model override was provided.")
    if selected_model not in models_cfg:
        available = ", ".join(sorted(models_cfg.keys())) or "<none>"
        raise ValueError(f"Unknown model key '{selected_model}'. Available models: {available}")

    model_cfg = models_cfg.get(selected_model, {})
    if not isinstance(model_cfg, dict):
        raise ValueError(f"models.{selected_model} must be a mapping.")

    backend = _normalize_model_type(model_cfg.get("backend", "yolo"))
    shared_image_size = int(train_cfg.get("image_size", 640))
    shared_epochs = int(train_cfg.get("epochs", 100))
    shared_batch_size = int(train_cfg.get("batch_size", 8))

    finetune_cfg = train_cfg.get("finetune", {}) if isinstance(train_cfg.get("finetune", {}), dict) else {}
    finetune_enabled = bool(finetune_cfg.get("enabled", train_cfg.get("finetune_mode", False)))
    finetune_weights = finetune_cfg.get("weights", train_cfg.get("pretrained_model_path"))
    finetune_lr = finetune_cfg.get("lr", train_cfg.get("finetune_lr"))
    finetune_epochs = finetune_cfg.get("epochs", train_cfg.get("finetune_epochs"))
    finetune_freeze_backbone = bool(
        finetune_cfg.get("freeze_backbone", train_cfg.get("freeze_backbone", False))
    )

    single_phase_overrides = {}
    for key in ("optimizer", "mosaic", "close_mosaic", "mixup", "cos_lr", "lrf", "patience"):
        value = finetune_cfg.get(key, train_cfg.get(key))
        if value is not None:
            single_phase_overrides[key] = value
    if not single_phase_overrides:
        single_phase_overrides = None

    model_checkpoint = str(model_cfg.get("checkpoint") or _default_yolo_checkpoint(str(selected_model)))
    fallback_checkpoint = model_checkpoint if backend == "yolo" else _first_yolo_fallback_checkpoint(models_cfg)
    if not fallback_checkpoint:
        fallback_checkpoint = "yolov8n.pt"

    resolved = {
        "model_key": str(selected_model),
        "backend": backend,
        "image_size": int(model_cfg.get("image_size", shared_image_size)),
        "epochs": int(model_cfg.get("epochs", shared_epochs)),
        "batch_size": int(model_cfg.get("batch_size", shared_batch_size)),
        "baseline_weights_path": eval_cfg.get("baseline_weights_path"),
        "fallback_checkpoint": str(fallback_checkpoint),
        "finetune_mode": finetune_enabled,
        "pretrained_model_path": finetune_weights,
        "finetune_lr": finetune_lr,
        "finetune_epochs": finetune_epochs,
        "freeze_backbone": finetune_freeze_backbone,
        "single_phase_overrides": single_phase_overrides,
        "params": params,
    }

    if backend == "yolo":
        resolved["checkpoint"] = model_checkpoint
        if finetune_enabled and finetune_epochs is not None:
            resolved["epochs"] = int(finetune_epochs)
        return resolved

    rfdetr_variant = str(model_cfg.get("variant") or model_cfg.get("model") or _infer_rfdetr_variant(str(selected_model)))
    rfdetr_batch_size = int(model_cfg.get("batch_size", shared_batch_size))
    explicit_grad_accum = model_cfg.get("grad_accum_steps")
    target_effective_batch = int(model_cfg.get("target_effective_batch", 16))
    if explicit_grad_accum is not None:
        rfdetr_grad_accum = int(explicit_grad_accum)
    else:
        rfdetr_grad_accum = max(1, target_effective_batch // rfdetr_batch_size)

    rfdetr_resolution = _normalize_rfdetr_resolution(
        rfdetr_variant,
        model_cfg.get("resolution", None),
        int(model_cfg.get("image_size", shared_image_size)),
    )

    resolved.update(
        {
            "rfdetr_variant": rfdetr_variant,
            "rfdetr_epochs": int(model_cfg.get("epochs", shared_epochs)),
            "rfdetr_batch_size": rfdetr_batch_size,
            "rfdetr_grad_accum": rfdetr_grad_accum,
            "rfdetr_grad_accum_explicit": explicit_grad_accum is not None,
            "rfdetr_target_effective_batch": target_effective_batch,
            "rfdetr_resolution": int(rfdetr_resolution),
            "rfdetr_lr": model_cfg.get("lr"),
            "rfdetr_pretrain": model_cfg.get("pretrain_weights"),
            "rfdetr_grad_ckpt": model_cfg.get("gradient_checkpointing"),
            "rfdetr_extra": model_cfg.get("extra_train_kwargs"),
        }
    )
    return resolved


def _get_rfdetr_model(
    model_variant: str,
    pretrain_weights: str | None = None,
    device: str | None = None,
    gradient_checkpointing: bool | None = None,
):
    """Return an initialized RF-DETR model based on a variant name."""
    try:
        import rfdetr
    except Exception as e:
        raise RuntimeError(
            "RF-DETR is not installed. Add it to your environment (see pyproject.toml)."
        ) from e

    variant = (model_variant or "base").lower()
    class_name_by_variant = {
        "nano": "RFDETRNano",
        "small": "RFDETRSmall",
        "medium": "RFDETRMedium",
        "base": "RFDETRBase",
        "large": "RFDETRLarge",
    }
    class_name = class_name_by_variant.get(variant)
    if class_name is None:
        raise ValueError(f"Unsupported RF-DETR model variant: {model_variant}")

    model_cls = getattr(rfdetr, class_name, None)
    if model_cls is None:
        raise RuntimeError(
            f"Your installed rfdetr package does not provide {class_name}. "
            "Either choose another train.rfdetr.model or upgrade rfdetr."
        )

    init_kwargs: dict[str, object] = {}
    if pretrain_weights:
        init_kwargs["pretrain_weights"] = pretrain_weights
    if device:
        init_kwargs["device"] = device
    if gradient_checkpointing is not None:
        init_kwargs["gradient_checkpointing"] = bool(gradient_checkpointing)
    return model_cls(**init_kwargs) if init_kwargs else model_cls()


def train_rfdetr(
    dataset_dir: Path,
    output_root: Path,
    experiment_name: str | None,
    model_variant: str,
    epochs: int,
    batch_size: int,
    grad_accum_steps: int,
    lr: float | None,
    resolution: int,
    pretrain_weights: str | None = None,
    gradient_checkpointing: bool | None = None,
    extra_train_kwargs: dict | None = None,
):
    """Train an RF-DETR model using a Roboflow-style COCO dataset export."""
    run_name = experiment_name or "rfdetr-train"
    output_dir = _resolve_unique_run_dir(output_root, run_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = _get_rfdetr_model(
        model_variant,
        pretrain_weights=pretrain_weights,
        device=device,
        gradient_checkpointing=gradient_checkpointing,
    )

    train_kwargs: dict[str, object] = {
        "dataset_dir": str(dataset_dir),
        "dataset_file": "roboflow",
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "grad_accum_steps": int(grad_accum_steps),
        "output_dir": str(output_dir),
        "resolution": int(resolution),
        "run_test": True,
    }
    if lr is not None:
        train_kwargs["lr"] = float(lr)
    if extra_train_kwargs:
        train_kwargs.update(extra_train_kwargs)

    model.train(**train_kwargs)
    return model, output_dir


def _save_rfdetr_weights(output_dir: Path) -> None:
    """Copy the best RF-DETR checkpoint into ``output_dir/weights/best.pt``.

    This mirrors the YOLO convention so that ``export_baseline.py`` (which looks
    for ``<run>/weights/best.pt``) works identically for both model families.
    """
    weights_dir = output_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)

    # RF-DETR typically saves the best checkpoint as checkpoint_best_total.pth
    candidates = [
        output_dir / "checkpoint_best_total.pth",
        output_dir / "checkpoint_best.pth",
        output_dir / "best.pth",
    ]
    # Also check for any .pth file as a last resort
    all_pth = sorted(output_dir.glob("*.pth"))

    source = None
    for c in candidates:
        if c.exists() and c.stat().st_size > 0:
            source = c
            break
    if source is None and all_pth:
        # Pick the largest .pth file (likely the full checkpoint)
        source = max(all_pth, key=lambda p: p.stat().st_size)

    if source is not None:
        dest = weights_dir / "best.pt"
        shutil.copy2(source, dest)
        print(f"RF-DETR best weights saved to {dest}")
    else:
        print("Warning: No RF-DETR checkpoint found to copy to weights/best.pt")


def _prepare_rfdetr_yolo_layout(training_path: Path, test_path: Path, dataset_name: str) -> Path:
    """Create a lightweight YOLO-format directory for RF-DETR.

    RF-DETR 1.4+ auto-detects YOLO datasets when it finds ``data.yaml`` +
    ``train/images/`` at the dataset root.  Our prepare stage produces a
    slightly different layout (``val/`` instead of ``valid/``, separate
    train/test roots, ``dataset.yaml`` instead of ``data.yaml``), so this
    helper bridges the gap with three symlinks and one tiny YAML file.

    This replaces the former ``export_coco_dataset_from_yolo`` step which
    had to read every image with ``cv2.imread`` to extract dimensions and
    then write COCO JSON annotation files — a process that could take
    minutes for large datasets.  Symlinks are instant.
    """
    output_dir = Path(".tmp") / "rfdetr_datasets" / str(dataset_name)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # RF-DETR expects train/, valid/, test/ under one root
    (output_dir / "train").symlink_to((training_path / "train").resolve())
    (output_dir / "valid").symlink_to((training_path / "val").resolve())
    (output_dir / "test").symlink_to((test_path / "val").resolve())

    # RF-DETR looks for data.yaml (not dataset.yaml)
    src_yaml = training_path / "dataset.yaml"
    with open(src_yaml, "r", encoding="utf-8") as f:
        ds_cfg = yaml.safe_load(f) or {}

    names_raw = ds_cfg.get("names", [])
    if isinstance(names_raw, dict):
        names_list = [names_raw[k] for k in sorted(names_raw.keys(), key=lambda x: int(x))]
    else:
        names_list = list(names_raw)

    data_yaml = {
        "names": names_list,
        "nc": len(names_list),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
    }
    with open(output_dir / "data.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(data_yaml, f, sort_keys=False)

    return output_dir



def _load_baseline_from_path(path_candidate: str | None) -> tuple[object | None, str | None]:
    """
    Load baseline model from a path if it exists and is valid.
    
    Args:
        path_candidate: Path to a .pt weights file (can be relative or absolute)
    
    Returns:
        (model, display_name) if successful, (None, None) otherwise
    
    IMPORTANT SIZE CHECK:
        Zero-byte files are always rejected as invalid weights. This supports
        DVC stub files that are created pre-run for clean-clone bootstrapping.
    """
    if not path_candidate:
        return None, None

    candidate_path = Path(path_candidate).expanduser()
    if not candidate_path.is_absolute():
        candidate_path = Path.cwd() / candidate_path

    # Check exists AND size > 0 to reject stub files
    if not candidate_path.exists() or candidate_path.stat().st_size == 0:
        print(f"Warning: Baseline weights not found at {candidate_path}")
        return None, None

    meta = {}
    try:
        meta_path = candidate_path.parent / "metadata.yaml"
        if meta_path.exists():
            with open(meta_path, "r") as mf:
                meta = yaml.safe_load(mf) or {}
    except Exception:
        meta = {}

    display_name = (
        meta.get("experiment_name")
        or meta.get("baseline_display_name")
        or meta.get("run_name")
        or candidate_path.stem
    )

    backend = str(meta.get("model_backend", "yolo")).strip().lower()
    if backend == "rfdetr":
        try:
            model_variant = str(meta.get("model_variant", "base")).strip().lower() or "base"
            resolution = int(meta.get("image_size", 640) or 640)

            device = "cuda" if torch.cuda.is_available() else "cpu"
            rfdetr_model = _get_rfdetr_model(
                model_variant=model_variant,
                pretrain_weights=str(candidate_path),
                device=device,
            )
            from yolov8_training.utils.rfdetr_adapter import RFDETRModelAdapter

            adapter = RFDETRModelAdapter(
                rfdetr_model,
                model_name=str(display_name),
                resolution=int(resolution),
                model_variant=model_variant,
            )
            return adapter, str(display_name)
        except Exception as load_error:
            print(f"Warning: Could not load RF-DETR baseline from {candidate_path}: {load_error}")
            return None, None

    try:
        model_instance = YOLO(str(candidate_path))
    except Exception as load_error:
        print(f"Warning: Could not load baseline model from {candidate_path}: {load_error}")
        return None, None

    return model_instance, str(display_name)

def train_model(
    dataset_path, checkpoint, image_size, batch_size, experiment_name, epochs=100,
    finetune_mode=False, pretrained_model_path=None, finetune_lr=None, freeze_backbone=False,
    single_phase_overrides: dict | None = None,
):
    """
    Train the YOLO model on the specified dataset.

    Args:
        dataset_path (Path): Path to the dataset directory.
        checkpoint (str): YOLO checkpoint to train from (e.g., 'yolov8m.pt').
        image_size (int): Size of images for training.
        batch_size (int): Batch size for training.
        experiment_name (str): Name for the experiment.
        epochs (int): Number of training epochs.
        finetune_mode (bool): Whether to use fine-tuning mode.
        pretrained_model_path (str): Path to pre-trained model for fine-tuning.
        finetune_lr (float): Learning rate for fine-tuning.
        freeze_backbone (bool): Whether to freeze backbone layers during fine-tuning.

    Returns:
        model (YOLO): The trained YOLO model.
        results: Training results.
        Path: Directory path of the training output.
    """
    # Choose model based on fine-tuning mode
    pretrained_model = None
    if pretrained_model_path:
        candidate = Path(pretrained_model_path)
        if not candidate.is_absolute():
            candidate = Path.cwd() / candidate
        if candidate.exists() and candidate.stat().st_size > 0:
            pretrained_model = candidate

    if finetune_mode:
        if not pretrained_model_path:
            raise ValueError(
                "Fine-tuning mode is enabled, but train.finetune.weights is not set."
            )
        if pretrained_model is None:
            raise FileNotFoundError(
                f"Fine-tuning mode is enabled, but weights were not found or are empty: "
                f"{pretrained_model_path}"
            )
        print(f"Fine-tuning mode enabled. Loading pre-trained model: {pretrained_model_path}")
        model = YOLO(str(pretrained_model))
        experiment_name = f"{experiment_name}-finetune"
    else:
        checkpoint_name = str(checkpoint)
        print(f"Using YOLO checkpoint: {checkpoint_name}")
        try:
            model = YOLO(checkpoint_name)
        except Exception as e:
            raise RuntimeError(
                "Failed to initialize training. "
                f"Tried to load official checkpoint '{checkpoint_name}' but it could not be "
                "downloaded or loaded. Ensure network access or provide a valid local checkpoint "
                "in models.<selected>.checkpoint."
            ) from e
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Define project name and ensure unique run name
    project = "runs"
    name = None
    if experiment_name:
        base_name = experiment_name
        name = base_name
        run_number = 1
        while (Path(project) / name).exists():
            name = f"{base_name}_{run_number}"
            run_number += 1

    # Prepare training arguments
    train_args = {
        "data": str(dataset_path / "dataset.yaml"),
        "epochs": epochs,
        "imgsz": image_size,
        "batch": batch_size,
        "device": device,
        "workers": 4,
        "amp": True,
        "project": project,
    }
    if name:
        train_args["name"] = name
    
    # Two-phase path removed for simplicity; single-phase fine-tune only

    # Single-phase (original) path
    if finetune_mode:
        if finetune_lr is not None:
            train_args["lr0"] = finetune_lr
            print(f"Using fine-tuning learning rate: {finetune_lr}")

        if freeze_backbone:
            # Freeze backbone layers (layers 0-9 typically for YOLOv8)
            train_args["freeze"] = list(range(10))
            print("Freezing backbone layers for fine-tuning")

        # Allow minimal, explicit overrides (optimizer, mosaic, mixup, close_mosaic, cos_lr, lrf, patience)
        if single_phase_overrides:
            sp = {k: v for k, v in single_phase_overrides.items() if v is not None}
            if sp:
                print(f"Applying single-phase overrides: {sorted(sp.keys())}")
                train_args.update(sp)

    results = model.train(**train_args)

    default_dir = Path(project) / (name if name else "train")
    output_dir = _resolve_save_dir(model, results, default_dir)

    # Keep return signature compatible with tests: (model, results, output_dir)
    return model, results, output_dir


def process_data(
    image_input_path,
    train_output_path,
    test_output_path,
    val_split,
    test_split,
    augment_multiplier,
    custom_classes=None,
    use_coco_classes=True,
    folder_subsets=None,
    class_mapping_config=None
):
    """
    Process image data for training and validation.

    Args:
        image_input_path (Path): Path to image input data.
        train_output_path (Path): Path to store training data.
        test_output_path (Path): Path to store test data.
        val_split (float): Validation split ratio.
        test_split (float): Test split ratio.
        augment_multiplier (int): Augmentation multiplier.
        custom_classes (list): List of custom class names.
        use_coco_classes (bool): Whether to use COCO classes when custom_classes is empty.
        folder_subsets (dict): Dictionary mapping folder names to subset ratios (0.0-1.0).
        class_mapping_config (dict): Dictionary mapping target classes to source classes for merging.

    Returns:
        int, int, int: Total frames for training, validation and test.
    """
    if image_input_path.exists():
        return process_single_images(
            image_input_path, train_output_path, test_output_path, val_split, test_split, 
            augment_multiplier, custom_classes, use_coco_classes, folder_subsets, class_mapping_config
        )
    return 0, 0, 0

def delete_unused_folders():
    """
    Deletes unused/empty folders in the runs/ directory 
    """
    print("Checking for unused folders in 'runs/' directory...")
    runs_dir = Path("runs")
    if not runs_dir.exists():
        return
    for folder in runs_dir.iterdir():
        if folder.is_dir() and not any(folder.iterdir()):
            print(f"Deleting empty folder: {folder}")
            shutil.rmtree(folder)



def run_prepare_stage(args):
    dataset_name = Path(args.dataset_name)
    val_split = float(args.val_split)
    recreate_dataset = args.recreate_dataset
    augment_multiplier=args.augment_multiplier
    
    # Load params once to avoid multiple file reads
    _params = _load_params_yaml()
    # Load class configuration
    custom_classes, use_coco_classes, class_mapping_config = load_class_config(_params)
    # Load folder subset configuration
    folder_subsets = load_folder_subset_config(_params)
    
    # Override with command-line arguments if provided
    if args.folder_subset:
        print("Overriding folder subset configuration with command-line arguments:")
        for folder_name, ratio_str in args.folder_subset:
            try:
                ratio = float(ratio_str)
                if ratio > 0:
                    folder_subsets[folder_name] = ratio
                    if ratio > 1:
                        print(f"  {folder_name}: {ratio*100:.1f}% (oversampling)")
                    else:
                        print(f"  {folder_name}: {ratio*100:.1f}%")
                else:
                    print(f"  Warning: Invalid ratio {ratio} for {folder_name}. Must be > 0.")
            except ValueError:
                print(f"  Warning: Invalid ratio '{ratio_str}' for {folder_name}. Must be a number.")
    
    if folder_subsets:
        print(f"Final folder subset configuration: {folder_subsets}")
    else:
        print("No folder subset configuration found. Using all images from all folders.")

    if custom_classes and use_coco_classes:
        raise ValueError(
            "Both 'custom_classes' and 'use_coco_classes' are set. "
            "Please choose one strategy."
        )
        
    
    # Define paths
    base_input_path = Path("raw_data")
    train_image_input_path = base_input_path / "train"
    test_image_input_path = base_input_path / "test"

    dataset_path = Path("datasets") / dataset_name
    training_path = dataset_path / "train"
    test_path = dataset_path / "test"

    test_data_exists = check_for_test_images(test_image_input_path)

    if not test_data_exists:
        test_split = args.test_split
        print(f"No dedicated test data found. Using {test_split} for test split.")
    else:
        test_split = 0
        print(f"Dedicated test data found. Using {test_split} for test split.")

    # Create or recreate dataset directory if specified
    if not dataset_path.exists() or recreate_dataset:
        if dataset_path.exists() and recreate_dataset:
            print(f"Recreating dataset '{dataset_name}'...")
            shutil.rmtree(dataset_path)

        dataset_path.mkdir(parents=True, exist_ok=True)
        for path in [training_path, test_path]:
            path.mkdir(parents=True, exist_ok=True)

        # Directories for training/validation frames and labels
        train_img_dir = training_path / "train" / "images"
        train_label_dir = training_path / "train" / "labels"
        val_img_dir = training_path / "val" / "images"
        val_label_dir = training_path / "val" / "labels"
        test_img_dir = test_path / "val" / "images"
        test_label_dir = test_path / "val" / "labels"

        for path in [
            train_img_dir,
            train_label_dir,
            val_img_dir,
            val_label_dir,
            test_img_dir,
            test_label_dir,
        ]:
            path.mkdir(parents=True, exist_ok=True)

        # Process training data
        total_train_frames, total_val_frames, total_test_frames = process_data(
            image_input_path=train_image_input_path,
            train_output_path=training_path,
            test_output_path=test_path,
            val_split=val_split,
            test_split=test_split,
            augment_multiplier=augment_multiplier,
            custom_classes=custom_classes,
            use_coco_classes=use_coco_classes,
            folder_subsets=folder_subsets,
            class_mapping_config=class_mapping_config
        )
        if total_train_frames <= 0:
            raise ValueError(
                "Prepare stage produced 0 training frames. "
                f"(train={total_train_frames}, val={total_val_frames}, test={total_test_frames}). "
                "Ensure raw_data/train contains labeled images and split/subset settings leave at least one training sample."
            )

        create_dataset_yaml(training_path, custom_classes, use_coco_classes, class_mapping_config)

        test_folder_frame_count = 0
        if test_data_exists:
            # Process test data
            _, test_folder_frame_count, _ = process_data(
                image_input_path=test_image_input_path,
                train_output_path=test_path,
                test_output_path=test_path,
                val_split=1,
                test_split=0,
                augment_multiplier=1,
                custom_classes=custom_classes,
                use_coco_classes=use_coco_classes,
                folder_subsets={},  # Don't apply subsets to test data
                class_mapping_config=class_mapping_config
            )

        create_dataset_yaml(test_path, custom_classes, use_coco_classes, class_mapping_config)

        print(f"Total training frames: {total_train_frames}")
        print(f"Total validation frames: {total_val_frames}")
        print(f"Total test frames: {total_test_frames + test_folder_frame_count}")
    else:
        print(f"Dataset '{dataset_name}' already exists. Skipping dataset creation.")

    print("Testing for duplicates between train and test folders...")
    # Initialize duplicate detector to compare the train and test folders
    detector = DuplicateDetector(phash_threshold=2, ssim_threshold=0.95)

    # Compare folders
    matches = detector.compare_folders(training_path, test_path)

    # Print results
    detector.print_folder_comparison_results(matches)

    _ensure_default_baseline_stub()
    return

def run_train_eval_stage(args):
    experiment_name = os.getenv("DVC_EXP_NAME")
    val_split = float(args.val_split)
    resolved_cfg = getattr(args, "resolved_training_config", None) or resolve_training_config(args)
    params = resolved_cfg.get("params", {})

    # Define paths
    dataset_name = Path(args.dataset_name)
    dataset_path = Path("datasets") / dataset_name
    training_path = dataset_path / "train"
    test_path = dataset_path / "test"

    if resolved_cfg["backend"] == "rfdetr":
        from yolov8_training.utils.rfdetr_adapter import RFDETRModelAdapter

        rfdetr_variant = resolved_cfg["rfdetr_variant"]
        rfdetr_epochs = int(resolved_cfg["rfdetr_epochs"])
        rfdetr_batch_size = int(resolved_cfg["rfdetr_batch_size"])
        rfdetr_grad_accum = int(resolved_cfg["rfdetr_grad_accum"])
        if not resolved_cfg.get("rfdetr_grad_accum_explicit", False):
            print(
                f"RF-DETR: auto-computed grad_accum_steps={rfdetr_grad_accum} "
                f"(target_effective_batch={resolved_cfg['rfdetr_target_effective_batch']} / batch_size={rfdetr_batch_size})"
            )

        rfdetr_lr = resolved_cfg.get("rfdetr_lr")
        rfdetr_resolution = int(resolved_cfg["rfdetr_resolution"])
        rfdetr_pretrain = resolved_cfg.get("rfdetr_pretrain")
        rfdetr_grad_ckpt = resolved_cfg.get("rfdetr_grad_ckpt")
        rfdetr_extra = resolved_cfg.get("rfdetr_extra")

        # Prepare a lightweight YOLO-format directory layout for RF-DETR.
        # RF-DETR 1.4+ auto-detects YOLO format (data.yaml + train/images/).
        # This replaces the old COCO export step — symlinks instead of copying.
        rfdetr_export_dir = _prepare_rfdetr_yolo_layout(
            training_path=training_path,
            test_path=test_path,
            dataset_name=str(dataset_name),
        )
        rfdetr_dataset_dir = rfdetr_export_dir

        runs_root = Path("runs") / "rfdetr"
        runs_root.mkdir(parents=True, exist_ok=True)
        display_name = f"{(experiment_name or resolved_cfg['model_key'])}-rfdetr-{rfdetr_variant}"

        rfdetr_model, train_output_dir = train_rfdetr(
            dataset_dir=rfdetr_dataset_dir,
            output_root=runs_root,
            experiment_name=display_name,
            model_variant=rfdetr_variant,
            epochs=rfdetr_epochs,
            batch_size=rfdetr_batch_size,
            grad_accum_steps=rfdetr_grad_accum,
            lr=float(rfdetr_lr) if rfdetr_lr is not None else None,
            resolution=rfdetr_resolution,
            pretrain_weights=str(rfdetr_pretrain) if rfdetr_pretrain is not None else None,
            gradient_checkpointing=rfdetr_grad_ckpt,
            extra_train_kwargs=rfdetr_extra if isinstance(rfdetr_extra, dict) else None,
        )

        # ── Save weights in the same layout as YOLO (weights/best.pt) ──
        _save_rfdetr_weights(train_output_dir)

        # ── Read class names from the dataset for the adapter ──
        from yolov8_training.utils.evaluate import get_dataset_classes
        test_yaml = test_path / "dataset.yaml"
        class_names_map, _ = get_dataset_classes(test_yaml)

        # ── Wrap in adapter so evaluate.py treats it like a YOLO model ──
        model = RFDETRModelAdapter(
            rfdetr_model,
            model_name=display_name,
            resolution=rfdetr_resolution,
            class_names=class_names_map,
            model_variant=rfdetr_variant,
        )

        # ── Clean up temporary YOLO layout (not needed after training) ──
        tmp_root = Path(".tmp")
        if rfdetr_export_dir.exists():
            shutil.rmtree(rfdetr_export_dir, ignore_errors=True)
        # Remove empty parent dirs (.tmp/rfdetr_datasets/, .tmp/) if nothing else uses them
        for parent in (rfdetr_export_dir.parent, tmp_root):
            try:
                parent.rmdir()  # only succeeds if empty
            except (OSError, FileNotFoundError):
                pass

        # ── Evaluate & finalize (shared with YOLO path) ──
        _evaluate_and_finalize(
            model=model,
            experiment_name=display_name,
            train_output_dir=train_output_dir,
            test_path=test_path,
            training_path=training_path,
            image_size=rfdetr_resolution,
            train_epochs=rfdetr_epochs,
            val_split=val_split,
            baseline_weights_path=resolved_cfg.get("baseline_weights_path"),
            fallback_checkpoint=resolved_cfg["fallback_checkpoint"],
            params=params,
        )
        return

    # Train the model
    model, results, train_output_dir = train_model(
        training_path,
        resolved_cfg["checkpoint"],
        int(resolved_cfg["image_size"]),
        int(resolved_cfg["batch_size"]),
        experiment_name,
        epochs=int(resolved_cfg["epochs"]),
        finetune_mode=bool(resolved_cfg.get("finetune_mode", False)),
        pretrained_model_path=resolved_cfg.get("pretrained_model_path"),
        finetune_lr=resolved_cfg.get("finetune_lr"),
        freeze_backbone=bool(resolved_cfg.get("freeze_backbone", False)),
        single_phase_overrides=resolved_cfg.get("single_phase_overrides"),
    )

    # Determine the final experiment display name (append suffix in finetune mode)
    final_experiment_name = (
        (
            f"{experiment_name}-finetune"
            if experiment_name
            and bool(resolved_cfg.get("finetune_mode", False))
            and resolved_cfg.get("pretrained_model_path")
            and Path(str(resolved_cfg.get("pretrained_model_path"))).exists()
            else experiment_name
        )
    )

    # ── Evaluate & finalize (shared with RF-DETR path) ──
    _evaluate_and_finalize(
        model=model,
        experiment_name=final_experiment_name,
        train_output_dir=train_output_dir,
        test_path=test_path,
        training_path=training_path,
        image_size=int(resolved_cfg["image_size"]),
        train_epochs=int(resolved_cfg["epochs"]),
        val_split=val_split,
        baseline_weights_path=resolved_cfg.get("baseline_weights_path"),
        fallback_checkpoint=resolved_cfg["fallback_checkpoint"],
        finetune_weights_path=(
            str(resolved_cfg.get("pretrained_model_path"))
            if bool(resolved_cfg.get("finetune_mode", False))
            and resolved_cfg.get("pretrained_model_path")
            else None
        ),
        params=params,
    )


def _resolve_baseline_model(
    baseline_weights_path: str | None,
    fallback_checkpoint: str,
    finetune_weights_path: str | None = None,
) -> tuple[object, str]:
    """Resolve the baseline model for evaluation comparison.

    1. Try ``baseline_weights_path`` (``evaluation.baseline_weights_path``).
    2. Fall back to ``fallback_checkpoint`` (official YOLO COCO checkpoint).

    Returns ``(model, display_name)``.
    """
    baseline_model, baseline_display_name = _load_baseline_from_path(baseline_weights_path)
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
            baseline_model, baseline_display_name = _load_baseline_from_path(finetune_weights_path)
            if baseline_model is not None:
                print(f"Using fine-tune weights as baseline: {finetune_weights_path}")
                return baseline_model, (baseline_display_name or Path(finetune_weights_path).stem)

    baseline_checkpoint = str(fallback_checkpoint)
    print(f"\n{'='*70}")
    print("No valid local baseline model found.")
    print(f"Using official YOLO COCO checkpoint: {baseline_checkpoint}")
    print(f"{'='*70}\n")
    try:
        baseline_model = YOLO(baseline_checkpoint)
    except Exception as e:
        raise RuntimeError(
            "Failed to load a baseline model. No local baseline weights were found "
            "and loading the official YOLO checkpoint also failed. "
            f"Attempted checkpoint: '{baseline_checkpoint}'. Ensure network access or set "
            "evaluation.baseline_weights_path to a valid local file."
        ) from e
    return baseline_model, f"{Path(baseline_checkpoint).stem}-coco"


def _evaluate_and_finalize(
    *,
    model,
    experiment_name: str,
    train_output_dir: Path,
    test_path: Path,
    training_path: Path,
    image_size: int,
    train_epochs: int,
    val_split: float,
    baseline_weights_path: str | None,
    fallback_checkpoint: str,
    finetune_weights_path: str | None = None,
    params: dict | None = None,
) -> None:
    """Shared evaluation + output finalisation for every training backend.

    Resolves the baseline model, evaluates both baseline and the trained model
    on the test set, writes comparison outputs, and prints export guidance.
    Called identically from the YOLO and RF-DETR code paths.
    """
    # ── Resolve baseline model ──
    baseline_model, baseline_display_name = _resolve_baseline_model(
        baseline_weights_path,
        fallback_checkpoint,
        finetune_weights_path=finetune_weights_path,
    )

    # ── Evaluate baseline model on the test set ──
    baseline_results = None
    if baseline_model is not None:
        _, baseline_results = evaluate_and_log_model_results(
            model=baseline_model,
            model_name=baseline_display_name or "baseline",
            test_path=test_path,
            image_size=image_size,
            output_dir=train_output_dir,
            val_split=val_split,
            train_epochs=0,
            is_original=True,
            baseline_model=None,
        )

    # ── Evaluate the trained model on the test set ──
    retrained_metadata, _ = evaluate_and_log_model_results(
        model=model,
        model_name=experiment_name,
        test_path=test_path,
        image_size=image_size,
        output_dir=train_output_dir,
        val_split=val_split,
        train_epochs=train_epochs,
        baseline_model=baseline_model,
        baseline_display_name=baseline_display_name,
        baseline_results=baseline_results,
    )

    # ── Organize output files (metadata, dataset yamls, plots) ──
    reorganize_output(train_output_dir, training_path, test_path, retrained_metadata)

    # ── Generate side-by-side comparisons ──
    if baseline_model is not None:
        generate_side_by_side_comparisons(
            original_model=baseline_model,
            retrained_model=model,
            test_img_dir=test_path / "val" / "images",
            output_dir=train_output_dir,
        )

    # ── Evaluate merged-class subsets (e.g. cigarette→waste) ──
    custom_classes, _, class_mapping_config = load_class_config(params)
    raw_test_path = Path("raw_data/test")
    if class_mapping_config and raw_test_path.exists():
        # Check if any source classes were actually merged
        has_merged = any(
            src != target
            for target, sources in class_mapping_config.items()
            for src in (sources if isinstance(sources, list) else [sources])
        )
        if has_merged:
            print(f"\n{'='*70}")
            print("Evaluating merged-class subsets...")
            print(f"{'='*70}")
            merged_class_results: list[tuple[str, dict]] = []

            if baseline_model is not None:
                baseline_merged = evaluate_merged_class_subsets(
                    baseline_model, baseline_display_name or "baseline",
                    test_path, raw_test_path,
                    class_mapping_config, custom_classes,
                    imgsz=image_size,
                )
                if baseline_merged:
                    merged_class_results.append(
                        (baseline_display_name or "baseline", baseline_merged)
                    )

            trained_merged = evaluate_merged_class_subsets(
                model, experiment_name,
                test_path, raw_test_path,
                class_mapping_config, custom_classes,
                imgsz=image_size,
            )
            if trained_merged:
                merged_class_results.append((experiment_name, trained_merged))

            if merged_class_results:
                os.makedirs("./results_comparison", exist_ok=True)
                write_merged_class_results("./results_comparison", merged_class_results)

                # Append flat metrics to metrics.json for DVC tracking
                metrics_json_path = Path("metrics.json")
                if metrics_json_path.exists():
                    try:
                        with open(metrics_json_path) as f:
                            metrics_data = json.load(f)
                        for src_class, m in trained_merged.items():
                            tgt = m["target_class"]
                            for k in ("ap50", "ap", "precision", "recall", "f1_score", "n_objects"):
                                metrics_data[f"{src_class}_as_{tgt}_{k}"] = m[k]
                        with open(metrics_json_path, "w") as f:
                            json.dump(metrics_data, f, indent=4)
                    except Exception as e:
                        print(f"Warning: Could not update metrics.json with merged-class metrics: {e}")

                # Regenerate results.txt to include merged-class section
                results_csv = "./results_comparison/results.csv"
                if os.path.exists(results_csv):
                    from yolov8_training.utils.evaluate import create_formatted_table
                    create_formatted_table(results_csv)

    # ── Optionally mine hard examples for replay ──
    if params is not None:
        auto_replay_cfg = (params.get("prepare", {}) or {}).get("auto_replay", {})
        if auto_replay_cfg and auto_replay_cfg.get("enabled", False):
            try:
                build_or_update_replay_set(
                    model=model,
                    training_path=training_path,
                    train_output_dir=train_output_dir,
                    config=auto_replay_cfg,
                )
            except Exception as e:
                print(f"Warning: Auto-replay mining failed: {e}")

    delete_unused_folders()
    _print_export_guidance(train_output_dir, experiment_name)


def _print_export_guidance(train_output_dir: Path, experiment_name: str) -> None:
    """
    Print helpful guidance for exporting the trained model as the new baseline.
    """
    print("\n" + "=" * 70)
    print("Training complete! Next steps:")
    print("=" * 70)
    print("\nTo make this run the baseline for future comparisons:\n")
    print(f"  python yolov8_training/utils/export_baseline.py --run-dir {train_output_dir}")
    print("\nThen track it with DVC:\n")
    print("  dvc add models/current_best/best.pt models/current_best/metadata.yaml")
    print("  dvc push")
    print("  git add models/current_best/best.pt.dvc models/current_best/metadata.yaml.dvc")
    print(f"  git commit -m \"Update baseline to {experiment_name}\"")
    print("\n" + "=" * 70 + "\n")

def main(args):
    """
    Main function to set up dataset, process data, and train the YOLO model.

    Args:
        args: Parsed command line arguments.
    """
    run_prepare_stage(args)
    run_train_eval_stage(args)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=["prepare", "train"],
        help="Specify which pipeline stage to run."
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model key to use from params.yaml models.* (overrides train.model).",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "-d", "--dataset-name", required=True, help="Name of the dataset"
    )
    """
    parser.add_argument(
        "-name", "--experiment-name", required=True, help="Name of the experiment"
    )
    """
    parser.add_argument(
        "-vs", "--val-split", type=float, default=0.1, help="Validation split ratio"
    )
    parser.add_argument(
        "-ts", "--test-split", type=float, default=0.1, help="Test split ratio"
    )
    parser.add_argument(
        "--recreate-dataset", action="store_true", help="Recreate dataset if it exists"
    )
    parser.add_argument(
        "--augment-multiplier", type=int, default=1, 
        help="By which factor the training data will be multiplied with augmented data. Default is 1, meaning there is no added augmentation by default."
    )
    parser.add_argument(
        "--folder-subset", action="append", nargs=2, metavar=('FOLDER', 'RATIO'),
        help="Use subset/oversample images from specific folders. Format: --folder-subset uavvaste 0.5 (50%%) --folder-subset small_dataset 2.0 (200%% = oversample)"
    )
    args = parser.parse_args()

    # Add seeds for reproducibility
    SEED = args.seed

    os.environ["PYTHONHASHSEED"] = str(SEED)

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    # When deterministic algorithms are enabled on CUDA, some ops (e.g. cuBLAS-backed
    # attention) require CUBLAS_WORKSPACE_CONFIG to be set for deterministic results.
    # RF-DETR hits this path quickly; set a safe default if the user hasn't.
    if torch.cuda.is_available():
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    # RF-DETR currently uses some CUDA ops (e.g. grid_sampler_2d_backward_cuda)
    # without a deterministic implementation. In strict deterministic mode PyTorch
    # raises; allow warn-only for RF-DETR so training can proceed.
    backend_hint = "yolo"
    if args.stage != "prepare":
        try:
            args.resolved_training_config = resolve_training_config(args)
            backend_hint = args.resolved_training_config["backend"]
        except Exception:
            backend_hint = "yolo"
    try:
        normalized_backend = _normalize_model_type(backend_hint)
    except Exception:
        normalized_backend = "yolo"

    torch.use_deterministic_algorithms(True, warn_only=(normalized_backend == "rfdetr"))
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

    # --- This logic calls the correct function based on the --stage argument ---
    if args.stage == "prepare":
        run_prepare_stage(args)
    elif args.stage == "train":
        run_train_eval_stage(args)
    else:  
        main(args)
