import argparse
import os
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
from yolov8_training.backends import rfdetr_backend, yolo_backend
from yolov8_training.backends.yolo_backend import train_model
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

    rfdetr_variant = str(
        model_cfg.get("variant")
        or model_cfg.get("model")
        or rfdetr_backend._infer_rfdetr_variant(str(selected_model))
    )
    rfdetr_batch_size = int(model_cfg.get("batch_size", shared_batch_size))
    explicit_grad_accum = model_cfg.get("grad_accum_steps")
    target_effective_batch = int(model_cfg.get("target_effective_batch", 16))
    if explicit_grad_accum is not None:
        rfdetr_grad_accum = int(explicit_grad_accum)
    else:
        rfdetr_grad_accum = max(1, target_effective_batch // rfdetr_batch_size)

    rfdetr_resolution = rfdetr_backend._normalize_rfdetr_resolution(
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
            rfdetr_model = rfdetr_backend._get_rfdetr_model(
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
        model, train_output_dir, display_name, image_size, train_epochs = (
            rfdetr_backend.train_rfdetr_backend(
                training_path=training_path,
                test_path=test_path,
                dataset_name=str(dataset_name),
                resolved_cfg=resolved_cfg,
                experiment_name=experiment_name,
            )
        )

        # ── Evaluate & finalize (shared with YOLO path) ──
        _evaluate_and_finalize(
            model=model,
            experiment_name=display_name,
            train_output_dir=train_output_dir,
            test_path=test_path,
            training_path=training_path,
            image_size=image_size,
            train_epochs=train_epochs,
            val_split=val_split,
            baseline_weights_path=resolved_cfg.get("baseline_weights_path"),
            fallback_checkpoint=resolved_cfg["fallback_checkpoint"],
            params=params,
        )
        return

    model, _results, train_output_dir, final_experiment_name = yolo_backend.train_yolo(
        training_path=training_path,
        resolved_cfg=resolved_cfg,
        experiment_name=experiment_name,
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
