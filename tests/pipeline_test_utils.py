from __future__ import annotations

"""Shared helpers for assembling the lightweight datasets/params used in tests."""

import copy
from pathlib import Path
from typing import Any, Dict

import cv2
import numpy as np
import yaml


BASE_PARAMS: Dict[str, Any] = {
    "data": {
        "dataset_name": "waste-detection",
        "experiment_name": "waste-detection",
        "custom_classes": ["waste"],
        "use_coco_classes": False,
    },
    "prepare": {
        "val_split": 0.5,
        "test_split": 0.0,
        "augment_multiplier": 1,
        "folder_subsets": {},
    },
    "train": {
        "model_size": "n",
        "image_size": 320,
        "epochs": 1,
        "batch_size": 1,
        "finetune_mode": False,
        "pretrained_model_path": "models/current_best/best.pt",
        "finetune_lr": 0.0001,
        "finetune_epochs": 1,
        "freeze_backbone": False,
    },
    "evaluation": {
        "baseline_weights_path": "models/current_best/best.pt",
    },
}


def _merge_dict(target: dict, overrides: dict) -> dict:
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            target[key] = _merge_dict(target[key], value)
        else:
            target[key] = value
    return target


def write_params_yaml(workspace: Path, overrides: dict | None = None) -> dict:
    params = _merge_dict(copy.deepcopy(BASE_PARAMS), overrides or {})
    with open(workspace / "params.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(params, f, sort_keys=False)
    return params


def create_minimal_dataset(base_dir: Path) -> None:
    """Create the small synthetic dataset the pipeline tests rely on."""

    # Training folder 1
    images_dir = base_dir / "raw_data" / "train" / "source1" / "images"
    labels_dir = base_dir / "raw_data" / "train" / "source1" / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    img1 = np.full((128, 128, 3), 100, dtype=np.uint8)
    cv2.rectangle(img1, (32, 32), (96, 96), (255, 255, 255), -1)
    cv2.imwrite(str(images_dir / "img1.jpg"), img1)
    with open(labels_dir / "img1.txt", "w", encoding="utf-8") as f:
        f.write("0 0.5 0.5 0.5 0.5\n")

    img2 = np.full((128, 128, 3), 150, dtype=np.uint8)
    cv2.imwrite(str(images_dir / "img2.jpg"), img2)
    with open(labels_dir / "img2.txt", "w", encoding="utf-8") as f:
        f.write("0 0.5 0.5 0.3 0.3\n")

    # Training folder 2
    images_dir2 = base_dir / "raw_data" / "train" / "source2" / "images"
    labels_dir2 = base_dir / "raw_data" / "train" / "source2" / "labels"
    images_dir2.mkdir(parents=True, exist_ok=True)
    labels_dir2.mkdir(parents=True, exist_ok=True)
    img3 = np.full((128, 128, 3), 60, dtype=np.uint8)
    cv2.line(img3, (0, 0), (127, 127), (255, 255, 255), 3)
    cv2.imwrite(str(images_dir2 / "img3.jpg"), img3)
    with open(labels_dir2 / "img3.txt", "w", encoding="utf-8") as f:
        f.write("0 0.5 0.5 0.2 0.2\n")

    # Held-out test scene
    test_images_dir = base_dir / "raw_data" / "test" / "sourceT" / "images"
    test_labels_dir = base_dir / "raw_data" / "test" / "sourceT" / "labels"
    test_images_dir.mkdir(parents=True, exist_ok=True)
    test_labels_dir.mkdir(parents=True, exist_ok=True)
    imgT = np.full((128, 128, 3), 80, dtype=np.uint8)
    cv2.circle(imgT, (64, 64), 20, (255, 255, 255), -1)
    cv2.imwrite(str(test_images_dir / "test1.jpg"), imgT)
    with open(test_labels_dir / "test1.txt", "w", encoding="utf-8") as f:
        f.write("0 0.5 0.5 0.25 0.25\n")


def build_args(dataset_name: str, overrides: dict | None = None):
    """Mirror the CLI arguments our tests pass into the train/prepare stages."""
    args = {
        "stage": None,
        "seed": 42,
        "dataset_name": dataset_name,
        "model_size": "n",
        "image_size": 320,
        "epochs": 1,
        "batch_size": 1,
        "val_split": 0.5,
        "test_split": 0.0,
        "recreate_dataset": True,
        "augment_multiplier": 1,
        "folder_subset": None,
    }
    if overrides:
        args.update(overrides)
    from types import SimpleNamespace

    return SimpleNamespace(**args)
