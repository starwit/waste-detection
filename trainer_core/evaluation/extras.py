from __future__ import annotations

import os
import shutil
import tempfile
import traceback
from pathlib import Path

import cv2
import numpy as np
import yaml


def generate_side_by_side_comparisons(
    original_model, retrained_model, test_img_dir, output_dir, conf_threshold=0.25
):
    side_by_side_dir = output_dir / "side_by_side_comparisons"
    side_by_side_dir.mkdir(exist_ok=True)

    for img_path in test_img_dir.glob("*"):
        if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
            continue
        original_results = original_model.predict(
            str(img_path), conf=conf_threshold, save=False, verbose=False
        )
        retrained_results = retrained_model.predict(
            str(img_path), conf=conf_threshold, save=False, verbose=False
        )

        original_img = original_results[0].plot()
        retrained_img = retrained_results[0].plot()
        comparison_img = np.hstack((original_img, retrained_img))
        save_path = side_by_side_dir / f"comparison_{img_path.name}"
        cv2.imwrite(str(save_path), comparison_img)


def _load_dataset_config_for_scenes(data: str) -> dict:
    try:
        with open(data, "r") as f:
            parsed = yaml.safe_load(f) or {}
    except FileNotFoundError:
        print(f"Error: Dataset YAML not found at {data}")
        return {}
    except (OSError, yaml.YAMLError) as e:
        print(f"Error reading dataset YAML {data}: {e}")
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _collect_scene_images(val_images_dir: Path, val_labels_dir: Path) -> dict:
    scene_images = {}
    for img_path in val_images_dir.glob("*"):
        if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
            continue
        if "__scene_" not in img_path.name:
            continue
        parts = img_path.stem.split("__scene_")
        if len(parts) <= 1:
            continue
        scene_name = parts[1]
        scene_images.setdefault(scene_name, []).append(
            (img_path, val_labels_dir / img_path.with_suffix(".txt").name)
        )
    return scene_images


def _copy_scene_to_temp(scene_name: str, images_labels: list) -> tuple[Path | None, int]:
    temp_path = Path(tempfile.mkdtemp())
    temp_images_dir = temp_path / "images"
    temp_labels_dir = temp_path / "labels"
    os.makedirs(temp_images_dir, exist_ok=True)
    os.makedirs(temp_labels_dir, exist_ok=True)

    copied_files = 0
    for img_path, label_path in images_labels:
        if not img_path.exists() or not label_path.exists():
            print(
                f"Warning: Source file missing during copy for scene '{scene_name}': {img_path} or {label_path}"
            )
            continue
        new_img_name = img_path.name.replace(f"__scene_{scene_name}", "")
        new_label_name = label_path.name.replace(f"__scene_{scene_name}", "")
        try:
            shutil.copy(img_path, temp_images_dir / new_img_name)
            shutil.copy(label_path, temp_labels_dir / new_label_name)
            copied_files += 1
        except OSError as copy_e:
            print(f"Error copying {img_path} or {label_path} to {temp_path}: {copy_e}")
    return (temp_path if copied_files > 0 else None), copied_files


def _write_scene_yaml(temp_path: Path, dataset_config: dict, scene_name: str) -> Path | None:
    temp_yaml_path = temp_path / f"scene_{scene_name}.yaml"
    scene_config = {
        "path": str(temp_path),
        "train": "images",
        "val": "images",
        "nc": len(dataset_config.get("names", [])),
        "names": dataset_config.get("names", {}),
    }
    try:
        with open(temp_yaml_path, "w") as f:
            yaml.dump(scene_config, f)
    except OSError as yaml_e:
        print(f"Error writing temporary YAML {temp_yaml_path}: {yaml_e}")
        return None
    return temp_yaml_path


def _validate_scene(model, temp_yaml_path: Path, class_ids: list | None, kwargs: dict) -> float:
    eval_kwargs = {k: v for k, v in kwargs.items() if k in ["imgsz", "batch", "conf", "iou"]}
    eval_kwargs["workers"] = 0
    eval_kwargs["batch"] = 1
    scene_results = model.val(
        data=str(temp_yaml_path),
        classes=class_ids if class_ids else None,
        verbose=False,
        save=False,
        plots=False,
        **eval_kwargs,
    )
    return float(getattr(scene_results, "fitness", 0.0))


def calculate_scene_metrics(model, data, **kwargs):
    """
    Calculate fitness metrics for each scene in the test dataset.
    Scenes are identified by the __scene_ suffix in image filenames.
    """
    dataset_config = _load_dataset_config_for_scenes(data)
    if not dataset_config:
        return {}

    if "path" not in dataset_config or "val" not in dataset_config:
        print(f"Error: Dataset YAML {data} is missing 'path' or 'val' key.")
        return {}

    dataset_path = Path(dataset_config["path"])
    val_images_dir = dataset_path / dataset_config["val"]
    val_labels_dir = val_images_dir.parent / "labels"

    from trainer_core.evaluation import validate as validate_core

    _, class_ids = validate_core.get_dataset_classes(data)
    scene_images = _collect_scene_images(val_images_dir, val_labels_dir)
    scene_metrics = {}

    for scene_name, images_labels in scene_images.items():
        temp_path = None
        try:
            temp_path, copied_files = _copy_scene_to_temp(scene_name, images_labels)
            if not temp_path or copied_files == 0:
                print(
                    f"Warning: No files were copied for scene '{scene_name}'. Skipping validation."
                )
                continue

            temp_yaml_path = _write_scene_yaml(temp_path, dataset_config, scene_name)
            if not temp_yaml_path:
                continue

            fitness = _validate_scene(model, temp_yaml_path, class_ids, kwargs)
            scene_metrics[f"scene_{scene_name}_fitness"] = fitness
        except (OSError, ValueError, RuntimeError) as e:
            print(f"Error evaluating scene {scene_name} using data {data}: {e}")
            traceback.print_exc()
            scene_metrics[f"scene_{scene_name}_fitness"] = 0.0
        finally:
            if temp_path and temp_path.exists():
                shutil.rmtree(temp_path, ignore_errors=True)

    return scene_metrics


def evaluate_merged_class_subsets(model, model_name, test_path, raw_test_path, class_mapping_config, custom_classes, **kwargs):
    """Evaluate model on test-image subsets that originally contained merged source classes."""
    if not class_mapping_config or not custom_classes:
        return {}
    if not raw_test_path or not Path(raw_test_path).exists():
        return {}

    raw_test_path = Path(raw_test_path)
    merged_sources = {}
    for target, sources in class_mapping_config.items():
        if isinstance(sources, str):
            sources = [sources]
        for src in sources:
            if src != target:
                merged_sources[src] = target

    if not merged_sources:
        return {}

    original_class_to_id = {cls: i for i, cls in enumerate(custom_classes)}
    test_images_dir = test_path / "val" / "images"
    test_labels_dir = test_path / "val" / "labels"
    dataset_yaml = test_path / "dataset.yaml"

    if not dataset_yaml.exists() or not test_images_dir.exists():
        return {}

    with open(dataset_yaml) as f:
        ds_config = yaml.safe_load(f) or {}

    results = {}

    for src_class, target_class in merged_sources.items():
        src_class_id = original_class_to_id.get(src_class)
        if src_class_id is None:
            continue

        matching_stems: set[str] = set()
        n_objects = 0
        for scene_dir in sorted(raw_test_path.iterdir()):
            if not scene_dir.is_dir():
                continue
            scene_name = scene_dir.name
            labels_dir = scene_dir / "labels"
            if not labels_dir.exists():
                continue
            for label_file in labels_dir.glob("*.txt"):
                if label_file.stat().st_size == 0:
                    continue
                file_hits = 0
                with open(label_file) as f:
                    for line in f:
                        parts = line.strip().split()
                        if not parts:
                            continue
                        try:
                            if int(parts[0]) == src_class_id:
                                file_hits += 1
                        except ValueError:
                            continue
                if file_hits:
                    prepared_stem = f"{label_file.stem}__scene_{scene_name}"
                    matching_stems.add(prepared_stem)
                    n_objects += file_hits

        if not matching_stems:
            print(f"  No test images found containing '{src_class}' annotations.")
            continue

        temp_dir = Path(tempfile.mkdtemp())
        try:
            temp_images = temp_dir / "images"
            temp_labels = temp_dir / "labels"
            temp_images.mkdir()
            temp_labels.mkdir()

            copied = 0
            for stem in matching_stems:
                for ext in (".jpg", ".jpeg", ".png", ".bmp"):
                    img_src = test_images_dir / f"{stem}{ext}"
                    if not img_src.exists():
                        continue
                    shutil.copy2(img_src, temp_images / img_src.name)
                    label_src = test_labels_dir / f"{stem}.txt"
                    if label_src.exists():
                        shutil.copy2(label_src, temp_labels / f"{stem}.txt")
                    else:
                        (temp_labels / f"{stem}.txt").touch()
                    copied += 1
                    break

            if copied == 0:
                print(f"  Warning: Could not locate prepared test images for '{src_class}' subset.")
                continue

            temp_yaml = temp_dir / "dataset.yaml"
            temp_config = {
                "path": str(temp_dir),
                "train": "images",
                "val": "images",
                "nc": ds_config.get("nc", len(ds_config.get("names", []))),
                "names": ds_config.get("names", {}),
            }
            with open(temp_yaml, "w") as f:
                yaml.dump(temp_config, f)

            eval_kwargs = {k: v for k, v in kwargs.items() if k in ["imgsz", "batch", "conf", "iou"]}
            eval_kwargs["workers"] = 0
            eval_kwargs["batch"] = 1

            from trainer_core.evaluation import validate as validate_core

            _, class_ids = validate_core.get_dataset_classes(str(temp_yaml))
            subset_metrics = model.val(
                data=str(temp_yaml),
                classes=class_ids if class_ids else None,
                verbose=False,
                save=False,
                plots=False,
                **eval_kwargs,
            )

            p = float(subset_metrics.results_dict.get("metrics/precision(B)", 0.0))
            r = float(subset_metrics.results_dict.get("metrics/recall(B)", 0.0))
            ap50 = float(subset_metrics.results_dict.get("metrics/mAP50(B)", 0.0))
            ap = float(subset_metrics.results_dict.get("metrics/mAP50-95(B)", 0.0))
            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0

            results[src_class] = {
                "target_class": target_class,
                "precision": p,
                "recall": r,
                "ap50": ap50,
                "ap": ap,
                "f1_score": f1,
                "n_objects": n_objects,
            }
            print(
                f"  Merged-class subset '{src_class}→{target_class}': "
                f"{copied} images, {n_objects} objects, mAP50={ap50:.4f}, mAP50-95={ap:.4f}"
            )
        except (OSError, ValueError, RuntimeError) as e:
            print(f"  Warning: Could not evaluate '{src_class}' subset: {e}")
            traceback.print_exc()
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    return results


__all__ = [
    "calculate_scene_metrics",
    "evaluate_merged_class_subsets",
    "generate_side_by_side_comparisons",
]
