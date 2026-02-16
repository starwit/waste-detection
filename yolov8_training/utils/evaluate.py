import argparse
import csv
import os
from pathlib import Path
import shutil
import tempfile
import traceback
import json

# Set matplotlib backend to non-GUI before any imports that might use it
# This prevents "Cannot load backend 'tkagg'" errors on headless systems
import matplotlib
matplotlib.use('Agg')

import cv2
import numpy as np
import yaml
from tabulate import tabulate
from ultralytics import YOLO


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

        # Plot both predictions
        original_img = original_results[0].plot()
        retrained_img = retrained_results[0].plot()

        # Stack images side-by-side
        comparison_img = np.hstack((original_img, retrained_img))

        # Save the side-by-side image
        save_path = side_by_side_dir / f"comparison_{img_path.name}"
        cv2.imwrite(str(save_path), comparison_img)


def evaluate_and_log_model_results(
    model,
    model_name,
    test_path,
    image_size,
    output_dir,
    val_split,
    train_epochs=0,
    is_original=False,
    baseline_model=None,
    baseline_display_name=None,
    baseline_results=None,
):
    """
    Evaluates the model on the test set, prepares metadata, and appends results to the CSV.

    Returns:
        tuple: ``(metadata, results)`` where *metadata* is the experiment info dict
        and *results* is the metrics dict produced by :func:`validate_model`.
    """
    # Get class information from dataset
    dataset_yaml_path = test_path / "dataset.yaml"
    class_names, class_ids = get_dataset_classes(dataset_yaml_path)

    # Run evaluation on the model
    results = validate_model(
        model, data=str(dataset_yaml_path), class_ids=class_ids, imgsz=image_size, workers=0
    )

    # Prepare metadata
    metadata = {
        "experiment_name": model_name,
        "split_parameters": {
            "val_split": val_split,
        },
        "num_epochs": train_epochs,
        "model_size": model.model_name if hasattr(model, "model_name") else "Unknown",
        "image_size": image_size,
    }

    # Append results to CSV
    append_results_to_csv(output_dir, results, metadata, is_original)

    # Also add to the global comparison table
    if not is_original:
        # Get the base model results for comparison if this is a retrained model
        try:
            # Use the provided baseline_model if available, otherwise fall back to COCO model
            if baseline_model is not None:
                base_model = baseline_model
                base_model_name = (
                    str(baseline_display_name)
                    if baseline_display_name
                    else getattr(baseline_model, "model_name", "Baseline Model")
                )
                print(f"Using provided baseline model for comparison: {base_model_name}")
            else:
                # Extract model size for fallback to COCO model
                model_size = "m"  # Default size if we can't determine it
                if hasattr(model, "model") and hasattr(model.model, "yaml"):
                    # Try to get size from model yaml
                    if "model_name" in model.model.yaml:
                        name = model.model.yaml["model_name"]
                        if "yolov8" in name and len(name) > 6:
                            model_size = name[6]  # Extract the size character (n, s, m, l, x)
                base_model = YOLO(f"yolov8{model_size}.pt")
                print(f"Using COCO baseline model: yolov8{model_size}.pt")
                base_model_name = f"YOLOv8{model_size} (COCO)"

            base_results = validate_model(
                base_model, data=str(dataset_yaml_path), class_ids=class_ids, imgsz=image_size, workers=0, write_json=False
            ) if baseline_results is None else baseline_results
            # Determine base model name for display
            mean_table(base_results, results, model_name, True, base_model_name)
        except Exception as e:
            print(f"Warning: Could not load base model for comparison: {e}")
            # Still add the retrained model results to the table
            mean_table(None, results, model_name, False, None)

    return metadata, results


def get_dataset_classes(dataset_yaml_path):
    """
    Get class information from dataset.yaml file.
    
    Args:
        dataset_yaml_path (Path): Path to dataset.yaml file
        
    Returns:
        dict: Class mapping {id: name}
        list: List of class IDs
    """
    try:
        with open(dataset_yaml_path, "r") as f:
            dataset_config = yaml.safe_load(f) or {}
    except Exception as e:
        print(f"Warning: Could not read dataset classes from {dataset_yaml_path}: {e}")
        return {}, []

    names_data = dataset_config.get("names", {})

    # Check if names is a list or dict and handle accordingly
    if isinstance(names_data, list):
        # Convert list to dictionary mapping indices to names
        class_names = {i: name for i, name in enumerate(names_data)}
        class_ids = list(range(len(names_data)))
    elif isinstance(names_data, dict):
        # Use existing dictionary format
        class_names = names_data
        class_ids = list(class_names.keys())
    else:
        # Fallback for unexpected format
        print(
            f"Warning: Unexpected format for 'names' in {dataset_yaml_path}. "
            f"Expected list or dict, got {type(names_data)}"
        )
        class_names = {}
        class_ids = []

    return class_names, class_ids


def _extract_per_class_metrics(metrics, data):
    """Extract per-class metrics from model validation results.

    Works with both Ultralytics YOLO metrics (``metrics.box``) and RF-DETR
    adapter metrics (``metrics.per_class``).

    Returns:
        dict: Mapping of class_name -> {precision, recall, map50, map, f1_score}
    """
    per_class = {}
    try:
        # Load class names from dataset yaml
        with open(data) as f:
            ds_config = yaml.safe_load(f)
        names = ds_config.get("names", {})
        if isinstance(names, list):
            names = {i: n for i, n in enumerate(names)}

        # Ultralytics-style: metrics.box has per-class arrays
        if hasattr(metrics, 'box') and hasattr(getattr(metrics, 'box', None), 'ap_class_index'):
            box = metrics.box
            ap_class_idx = box.ap_class_index
            if hasattr(ap_class_idx, '__len__') and len(ap_class_idx) > 0:
                ap50_vals = box.ap50
                ap_vals = box.ap  # mAP50-95 per class
                for i, cls_idx in enumerate(ap_class_idx):
                    cls_name = names.get(int(cls_idx), f"class_{int(cls_idx)}")
                    p = float(box.p[i])
                    r = float(box.r[i])
                    a50 = float(ap50_vals[i])
                    a = float(ap_vals[i])
                    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
                    per_class[cls_name] = {
                        "precision": p,
                        "recall": r,
                        "map50": a50,
                        "map": a,
                        "f1_score": f1,
                    }
        # RF-DETR adapter: metrics.per_class dict
        elif hasattr(metrics, 'per_class') and metrics.per_class:
            return dict(metrics.per_class)
    except Exception as e:
        print(f"Warning: Could not extract per-class metrics: {e}")

    return per_class


def validate_model(model, data, class_ids=None, write_json=True, **kwargs):
    """
    Validate model with dynamic class selection.
    
    Args:
        model: YOLO model to validate
        data: Path to dataset.yaml or dataset configuration
        class_ids: List of class IDs to validate (if None, validate all classes)
        write_json: Whether to write metrics.json
        **kwargs: Additional validation arguments
    """
    # If class_ids is provided, use it; otherwise validate all classes
    validation_kwargs = kwargs.copy()
    if class_ids is not None:
        validation_kwargs["classes"] = class_ids
    
    metrics = model.val(
        data=data, verbose=False, save=False, plots=False, **validation_kwargs
    )

    # Compute average per-frame time in ms (preprocess + inference + postprocess)
    spd = metrics.speed
    ms_per_frame = (
        float(spd.get("preprocess", 0.0))
        + float(spd.get("inference", 0.0))
        + float(spd.get("postprocess", 0.0))
    )

    # Extract values directly from results_dict
    precision = float(metrics.results_dict["metrics/precision(B)"])
    recall = float(metrics.results_dict["metrics/recall(B)"])
    map50 = float(metrics.results_dict["metrics/mAP50(B)"])
    map50_95 = float(metrics.results_dict["metrics/mAP50-95(B)"])
    fitness = float(metrics.fitness)

    # Calculate F1 score
    f1_score = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0
    )

    # Create metrics dictionary
    metrics_dict = {
        "img_size": kwargs.get("imgsz"),     
        "precision": precision,
        "recall": recall,
        "map": map50_95,
        "map50": map50,
        "fitness": fitness,
        "f1_score": f1_score,
        "ms/frame": ms_per_frame,
    }

    # Extract per-class metrics
    per_class = _extract_per_class_metrics(metrics, data)
    if per_class:
        metrics_dict["per_class"] = per_class

    # Get scene-specific metrics
    scene_metrics = calculate_scene_metrics(model, data, **kwargs)
    metrics_dict.update(scene_metrics)

    if write_json:
        with open("metrics.json", "w") as f:
            json.dump(metrics_dict, f, indent=4)

    return metrics_dict


def _load_dataset_config_for_scenes(data: str) -> dict:
    try:
        with open(data, "r") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Dataset YAML not found at {data}")
        return {}
    except Exception as e:
        print(f"Error reading dataset YAML {data}: {e}")
        return {}


def _collect_scene_images(val_images_dir: Path, val_labels_dir: Path) -> dict:
    scene_images = {}
    for img_path in val_images_dir.glob("*"):
        if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
            continue
        if "__scene_" in img_path.name:
            parts = img_path.stem.split("__scene_")
            if len(parts) > 1:
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
        if img_path.exists() and label_path.exists():
            new_img_name = img_path.name.replace(f"__scene_{scene_name}", "")
            new_label_name = label_path.name.replace(f"__scene_{scene_name}", "")
            try:
                shutil.copy(img_path, temp_images_dir / new_img_name)
                shutil.copy(label_path, temp_labels_dir / new_label_name)
                copied_files += 1
            except Exception as copy_e:
                print(f"Error copying {img_path} or {label_path} to {temp_path}: {copy_e}")
        else:
            print(
                f"Warning: Source file missing during copy for scene '{scene_name}': {img_path} or {label_path}"
            )
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
        return temp_yaml_path
    except Exception as yaml_e:
        print(f"Error writing temporary YAML {temp_yaml_path}: {yaml_e}")
        return None


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
    # Load the dataset yaml to get the path to test images
    dataset_config = _load_dataset_config_for_scenes(data)
    if not dataset_config:
        return {}

    if "path" not in dataset_config or "val" not in dataset_config:
        print(f"Error: Dataset YAML {data} is missing 'path' or 'val' key.")
        return {}

    dataset_path = Path(dataset_config["path"])
    val_images_dir = dataset_path / dataset_config["val"]
    val_labels_dir = val_images_dir.parent / "labels"

    # Get class information
    class_names, class_ids = get_dataset_classes(data)

    # Group images by scene
    scene_images = _collect_scene_images(val_images_dir, val_labels_dir)

    # Results dictionary to store scene metrics
    scene_metrics = {}

    # Process each scene separately
    for scene_name, images_labels in scene_images.items():
        temp_path = None
        try:
            temp_path, copied_files = _copy_scene_to_temp(scene_name, images_labels)
            if not temp_path or copied_files == 0:
                print(
                    f"Warning: No files were copied for scene '{scene_name}'. Skipping validation."
                )
                continue

            # Create temp yaml for this scene
            temp_yaml_path = _write_scene_yaml(temp_path, dataset_config, scene_name)
            if not temp_yaml_path:
                continue

            # Run validation on this scene
            try:
                # Run validation and store fitness
                fitness = _validate_scene(model, temp_yaml_path, class_ids, kwargs)
                scene_metrics[f"scene_{scene_name}_fitness"] = fitness

            except Exception as e:
                # Print details if validation for a specific scene fails
                print(
                    f"Error evaluating scene {scene_name} using data {temp_yaml_path}:"
                )
                traceback.print_exc()
                scene_metrics[f"scene_{scene_name}_fitness"] = 0.0

        finally:
            # Ensure cleanup happens regardless of success or failure after evaluation is finished
            if temp_path and temp_path.exists():
                shutil.rmtree(temp_path, ignore_errors=True)

    return scene_metrics


def evaluate_merged_class_subsets(model, model_name, test_path, raw_test_path,
                                   class_mapping_config, custom_classes, **kwargs):
    """Evaluate model on test-image subsets that originally contained merged source classes.

    When classes are merged (e.g. cigarette → waste), this evaluates the model
    specifically on images that originally had cigarette annotations.  This enables
    direct comparison between:

    * A model with cigarette as a separate class → per-class AP already shows this.
    * A model with cigarette merged into waste → this subset evaluation shows it.

    Args:
        model: The model to evaluate.
        model_name: Display name for the model.
        test_path: Path to the prepared test dataset (mapped labels).
        raw_test_path: Path to ``raw_data/test/`` (original labels before mapping).
        class_mapping_config: ``data.class_mapping`` from params.yaml.
        custom_classes: ``data.custom_classes`` from params.yaml.
        **kwargs: Extra eval kwargs forwarded to ``model.val()`` (e.g. ``imgsz``).

    Returns:
        dict: ``source_class_name`` → ``{target_class, precision, recall, ap50, ap, f1_score, n_objects}``
    """
    if not class_mapping_config or not custom_classes:
        return {}
    if not raw_test_path or not Path(raw_test_path).exists():
        return {}

    raw_test_path = Path(raw_test_path)

    # Find source classes that were merged into a different target
    merged_sources = {}  # source_class → target_class
    for target, sources in class_mapping_config.items():
        if isinstance(sources, str):
            sources = [sources]
        for src in sources:
            if src != target:
                merged_sources[src] = target

    if not merged_sources:
        return {}

    # Build original class name → ID mapping (before mapping was applied)
    original_class_to_id = {cls: i for i, cls in enumerate(custom_classes)}

    # Prepared test set paths
    test_images_dir = test_path / "val" / "images"
    test_labels_dir = test_path / "val" / "labels"
    dataset_yaml = test_path / "dataset.yaml"

    if not dataset_yaml.exists() or not test_images_dir.exists():
        return {}

    with open(dataset_yaml) as f:
        ds_config = yaml.safe_load(f)

    results = {}

    for src_class, target_class in merged_sources.items():
        src_class_id = original_class_to_id.get(src_class)
        if src_class_id is None:
            continue

        # Scan raw test labels for images containing this source class
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

        # Create temp directory with matching images + their MAPPED labels
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
                    if img_src.exists():
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

            # Write temp dataset.yaml
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

            # Evaluate
            eval_kwargs = {k: v for k, v in kwargs.items()
                           if k in ["imgsz", "batch", "conf", "iou"]}
            eval_kwargs["workers"] = 0
            eval_kwargs["batch"] = 1

            _, class_ids = get_dataset_classes(str(temp_yaml))

            subset_metrics = model.val(
                data=str(temp_yaml),
                classes=class_ids if class_ids else None,
                verbose=False, save=False, plots=False,
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

            print(f"  Merged-class subset '{src_class}→{target_class}': "
                  f"{copied} images, {n_objects} objects, mAP50={ap50:.4f}, mAP50-95={ap:.4f}")

        except Exception as e:
            print(f"  Warning: Could not evaluate '{src_class}' subset: {e}")
            traceback.print_exc()
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    return results


def write_merged_class_results(output_dir, all_model_results):
    """Write merged-class subset results to CSV and return formatted table.

    Args:
        output_dir: Directory to write ``merged_class_results.csv``.
        all_model_results: list of ``(model_name, results_dict)`` tuples.
    """
    csv_path = Path(output_dir) / "merged_class_results.csv"
    columns = ["MODEL", "source_class", "target_class", "n_objects",
               "precision", "recall", "ap50", "ap", "f1_score"]

    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        if write_header:
            writer.writeheader()
        for model_name, res in all_model_results:
            for src_class, m in res.items():
                row = {
                    "MODEL": model_name,
                    "source_class": src_class,
                    "target_class": m["target_class"],
                    "n_objects": m["n_objects"],
                }
                for col in ("precision", "recall", "ap50", "ap", "f1_score"):
                    row[col] = f"{m[col]:.4f}"
                writer.writerow(row)


def _format_merged_class_table(csv_path):
    """Format merged-class CSV into a readable table grouped by model."""
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        return "No merged-class metrics available.\n"

    from collections import OrderedDict
    models: dict[str, list[dict]] = OrderedDict()
    for row in rows:
        models.setdefault(row["MODEL"], []).append(row)

    metric_cols = ["n_objects", "precision", "recall", "ap50", "ap", "f1_score"]
    output_parts: list[str] = []

    for model_name, model_rows in models.items():
        output_parts.append(f"  Model: {model_name}")
        table_data = []
        for row in model_rows:
            table_row = [f"{row['source_class']}→{row['target_class']}"]
            for col in metric_cols:
                val = row.get(col, "-")
                try:
                    fval = float(val)
                    table_row.append(f"{fval:.4f}" if col != "n_objects" else str(int(fval)))
                except (ValueError, TypeError):
                    table_row.append(str(val))
            table_data.append(table_row)

        headers = ["merged_class"] + metric_cols
        table_str = tabulate(table_data, headers=headers, tablefmt="simple")
        output_parts.append(table_str)
        output_parts.append("")

    return "\n".join(output_parts)


def append_results_to_csv(train_output_dir, results, metadata, is_original=False):
    """
    Append test results to CSV.
    If `is_original` is True, mark the model as the original (not retrained) in the CSV.
    """
    model_type = (
        f"{metadata['experiment_name']} (original)"
        if is_original
        else metadata["experiment_name"]
    )

    # Prepare data for CSV
    csv_data = {
            "MODEL": model_type,
            "img_size": metadata["image_size"], 
            "precision": results["precision"],
            "recall":    results["recall"],
            "map":       results["map"],
            "map50":     results["map50"],
            "fitness":   results["fitness"],
            "f1_score":  results["f1_score"],
            "ms/frame":  results["ms/frame"],
            "val_split": metadata["split_parameters"]["val_split"],
        }
    # Add scene-specific fitness values
    for key, value in results.items():
        if key.startswith("scene_") and key.endswith("_fitness"):
            csv_data[key] = value

    # Append to results CSV
    csv_path = train_output_dir / "test_results.csv"
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_data.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(csv_data)

    # Write per-class metrics to a separate CSV
    per_class = results.get("per_class", {})
    if per_class:
        _append_per_class_to_csv(
            train_output_dir / "test_per_class_results.csv",
            per_class,
            model_type,
        )


def _append_per_class_to_csv(csv_path, per_class, model_name):
    """Append per-class metrics rows to a CSV file.

    Args:
        csv_path: Path to the per-class CSV file.
        per_class: dict mapping class_name -> {precision, recall, map50, map, f1_score}.
        model_name: Display name for the model.
    """
    per_class_columns = ["MODEL", "CLASS", "precision", "recall", "ap50", "ap", "f1_score"]
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=per_class_columns)
        if write_header:
            writer.writeheader()
        for cls_name, cls_metrics in per_class.items():
            row = {
                "MODEL": model_name,
                "CLASS": cls_name,
            }
            # Map internal keys to CSV column names
            key_map = {"ap50": "map50", "ap": "map"}
            for col in per_class_columns[2:]:
                src_key = key_map.get(col, col)
                val = cls_metrics.get(src_key, cls_metrics.get(col, 0.0))
                row[col] = f"{val:.4f}" if isinstance(val, (int, float)) else val
            writer.writerow(row)


def _format_per_class_table(per_class_csv_path):
    """Format per-class CSV into a readable table grouped by model."""
    with open(per_class_csv_path, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        return "No per-class metrics available.\n"

    # Group by model, preserving insertion order
    from collections import OrderedDict
    models: dict[str, list[dict]] = OrderedDict()
    for row in rows:
        model_name = row["MODEL"]
        models.setdefault(model_name, []).append(row)

    metric_cols = ["precision", "recall", "ap50", "ap", "f1_score"]
    output_parts: list[str] = []

    for model_name, model_rows in models.items():
        output_parts.append(f"  Model: {model_name}")
        table_data = []
        for row in model_rows:
            table_row = [row["CLASS"]]
            for col in metric_cols:
                val = row.get(col, "-")
                try:
                    val = float(val)
                    table_row.append(f"{val:.4f}")
                except (ValueError, TypeError):
                    table_row.append(str(val))
            table_data.append(table_row)

        headers = ["CLASS"] + metric_cols
        table_str = tabulate(table_data, headers=headers, tablefmt="simple")
        output_parts.append(table_str)
        output_parts.append("")  # blank line between models

    return "\n".join(output_parts)


def create_formatted_table(csv_path):
    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        headers = next(reader)  # Read headers
        data = list(reader)  # Read all rows

    # Find the best values for each metric column (skip MODEL column)
    best_values = {}
    for col_index, header in enumerate(headers[1:], start=1):  # Skip MODEL
        column_values = []
        for row in data:
            if col_index >= len(row):
                continue
            raw = row[col_index].strip()
            if not raw or raw == "-":
                continue
            try:
                column_values.append(float(raw))
            except ValueError:
                continue

        if not column_values:
            best_values[header] = None
        elif header == "ms/frame":  # Lower is better for time
            best_values[header] = min(column_values)
        else:  # Higher is better for other metrics
            best_values[header] = max(column_values)

    # Format the data with the best values in bold
    formatted_data = []
    for row in data:
        if not row:
            continue
        formatted_row = [row[0]]  # Model name
        for col_index, header in enumerate(headers[1:], start=1):  # Skip MODEL
            raw = row[col_index].strip() if col_index < len(row) else ""
            if not raw or raw == "-":
                formatted_row.append("-")
                continue
            try:
                value = float(raw)
            except ValueError:
                formatted_row.append("-")
                continue

            is_best = best_values.get(header) is not None and value == best_values[header]
            if is_best:
                formatted_row.append(f"*{value:.4f}*")
            else:
                formatted_row.append(f"{value:.4f}")
        formatted_data.append(formatted_row)

    # Create table
    formatted_table = tabulate(formatted_data, headers=headers, tablefmt="simple")

    # Metric descriptions
    descriptions = [
        "precision (higher is better): Fraction of correct positive predictions out of all positive predictions.",
        "recall (higher is better): Fraction of actual positive cases correctly identified.",
        "map (higher is better): Mean Average Precision across IoU thresholds 0.5-0.95.",
        "map50 (higher is better): Mean Average Precision at IoU threshold 0.5.",
        "fitness (higher is better): Combined metric (0.1*mAP50 + 0.9*mAP50-95).",
        "f1_score (higher is better): Harmonic mean of precision and recall.",
        "ms/frame (lower is better): Average inference time per frame in milliseconds (preprocess + inference + postprocess).",
    ]

    # Add scene metric descriptions
    for header in headers:
        if header.startswith("scene_") and header.endswith("_fitness"):
            scene_name = header[6:-8]  # Extract scene name from header
            descriptions.append(
                f"{header} (higher is better): Fitness score for scene '{scene_name}'."
            )

    formatted_descriptions = "\n\n".join(descriptions)

    # Write results to file
    txt_path = "./results_comparison/results.txt"
    with open(txt_path, "w") as f:
        f.write("Overall Metrics:\n")
        f.write("=" * 80)
        f.write("\n\n")
        f.write(formatted_table)
        f.write("\n\n")
        f.write("Metric Descriptions:\n")
        f.write(formatted_descriptions)

        # Append per-class table if available
        per_class_csv_path = "./results_comparison/per_class_results.csv"
        if os.path.exists(per_class_csv_path):
            f.write("\n\n")
            f.write("=" * 80)
            f.write("\nPer-Class Metrics:\n")
            f.write("=" * 80)
            f.write("\n\n")
            per_class_table = _format_per_class_table(per_class_csv_path)
            f.write(per_class_table)
            f.write("\n")
            f.write("Per-class metric descriptions:\n")
            f.write("  precision: Fraction of correct positive predictions for this class (at best-F1 confidence).\n")
            f.write("  recall:    Fraction of actual positives correctly detected for this class (at best-F1 confidence).\n")
            f.write("  ap50:      Average Precision at IoU threshold 0.5 for this class.\n")
            f.write("  ap:        Average Precision across IoU 0.5-0.95 for this class.\n")
            f.write("  f1_score:  Harmonic mean of precision and recall for this class (at best-F1 confidence).\n")

        # Append merged-class subset table if available
        merged_csv_path = "./results_comparison/merged_class_results.csv"
        if os.path.exists(merged_csv_path):
            f.write("\n\n")
            f.write("=" * 80)
            f.write("\nMerged-Class Subset Metrics:\n")
            f.write("(Evaluation on test images that originally contained the source class\n")
            f.write(" before it was merged into the target class during training)\n")
            f.write("=" * 80)
            f.write("\n\n")
            merged_table = _format_merged_class_table(merged_csv_path)
            f.write(merged_table)
            f.write("\n")
            f.write("Merged-class metric descriptions:\n")
            f.write("  merged_class: Source class → target class it was merged into.\n")
            f.write("  n_objects:    Number of source-class annotations (objects) in the subset.\n")
            f.write("  precision:    Precision on this image subset (all classes).\n")
            f.write("  recall:       Recall on this image subset (all classes).\n")
            f.write("  ap50:         mAP@50 on this image subset (all classes).\n")
            f.write("  ap:           mAP@50-95 on this image subset (all classes).\n")
            f.write("  f1_score:     F1 score on this image subset (all classes).\n")


def _collect_scene_columns(path_results2: dict | None) -> list:
    scene_columns = []
    if path_results2 is not None:
        for key in path_results2:
            if key.startswith("scene_") and key.endswith("_fitness"):
                scene_columns.append(key)
    return scene_columns


def _format_values(rows: list) -> list:
    formatted_data = []
    for row in rows:
        formatted_row = [row[0]]
        formatted_values = []
        for value in row[1:]:
            if isinstance(value, (int, float)):
                formatted_values.append(f"{value:.4f}")
            else:
                formatted_values.append(value)
        formatted_row.extend(formatted_values)
        formatted_data.append(formatted_row)
    return formatted_data


def _build_mean_table_rows(basic_columns, scene_columns, path_results1, path_results2, experiment_name, base_run, base_model_name):
    if base_run and path_results1 is not None:
        base_model_display_name = base_model_name if base_model_name else "YOLOv8m (base run)"
        base_row = [base_model_display_name]
        for col in basic_columns[1:]:
            base_row.append(path_results1[col.lower()])
        for col in scene_columns:
            base_row.append(path_results1.get(col, 0.0))

        retrained_row = [experiment_name]
        for col in basic_columns[1:]:
            retrained_row.append(path_results2[col.lower()])
        for col in scene_columns:
            retrained_row.append(path_results2.get(col, 0.0))
        return [base_row, retrained_row]
    else:
        retrained_row = [experiment_name]
        for col in basic_columns[1:]:
            retrained_row.append(path_results2[col.lower()])
        for col in scene_columns:
            retrained_row.append(path_results2.get(col, 0.0))
        return [retrained_row]


def mean_table(path_results1, path_results2, experiment_name, base_run, base_model_name=None):
    # Collect all columns, including scene metrics
    basic_columns = [
        "MODEL",
        "img_size",       
        "precision",
        "recall",
        "map",
        "map50",
        "fitness",
        "f1_score",
        "ms/frame",
]

    # Add scene metrics columns
    scene_columns = _collect_scene_columns(path_results2)

    columns = basic_columns + scene_columns

    data = _build_mean_table_rows(basic_columns, scene_columns, path_results1, path_results2, experiment_name, base_run, base_model_name)

    # Format the data values
    formatted_data = _format_values(data)

    # Create results directory if it doesn't exist
    os.makedirs("./results_comparison", exist_ok=True)

    csv_path = "./results_comparison/results.csv"

    existing_columns: list[str] = []
    existing_rows: list[dict[str, str]] = []
    if os.path.exists(csv_path):
        with open(csv_path, newline="") as f:
            reader = csv.reader(f)
            try:
                existing_columns = next(reader)
            except StopIteration:
                existing_columns = []
            else:
                for row in reader:
                    row_dict: dict[str, str] = {}
                    for idx, col in enumerate(existing_columns):
                        if idx < len(row):
                            row_dict[col] = row[idx]
                    existing_rows.append(row_dict)

    merged_columns = list(existing_columns) if existing_columns else []
    for col in columns:
        if col not in merged_columns:
            merged_columns.append(col)
    if not merged_columns:
        merged_columns = list(columns)

    for row in formatted_data:
        row_dict: dict[str, str] = {}
        for idx, col in enumerate(columns):
            if idx < len(row):
                row_dict[col] = row[idx]
        existing_rows.append(row_dict)

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(merged_columns)
        for row_dict in existing_rows:
            writer.writerow([row_dict.get(col, "") for col in merged_columns])

    # Write per-class metrics CSV
    per_class_csv_path = "./results_comparison/per_class_results.csv"
    if base_run and path_results1 is not None:
        base_display = base_model_name if base_model_name else "YOLOv8m (base run)"
        per_class_base = path_results1.get("per_class", {})
        if per_class_base:
            _append_per_class_to_csv(per_class_csv_path, per_class_base, base_display)

    per_class_retrained = path_results2.get("per_class", {}) if path_results2 else {}
    if per_class_retrained:
        _append_per_class_to_csv(per_class_csv_path, per_class_retrained, experiment_name)

    # Create formatted text table from CSV
    create_formatted_table(csv_path)


def main(args):
    dataset = args.dataset_yaml
    weights_dir = Path(args.weights_dir)
    experiment_name = weights_dir.parts[1]

    model_retrained = YOLO(f"{args.weights_dir}")

    if args.run_base:
        model_coco = YOLO(f"yolov8{args.yolo_size}.pt")
        path_results_coco = validate_model(
            model_coco, data=dataset, save_json=True, imgsz=args.image_size, workers=0
        )
        path_results_carmel = validate_model(
            model_retrained, data=dataset, save_json=True, imgsz=args.image_size, workers=0
        )

        # f1_pre_rec_curves(path_results_coco, path_results_carmel)
        # confusion_matrices(path_results_coco, path_results_carmel)
        mean_table(path_results_coco, path_results_carmel, experiment_name, True, f"YOLOv8{args.yolo_size} (COCO)")
        # ap_curve(path_results_coco, path_results_carmel)
    else:
        path_results_carmel = validate_model(
            model_retrained, data=dataset, save_json=True, imgsz=args.image_size
        )

        generate_side_by_side_comparisons(
            YOLO(f"yolov8{args.yolo_size}.pt"),
            model_retrained,
            Path("datasets/tspwob-yolotest-101224/test/val/images"),
            weights_dir.parent.parent,
        )
        mean_table(None, path_results_carmel, experiment_name, False, None)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv8 Model Comparison")
    parser.add_argument(
        "--yolo_size", type=str, default="m", help="YOLOv8 model size (n, s, m, l, x)"
    )
    parser.add_argument(
        "--dataset_yaml", type=str, required=True, help="Path to dataset YAML file"
    )
    parser.add_argument(
        "--weights_dir",
        type=str,
        required=True,
        help="Directory of trained model (e.g., runs/detect/train0011/weights/best.pt)",
    )
    parser.add_argument(
        "--image_size", type=int, default=640, help="Image size for validation"
    )
    parser.add_argument(
        "--run_base",
        action="store_true",
        help="Whether to run the base (COCO) validation",
    )
    args = parser.parse_args()

    main(args)
