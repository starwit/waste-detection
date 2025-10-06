import argparse
import csv
import os
from pathlib import Path
import shutil
import tempfile
import traceback
import json

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
):
    """
    Evaluates the model on the test set, prepares metadata, and appends results to the CSV.
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
            )
            # Determine base model name for display
            mean_table(base_results, results, model_name, True, base_model_name)
        except Exception as e:
            print(f"Warning: Could not load base model for comparison: {e}")
            # Still add the retrained model results to the table
            mean_table(None, results, model_name, False, None)

    return metadata


def save_false_predictions(model, test_path, output_dir, conf_threshold=0.25):
    # Create directory for false predictions
    false_pred_dir = output_dir / "false_predictions"
    false_pred_dir.mkdir(exist_ok=True)

    # Get test image directory
    test_img_dir = test_path / "val" / "images"
    test_label_dir = test_path / "val" / "labels"

    # Process each image in test set
    for img_path in test_img_dir.glob("*"):
        # Get corresponding label file
        label_path = test_label_dir / (img_path.stem + ".txt")

        # Run prediction
        results = model.predict(str(img_path), conf=conf_threshold, save=False)
        result = results[0]

        # Get ground truth boxes
        gt_boxes = []
        if label_path.exists():
            with open(label_path, "r") as f:
                for line in f:
                    class_id, x, y, w, h = map(float, line.strip().split())
                    gt_boxes.append({"class": int(class_id), "box": [x, y, w, h]})

        # Get predicted boxes
        pred_boxes = []
        for box in result.boxes:
            pred_boxes.append(
                {
                    "class": int(box.cls),
                    "conf": float(box.conf),
                    "box": box.xywhn[0].tolist(),  # normalized coordinates
                }
            )

        # Check for false predictions
        has_false_prediction = False

        # Simple check for mismatches between predictions and ground truth
        if len(gt_boxes) != len(pred_boxes):
            has_false_prediction = True
        else:
            # More detailed comparison could be implemented here
            # This is a simplified version
            for gt_box in gt_boxes:
                match_found = False
                for pred_box in pred_boxes:
                    if gt_box["class"] == pred_box["class"] and all(
                        abs(g - p) < 0.1 for g, p in zip(gt_box["box"], pred_box["box"])
                    ):
                        match_found = True
                        break
                if not match_found:
                    has_false_prediction = True
                    break

        # If false prediction found, save the image with annotations
        if has_false_prediction:
            # Plot predictions and ground truth
            result_plotted = result.plot()

            # Save the annotated image
            save_path = false_pred_dir / f"false_pred_{img_path.name}"
            cv2.imwrite(str(save_path), result_plotted)


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
            dataset_config = yaml.safe_load(f)
        
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
            print(f"Warning: Unexpected format for 'names' in {dataset_yaml_path}. Expected list or dict, got {type(names_data)}")
            class_names = {}
            class_ids = []
        
        return class_names, class_ids
    except Exception as e:
        print(f"Warning: Could not read dataset classes from {dataset_yaml_path}: {e}")
        return {}, []


def calculate_fp_fn_tp(confusion_matrix, class_ids=None):
    """
    Calculate false positives, false negatives, and true positives.
    
    Args:
        confusion_matrix: Confusion matrix from model validation
        class_ids: List of class IDs to consider (if None, use all classes)
    """
    if class_ids is None:
        # Use all available classes except background
        class_ids = list(range(len(confusion_matrix) - 1))
    
    # Create mask for classes to consider
    mask = [i in class_ids for i in range(len(confusion_matrix) - 1)] + [True]
    filtered_matrix = confusion_matrix[mask][:, mask]
    fp = filtered_matrix[:-1, -1].sum()
    fn = filtered_matrix[-1, :-1].sum()
    tp = filtered_matrix.diagonal()[:-1].sum()
    return fp, fn, tp


def calculate_additional_metrics(fp, fn, tp):
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0
    )
    fpr = fp / (fp + tp + fn) if (fp + tp + fn) > 0 else 0
    return precision, recall, f1_score, fpr


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
    time_taken = float(metrics.speed["inference"])

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
        "time": time_taken,
    }

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
            "time":      results["time"],
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


def create_formatted_table(csv_path):
    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        headers = next(reader)  # Read headers
        data = list(reader)  # Read all rows

    # Find the best values for each metric column (skip MODEL column)
    best_values = {}
    for i, header in enumerate(headers[1:]):  # Skip first column
        try:
            column_values = [
                float(row[i + 1])
                for row in data
                if i + 1 < len(row) and row[i + 1].strip() and row[i + 1] != "-"
            ]
            if column_values:
                if header == "time":  # Lower is better for time
                    best_values[header] = min(column_values)
                else:  # Higher is better for other metrics
                    best_values[header] = max(column_values)
        except (IndexError, ValueError) as e:
            print(f"Warning: Error processing column {header}: {e}")
            best_values[header] = None

    # Format the data with the best values in bold
    formatted_data = []
    for row in data:
        try:
            formatted_row = [row[0]]  # Model name
            for i, header in enumerate(headers[1:]):  # Skip MODEL
                try:
                    if i + 1 < len(row) and row[i + 1].strip() and row[i + 1] != "-":
                        value = float(row[i + 1])
                        is_best = (
                            best_values[header] is not None
                            and value == best_values[header]
                        )
                        if is_best:
                            formatted_row.append(f"*{value:.4f}*")
                        else:
                            formatted_row.append(f"{value:.4f}")
                    else:
                        formatted_row.append("-")
                except (IndexError, ValueError):
                    formatted_row.append("-")
            formatted_data.append(formatted_row)
        except Exception as e:
            print(f"Warning: Error processing row {row}: {e}")
            continue

    # Create table
    formatted_table = tabulate(formatted_data, headers=headers, tablefmt="simple")

    # Metric descriptions
    descriptions = [
        "precision (higher is better): The percentage of correct positive predictions out of all positive predictions.",
        "recall (higher is better): The percentage of actual positive cases correctly identified.",
        "map (higher is better): Mean Average Precision across IoU thresholds 0.5-0.95.",
        "map50 (higher is better): Mean Average Precision at IoU threshold 0.5.",
        "fitness (higher is better): Combined metric (0.1*precision + 0.1*recall + 0.8*mAP50).",
        "f1_score (higher is better): Harmonic mean of precision and recall.",
        "time (lower is better): Total inference time in seconds.",
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
        f.write(formatted_table)
        f.write("\n\n")
        f.write("Metric Descriptions:\n")
        f.write(formatted_descriptions)


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
        "time",
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
    write_header = not os.path.exists(csv_path)

    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(columns)
        writer.writerows(formatted_data)

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
