from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
import yaml
from PIL import Image
import sam3
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from tqdm import tqdm

try:
    import cv2
    from ultralytics import YOLO
    from yolov8_training.utils.evaluate import (
        get_dataset_classes,
        mean_table,
        validate_model,
    )
except Exception:  # pragma: no cover - optional YOLO comparison
    cv2 = None
    YOLO = None
    get_dataset_classes = None
    mean_table = None
    validate_model = None

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

# ---- SAM3 tokenizer asset ----
BPE_FILENAME = "bpe_simple_vocab_16e6.txt.gz"
BPE_URL = (
    "https://github.com/facebookresearch/sam3/"
    "raw/main/assets/bpe_simple_vocab_16e6.txt.gz"
)


@dataclass
class Detection:
    box: np.ndarray  # xyxy in pixel space
    score: float


def ensure_bpe_vocab() -> Path:
    """
    Ensure the SAM3 BPE vocab exists in the location expected by the library.

    We place it in `<site-packages>/assets/bpe_simple_vocab_16e6.txt.gz`,
    which matches what the internal tokenizer code looks for.
    """
    sam3_root = Path(sam3.__file__).resolve().parent.parent  # parent of sam3 package
    assets_dir = sam3_root / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    bpe_path = assets_dir / BPE_FILENAME

    if not bpe_path.exists():
        print(f"[sam3-eval] BPE vocab not found at {bpe_path}, downloading...")
        try:
            import urllib.request

            with urllib.request.urlopen(BPE_URL) as resp, open(
                bpe_path, "wb"
            ) as out_f:
                out_f.write(resp.read())
        except Exception as e:
            raise RuntimeError(
                "Failed to download SAM3 BPE vocab. "
                f"Please download it manually from:\n{BPE_URL}\n"
                f"and place it at:\n{bpe_path}"
            ) from e

    return bpe_path


def _infer_dataset_yaml_from_params() -> Path:
    """
    Infer the test dataset.yaml path from params.yaml so we
    automatically match the train/eval pipeline's test split.
    """
    params_path = Path("params.yaml")
    if not params_path.exists():
        raise FileNotFoundError(
            "params.yaml not found. Please pass --dataset explicitly."
        )

    with open(params_path, "r") as f:
        params = yaml.safe_load(f) or {}

    dataset_name = (params.get("data") or {}).get("dataset_name")
    if not dataset_name:
        raise RuntimeError(
            "data.dataset_name missing in params.yaml. "
            "Please pass --dataset explicitly."
        )

    candidate = Path("datasets") / dataset_name / "test" / "dataset.yaml"
    if not candidate.exists():
        raise FileNotFoundError(
            f"Inferred test dataset YAML '{candidate}' does not exist. "
            "Run the prepare stage or pass --dataset explicitly."
        )
    return candidate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run SAM 3 on the test set and score detections like the YOLO evaluator.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help=(
            "Path to dataset.yaml describing the test split. "
            "If omitted, uses datasets/<data.dataset_name>/test/dataset.yaml "
            "from params.yaml."
        ),
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="trash on the ground",
        help="Text prompt to feed into SAM 3.",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="sam3-image-official",
        help=(
            "Label for logging only. SAM3 weights are loaded via the official "
            "`sam3` package, not via this string."
        ),
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Computation device. 'auto' picks CUDA when available.",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.05,
        help="Minimum SAM 3 score to keep a detection.",
    )
    parser.add_argument(
        "--max-detections",
        type=int,
        default=100,
        help="Maximum detections per image after sorting by score.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("tools/sam3_eval/output"),
        help="Directory to write metrics and prediction dumps.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on number of images to process (useful for quick smoke tests).",
    )
    parser.add_argument(
        "--save-predictions",
        type=Path,
        default=None,
        help="Path to save raw predictions JSON. Defaults to <output-dir>/predictions.json.",
    )
    return parser.parse_args()


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available. Falling back to CPU.")
        return torch.device("cpu")
    return torch.device(requested)


def load_dataset_paths(dataset_yaml: Path) -> Tuple[Path, Path]:
    with open(dataset_yaml, "r") as f:
        cfg = yaml.safe_load(f)

    base_path = Path(cfg.get("path", dataset_yaml.parent))
    if not base_path.is_absolute():
        base_path = (dataset_yaml.parent / base_path).resolve()

    images_subdir = cfg.get("val", "val/images")
    images_dir = (base_path / images_subdir).resolve()
    labels_dir = images_dir.parent / "labels"

    if not images_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {images_dir}")
    if not labels_dir.exists():
        raise FileNotFoundError(f"Label directory not found: {labels_dir}")
    return images_dir, labels_dir


def list_images(images_dir: Path, limit: int | None) -> List[Path]:
    images = sorted(
        p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS
    )
    return images[:limit] if limit is not None else images


def load_ground_truth_boxes(
    label_path: Path, width: int, height: int
) -> List[np.ndarray]:
    boxes: List[np.ndarray] = []
    if not label_path.exists():
        return boxes
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            _, x_c, y_c, w, h = map(float, parts[:5])
            x1 = (x_c - w / 2) * width
            y1 = (y_c - h / 2) * height
            x2 = (x_c + w / 2) * width
            y2 = (y_c + h / 2) * height
            boxes.append(np.array([x1, y1, x2, y2], dtype=float))
    return boxes


def bbox_iou(box: np.ndarray, boxes: Iterable[np.ndarray]) -> np.ndarray:
    boxes_arr = np.array(list(boxes), dtype=float)
    if boxes_arr.size == 0:
        return np.zeros(0, dtype=float)

    inter_x1 = np.maximum(box[0], boxes_arr[:, 0])
    inter_y1 = np.maximum(box[1], boxes_arr[:, 1])
    inter_x2 = np.minimum(box[2], boxes_arr[:, 2])
    inter_y2 = np.minimum(box[3], boxes_arr[:, 3])

    inter_w = np.maximum(0.0, inter_x2 - inter_x1)
    inter_h = np.maximum(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_pred = (box[2] - box[0]) * (box[3] - box[1])
    area_gt = (boxes_arr[:, 2] - boxes_arr[:, 0]) * (boxes_arr[:, 3] - boxes_arr[:, 1])
    union = area_pred + area_gt - inter_area
    union = np.maximum(union, 1e-9)
    return inter_area / union


def compute_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1])
    return float(ap)


def evaluate_predictions(
    predictions: Dict[str, List[Detection]],
    ground_truths: Dict[str, List[np.ndarray]],
    iou_thresholds: Iterable[float],
) -> Dict[str, float]:
    total_gts = sum(len(v) for v in ground_truths.values())
    if total_gts == 0:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "map": 0.0,
            "map50": 0.0,
            "fitness": 0.0,
            "f1_score": 0.0,
        }

    aps: List[float] = []
    precision_at_05 = 0.0
    recall_at_05 = 0.0

    for iou_thr in iou_thresholds:
        scores: List[float] = []
        tps: List[int] = []
        fps: List[int] = []
        matched = {
            img_id: np.zeros(len(gts), dtype=bool)
            for img_id, gts in ground_truths.items()
        }

        for image_id, dets in predictions.items():
            ordered = sorted(dets, key=lambda d: d.score, reverse=True)
            gts = ground_truths.get(image_id, [])
            for det in ordered:
                scores.append(det.score)
                if not gts:
                    tps.append(0)
                    fps.append(1)
                    continue
                ious = bbox_iou(det.box, gts)
                best_idx = int(np.argmax(ious)) if ious.size else -1
                best_iou = float(ious[best_idx]) if best_idx >= 0 else 0.0
                if best_iou >= iou_thr and not matched[image_id][best_idx]:
                    matched[image_id][best_idx] = True
                    tps.append(1)
                    fps.append(0)
                else:
                    tps.append(0)
                    fps.append(1)

        if not scores:
            aps.append(0.0)
            if np.isclose(iou_thr, 0.5):
                precision_at_05 = 0.0
                recall_at_05 = 0.0
            continue

        order = np.argsort(-np.array(scores))
        tp_sorted = np.cumsum(np.array(tps)[order])
        fp_sorted = np.cumsum(np.array(fps)[order])

        recall_curve = tp_sorted / (total_gts + 1e-16)
        precision_curve = tp_sorted / (tp_sorted + fp_sorted + 1e-16)
        aps.append(compute_ap(recall_curve, precision_curve))

        if np.isclose(iou_thr, 0.5):
            precision_at_05 = float(precision_curve[-1])
            recall_at_05 = float(recall_curve[-1])

    map50 = aps[0] if aps else 0.0
    map50_95 = float(np.mean(aps)) if aps else 0.0
    f1 = (
        2 * (precision_at_05 * recall_at_05) / (precision_at_05 + precision_at_05 + 1e-16)
    )
    fitness = 0.1 * precision_at_05 + 0.1 * recall_at_05 + 0.8 * map50

    return {
        "precision": precision_at_05,
        "recall": recall_at_05,
        "map": map50_95,
        "map50": map50,
        "fitness": fitness,
        "f1_score": f1,
    }


def xyxy_to_xywhn(box: np.ndarray, width: int, height: int) -> List[float]:
    x1, y1, x2, y2 = box.tolist()
    w = x2 - x1
    h = y2 - y1
    x_c = x1 + w / 2
    y_c = y1 + h / 2
    return [x_c / width, y_c / height, w / width, h / height]


def _get_output_field(output: dict, keys: List[str]):
    """
    Safely extract a field from SAM3 output without using `or` on tensors.

    Returns the first non-None value for any of the given keys.
    """
    for k in keys:
        if k in output:
            v = output[k]
            # Accept tensors/arrays/lists; explicit None check avoids bool(tensor)
            if v is not None:
                return v
    return None


def run_sam3_inference(
    processor: Sam3Processor,
    image: Image.Image,
    prompt: str,
    score_threshold: float,
    max_detections: int,
) -> Tuple[List[Detection], float]:
    """
    Run SAM3 on a single image using the official `sam3` API.

    Steps:
      * set the image
      * apply a text prompt
      * pull boxes + scores from the output
      * threshold + sort + truncate
    """
    start = time.perf_counter()
    with torch.no_grad():
        state = processor.set_image(image)
        output = processor.set_text_prompt(state=state, prompt=prompt)
    inference_time = time.perf_counter() - start

    if output is None:
        return [], inference_time

    # output["masks"], output["boxes"], output["scores"]
    boxes = _get_output_field(output, ["boxes", "bboxes", "boxes_xyxy"])
    scores = _get_output_field(output, ["scores", "confidence_scores"])

    if boxes is None or scores is None:
        return [], inference_time

    boxes_np = np.asarray(
        boxes.detach().cpu() if hasattr(boxes, "detach") else boxes, dtype=float
    )
    scores_np = np.asarray(
        scores.detach().cpu() if hasattr(scores, "detach") else scores, dtype=float
    )

    # No detections: boxes shape (0, 4) or size 0
    if boxes_np.size == 0 or scores_np.size == 0:
        return [], inference_time

    if boxes_np.ndim != 2 or boxes_np.shape[1] != 4:
        raise ValueError(f"Expected boxes with shape (N, 4); got {boxes_np.shape}")

    order = np.argsort(-scores_np)
    detections: List[Detection] = []
    for idx in order:
        score = float(scores_np[idx])
        if score < score_threshold:
            continue
        detections.append(Detection(box=boxes_np[idx], score=score))
        if len(detections) >= max_detections:
            break

    return detections, inference_time


def save_predictions(
    path: Path,
    predictions: Dict[str, List[Detection]],
    image_sizes: Dict[str, Tuple[int, int]],
) -> None:
    serialized = []
    for image_id, dets in predictions.items():
        width, height = image_sizes[image_id]
        for det in dets:
            serialized.append(
                {
                    "image_id": image_id,
                    "class_name": "waste",
                    "class_id": 0,
                    "score": det.score,
                    "bbox_xyxy": [float(x) for x in det.box.tolist()],
                    "bbox_xywh_normalized": xyxy_to_xywhn(det.box, width, height),
                    "width": width,
                    "height": height,
                }
            )
    with open(path, "w") as f:
        json.dump(serialized, f, indent=2)


def _build_sam3_metrics_dict(
    base_metrics: Dict[str, float],
    avg_inference_time: float,
    img_size: int | None,
) -> Dict[str, float]:
    """
    Build a metrics dict compatible with yolov8_training.utils.evaluate.mean_table.
    """
    return {
        "img_size": img_size,
        "precision": float(base_metrics.get("precision", 0.0)),
        "recall": float(base_metrics.get("recall", 0.0)),
        "map": float(base_metrics.get("map", 0.0)),
        "map50": float(base_metrics.get("map50", 0.0)),
        "fitness": float(base_metrics.get("fitness", 0.0)),
        "f1_score": float(base_metrics.get("f1_score", 0.0)),
        "time": float(avg_inference_time),
    }


def _evaluate_current_best_and_write_csv(
    dataset_yaml: Path,
    sam3_metrics: Dict[str, float],
    experiment_name: str,
) -> None:
    """
    Evaluate the current_best YOLO model and append a comparison row to
    results_comparison/results.csv using the shared mean_table().
    """
    if (
        YOLO is None
        or validate_model is None
        or get_dataset_classes is None
        or mean_table is None
    ):
        print(
            "[sam3-eval] YOLO or evaluation utilities not available; "
            "skipping CSV comparison against current_best."
        )
        return

    current_best_dir = Path("models/current_best")
    weights_path = current_best_dir / "best.pt"
    metadata_path = current_best_dir / "metadata.yaml"

    if not weights_path.exists():
        print(
            f"[sam3-eval] No current_best weights found at {weights_path}; "
            "skipping CSV comparison."
        )
        return

    metadata: Dict[str, object] = {}
    if metadata_path.exists():
        try:
            with open(metadata_path, "r") as f:
                metadata = yaml.safe_load(f) or {}
        except Exception as e:
            print(f"[sam3-eval] Warning: failed to read {metadata_path}: {e}")

    image_size = metadata.get("image_size")
    base_experiment_name = metadata.get("experiment_name", "current_best")
    base_display_name = f"{base_experiment_name} (current best)"

    print(f"[sam3-eval] Loading current_best YOLO model from {weights_path}...")
    base_model = YOLO(str(weights_path))

    # Determine the class IDs to evaluate on
    class_names, class_ids = get_dataset_classes(dataset_yaml)
    # get_dataset_classes may return IDs as strings – coerce to ints if needed
    try:
        class_ids = [int(c) for c in class_ids]
    except Exception:
        pass

    print("[sam3-eval] Evaluating current_best model on the same dataset...")
    base_results = validate_model(
        base_model,
        data=str(dataset_yaml),
        class_ids=class_ids or None,
        write_json=False,
        imgsz=image_size,
        workers=0,
    )

    print("[sam3-eval] Writing CSV comparison to results_comparison/results.csv...")
    mean_table(
        base_results,
        sam3_metrics,
        experiment_name,
        base_run=True,
        base_model_name=base_display_name,
    )


def _draw_sam3_boxes_on_image(
    img_path: Path,
    detections: List[Detection],
) -> np.ndarray | None:
    """
    Create an annotated image for SAM3 detections using OpenCV, similar to YOLO's plot().
    """
    if cv2 is None:
        return None

    img = cv2.imread(str(img_path))
    if img is None:
        return None

    for det in detections:
        x1, y1, x2, y2 = det.box.astype(int)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        label = f"{det.score:.2f}"
        cv2.putText(
            img,
            label,
            (x1, max(y1 - 5, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )

    return img


def _generate_side_by_side_comparisons_sam3_vs_current_best(
    base_model,
    image_paths: List[Path],
    predictions: Dict[str, List[Detection]],
    output_dir: Path,
    conf_threshold: float = 0.25,
) -> None:
    """
    Generate side-by-side comparison images of current_best YOLO vs SAM3.
    """
    if cv2 is None:
        print(
            "[sam3-eval] OpenCV not available; skipping side-by-side visual comparisons."
        )
        return

    side_by_side_dir = output_dir / "side_by_side_current_best"
    side_by_side_dir.mkdir(exist_ok=True)

    for img_path in image_paths:
        if img_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        image_id = img_path.stem
        sam3_dets = predictions.get(image_id, [])

        # YOLO current_best prediction and plot
        base_results = base_model.predict(
            str(img_path), conf=conf_threshold, save=False, verbose=False
        )
        base_img = base_results[0].plot()

        # SAM3 drawing based on already computed detections
        sam3_img = _draw_sam3_boxes_on_image(img_path, sam3_dets)
        if sam3_img is None:
            continue

        # Resize SAM3 image to match YOLO output if needed
        if sam3_img.shape[:2] != base_img.shape[:2]:
            sam3_img = cv2.resize(
                sam3_img, (base_img.shape[1], base_img.shape[0]), interpolation=cv2.INTER_LINEAR
            )

        comparison_img = np.hstack((base_img, sam3_img))
        save_path = side_by_side_dir / f"comparison_{img_path.name}"
        cv2.imwrite(str(save_path), comparison_img)


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    preds_path = args.save_predictions or (output_dir / "predictions.json")

    dataset_yaml = args.dataset or _infer_dataset_yaml_from_params()
    images_dir, labels_dir = load_dataset_paths(dataset_yaml)
    image_paths = list_images(images_dir, args.limit)
    if not image_paths:
        raise RuntimeError(f"No images found in {images_dir}")

    # --- SAM3: make sure tokenizer vocab exists, then build image model + processor ---
    bpe_path = ensure_bpe_vocab()
    model = build_sam3_image_model(bpe_path=str(bpe_path))
    model.to(device)
    model.eval()
    processor = Sam3Processor(model)

    predictions: Dict[str, List[Detection]] = {}
    ground_truths: Dict[str, List[np.ndarray]] = {}
    image_sizes: Dict[str, Tuple[int, int]] = {}
    inference_times: List[float] = []

    for img_path in tqdm(image_paths, desc="Running SAM 3"):
        with Image.open(img_path) as img_raw:
            image = img_raw.convert("RGB")
            width, height = image.size

        label_path = labels_dir / f"{img_path.stem}.txt"
        gt_boxes = load_ground_truth_boxes(label_path, width, height)

        dets, elapsed = run_sam3_inference(
            processor,
            image,
            args.prompt,
            args.score_threshold,
            args.max_detections,
        )

        predictions[img_path.stem] = dets
        ground_truths[img_path.stem] = gt_boxes
        image_sizes[img_path.stem] = (width, height)
        inference_times.append(elapsed)

    iou_thresholds = np.arange(0.5, 0.96, 0.05)
    metrics = evaluate_predictions(predictions, ground_truths, iou_thresholds)

    avg_inference_time = float(np.mean(inference_times)) if inference_times else 0.0

    metrics_payload = {
        "prompt": args.prompt,
        "model_id": args.model_id,  # informational only
        "dataset_yaml": str(dataset_yaml),
        "num_images": len(image_paths),
        "num_ground_truth_boxes": int(sum(len(v) for v in ground_truths.values())),
        "num_predictions": int(sum(len(v) for v in predictions.values())),
        "score_threshold": args.score_threshold,
        "max_detections": args.max_detections,
        "device": str(device),
        "iou_thresholds": [float(x) for x in iou_thresholds],
        "metrics": {
            "img_size": None,
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "map": metrics["map"],
            "map50": metrics["map50"],
            "fitness": metrics["fitness"],
            "f1_score": metrics["f1_score"],
            "time": avg_inference_time,
        },
    }

    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics_payload, f, indent=2)

    save_predictions(preds_path, predictions, image_sizes)

    print(f"Saved metrics to {metrics_path}")
    print(f"Saved predictions to {preds_path}")

    # --- Optional: CSV logging and side-by-side comparisons vs current_best YOLO ---
    sam3_metrics_dict = _build_sam3_metrics_dict(
        metrics, avg_inference_time=avg_inference_time, img_size=None
    )
    sam3_experiment_name = f"SAM3 ({args.prompt})"

    try:
        _evaluate_current_best_and_write_csv(dataset_yaml, sam3_metrics_dict, sam3_experiment_name)
    except Exception as e:
        print(f"[sam3-eval] Warning: failed to write CSV comparison: {e}")

    if YOLO is not None and cv2 is not None:
        current_best_dir = Path("models/current_best")
        weights_path = current_best_dir / "best.pt"
        if weights_path.exists():
            try:
                base_model = YOLO(str(weights_path))
                _generate_side_by_side_comparisons_sam3_vs_current_best(
                    base_model,
                    image_paths,
                    predictions,
                    output_dir,
                )
                print(
                    f"[sam3-eval] Saved side-by-side comparisons to "
                    f"{output_dir / 'side_by_side_current_best'}"
                )
            except Exception as e:
                print(
                    f"[sam3-eval] Warning: failed to generate side-by-side "
                    f"comparisons vs current_best: {e}"
                )


if __name__ == "__main__":
    main()
