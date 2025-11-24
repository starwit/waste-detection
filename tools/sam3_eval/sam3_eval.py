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
from transformers import Sam3Model, Sam3Processor
from tqdm import tqdm

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


@dataclass
class Detection:
    box: np.ndarray  # xyxy in pixel space
    score: float


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
        default="facebook/sam3-hiera-large",
        help="Hugging Face model id or local path for the SAM 3 checkpoint.",
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


def load_ground_truth_boxes(label_path: Path, width: int, height: int) -> List[np.ndarray]:
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
        matched = {img_id: np.zeros(len(gts), dtype=bool) for img_id, gts in ground_truths.items()}

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
        2 * (precision_at_05 * recall_at_05) / (precision_at_05 + recall_at_05 + 1e-16)
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


def run_sam3_inference(
    processor: Sam3Processor,
    model: Sam3Model,
    image: Image.Image,
    prompt: str,
    device: torch.device,
    score_threshold: float,
    max_detections: int,
) -> Tuple[List[Detection], float]:
    inputs = processor(text=[prompt], images=image, return_tensors="pt").to(device)

    start = time.perf_counter()
    with torch.no_grad():
        outputs = model(**inputs)
    inference_time = time.perf_counter() - start

    if not hasattr(processor, "post_process_instance_segmentation"):
        raise RuntimeError("Loaded processor does not support instance segmentation post-processing.")

    try:
        processed = processor.post_process_instance_segmentation(
            outputs=outputs, input_size=image.size, threshold=score_threshold
        )
    except TypeError:
        processed = processor.post_process_instance_segmentation(
            outputs=outputs, input_size=image.size
        )

    if not processed:
        return [], inference_time

    inst = processed[0]
    boxes = inst.get("bboxes") or inst.get("boxes") or inst.get("boxes_xyxy")
    scores = inst.get("scores") or inst.get("confidence_scores")

    if boxes is None or scores is None:
        return [], inference_time

    boxes_np = np.asarray(
        boxes.detach().cpu() if hasattr(boxes, "detach") else boxes, dtype=float
    )
    scores_np = np.asarray(
        scores.detach().cpu() if hasattr(scores, "detach") else scores, dtype=float
    )

    if boxes_np.ndim != 2 or boxes_np.shape[1] != 4:
        raise ValueError(f"Expected boxes with shape (N, 4); got {boxes_np.shape}")

    order = np.argsort(-scores_np)
    detections: List[Detection] = []
    for idx in order[:max_detections]:
        if score_threshold and scores_np[idx] < score_threshold:
            continue
        detections.append(Detection(box=boxes_np[idx], score=float(scores_np[idx])))
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

    processor = Sam3Processor.from_pretrained(args.model_id)
    model = Sam3Model.from_pretrained(args.model_id).to(device)
    model.eval()

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
            model,
            image,
            args.prompt,
            device,
            args.score_threshold,
            args.max_detections,
        )

        predictions[img_path.stem] = dets
        ground_truths[img_path.stem] = gt_boxes
        image_sizes[img_path.stem] = (width, height)
        inference_times.append(elapsed)

    iou_thresholds = np.arange(0.5, 0.96, 0.05)
    metrics = evaluate_predictions(predictions, ground_truths, iou_thresholds)

    metrics_payload = {
        "prompt": args.prompt,
        "model_id": args.model_id,
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
            "time": float(np.mean(inference_times)) if inference_times else 0.0,
        },
    }

    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics_payload, f, indent=2)

    save_predictions(preds_path, predictions, image_sizes)

    print(f"Saved metrics to {metrics_path}")
    print(f"Saved predictions to {preds_path}")


if __name__ == "__main__":
    main()
