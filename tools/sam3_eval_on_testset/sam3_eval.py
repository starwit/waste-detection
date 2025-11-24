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

# Hyperparameters for leaf regions
# --- region-of-leaves hyperparams ---
MIN_LEAVES_IN_REGION = 8              # was 10
LEAF_REGION_CENTER_DIST_RATIO = 0.15  # was 0.10
MERGE_REGION_GAP_RATIO = 0.05         # how close two region boxes can be to get merged

"""
# ---- MULTI-PROMPT CONFIG ---- 
WASTE_PROMPTS: List[str] = [ 
"plastic bottle on the ground", "plastic bag on the ground", "plastic wrapper on the ground", "plastic bottle cap on the ground",  "small piece of plastic on the ground", "crumpled plastic trash on the ground", "piece of plastic wrapper on the ground", "white plastic trash on the ground", "plastic packaging trash on the ground", "paper cup on the ground", "paper packaging on the ground", "crumpled paper trash on the ground", "piece of paper trash on the ground", "paper tissue on the ground", "paper napkin on the ground", "cardboard box on the ground","aluminum drink can on the ground", "crushed drink can on the ground", "glass bottle on the ground", "broken glass on the ground", "shards of glass on the ground", "cigarette butt on the ground", "cigarette on the ground", ]
"""

# Your custom waste prompts (kept + extra cig variants)
WASTE_PROMPTS = [
    # core
    "paper packaging on the ground",
    "piece of paper trash on the ground",
    "cigarette butt on the ground",
    "plastic packaging trash on the ground",
    "piece of plastic trash on the ground",
    "piece of plastic wrapper on the ground",
    "white plastic trash on the ground",
    "crushed drink can on the ground",
    # extra cigarette-specific prompts
    "discarded cigarette butt on the ground",
    "small white and brown cigarette butt on the street",

    # single generic booster
    "piece of trash on the ground",
]

# For now only two leaf prompts: pile + generic region-ish leaves
LEAF_PROMPTS: List[str] = [
    "pile of leaves on the ground",
    "leaf lying on the street",
]

# Prompts that correspond to cigarettes / tiny things we never want to drop
CIGARETTE_PROMPTS: List[str] = [
    "cigarette butt on the ground",
    "discarded cigarette butt on the ground",
    "small white and brown cigarette butt on the street",
]

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
    label: str       # which prompt produced this detection


@dataclass
class RemovalByLeaf:
    """
    Records which waste detection was removed due to overlap
    with which leaf detection (and the IoU).
    """
    waste_det: Detection
    leaf_det: Detection
    iou: float


def ensure_bpe_vocab() -> Path:
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
        help=(
            "Base prompt (kept for logging); the actual inference uses "
            "the fixed lists WASTE_PROMPTS and LEAF_PROMPTS."
        ),
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
        help="Minimum SAM 3 score to keep a *waste* detection (non-cigarettes).",
    )
    parser.add_argument(
        "--leaf-score-threshold",
        type=float,
        default=None,
        help=(
            "Minimum SAM 3 score to keep a *leaf* detection. "
            "Defaults to --score-threshold if not set."
        ),
    )
    parser.add_argument(
        "--cigarette-score-threshold",
        type=float,
        default=None,
        help=(
            "Minimum SAM 3 score to keep *cigarette* detections. "
            "Defaults to 0.5 * --score-threshold if not set."
        ),
    )
    parser.add_argument(
        "--max-detections",
        type=int,
        default=100,
        help="Maximum *waste* detections per image after NMS and sorting by score.",
    )
    parser.add_argument(
        "--nms-iou-threshold",
        type=float,
        default=0.5,
        help="IoU threshold for NMS across prompts per image.",
    )
    parser.add_argument(
        "--leaf-iou-threshold",
        type=float,
        default=0.5,
        help=(
            "If a waste box has IoU >= this with any leaf box, "
            "the waste box is discarded (except cigarette prompts)."
        ),
    )
    parser.add_argument(
        "--min-area-ratio",
        type=float,
        default=2e-4,
        help=(
            "Minimum box area as a fraction of image area. "
            "Detections smaller than this are dropped, except cigarette prompts. "
            "Example: 2e-4 ~ 200 px on a 1280x720 image."
        ),
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

    # NOTE: this uses the 'val' split; change to 'train' if you want to run on train
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
        2 * (precision_at_05 * recall_at_05)
        / (precision_at_05 + recall_at_05 + 1e-16)
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
    for k in keys:
        if k in output:
            v = output[k]
            if v is not None:
                return v
    return None


def nms_detections(
    detections: List[Detection],
    iou_threshold: float,
) -> List[Detection]:
    """
    Non-maximum suppression across prompts.

    Cigarette detections are never suppressed by other detections, and
    cigarettes do not suppress each other either. This guarantees that
    once a cigarette detection passes the score threshold, NMS will not
    remove it.
    """
    if not detections or iou_threshold <= 0:
        return detections

    boxes = np.stack([d.box for d in detections], axis=0)
    scores = np.array([d.score for d in detections], dtype=float)

    order = scores.argsort()[::-1]
    keep_indices: List[int] = []

    while order.size > 0:
        i = int(order[0])
        keep_indices.append(i)
        if order.size == 1:
            break

        rest = order[1:]
        ious = bbox_iou(boxes[i], boxes[rest])

        new_rest: List[int] = []
        for k, j in enumerate(rest):
            iou = ious[k]
            if iou < iou_threshold:
                # no suppression needed, keep this index in play
                new_rest.append(int(j))
            else:
                # would normally suppress j, but NEVER suppress cigarettes
                if detections[int(j)].label in CIGARETTE_PROMPTS:
                    new_rest.append(int(j))

        order = np.array(new_rest, dtype=int) if new_rest else np.array([], dtype=int)

    return [detections[i] for i in keep_indices]


def run_sam3_multi_prompt_inference(
    processor: Sam3Processor,
    image: Image.Image,
    prompts: List[str],
    score_threshold: float,
    max_detections: int,
    nms_iou_threshold: float,
    cigarette_score_threshold: float | None = None,
) -> Tuple[List[Detection], float]:
    """
    Run SAM3 for a list of prompts on one image.

    - Uses a separate (lower) score threshold for cigarette prompts if provided.
    - Runs NMS across prompts while protecting cigarettes in nms_detections.
    - Respects max_detections but never drops cigarettes in favor of non-cigarettes.
    """
    start = time.perf_counter()
    all_detections: List[Detection] = []

    with torch.no_grad():
        state = processor.set_image(image)
        for prompt in prompts:
            output = processor.set_text_prompt(state=state, prompt=prompt)
            if output is None:
                continue

            boxes = _get_output_field(output, ["boxes", "bboxes", "boxes_xyxy"])
            scores = _get_output_field(output, ["scores", "confidence_scores"])

            if boxes is None or scores is None:
                continue

            boxes_np = np.asarray(
                boxes.detach().cpu() if hasattr(boxes, "detach") else boxes,
                dtype=float,
            )
            scores_np = np.asarray(
                scores.detach().cpu() if hasattr(scores, "detach") else scores,
                dtype=float,
            )

            if boxes_np.size == 0 or scores_np.size == 0:
                continue
            if boxes_np.ndim != 2 or boxes_np.shape[1] != 4:
                continue

            # Use a lower threshold for cigarette prompts if configured
            if cigarette_score_threshold is not None and prompt in CIGARETTE_PROMPTS:
                this_threshold = cigarette_score_threshold
            else:
                this_threshold = score_threshold

            order = np.argsort(-scores_np)
            for idx in order:
                score = float(scores_np[idx])
                if score < this_threshold:
                    continue
                all_detections.append(
                    Detection(box=boxes_np[idx], score=score, label=prompt)
                )

    if all_detections and nms_iou_threshold > 0:
        all_detections = nms_detections(all_detections, nms_iou_threshold)

    inference_time = time.perf_counter() - start

    if not all_detections:
        return [], inference_time

    # Sort by score (descending)
    all_detections.sort(key=lambda d: d.score, reverse=True)

    # Respect max_detections, but NEVER drop cigarettes in favor of non-cigarettes.
    if max_detections > 0 and len(all_detections) > max_detections:
        cig_dets = [d for d in all_detections if d.label in CIGARETTE_PROMPTS]
        non_cig_dets = [d for d in all_detections if d.label not in CIGARETTE_PROMPTS]

        # If there are more cigarettes than max_detections, keep the best ones
        if len(cig_dets) >= max_detections:
            cig_dets.sort(key=lambda d: d.score, reverse=True)
            all_detections = cig_dets[:max_detections]
        else:
            remaining_slots = max_detections - len(cig_dets)
            non_cig_dets.sort(key=lambda d: d.score, reverse=True)
            all_detections = cig_dets + non_cig_dets[:remaining_slots]

    return all_detections, inference_time


def filter_leaf_overlaps(
    waste_dets: List[Detection],
    leaf_dets: List[Detection],
    iou_threshold: float,
) -> Tuple[List[Detection], List[RemovalByLeaf]]:
    """
    Remove waste detections that strongly overlap SAM leaf detections.

    Cigarette prompts are *never* removed here.

    Returns:
      kept_waste, removed_records
    """
    if not waste_dets or not leaf_dets or iou_threshold <= 0:
        return waste_dets, []

    leaf_boxes = [d.box for d in leaf_dets]
    kept: List[Detection] = []
    removed: List[RemovalByLeaf] = []

    for w_det in waste_dets:
        # Never remove cigarette(-butt) detections due to leaves
        if w_det.label in CIGARETTE_PROMPTS:
            kept.append(w_det)
            continue

        ious = bbox_iou(w_det.box, leaf_boxes)
        if ious.size == 0:
            kept.append(w_det)
            continue

        best_idx = int(np.argmax(ious))
        best_iou = float(ious[best_idx])

        if best_iou >= iou_threshold:
            removed.append(
                RemovalByLeaf(
                    waste_det=w_det,
                    leaf_det=leaf_dets[best_idx],
                    iou=best_iou,
                )
            )
        else:
            kept.append(w_det)

    return kept, removed


def filter_small_detections(
    dets: List[Detection],
    width: int,
    height: int,
    min_area_ratio: float,
) -> List[Detection]:
    """
    Remove detections whose area is smaller than min_area_ratio * image_area,
    except for cigarette prompts which are always kept.
    """
    if min_area_ratio <= 0 or not dets:
        return dets

    img_area = float(width * height)
    min_area = img_area * float(min_area_ratio)

    kept: List[Detection] = []
    for det in dets:
        # Cigarettes are always kept, regardless of size
        if det.label in CIGARETTE_PROMPTS:
            kept.append(det)
            continue

        x1, y1, x2, y2 = det.box.tolist()
        w = max(0.0, x2 - x1)
        h = max(0.0, y2 - y1)
        area = w * h

        if area >= min_area:
            kept.append(det)
        # else: too small, drop it

    return kept


def _boxes_close_or_overlap(a: np.ndarray, b: np.ndarray, gap: float) -> bool:
    """
    Returns True if two boxes overlap or are at most `gap` pixels apart
    in both x and y directions.
    Boxes are [x1, y1, x2, y2].
    """
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    # If they overlap, distance is 0 in that dimension.
    dx = max(0.0, max(bx1 - ax2, ax1 - bx2))
    dy = max(0.0, max(by1 - ay2, ay1 - by2))

    return dx <= gap and dy <= gap


def _merge_close_boxes(boxes: List[np.ndarray], gap: float) -> List[np.ndarray]:
    """
    Iteratively merge boxes that are closer than `gap` or overlapping.
    """
    result: List[np.ndarray] = []
    boxes = [np.array(b, dtype=float) for b in boxes]

    # First pass: greedily merge into base boxes
    while boxes:
        base = boxes.pop(0)
        keep: List[np.ndarray] = []

        for b in boxes:
            if _boxes_close_or_overlap(base, b, gap):
                x1 = min(base[0], b[0])
                y1 = min(base[1], b[1])
                x2 = max(base[2], b[2])
                y2 = max(base[3], b[3])
                base = np.array([x1, y1, x2, y2], dtype=float)
            else:
                keep.append(b)

        boxes = keep
        result.append(base)

    # Second pass: in case merges should chain indirectly
    changed = True
    while changed and len(result) > 1:
        changed = False
        new_result: List[np.ndarray] = []

        while result:
            base = result.pop(0)
            keep: List[np.ndarray] = []
            for b in result:
                if _boxes_close_or_overlap(base, b, gap):
                    x1 = min(base[0], b[0])
                    y1 = min(base[1], b[1])
                    x2 = max(base[2], b[2])
                    y2 = max(base[3], b[3])
                    base = np.array([x1, y1, x2, y2], dtype=float)
                    changed = True
                else:
                    keep.append(b)
            result = keep
            new_result.append(base)

        result = new_result

    return result


def build_leaf_regions_from_detections(
    leaf_dets: List[Detection],
    width: int,
    height: int,
    min_leaf_count: int = MIN_LEAVES_IN_REGION,
    center_dist_ratio: float = LEAF_REGION_CENTER_DIST_RATIO,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Build 'leaf region' and 'leaf pile' boxes from SAM3 leaf detections.

    - Regions: clusters of >= min_leaf_count boxes from 'leaf lying on the street',
      clustered by center distance, then merged if their cluster boxes are
      close or overlapping.
    - Piles: direct boxes from 'pile of leaves on the ground'.
    """
    region_seed_prompt = "leaf lying on the street"
    pile_prompt = "pile of leaves on the ground"

    centers = []
    boxes = []
    pile_boxes: List[np.ndarray] = []

    for det in leaf_dets:
        if det.label == region_seed_prompt:
            x1, y1, x2, y2 = det.box.tolist()
            cx = 0.5 * (x1 + x2)
            cy = 0.5 * (y1 + y2)
            centers.append((cx, cy))
            boxes.append(det.box.astype(float))
        elif det.label == pile_prompt:
            pile_boxes.append(det.box.astype(float))

    if not centers:
        return [], pile_boxes

    centers = np.array(centers, dtype=float)
    n = len(centers)

    # Distance threshold in pixels
    max_dist = center_dist_ratio * float(min(width, height))
    max_dist_sq = max_dist * max_dist

    remaining = set(range(n))
    clusters: List[List[int]] = []

    # BFS clustering by center distance
    while remaining:
        seed = remaining.pop()
        queue = [seed]
        cluster = [seed]

        while queue:
            i = queue.pop()
            ci = centers[i]
            for j in list(remaining):
                cj = centers[j]
                dx = ci[0] - cj[0]
                dy = ci[1] - cj[1]
                if dx * dx + dy * dy <= max_dist_sq:
                    remaining.remove(j)
                    queue.append(j)
                    cluster.append(j)

        clusters.append(cluster)

    region_boxes: List[np.ndarray] = []
    for cluster in clusters:
        if len(cluster) < min_leaf_count:
            continue

        xs1, ys1, xs2, ys2 = [], [], [], []
        for idx in cluster:
            x1, y1, x2, y2 = boxes[idx]
            xs1.append(x1)
            ys1.append(y1)
            xs2.append(x2)
            ys2.append(y2)

        region_box = np.array(
            [min(xs1), min(ys1), max(xs2), max(ys2)], dtype=float
        )
        region_boxes.append(region_box)

    # Merge nearby region boxes so clusters next to each other
    # become a single big 'leaf region'
    if region_boxes:
        gap = MERGE_REGION_GAP_RATIO * float(min(width, height))
        region_boxes = _merge_close_boxes(region_boxes, gap)

    return region_boxes, pile_boxes

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
                    "sam3_prompt": det.label,
                }
            )
    with open(path, "w") as f:
        json.dump(serialized, f, indent=2)


def _build_sam3_metrics_dict(
    base_metrics: Dict[str, float],
    avg_inference_time: float,
    img_size: int | None,
) -> Dict[str, float]:
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

    class_names, class_ids = get_dataset_classes(dataset_yaml)
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


def _strip_on_the_ground(prompt: str) -> str:
    suffix = " on the ground"
    if prompt.endswith(suffix):
        return prompt[: -len(suffix)]
    return prompt


def _draw_sam3_boxes_on_image(
    img_path: Path,
    detections: List[Detection],
) -> np.ndarray | None:
    if cv2 is None:
        return None

    img = cv2.imread(str(img_path))
    if img is None:
        return None

    for det in detections:
        x1, y1, x2, y2 = det.box.astype(int)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

        display_label = _strip_on_the_ground(det.label)
        label_text = f"{display_label} {det.score:.2f}"

        cv2.putText(
            img,
            label_text,
            (x1, max(y1 - 10, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    return img


def _draw_removed_boxes_on_image(
    img_path: Path,
    removed: List[RemovalByLeaf],
) -> np.ndarray | None:
    """
    Third panel: show waste boxes that were removed, and which leaf prompt removed them.
    Waste box: red, leaf box: green. Label shows waste_prompt -> leaf_prompt + IoU.
    """
    if cv2 is None:
        return None

    img = cv2.imread(str(img_path))
    if img is None:
        return None

    for r in removed:
        # Waste box in red
        wx1, wy1, wx2, wy2 = r.waste_det.box.astype(int)
        cv2.rectangle(img, (wx1, wy1), (wx2, wy2), (0, 0, 255), 2)

        # Leaf box in green
        lx1, ly1, lx2, ly2 = r.leaf_det.box.astype(int)
        cv2.rectangle(img, (lx1, ly1), (lx2, ly2), (0, 255, 0), 2)

        waste_label = _strip_on_the_ground(r.waste_det.label)
        leaf_label = _strip_on_the_ground(r.leaf_det.label)

        text = (
            f"REM {waste_label} {r.waste_det.score:.2f} "
            f"by {leaf_label} {r.leaf_det.score:.2f} IoU={r.iou:.2f}"
        )

        cv2.putText(
            img,
            text,
            (wx1, max(wy1 - 10, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    return img


def _draw_leaf_regions_on_image(
    img_path: Path,
    region_boxes: List[np.ndarray],
    pile_boxes: List[np.ndarray],
) -> np.ndarray | None:
    """
    Fourth panel: show leaf regions and piles.

    - leaf_region boxes: green
    - leaf_pile boxes: blue
    """
    if cv2 is None:
        return None

    img = cv2.imread(str(img_path))
    if img is None:
        return None

    # Draw leaf regions (clusters)
    for box in region_boxes:
        x1, y1, x2, y2 = box.astype(int)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            img,
            "leaf_region",
            (x1, max(y1 - 10, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    # Draw leaf piles
    for box in pile_boxes:
        x1, y1, x2, y2 = box.astype(int)
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)  # blue
        cv2.putText(
            img,
            "leaf_pile",
            (x1, max(y1 - 10, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 0, 0),
            2,
            cv2.LINE_AA,
        )

    return img


def _generate_side_by_side_comparisons_sam3_vs_current_best(
    base_model,
    image_paths: List[Path],
    predictions: Dict[str, List[Detection]],
    removed_by_leaf: Dict[str, List[RemovalByLeaf]],
    leaf_regions: Dict[str, Dict[str, List[np.ndarray]]],
    output_dir: Path,
    conf_threshold: float = 0.25,
) -> None:
    """
    Create a 2x2 grid per image:

        [ YOLO current_best | SAM3 waste ]
        [ Removed-by-leaf   | Leaf regions (region + pile) ]
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
        removed = removed_by_leaf.get(image_id, [])
        lr = leaf_regions.get(image_id, {})
        region_boxes = lr.get("regions", [])
        pile_boxes = lr.get("piles", [])

        # YOLO panel
        base_results = base_model.predict(
            str(img_path), conf=conf_threshold, save=False, verbose=False
        )
        base_img = base_results[0].plot()

        # SAM3 waste panel
        sam3_img = _draw_sam3_boxes_on_image(img_path, sam3_dets)
        if sam3_img is None:
            continue

        # Removed-by-leaf panel
        removed_img = _draw_removed_boxes_on_image(img_path, removed)
        if removed_img is None:
            continue

        # Leaf regions / piles panel
        leaf_img = _draw_leaf_regions_on_image(img_path, region_boxes, pile_boxes)
        if leaf_img is None:
            continue

        # Resize all panels to match YOLO panel size
        h, w = base_img.shape[:2]

        def _resize_to_base(img: np.ndarray) -> np.ndarray:
            if img.shape[:2] != (h, w):
                img = cv2.resize(
                    img,
                    (w, h),
                    interpolation=cv2.INTER_LINEAR,
                )
            return img

        sam3_img = _resize_to_base(sam3_img)
        removed_img = _resize_to_base(removed_img)
        leaf_img = _resize_to_base(leaf_img)

        # 2x2 mosaic:
        # [ base_img | sam3_img ]
        # [ removed_img | leaf_img ]
        top_row = np.hstack((base_img, sam3_img))
        bottom_row = np.hstack((removed_img, leaf_img))
        comparison_img = np.vstack((top_row, bottom_row))

        save_path = side_by_side_dir / f"comparison_{img_path.name}"
        cv2.imwrite(str(save_path), comparison_img)


def main() -> None:
    args = parse_args()
    if args.leaf_score_threshold is None:
        args.leaf_score_threshold = args.score_threshold
    if args.cigarette_score_threshold is None:
        # Use a lower default threshold for cigarettes
        args.cigarette_score_threshold = max(0.0, args.score_threshold * 0.5)

    device = resolve_device(args.device)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    preds_path = args.save_predictions or (output_dir / "predictions.json")

    dataset_yaml = args.dataset or _infer_dataset_yaml_from_params()
    images_dir, labels_dir = load_dataset_paths(dataset_yaml)
    image_paths = list_images(images_dir, args.limit)
    if not image_paths:
        raise RuntimeError(f"No images found in {images_dir}")

    bpe_path = ensure_bpe_vocab()
    model = build_sam3_image_model(bpe_path=str(bpe_path))
    model.to(device)
    model.eval()
    processor = Sam3Processor(model)

    predictions: Dict[str, List[Detection]] = {}
    removed_by_leaf: Dict[str, List[RemovalByLeaf]] = {}
    ground_truths: Dict[str, List[np.ndarray]] = {}
    image_sizes: Dict[str, Tuple[int, int]] = {}
    leaf_regions: Dict[str, Dict[str, List[np.ndarray]]] = {}
    inference_times: List[float] = []

    for img_path in tqdm(
        image_paths, desc="Running SAM 3 (waste + leaves, multi-prompt + NMS)"
    ):
        with Image.open(img_path) as img_raw:
            image = img_raw.convert("RGB")
            width, height = image.size

        label_path = labels_dir / f"{img_path.stem}.txt"
        gt_boxes = load_ground_truth_boxes(label_path, width, height)

        # Waste detections
        waste_dets, t_waste = run_sam3_multi_prompt_inference(
            processor,
            image,
            WASTE_PROMPTS,
            args.score_threshold,
            args.max_detections,
            args.nms_iou_threshold,
            cigarette_score_threshold=args.cigarette_score_threshold,
        )

        # Leaf detections (used for suppression + region building)
        leaf_dets, t_leaf = run_sam3_multi_prompt_inference(
            processor,
            image,
            LEAF_PROMPTS,
            args.leaf_score_threshold,
            args.max_detections,
            args.nms_iou_threshold,
        )

        # Build leaf regions (clustered) + piles
        region_boxes, pile_boxes = build_leaf_regions_from_detections(
            leaf_dets,
            width,
            height,
            min_leaf_count=MIN_LEAVES_IN_REGION,
            center_dist_ratio=LEAF_REGION_CENTER_DIST_RATIO,
        )

        # Remove waste boxes that overlap leaf boxes, but keep track of what we dropped
        dets_leaf_filtered, removed = filter_leaf_overlaps(
            waste_dets, leaf_dets, iou_threshold=args.leaf_iou_threshold
        )

        # Apply minimum area filter (but never drop cigarette prompts)
        dets = filter_small_detections(
            dets_leaf_filtered,
            width,
            height,
            min_area_ratio=args.min_area_ratio,
        )

        image_id = img_path.stem
        predictions[image_id] = dets
        removed_by_leaf[image_id] = removed
        ground_truths[image_id] = gt_boxes
        image_sizes[image_id] = (width, height)
        leaf_regions[image_id] = {
            "regions": region_boxes,
            "piles": pile_boxes,
        }

        inference_times.append(t_waste + t_leaf)

    # ---- Serialize leaf removals for prompt analyzer ----
    leaf_removals: List[dict] = []
    for image_id, removals in removed_by_leaf.items():
        width, height = image_sizes[image_id]
        for r in removals:
            leaf_removals.append(
                {
                    "image_id": image_id,
                    "waste_bbox_xyxy": [float(x) for x in r.waste_det.box.tolist()],
                    "waste_score": float(r.waste_det.score),
                    "waste_prompt": r.waste_det.label,
                    "leaf_bbox_xyxy": [float(x) for x in r.leaf_det.box.tolist()],
                    "leaf_score": float(r.leaf_det.score),
                    "leaf_prompt": r.leaf_det.label,
                    "iou_waste_leaf": float(r.iou),
                    "width": width,
                    "height": height,
                }
            )

    leaf_removals_path = output_dir / "leaf_removals.json"
    with open(leaf_removals_path, "w") as f:
        json.dump(leaf_removals, f, indent=2)
    print(f"Saved leaf removals to {leaf_removals_path}")

    # ---- Serialize leaf regions (for future leaf model training) ----
    leaf_regions_serialized: List[dict] = []
    for image_id, lr in leaf_regions.items():
        width, height = image_sizes[image_id]
        for box in lr["regions"]:
            leaf_regions_serialized.append(
                {
                    "image_id": image_id,
                    "type": "leaf_region",
                    "bbox_xyxy": [float(x) for x in box.tolist()],
                    "width": width,
                    "height": height,
                }
            )
        for box in lr["piles"]:
            leaf_regions_serialized.append(
                {
                    "image_id": image_id,
                    "type": "leaf_pile",
                    "bbox_xyxy": [float(x) for x in box.tolist()],
                    "width": width,
                    "height": height,
                }
            )

    leaf_regions_path = output_dir / "leaf_regions.json"
    with open(leaf_regions_path, "w") as f:
        json.dump(leaf_regions_serialized, f, indent=2)
    print(f"Saved leaf regions to {leaf_regions_path}")

    # ---- existing metric / prediction saving ----
    iou_thresholds = np.arange(0.5, 0.96, 0.05)
    metrics = evaluate_predictions(predictions, ground_truths, iou_thresholds)
    avg_inference_time = float(np.mean(inference_times)) if inference_times else 0.0

    metrics_payload = {
        "prompt": "; ".join(WASTE_PROMPTS),
        "leaf_prompts": "; ".join(LEAF_PROMPTS),
        "model_id": args.model_id,
        "dataset_yaml": str(dataset_yaml),
        "num_images": len(image_paths),
        "num_ground_truth_boxes": int(sum(len(v) for v in ground_truths.values())),
        "num_predictions": int(sum(len(v) for v in predictions.values())),
        "score_threshold": args.score_threshold,
        "leaf_score_threshold": args.leaf_score_threshold,
        "cigarette_score_threshold": args.cigarette_score_threshold,
        "max_detections": args.max_detections,
        "nms_iou_threshold": args.nms_iou_threshold,
        "leaf_iou_threshold": args.leaf_iou_threshold,
        "min_area_ratio": args.min_area_ratio,
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

    sam3_metrics_dict = _build_sam3_metrics_dict(
        metrics, avg_inference_time=avg_inference_time, img_size=None
    )
    sam3_experiment_name = "SAM3 multi-prompt + leaf-suppression + min-area + NMS"

    try:
        _evaluate_current_best_and_write_csv(
            dataset_yaml, sam3_metrics_dict, sam3_experiment_name
        )
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
                    removed_by_leaf,
                    leaf_regions,
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
