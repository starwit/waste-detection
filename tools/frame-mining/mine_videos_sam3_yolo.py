from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

from tqdm import tqdm

import cv2
import numpy as np
import sam3
import torch
from PIL import Image
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from ultralytics import YOLO

# ---------------------------
#   PROMPTS / CONSTANTS
# ---------------------------

# Your custom waste prompts
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

# Leaf prompts
LEAF_PROMPTS: List[str] = [
    "pile of leaves on the ground",
    "leaf lying on the street",
]

# Prompts that correspond to cigarettes
CIGARETTE_PROMPTS: List[str] = [
    "cigarette butt on the ground",
    "discarded cigarette butt on the ground",
    "small white and brown cigarette butt on the street",
]

# Leaf region hyperparams
MIN_LEAVES_IN_REGION = 8
LEAF_REGION_CENTER_DIST_RATIO = 0.15
MERGE_REGION_GAP_RATIO = 0.05

# SAM3 tokenizer asset
BPE_FILENAME = "bpe_simple_vocab_16e6.txt.gz"
BPE_URL = (
    "https://github.com/facebookresearch/sam3/"
    "raw/main/assets/bpe_simple_vocab_16e6.txt.gz"
)


# ---------------------------
#   DATA STRUCTURES
# ---------------------------

@dataclass
class Detection:
    box: np.ndarray  # xyxy
    score: float
    label: str       # SAM3 prompt that produced this detection


# ---------------------------
#   UTILS
# ---------------------------

def ensure_bpe_vocab() -> Path:
    sam3_root = Path(sam3.__file__).resolve().parent.parent
    assets_dir = sam3_root / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    bpe_path = assets_dir / BPE_FILENAME

    if not bpe_path.exists():
        print(f"[sam3-video] BPE vocab not found at {bpe_path}, downloading...")
        import urllib.request

        with urllib.request.urlopen(BPE_URL) as resp, open(bpe_path, "wb") as out_f:
            out_f.write(resp.read())

    return bpe_path


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available. Falling back to CPU.")
        return torch.device("cpu")
    return torch.device(requested)


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


def xyxy_to_xywhn(box: np.ndarray, width: int, height: int) -> Tuple[float, float, float, float]:
    x1, y1, x2, y2 = box.tolist()
    w = x2 - x1
    h = y2 - y1
    x_c = x1 + w / 2
    y_c = y1 + h / 2
    return (
        x_c / width,
        y_c / height,
        w / width,
        h / height,
    )


def _get_output_field(output: dict, keys: List[str]):
    for k in keys:
        if k in output:
            v = output[k]
            if v is not None:
                return v
    return None


# ---------------------------
#   NMS & FILTERING
# ---------------------------

def nms_detections(detections: List[Detection], iou_threshold: float) -> List[Detection]:
    """
    NMS across prompts. Cigarette detections are never suppressed.
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
                new_rest.append(int(j))
            else:
                # never suppress cigarettes
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
) -> List[Detection]:
    """
    Run SAM3 for a list of prompts on one image.
    """
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
                boxes.detach().cpu() if hasattr(boxes, "detach") else boxes, dtype=float
            )
            scores_np = np.asarray(
                scores.detach().cpu() if hasattr(scores, "detach") else scores, dtype=float
            )

            if boxes_np.size == 0 or scores_np.size == 0:
                continue
            if boxes_np.ndim != 2 or boxes_np.shape[1] != 4:
                continue

            if cigarette_score_threshold is not None and prompt in CIGARETTE_PROMPTS:
                this_thr = cigarette_score_threshold
            else:
                this_thr = score_threshold

            order = np.argsort(-scores_np)
            for idx in order:
                score = float(scores_np[idx])
                if score < this_thr:
                    continue
                all_detections.append(
                    Detection(box=boxes_np[idx], score=score, label=prompt)
                )

    if all_detections and nms_iou_threshold > 0:
        all_detections = nms_detections(all_detections, nms_iou_threshold)

    if not all_detections:
        return []

    all_detections.sort(key=lambda d: d.score, reverse=True)

    if max_detections > 0 and len(all_detections) > max_detections:
        cig_dets = [d for d in all_detections if d.label in CIGARETTE_PROMPTS]
        non_cig_dets = [d for d in all_detections if d.label not in CIGARETTE_PROMPTS]

        if len(cig_dets) >= max_detections:
            cig_dets.sort(key=lambda d: d.score, reverse=True)
            all_detections = cig_dets[:max_detections]
        else:
            remaining_slots = max_detections - len(cig_dets)
            non_cig_dets.sort(key=lambda d: d.score, reverse=True)
            all_detections = cig_dets + non_cig_dets[:remaining_slots]

    return all_detections


def filter_leaf_overlaps(
    waste_dets: List[Detection],
    leaf_dets: List[Detection],
    iou_threshold: float,
) -> List[Detection]:
    """
    - Remove waste detections that overlap leaf detections at IoU >= threshold.
    - NEVER remove cigarette prompts.
    """
    if not waste_dets or not leaf_dets or iou_threshold <= 0:
        return waste_dets

    leaf_boxes = [d.box for d in leaf_dets]
    kept: List[Detection] = []

    for w_det in waste_dets:
        # Never remove cigarette(-butt) detections due to leaves
        if w_det.label in CIGARETTE_PROMPTS:
            kept.append(w_det)
            continue

        ious = bbox_iou(w_det.box, leaf_boxes)
        if ious.size == 0:
            kept.append(w_det)
            continue

        best_iou = float(ious.max())
        if best_iou < iou_threshold:
            kept.append(w_det)
        # else: drop, overlaps leaf too much

    return kept


def filter_small_detections(
    dets: List[Detection],
    width: int,
    height: int,
    min_area_ratio: float,
) -> List[Detection]:
    """
    Remove detections below min area (except cigarettes).
    """
    if min_area_ratio <= 0 or not dets:
        return dets

    img_area = float(width * height)
    min_area = img_area * float(min_area_ratio)

    kept: List[Detection] = []
    for det in dets:
        if det.label in CIGARETTE_PROMPTS:
            kept.append(det)
            continue

        x1, y1, x2, y2 = det.box.tolist()
        w = max(0.0, x2 - x1)
        h = max(0.0, y2 - y1)
        area = w * h

        if area >= min_area:
            kept.append(det)

    return kept


def _boxes_close_or_overlap(a: np.ndarray, b: np.ndarray, gap: float) -> bool:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    dx = max(0.0, max(bx1 - ax2, ax1 - bx2))
    dy = max(0.0, max(by1 - ay2, ay1 - by2))

    return dx <= gap and dy <= gap


def _merge_close_boxes(boxes: List[np.ndarray], gap: float) -> List[np.ndarray]:
    result: List[np.ndarray] = []
    boxes = [np.array(b, dtype=float) for b in boxes]

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

    # second pass to chain merges
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

    max_dist = center_dist_ratio * float(min(width, height))
    max_dist_sq = max_dist * max_dist

    remaining = set(range(n))
    clusters: List[List[int]] = []

    # BFS clustering
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

    if region_boxes:
        gap = MERGE_REGION_GAP_RATIO * float(min(width, height))
        region_boxes = _merge_close_boxes(region_boxes, gap)

    return region_boxes, pile_boxes


# ---------------------------
#   DRAWING / VIS
# ---------------------------

def draw_annotated_image(
    frame_bgr: np.ndarray,
    waste_dets: List[Detection],
    leaf_region_boxes: List[np.ndarray],
    leaf_pile_boxes: List[np.ndarray],
) -> np.ndarray:
    """
    Draw final mined detections (waste, cigarette, leaf_pile, leaf_region).
    """
    img = frame_bgr.copy()

    # waste / cigarette (with confidence)
    for det in waste_dets:
        x1, y1, x2, y2 = det.box.astype(int)
        score = det.score

        if det.label in CIGARETTE_PROMPTS:
            cls_name = "cigarette"
            color = (0, 255, 255)  # yellow
        else:
            cls_name = "waste"
            color = (0, 0, 255)    # red

        label_text = f"{cls_name} {score:.2f}"

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            img,
            label_text,
            (x1, max(y1 - 5, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2,
            cv2.LINE_AA,
        )

    # leaf piles (class 2) – blue
    for box in leaf_pile_boxes:
        x1, y1, x2, y2 = box.astype(int)
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(
            img,
            "leaf_pile",
            (x1, max(y1 - 5, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 0, 0),
            2,
            cv2.LINE_AA,
        )

    # leaf regions (class 3) – green
    for box in leaf_region_boxes:
        x1, y1, x2, y2 = box.astype(int)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            img,
            "leaf_region",
            (x1, max(y1 - 5, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    return img


def draw_yolo_annotated_image(
    frame_bgr: np.ndarray,
    yolo_boxes: List[np.ndarray],
    yolo_cls_ids: List[int],
    yolo_confs: List[float],
    waste_class_ids: Iterable[int],
    cigarette_class_ids: Iterable[int],
) -> np.ndarray:
    """
    Draw YOLO detections (waste / cigarette) with their confidence.
    Used for the false_positives export.
    """
    img = frame_bgr.copy()
    waste_set = set(waste_class_ids)
    cig_set = set(cigarette_class_ids)

    for box, cls_id, conf in zip(yolo_boxes, yolo_cls_ids, yolo_confs):
        x1, y1, x2, y2 = np.asarray(box, dtype=float).astype(int)

        if cls_id in cig_set:
            cls_name = "cigarette"
            color = (0, 255, 255)  # yellow
        elif cls_id in waste_set:
            cls_name = "waste"
            color = (0, 0, 255)    # red
        else:
            continue

        label_text = f"{cls_name} {conf:.2f}"
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            img,
            label_text,
            (x1, max(y1 - 5, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2,
            cv2.LINE_AA,
        )

    return img


# ---------------------------
#   MAIN VIDEO PIPELINE
# ---------------------------

def parse_args() -> argparse.Namespace:
    script_root = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(
        description=(
            "Mine pseudo-labels from video(s) using SAM3 + YOLO.\n"
            "Exports two sets:\n"
            "  - sam3_gated: SAM3-only + low-confidence YOLO matches\n"
            "  - false_positives: YOLO confident detections with no SAM3 support"
        )
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--video",
        type=Path,
        help="Single video file to process",
    )
    group.add_argument(
        "--video-dir",
        type=Path,
        help="Directory containing .mkv videos to process",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=script_root / "sam3" /"video_mining_output",
        help="Root output directory (default: next to this script)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device for SAM3",
    )
    parser.add_argument(
        "--sample-fps",
        type=float,
        default=1.0,
        help="How many frames per second to sample from the video(s)",
    )
    # SAM3 thresholds
    parser.add_argument("--score-threshold", type=float, default=0.6)
    parser.add_argument("--leaf-score-threshold", type=float, default=0.5)
    parser.add_argument("--cigarette-score-threshold", type=float, default=None)
    parser.add_argument("--max-detections", type=int, default=50)
    parser.add_argument("--nms-iou-threshold", type=float, default=0.5)
    parser.add_argument("--min-area-ratio", type=float, default=1e-3)
    parser.add_argument(
        "--leaf-iou-threshold",
        type=float,
        default=0.5,
        help="IoU threshold: waste vs leaf; overlapping waste boxes are dropped (except cigarettes).",
    )

    # YOLO stuff
    parser.add_argument(
        "--yolo-weights",
        type=Path,
        required=True,
        help="Path to YOLO weights (.pt) to compare against.",
    )
    parser.add_argument(
        "--yolo-waste-class-ids",
        type=int,
        nargs="*",
        default=[0],
        help="YOLO class IDs that count as waste (for matching).",
    )
    parser.add_argument(
        "--yolo-cigarette-class-ids",
        type=int,
        nargs="*",
        default=[1],
        help="YOLO class IDs that count as cigarette (for matching).",
    )
    parser.add_argument(
        "--match-iou-threshold",
        type=float,
        default=0.5,
        help="IoU threshold to consider SAM and YOLO as 'same object' (for mining logic).",
    )
    parser.add_argument(
        "--yolo-unsure-threshold",
        type=float,
        default=0.5,
        help=(
            "YOLO confidence below this is treated as 'unsure' when there is a SAM3 match "
            "(frame goes to sam3_gated). YOLO detections above this, with no SAM3 match, "
            "go to false_positives."
        ),
    )

    return parser.parse_args()


def process_video(
    video_path: Path,
    processor: Sam3Processor,
    yolo_model: YOLO,
    args: argparse.Namespace,
    sam3_images_dir: Path,
    sam3_labels_dir: Path,
    sam3_annotated_dir: Path,
    fp_images_dir: Path,
    fp_labels_dir: Path,
    fp_annotated_dir: Path,
) -> Tuple[int, int]:
    """
    Run the full mining pipeline on a single video.

    Returns
    -------
    sam_frames_saved : int
        Number of frames saved in the sam3_gated set.
    fp_frames_saved : int
        Number of frames saved in the false_positives set.
    """
    print(f"\n[sam3-video] Processing video: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps <= 0:
        print("[sam3-video] Warning: FPS not available, assuming 25.")
        video_fps = 25.0

    frame_interval = max(1, int(round(video_fps / max(args.sample_fps, 1e-3))))
    print(
        f"[sam3-video] {video_path.name}: FPS={video_fps:.2f}, "
        f"sampling every {frame_interval} frames (~{args.sample_fps:.2f} fps)"
    )

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        total_frames = None  # some codecs don't expose this

    frame_idx = 0
    sam_saved_counter = 0
    fp_saved_counter = 0

    yolo_relevant_classes = set(
        args.yolo_waste_class_ids + args.yolo_cigarette_class_ids
    )

    with tqdm(total=total_frames, desc=f"{video_path.name}") as pbar:
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break

            pbar.update(1)

            # sampling at desired FPS
            if frame_idx % frame_interval != 0:
                frame_idx += 1
                continue

            height, width = frame_bgr.shape[:2]
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)

            # --- SAM3: waste / cigarettes (raw) ---
            waste_dets = run_sam3_multi_prompt_inference(
                processor=processor,
                image=pil_image,
                prompts=WASTE_PROMPTS,
                score_threshold=args.score_threshold,
                max_detections=args.max_detections,
                nms_iou_threshold=args.nms_iou_threshold,
                cigarette_score_threshold=args.cigarette_score_threshold,
            )

            # --- SAM3: leaves ---
            leaf_dets = run_sam3_multi_prompt_inference(
                processor=processor,
                image=pil_image,
                prompts=LEAF_PROMPTS,
                score_threshold=args.leaf_score_threshold,
                max_detections=args.max_detections,
                nms_iou_threshold=args.nms_iou_threshold,
                cigarette_score_threshold=None,
            )

            # --- Leaf-based suppression of waste ---
            waste_dets = filter_leaf_overlaps(
                waste_dets,
                leaf_dets,
                iou_threshold=args.leaf_iou_threshold,
            )

            # --- Min-area filter (after leaf suppression) ---
            waste_dets = filter_small_detections(
                waste_dets,
                width=width,
                height=height,
                min_area_ratio=args.min_area_ratio,
            )

            # --- Build leaf regions + piles from leaf_dets ---
            leaf_region_boxes, leaf_pile_boxes = build_leaf_regions_from_detections(
                leaf_dets,
                width=width,
                height=height,
                min_leaf_count=MIN_LEAVES_IN_REGION,
                center_dist_ratio=LEAF_REGION_CENTER_DIST_RATIO,
            )

            # --- YOLO inference on this frame (always run) ---
            yolo_results = yolo_model.predict(
                frame_bgr, conf=0.25, verbose=False
            )[0]  # single frame

            if yolo_results.boxes is not None and len(yolo_results.boxes) > 0:
                all_boxes = yolo_results.boxes.xyxy.cpu().numpy()
                all_cls = yolo_results.boxes.cls.cpu().numpy().astype(int)
                all_conf = yolo_results.boxes.conf.cpu().numpy()
            else:
                all_boxes = np.zeros((0, 4), dtype=float)
                all_cls = np.zeros((0,), dtype=int)
                all_conf = np.zeros((0,), dtype=float)

            yolo_boxes_relevant: List[np.ndarray] = []
            yolo_cls_relevant: List[int] = []
            yolo_conf_relevant: List[float] = []
            for box, cls_id, conf in zip(all_boxes, all_cls, all_conf):
                if cls_id in yolo_relevant_classes:
                    yolo_boxes_relevant.append(np.array(box, dtype=float))
                    yolo_cls_relevant.append(int(cls_id))
                    yolo_conf_relevant.append(float(conf))

            sam_wc_dets = waste_dets  # waste + cigarette prompts after leaf suppression

            # --- Compute IoU matrix SAM vs YOLO (for relevant classes) ---
            ious_mat = None
            if sam_wc_dets and yolo_boxes_relevant:
                yolo_boxes_arr = np.stack(yolo_boxes_relevant, axis=0)
                ious_mat = np.zeros((len(sam_wc_dets), len(yolo_boxes_relevant)), dtype=float)
                for i, det in enumerate(sam_wc_dets):
                    ious_mat[i] = bbox_iou(det.box, yolo_boxes_arr)

            # --- Decide whether this frame goes into the sam3_gated set ---
            # Condition: there exists at least one SAM detection that YOLO misses,
            # or YOLO sees it but is not confident.
            interesting_sam_exist = False
            if sam_wc_dets:
                if not yolo_boxes_relevant:
                    # YOLO completely missed all waste/cigarettes
                    interesting_sam_exist = True
                else:
                    assert ious_mat is not None
                    for i in range(len(sam_wc_dets)):
                        row = ious_mat[i]
                        if row.size == 0:
                            continue
                        best_j = int(row.argmax())
                        best_iou = float(row[best_j])
                        best_conf = float(yolo_conf_relevant[best_j])
                        # keep frame if SAM has an object that YOLO does not see well enough
                        if (best_iou < args.match_iou_threshold) or (best_conf < args.yolo_unsure_threshold):
                            interesting_sam_exist = True
                            break

            if interesting_sam_exist:
                final_waste_dets = sam_wc_dets
            else:
                final_waste_dets = []

            # --- Find YOLO-only potential false positives ---
            # Condition: YOLO detects waste/cigarette with high confidence
            # but there is no matching SAM detection (IoU < match_iou_threshold).
            fp_indices: List[int] = []
            if yolo_boxes_relevant:
                if ious_mat is None or ious_mat.shape[0] == 0:
                    # no SAM detections: every YOLO detection is a potential FP
                    for j, conf in enumerate(yolo_conf_relevant):
                        if conf >= args.yolo_unsure_threshold:
                            fp_indices.append(j)
                else:
                    for j in range(len(yolo_boxes_relevant)):
                        col = ious_mat[:, j]
                        best_iou = float(col.max()) if col.size > 0 else 0.0
                        if (best_iou < args.match_iou_threshold) and (
                            yolo_conf_relevant[j] >= args.yolo_unsure_threshold
                        ):
                            fp_indices.append(j)

            file_stem = f"{video_path.stem}_f{frame_idx:06d}"

            # --- Save sam3_gated image/labels/annotated ---
            if interesting_sam_exist and (final_waste_dets or leaf_region_boxes or leaf_pile_boxes):
                img_out_path = sam3_images_dir / f"{file_stem}.jpg"
                cv2.imwrite(str(img_out_path), frame_bgr)

                label_lines: List[str] = []

                # class 0: waste, class 1: cigarette
                for det in final_waste_dets:
                    x_c, y_c, w, h = xyxy_to_xywhn(det.box, width, height)
                    if det.label in CIGARETTE_PROMPTS:
                        cls_id = 1
                    else:
                        cls_id = 0
                    label_lines.append(
                        f"{cls_id} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}"
                    )

                # class 2: leaf_pile
                for box in leaf_pile_boxes:
                    x_c, y_c, w, h = xyxy_to_xywhn(box, width, height)
                    label_lines.append(
                        f"2 {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}"
                    )

                # class 3: leaf_region
                for box in leaf_region_boxes:
                    x_c, y_c, w, h = xyxy_to_xywhn(box, width, height)
                    label_lines.append(
                        f"3 {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}"
                    )

                if label_lines:
                    label_out_path = sam3_labels_dir / f"{file_stem}.txt"
                    with open(label_out_path, "w") as f:
                        f.write("\n".join(label_lines))

                    annotated = draw_annotated_image(
                        frame_bgr,
                        final_waste_dets,
                        leaf_region_boxes,
                        leaf_pile_boxes,
                    )
                    anno_out_path = sam3_annotated_dir / f"{file_stem}.jpg"
                    cv2.imwrite(str(anno_out_path), annotated)

                    sam_saved_counter += 1

            # --- Save false_positives (YOLO potential FPs) ---
            if fp_indices:
                img_out_path_fp = fp_images_dir / f"{file_stem}.jpg"
                cv2.imwrite(str(img_out_path_fp), frame_bgr)

                fp_label_lines: List[str] = []
                for j in fp_indices:
                    box = yolo_boxes_relevant[j]
                    cls_id_raw = yolo_cls_relevant[j]
                    x_c, y_c, w, h = xyxy_to_xywhn(box, width, height)

                    # Map YOLO's class IDs to 0/1 convention
                    if cls_id_raw in args.yolo_cigarette_class_ids:
                        cls_id = 1
                    else:
                        cls_id = 0

                    fp_label_lines.append(
                        f"{cls_id} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}"
                    )

                if fp_label_lines:
                    label_out_path_fp = fp_labels_dir / f"{file_stem}.txt"
                    with open(label_out_path_fp, "w") as f:
                        f.write("\n".join(fp_label_lines))

                    annotated_fp = draw_yolo_annotated_image(
                        frame_bgr,
                        [yolo_boxes_relevant[j] for j in fp_indices],
                        [yolo_cls_relevant[j] for j in fp_indices],
                        [yolo_conf_relevant[j] for j in fp_indices],
                        args.yolo_waste_class_ids,
                        args.yolo_cigarette_class_ids,
                    )
                    anno_out_path_fp = fp_annotated_dir / f"{file_stem}.jpg"
                    cv2.imwrite(str(anno_out_path_fp), annotated_fp)

                    fp_saved_counter += 1

            frame_idx += 1

    cap.release()
    print(f"[sam3-video] {video_path.name}: saved {sam_saved_counter} sam3_gated frames")
    print(f"[sam3-video] {video_path.name}: saved {fp_saved_counter} false_positives frames")
    return sam_saved_counter, fp_saved_counter


def main() -> None:
    args = parse_args()

    if args.leaf_score_threshold is None:
        args.leaf_score_threshold = args.score_threshold
    if args.cigarette_score_threshold is None:
        args.cigarette_score_threshold = max(0.0, args.score_threshold * 0.5)

    device = resolve_device(args.device)

    # --- Prepare output dirs (shared across all videos) ---
    output_dir = args.output_dir

    sam3_root = output_dir / "sam3_gated"
    fp_root = output_dir / "false_positives"

    sam3_images_dir = sam3_root / "images"
    sam3_labels_dir = sam3_root / "labels"
    sam3_annotated_dir = sam3_root / "annotated"

    fp_images_dir = fp_root / "images"
    fp_labels_dir = fp_root / "labels"
    fp_annotated_dir = fp_root / "annotated"

    for d in (
        sam3_images_dir,
        sam3_labels_dir,
        sam3_annotated_dir,
        fp_images_dir,
        fp_labels_dir,
        fp_annotated_dir,
    ):
        d.mkdir(parents=True, exist_ok=True)

    # --- Build SAM3 model (shared) ---
    bpe_path = ensure_bpe_vocab()
    sam3_model = build_sam3_image_model(bpe_path=str(bpe_path))
    sam3_model.to(device)
    sam3_model.eval()
    processor = Sam3Processor(sam3_model)

    # --- YOLO model (shared) ---
    yolo_model = YOLO(str(args.yolo_weights))

    # --- Collect videos to process ---
    if args.video is not None:
        video_paths = [args.video]
    else:
        video_dir: Path = args.video_dir
        if not video_dir.is_dir():
            raise RuntimeError(f"--video-dir {video_dir} is not a directory")
        video_paths = sorted(
            p for p in video_dir.iterdir()
            if p.suffix.lower() == ".mkv"
        )
        if not video_paths:
            raise RuntimeError(f"No .mkv files found in {video_dir}")

    total_sam_saved = 0
    total_fp_saved = 0
    for vp in video_paths:
        sam_saved, fp_saved = process_video(
            video_path=vp,
            processor=processor,
            yolo_model=yolo_model,
            args=args,
            sam3_images_dir=sam3_images_dir,
            sam3_labels_dir=sam3_labels_dir,
            sam3_annotated_dir=sam3_annotated_dir,
            fp_images_dir=fp_images_dir,
            fp_labels_dir=fp_labels_dir,
            fp_annotated_dir=fp_annotated_dir,
        )
        total_sam_saved += sam_saved
        total_fp_saved += fp_saved

    print(f"\n[sam3-video] All done. Total sam3_gated frames saved: {total_sam_saved}")
    print(f"[sam3-video] Total false_positives frames saved: {total_fp_saved}")
    print(f"[sam3-video] Output root: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
