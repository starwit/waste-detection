from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import yaml


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


@dataclass
class PredDet:
    box: np.ndarray  # xyxy
    score: float
    prompt: str


@dataclass
class LeafRemoval:
    image_id: str
    waste_box: np.ndarray
    waste_score: float
    waste_prompt: str
    leaf_box: np.ndarray
    leaf_score: float
    leaf_prompt: str
    iou_waste_leaf: float


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
    union = np.maximum(area_pred + area_gt - inter_area, 1e-9)
    return inter_area / union


def load_dataset_paths(dataset_yaml: Path):
    with open(dataset_yaml, "r") as f:
        cfg = yaml.safe_load(f)

    base_path = Path(cfg.get("path", dataset_yaml.parent))
    if not base_path.is_absolute():
        base_path = (dataset_yaml.parent / base_path).resolve()

    # If you want to run this on the train split, change "val" -> "train" here:
    images_subdir = cfg.get("val", "val/images")
    images_dir = (base_path / images_subdir).resolve()
    labels_dir = images_dir.parent / "labels"

    if not images_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {images_dir}")
    if not labels_dir.exists():
        raise FileNotFoundError(f"Label directory not found: {labels_dir}")
    return images_dir, labels_dir


def load_ground_truth_boxes_per_image(
    labels_dir: Path, image_ids: List[str], images_dir: Path
) -> Dict[str, List[np.ndarray]]:
    gt: Dict[str, List[np.ndarray]] = {}
    for img_id in image_ids:
        label_path = labels_dir / f"{img_id}.txt"
        boxes: List[np.ndarray] = []
        if label_path.exists():
            img_path = None
            for ext in IMAGE_EXTENSIONS:
                cand = images_dir / f"{img_id}{ext}"
                if cand.exists():
                    img_path = cand
                    break

            if img_path is None:
                # Fallback: treat coords as already absolute
                with open(label_path, "r") as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) < 5:
                            continue
                        _, x_c, y_c, w, h = map(float, parts[:5])
                        x1 = x_c - w / 2
                        y1 = y_c - h / 2
                        x2 = x_c + w / 2
                        y2 = y_c + h / 2
                        boxes.append(np.array([x1, y1, x2, y2], dtype=float))
            else:
                from PIL import Image

                with Image.open(img_path) as img:
                    width, height = img.size

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
        gt[img_id] = boxes
    return gt


def parse_predictions(pred_path: Path) -> Dict[str, List[PredDet]]:
    with open(pred_path, "r") as f:
        data = json.load(f)

    per_image: Dict[str, List[PredDet]] = {}
    for item in data:
        img_id = item["image_id"]
        box_xyxy = np.array(item["bbox_xyxy"], dtype=float)
        score = float(item["score"])
        prompt = item.get("sam3_prompt", "unknown")
        per_image.setdefault(img_id, []).append(
            PredDet(box=box_xyxy, score=score, prompt=prompt)
        )
    return per_image


def parse_leaf_removals(leaf_path: Path) -> List[LeafRemoval]:
    with open(leaf_path, "r") as f:
        data = json.load(f)

    out: List[LeafRemoval] = []
    for item in data:
        out.append(
            LeafRemoval(
                image_id=item["image_id"],
                waste_box=np.array(item["waste_bbox_xyxy"], dtype=float),
                waste_score=float(item["waste_score"]),
                waste_prompt=item["waste_prompt"],
                leaf_box=np.array(item["leaf_bbox_xyxy"], dtype=float),
                leaf_score=float(item["leaf_score"]),
                leaf_prompt=item["leaf_prompt"],
                iou_waste_leaf=float(item.get("iou_waste_leaf", 0.0)),
            )
        )
    return out


def analyze_prompts(
    preds_by_image: Dict[str, List[PredDet]],
    gts_by_image: Dict[str, List[np.ndarray]],
    iou_thr: float,
):
    # Init structures
    prompt_stats: Dict[str, Dict[str, float]] = {}
    gt_prompts_by_image: Dict[str, List[set]] = {}

    # Initialize GT prompt sets per image
    for img_id, gts in gts_by_image.items():
        gt_prompts_by_image[img_id] = [set() for _ in gts]

    # Pass over predictions
    for img_id, preds in preds_by_image.items():
        gts = gts_by_image.get(img_id, [])
        gt_sets = gt_prompts_by_image.get(img_id, [])
        for pred in preds:
            p = pred.prompt
            stats = prompt_stats.setdefault(
                p,
                {
                    "num_dets": 0,
                    "num_tp_preds": 0,
                    "num_fp_preds": 0,
                },
            )
            stats["num_dets"] += 1

            if not gts:
                stats["num_fp_preds"] += 1
                continue

            ious = bbox_iou(pred.box, gts)
            if ious.size == 0:
                stats["num_fp_preds"] += 1
                continue

            best_idx = int(np.argmax(ious))
            best_iou = float(ious[best_idx])
            if best_iou >= iou_thr:
                stats["num_tp_preds"] += 1
                if gt_sets:
                    gt_sets[best_idx].add(p)
            else:
                stats["num_fp_preds"] += 1

    # Compute per-prompt coverage / unique coverage
    total_gts = sum(len(v) for v in gts_by_image.values())
    for p, stats in prompt_stats.items():
        covered = 0
        unique = 0
        for img_id, sets_list in gt_prompts_by_image.items():
            for s in sets_list:
                if p in s:
                    covered += 1
                    if len(s) == 1:
                        unique += 1
        stats["covered_gts"] = covered
        stats["unique_gts"] = unique
        stats["recall_coverage"] = covered / total_gts if total_gts > 0 else 0.0
        stats["unique_recall"] = unique / total_gts if total_gts > 0 else 0.0
        tp = stats["num_tp_preds"]
        fp = stats["num_fp_preds"]
        stats["precision"] = tp / (tp + fp) if (tp + fp) > 0 else 0.0

    return prompt_stats, total_gts


def analyze_leaf_prompts(
    leaf_removals: List[LeafRemoval],
    gts_by_image: Dict[str, List[np.ndarray]],
    iou_thr: float,
):
    """
    For each LEAF prompt, measure how often it removes:
      - a false-positive waste detection (good)
      - a true-positive waste detection (bad)

    Classification is based on IoU of the *waste* box vs GT.
    """
    stats: Dict[str, Dict[str, float]] = {}

    for r in leaf_removals:
        p = r.leaf_prompt
        s = stats.setdefault(
            p,
            {
                "num_waste_removed": 0,
                "num_removed_tp": 0,  # waste that overlapped GT >= iou_thr
                "num_removed_fp": 0,  # waste that did not overlap GT
            },
        )
        s["num_waste_removed"] += 1

        gts = gts_by_image.get(r.image_id, [])
        if not gts:
            # No GT boxes in this image: treat removed waste as FP
            s["num_removed_fp"] += 1
            continue

        ious = bbox_iou(r.waste_box, gts)
        best_iou = float(ious.max()) if ious.size else 0.0

        if best_iou >= iou_thr:
            s["num_removed_tp"] += 1
        else:
            s["num_removed_fp"] += 1

    for p, s in stats.items():
        n = s["num_waste_removed"]
        tp = s["num_removed_tp"]
        fp = s["num_removed_fp"]
        s["fp_fraction"] = fp / n if n > 0 else 0.0
        s["tp_fraction"] = tp / n if n > 0 else 0.0

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Analyze which SAM3 prompts are useful vs redundant, "
        "and how leaf prompts clean false positives."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Path to dataset.yaml (same as for sam3_eval).",
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        required=True,
        help="Path to predictions.json saved by sam3_eval.py.",
    )
    parser.add_argument(
        "--leaf-removals",
        type=Path,
        default=None,
        help="Optional path to leaf_removals.json saved by sam3_eval.py "
        "to analyze leaf prompts as FP cleaners.",
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.5,
        help="IoU threshold to consider a prediction a TP (default: 0.5).",
    )
    args = parser.parse_args()

    preds_by_image = parse_predictions(args.predictions)
    image_ids = sorted(preds_by_image.keys())

    images_dir, labels_dir = load_dataset_paths(args.dataset)
    gts_by_image = load_ground_truth_boxes_per_image(labels_dir, image_ids, images_dir)

    # ---- Original prompt (waste prompt) analysis ----
    stats, total_gts = analyze_prompts(
        preds_by_image, gts_by_image, iou_thr=args.iou_threshold
    )

    print(f"Total GT boxes: {total_gts}")
    print(
        "Prompt summary (sorted by unique GTs desc):\n"
        "unique_gts\tcovered_gts\tprec\tuniq_rec\trec_cov\tnum_dets\tprompt"
    )

    for prompt, s in sorted(
        stats.items(), key=lambda kv: kv[1]["unique_gts"], reverse=True
    ):
        print(
            f"{int(s['unique_gts'])}\t"
            f"{int(s['covered_gts'])}\t"
            f"{s['precision']:.3f}\t"
            f"{s['unique_recall']:.3f}\t"
            f"{s['recall_coverage']:.3f}\t"
            f"{int(s['num_dets'])}\t"
            f"{prompt}"
        )

    # ---- NEW: leaf prompt cleaning analysis ----
    if args.leaf_removals is not None:
        leaf_removals = parse_leaf_removals(args.leaf_removals)
        leaf_stats = analyze_leaf_prompts(
            leaf_removals, gts_by_image, iou_thr=args.iou_threshold
        )

        print(
            "\nLeaf prompt cleaning summary (per leaf prompt):\n"
            "n_removed\tremoved_fp\tremoved_tp\tfp_frac\ttp_frac\tleaf_prompt"
        )
        # Sort by "goodness": more FP removed, and higher FP fraction
        for prompt, s in sorted(
            leaf_stats.items(),
            key=lambda kv: (kv[1]["num_removed_fp"], kv[1]["fp_fraction"]),
            reverse=True,
        ):
            print(
                f"{int(s['num_waste_removed'])}\t"
                f"{int(s['num_removed_fp'])}\t"
                f"{int(s['num_removed_tp'])}\t"
                f"{s['fp_fraction']:.3f}\t"
                f"{s['tp_fraction']:.3f}\t"
                f"{prompt}"
            )
    else:
        print(
            "\n(No --leaf-removals provided, skipping leaf prompt cleaning analysis.)"
        )


if __name__ == "__main__":
    main()
