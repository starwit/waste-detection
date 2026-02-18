"""Adapter that wraps an RF-DETR model to expose *.val()* and *.predict()*
compatible with ``ultralytics.YOLO``, so that ``evaluate.py`` can treat
both model families identically.

Usage::

    from yolov8_training.utils.rfdetr_adapter import RFDETRModelAdapter

    adapter = RFDETRModelAdapter(trained_rfdetr_model, model_name="RF-DETR-medium",
                                  resolution=1280, class_names={0: "waste", 1: "leaves"})
    # Now usable everywhere a YOLO model is expected:
    results = adapter.predict(img_path, conf=0.25)
    metrics = adapter.val(data="path/to/dataset.yaml", imgsz=1280)
"""

from __future__ import annotations

import io
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

import cv2
import numpy as np
import yaml


# ---------------------------------------------------------------------------
# Lightweight containers that mimic Ultralytics return types
# ---------------------------------------------------------------------------


@dataclass
class _Boxes:
    """Mimics ``ultralytics.engine.results.Boxes`` for a single image."""

    xyxy: np.ndarray  # (N, 4) absolute pixel coords
    xywhn: np.ndarray  # (N, 4) normalised centre-x, centre-y, w, h
    conf: np.ndarray  # (N,)
    cls: np.ndarray  # (N,)

    def __len__(self) -> int:
        return len(self.cls)

    def __iter__(self):
        """Iterate over individual detection boxes (mirrors Ultralytics API)."""
        for i in range(len(self)):
            yield _SingleBox(
                xyxy=self.xyxy[i : i + 1],
                xywhn=self.xywhn[i : i + 1],
                conf=self.conf[i],
                cls=self.cls[i],
            )


@dataclass
class _SingleBox:
    """A single detection box – yielded when iterating over ``_Boxes``."""

    xyxy: np.ndarray
    xywhn: np.ndarray
    conf: float
    cls: float


@dataclass
class _PredictionResult:
    """Mimics a single element of the list returned by ``YOLO.predict()``."""

    boxes: _Boxes
    orig_img: np.ndarray  # BGR
    names: Dict[int, str]

    def plot(
        self,
        line_width: int = 2,
        font_scale: float = 0.5,
        font_thickness: int = 1,
    ) -> np.ndarray:
        """Draw detection boxes on the original image and return a BGR copy."""
        img = self.orig_img.copy()
        for i in range(len(self.boxes)):
            x1, y1, x2, y2 = self.boxes.xyxy[i].astype(int)
            cls_id = int(self.boxes.cls[i])
            conf = float(self.boxes.conf[i])
            label = self.names.get(cls_id, str(cls_id))
            text = f"{label} {conf:.2f}"
            color = _class_color(cls_id)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, line_width)
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
            cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw, y1), color, -1)
            cv2.putText(
                img,
                text,
                (x1, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),
                font_thickness,
            )
        return img


def _class_color(cls_id: int) -> tuple[int, int, int]:
    """Deterministic colour per class id (BGR)."""
    rng = np.random.RandomState(cls_id + 1)
    return tuple(int(c) for c in rng.randint(60, 255, size=3))


@dataclass
class _ValMetrics:
    """Mimics the object returned by ``YOLO.val()``."""

    results_dict: Dict[str, float]
    fitness: float
    speed: Dict[str, float]
    per_class: Dict[str, Dict[str, float]] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Main adapter
# ---------------------------------------------------------------------------


class RFDETRModelAdapter:
    """Makes an RF-DETR model look like ``ultralytics.YOLO`` for evaluation.

    After training, wrap the model:

    >>> adapter = RFDETRModelAdapter(model, ...)
    >>> evaluate_and_log_model_results(adapter, ...)       # works
    >>> generate_side_by_side_comparisons(adapter, ...)    # works
    """

    def __init__(
        self,
        rfdetr_model: Any,
        model_name: str = "RF-DETR",
        resolution: int = 560,
        class_names: Dict[int, str] | None = None,
        model_variant: str | None = None,
    ):
        self._model = rfdetr_model
        self.model_name = model_name
        self.model_backend = "rfdetr"
        self.model_variant = str(model_variant).lower() if model_variant else None
        self._resolution = resolution
        self._class_names = class_names or {}
        # The adapter also needs to look like it has a `model` attr (see train_model code).
        self.model = type("_Stub", (), {"yaml": {"model_name": model_name}})()
        self.trainer = None

    # ------------------------------------------------------------------
    # predict()  — compatible with ultralytics.YOLO.predict()
    # ------------------------------------------------------------------

    def predict(
        self,
        source: str | Path | np.ndarray,
        conf: float = 0.25,
        save: bool = False,
        verbose: bool = False,
        **kwargs: Any,
    ) -> list[_PredictionResult]:
        """Run inference on a single image.

        Returns a list with one ``_PredictionResult`` per image (always length 1).
        """
        if isinstance(source, (str, Path)):
            img = cv2.imread(str(source))
            if img is None:
                raise FileNotFoundError(f"Could not read image: {source}")
        else:
            img = source

        detections = self._model.predict(img, threshold=conf)
        return [self._wrap_detections(detections, img)]

    def _wrap_detections(
        self, detections: Any, img: np.ndarray
    ) -> _PredictionResult:
        """Convert ``supervision.Detections`` to our Ultralytics-compatible wrapper."""
        h, w = img.shape[:2]

        if detections is None or len(detections) == 0:
            empty = np.empty((0, 4), dtype=np.float32)
            return _PredictionResult(
                boxes=_Boxes(
                    xyxy=empty,
                    xywhn=empty,
                    conf=np.empty(0, dtype=np.float32),
                    cls=np.empty(0, dtype=np.float32),
                ),
                orig_img=img,
                names=self._class_names,
            )

        xyxy = np.asarray(detections.xyxy, dtype=np.float32)
        confs = np.asarray(detections.confidence, dtype=np.float32)
        cls_ids = np.asarray(detections.class_id, dtype=np.float32)

        # Compute normalised centre-xywh
        cx = ((xyxy[:, 0] + xyxy[:, 2]) / 2) / w
        cy = ((xyxy[:, 1] + xyxy[:, 3]) / 2) / h
        bw = (xyxy[:, 2] - xyxy[:, 0]) / w
        bh = (xyxy[:, 3] - xyxy[:, 1]) / h
        xywhn = np.stack([cx, cy, bw, bh], axis=1)

        return _PredictionResult(
            boxes=_Boxes(xyxy=xyxy, xywhn=xywhn, conf=confs, cls=cls_ids),
            orig_img=img,
            names=self._class_names,
        )

    # ------------------------------------------------------------------
    # val()  — compatible with ultralytics.YOLO.val()
    # ------------------------------------------------------------------

    def val(
        self,
        data: str | None = None,
        verbose: bool = False,
        save: bool = False,
        plots: bool = False,
        **kwargs: Any,
    ) -> _ValMetrics:
        """Evaluate the model on a YOLO-format dataset.

        Parameters match the keyword-style call used by ``validate_model()``:

            model.val(data=..., classes=[...], imgsz=1280, workers=0, ...)

        Metrics are computed via ``pycocotools`` to stay comparable with standard
        COCO evaluation.
        """
        if data is None:
            raise ValueError("data (path to dataset.yaml) is required for val()")

        ds_cfg = _load_dataset_yaml(data)
        dataset_path = Path(ds_cfg["path"])
        val_rel = ds_cfg.get("val", "val/images")
        images_dir = dataset_path / val_rel
        labels_dir = images_dir.parent / "labels"

        class_names = _parse_class_names(ds_cfg)
        num_classes = len(class_names)

        conf_threshold = kwargs.get("conf", 0.001)
        classes_filter: set[int] | None = None
        if kwargs.get("classes") is not None:
            classes_filter = set(int(c) for c in kwargs["classes"])

        # Collect ground-truth and predictions in COCO format
        gt_images: list[dict] = []
        gt_annotations: list[dict] = []
        dt_results: list[dict] = []

        ann_id = 1
        image_id = 0
        total_inference_ms = 0.0

        image_files = sorted(
            p
            for p in images_dir.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
        ) if images_dir.exists() else []

        for img_path in image_files:
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            image_id += 1
            h, w = img.shape[:2]

            gt_images.append(
                {"id": image_id, "file_name": img_path.name, "width": w, "height": h}
            )

            # --- Ground truth from YOLO labels ---
            label_path = labels_dir / f"{img_path.stem}.txt"
            if label_path.exists():
                with open(label_path) as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) < 5:
                            continue
                        cls_id = int(parts[0])
                        if classes_filter and cls_id not in classes_filter:
                            continue
                        cx, cy, bw, bh = (
                            float(parts[1]),
                            float(parts[2]),
                            float(parts[3]),
                            float(parts[4]),
                        )
                        x = (cx - bw / 2) * w
                        y = (cy - bh / 2) * h
                        box_w = bw * w
                        box_h = bh * h
                        gt_annotations.append(
                            {
                                "id": ann_id,
                                "image_id": image_id,
                                "category_id": cls_id,
                                "bbox": [x, y, box_w, box_h],
                                "area": box_w * box_h,
                                "iscrowd": 0,
                            }
                        )
                        ann_id += 1

            # --- Inference ---
            t0 = time.time()
            detections = self._model.predict(img, threshold=conf_threshold)
            t1 = time.time()
            total_inference_ms += (t1 - t0) * 1000

            if detections is not None and len(detections) > 0:
                for i in range(len(detections)):
                    det_cls = int(detections.class_id[i])
                    if classes_filter and det_cls not in classes_filter:
                        continue
                    x1, y1, x2, y2 = detections.xyxy[i]
                    score = float(detections.confidence[i])
                    dt_results.append(
                        {
                            "image_id": image_id,
                            "category_id": det_cls,
                            "bbox": [
                                float(x1),
                                float(y1),
                                float(x2 - x1),
                                float(y2 - y1),
                            ],
                            "score": score,
                        }
                    )

        # --- Compute metrics via pycocotools ---
        categories = [{"id": i, "name": n} for i, n in class_names.items()]
        precision, recall, map50, map50_95, per_class = _compute_coco_metrics(
            gt_images, gt_annotations, dt_results, categories
        )

        # Ultralytics fitness formula: 0.1 * mAP50 + 0.9 * mAP50-95
        fitness = 0.1 * map50 + 0.9 * map50_95
        avg_inference = total_inference_ms / max(image_id, 1)

        return _ValMetrics(
            results_dict={
                "metrics/precision(B)": precision,
                "metrics/recall(B)": recall,
                "metrics/mAP50(B)": map50,
                "metrics/mAP50-95(B)": map50_95,
            },
            fitness=fitness,
            # RF-DETR timing includes the full predict() call (preprocess+inference+postprocess)
            # so we report it all under "inference" with the others zeroed out.
            speed={"preprocess": 0.0, "inference": avg_inference, "postprocess": 0.0},
            per_class=per_class,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_dataset_yaml(path: str | Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def _parse_class_names(ds_cfg: dict) -> Dict[int, str]:
    names = ds_cfg.get("names", {})
    if isinstance(names, list):
        return {i: n for i, n in enumerate(names)}
    if isinstance(names, dict):
        parsed: dict[int, str] = {}
        for raw_key, raw_name in names.items():
            try:
                cls_id = int(raw_key)
            except (TypeError, ValueError) as e:
                raise ValueError(
                    f"Invalid class id in dataset.yaml names mapping: {raw_key!r}. "
                    "Expected integer keys (e.g. 0, 1, 2) or numeric strings (e.g. '0', '1')."
                ) from e
            parsed[cls_id] = str(raw_name)
        # Ensure deterministic ordering by class id
        return {k: parsed[k] for k in sorted(parsed)}
    return {}


def _compute_coco_metrics(
    images: list[dict],
    annotations: list[dict],
    detections: list[dict],
    categories: list[dict],
) -> tuple[float, float, float, float, dict]:
    """Run COCO evaluation via pycocotools and return (precision, recall, mAP50, mAP50-95, per_class)."""
    if len(images) == 0:
        return 0.0, 0.0, 0.0, 0.0, {}
    return _coco_eval_pycocotools(images, annotations, detections, categories)


def _coco_eval_pycocotools(
    images: list[dict],
    annotations: list[dict],
    detections: list[dict],
    categories: list[dict],
) -> tuple[float, float, float, float, dict]:
    """COCO evaluation using pycocotools."""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    # Build ground-truth COCO object
    coco_gt = COCO()
    coco_gt.dataset = {
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }
    coco_gt.createIndex()

    if len(detections) == 0 or len(annotations) == 0:
        # No detections or no ground-truth → everything is zero
        return 0.0, 0.0, 0.0, 0.0, {}

    # Suppress pycocotools stdout spam
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    try:
        coco_dt = coco_gt.loadRes(detections)
        coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
    finally:
        sys.stdout = old_stdout

    # stats: [AP@.5:.95, AP@.5, AP@.75, APsmall, APmed, APlarge,
    #         AR@1, AR@10, AR@100, ARsmall, ARmed, ARlarge]
    map50_95 = float(coco_eval.stats[0])
    map50 = float(coco_eval.stats[1])

    # Extract precision / recall from the evaluation arrays.
    # precision shape: (T=10, R=101, K, A=4, M=3)
    # recall shape:    (T=10, K, A=4, M=3)
    # IoU=0.5 is index 0, area=all is index 0, maxDets=100 is index 2.
    prec_array = coco_eval.eval["precision"]  # (T, R, K, A, M)
    rec_array = coco_eval.eval["recall"]  # (T, K, A, M)

    # Mean precision across recall thresholds & classes at IoU=0.5
    p50 = prec_array[0, :, :, 0, 2]  # (101, K)
    valid_p = p50[p50 > -1]
    precision = float(np.mean(valid_p)) if valid_p.size > 0 else 0.0

    # Mean recall across classes at IoU=0.5
    r50 = rec_array[0, :, 0, 2]  # (K,)
    valid_r = r50[r50 > -1]
    recall = float(np.mean(valid_r)) if valid_r.size > 0 else 0.0

    # --- Per-class metrics ---
    cat_id_to_name = {cat["id"]: cat["name"] for cat in categories}
    cat_ids = coco_eval.params.catIds  # category ids used in evaluation
    per_class: dict[str, dict[str, float]] = {}

    for k_idx, cat_id in enumerate(cat_ids):
        cls_name = cat_id_to_name.get(cat_id, f"class_{cat_id}")

        # Per-class AP50: mean precision at IoU=0.5 across 101 recall thresholds
        p50_cls = prec_array[0, :, k_idx, 0, 2]  # (101,)
        valid_p50_cls = p50_cls[p50_cls > -1]
        cls_ap50 = float(np.mean(valid_p50_cls)) if valid_p50_cls.size > 0 else 0.0

        # Per-class mAP50-95: mean AP across all 10 IoU thresholds
        ap_per_iou = []
        for t_idx in range(prec_array.shape[0]):
            p_t = prec_array[t_idx, :, k_idx, 0, 2]
            valid_p_t = p_t[p_t > -1]
            ap_per_iou.append(float(np.mean(valid_p_t)) if valid_p_t.size > 0 else 0.0)
        cls_map = float(np.mean(ap_per_iou)) if ap_per_iou else 0.0

        # Per-class recall at IoU=0.5
        cls_recall_val = float(rec_array[0, k_idx, 0, 2]) if rec_array[0, k_idx, 0, 2] > -1 else 0.0

        # Precision at best-F1 operating point on the P-R curve at IoU=0.5
        recall_thresholds = np.linspace(0, 1, 101)
        best_f1, best_p, best_r = 0.0, 0.0, 0.0
        for r_idx, r_thr in enumerate(recall_thresholds):
            p_val = p50_cls[r_idx]
            if p_val < 0:
                continue
            f1_val = 2 * p_val * r_thr / (p_val + r_thr) if (p_val + r_thr) > 0 else 0.0
            if f1_val > best_f1:
                best_f1, best_p, best_r = f1_val, float(p_val), float(r_thr)

        per_class[cls_name] = {
            "precision": best_p,
            "recall": best_r if best_r > 0 else cls_recall_val,
            "map50": cls_ap50,
            "map": cls_map,
            "f1_score": best_f1,
        }

    return precision, recall, map50, map50_95, per_class
