from __future__ import annotations

import io
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from trainer_core.wrappers.shared import (
    CocoEvalResults as _CocoEvalResults,
    PredictionResult as _PredictionResult,
    ValMetrics as _ValMetrics,
    build_prediction_result,
    load_dataset_yaml as _load_dataset_yaml,
    load_image,
    parse_class_names as _parse_class_names,
)


def _default_infer_fn(model: Any, image: np.ndarray):
    from mmdet.apis import inference_detector

    return inference_detector(model, image)


def _extract_prediction_arrays(pred: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if isinstance(pred, dict):
        bboxes = np.asarray(pred.get("bboxes", []), dtype=np.float32)
        scores = np.asarray(pred.get("scores", []), dtype=np.float32)
        labels = np.asarray(pred.get("labels", []), dtype=np.int64)
        return bboxes, scores, labels

    pred_instances = getattr(pred, "pred_instances", pred)
    bboxes = getattr(pred_instances, "bboxes", None)
    scores = getattr(pred_instances, "scores", None)
    labels = getattr(pred_instances, "labels", None)

    def _to_numpy(v, dtype):
        if v is None:
            return np.empty((0,), dtype=dtype)
        if hasattr(v, "detach"):
            v = v.detach()
        if hasattr(v, "cpu"):
            v = v.cpu()
        if hasattr(v, "numpy"):
            v = v.numpy()
        return np.asarray(v, dtype=dtype)

    boxes = _to_numpy(bboxes, np.float32)
    if boxes.size == 0:
        boxes = np.empty((0, 4), dtype=np.float32)
    elif boxes.ndim == 1:
        boxes = boxes.reshape(-1, 4)
    return boxes, _to_numpy(scores, np.float32), _to_numpy(labels, np.int64)


def _mean_valid(values: np.ndarray) -> float:
    valid = values[values > -1]
    if valid.size == 0:
        return 0.0
    return float(np.mean(valid))


def _coco_eval_pycocotools(
    images: list[dict],
    annotations: list[dict],
    detections: list[dict],
    categories: list[dict],
) -> _CocoEvalResults:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    coco_gt = COCO()
    coco_gt.dataset = {
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }
    coco_gt.createIndex()

    if len(detections) == 0 or len(annotations) == 0:
        return _CocoEvalResults(0.0, 0.0, 0.0, 0.0, {}, macro_f1=0.0)

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

    precision_tensor = coco_eval.eval["precision"]
    recall_tensor = coco_eval.eval["recall"]
    iou_50_idx = 0
    area_idx = 0
    max_det_idx = 2

    precision = _mean_valid(precision_tensor[iou_50_idx, :, :, area_idx, max_det_idx])
    recall = _mean_valid(recall_tensor[iou_50_idx, :, area_idx, max_det_idx])
    map50_95 = float(coco_eval.stats[0])
    map50 = float(coco_eval.stats[1])
    macro_f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    per_class: dict[str, dict[str, float]] = {}
    for class_idx, category in enumerate(categories):
        cls_name = str(category.get("name", f"class_{class_idx}"))
        cls_precision_50 = _mean_valid(
            precision_tensor[iou_50_idx, :, class_idx, area_idx, max_det_idx]
        )
        cls_precision = _mean_valid(precision_tensor[:, :, class_idx, area_idx, max_det_idx])
        cls_recall = _mean_valid(recall_tensor[:, class_idx, area_idx, max_det_idx])
        cls_f1 = (
            2 * cls_precision * cls_recall / (cls_precision + cls_recall)
            if (cls_precision + cls_recall) > 0
            else 0.0
        )
        per_class[cls_name] = {
            "precision": cls_precision,
            "recall": cls_recall,
            "map50": cls_precision_50,
            "map": cls_precision,
            "f1_score": cls_f1,
        }

    return _CocoEvalResults(precision, recall, map50, map50_95, per_class, macro_f1=macro_f1)


def _compute_coco_metrics(
    images: list[dict],
    annotations: list[dict],
    detections: list[dict],
    categories: list[dict],
) -> _CocoEvalResults:
    if len(images) == 0:
        return _CocoEvalResults(0.0, 0.0, 0.0, 0.0, {}, macro_f1=0.0)
    return _coco_eval_pycocotools(images, annotations, detections, categories)


class RTMDetModelAdapter:
    def __init__(
        self,
        model: Any,
        model_name: str = "RTMDet",
        resolution: int = 640,
        class_names: dict[int, str] | None = None,
        model_variant: str | None = None,
        model_config_path: str | None = None,
        config_name: str | None = None,
        cache_dir: str | None = None,
        allow_download: bool | None = None,
        infer_fn=None,
    ):
        self._model = model
        self._infer_fn = infer_fn or _default_infer_fn
        self.model_name = model_name
        self.model_backend = "mmdet"
        self.model_variant = str(model_variant) if model_variant else None
        self.resolution = int(resolution)
        self._class_names = class_names or {}
        self.class_names = dict(self._class_names)
        self.model_config_path = model_config_path
        self.mmdet_config_name = config_name
        self.mmdet_cache_dir = cache_dir
        self.mmdet_allow_download = allow_download
        self.model = type("_Stub", (), {"yaml": {"model_name": model_name}})()
        self.trainer = None

    def predict(
        self,
        source: str | Path | np.ndarray,
        conf: float = 0.25,
        save: bool = False,
        verbose: bool = False,
        **kwargs: Any,
    ) -> list[_PredictionResult]:
        img = load_image(source)

        bboxes, scores, labels = _extract_prediction_arrays(self._infer_fn(self._model, img))
        keep = scores >= float(conf)
        if self._class_names:
            valid_ids = set(int(k) for k in self._class_names.keys())
            keep = keep & np.array([int(c) in valid_ids for c in labels], dtype=bool)

        bboxes = bboxes[keep]
        scores = scores[keep]
        labels = labels[keep]
        return [self._wrap_arrays(bboxes, scores, labels, img)]

    def _wrap_arrays(
        self,
        bboxes: np.ndarray,
        scores: np.ndarray,
        labels: np.ndarray,
        img: np.ndarray,
    ) -> _PredictionResult:
        return build_prediction_result(
            xyxy=bboxes,
            scores=scores,
            labels=labels,
            img=img,
            class_names=self._class_names,
        )

    def val(
        self,
        data: str | None = None,
        verbose: bool = False,
        save: bool = False,
        plots: bool = False,
        **kwargs: Any,
    ) -> _ValMetrics:
        if data is None:
            raise ValueError("data (path to dataset.yaml) is required for val()")

        imgsz = kwargs.get("imgsz")
        if imgsz is not None and int(imgsz) != int(self.resolution):
            print(
                f"Warning: RTMDet adapter ignores imgsz={int(imgsz)}; "
                f"using resolution={self.resolution}."
            )

        ds_cfg = _load_dataset_yaml(data)
        if "path" not in ds_cfg:
            raise ValueError(f"Dataset YAML at {data} is missing required 'path'.")
        dataset_path = Path(ds_cfg["path"])
        val_rel = ds_cfg.get("val", "val/images")
        images_dir = dataset_path / val_rel
        labels_dir = images_dir.parent / "labels"

        class_names = _parse_class_names(ds_cfg)
        valid_class_ids = set(int(k) for k in class_names.keys())

        conf_threshold = float(kwargs.get("conf", 0.001))
        classes_filter: set[int] | None = None
        if kwargs.get("classes") is not None:
            classes_filter = {int(c) for c in kwargs["classes"]}

        gt_images: list[dict] = []
        gt_annotations: list[dict] = []
        dt_results: list[dict] = []
        ann_id = 1
        image_id = 0
        total_inference_ms = 0.0

        image_files = sorted(
            p for p in images_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
        ) if images_dir.exists() else []

        for img_path in image_files:
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            image_id += 1
            height, width = img.shape[:2]
            gt_images.append(
                {"id": image_id, "file_name": img_path.name, "width": width, "height": height}
            )

            label_path = labels_dir / f"{img_path.stem}.txt"
            if label_path.exists():
                with open(label_path, "r", encoding="utf-8") as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) < 5:
                            continue
                        try:
                            cls_id = int(float(parts[0]))
                            cx = float(parts[1])
                            cy = float(parts[2])
                            bw = float(parts[3])
                            bh = float(parts[4])
                        except ValueError:
                            continue
                        if classes_filter and cls_id not in classes_filter:
                            continue
                        x = (cx - bw / 2.0) * width
                        y = (cy - bh / 2.0) * height
                        box_w = bw * width
                        box_h = bh * height
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

            t0 = time.time()
            bboxes, scores, labels = _extract_prediction_arrays(self._infer_fn(self._model, img))
            t1 = time.time()
            total_inference_ms += (t1 - t0) * 1000.0

            keep = scores >= conf_threshold
            if classes_filter is not None:
                keep = keep & np.array([int(c) in classes_filter for c in labels], dtype=bool)
            else:
                keep = keep & np.array([int(c) in valid_class_ids for c in labels], dtype=bool)

            for box, score, cls_id in zip(bboxes[keep], scores[keep], labels[keep]):
                x1, y1, x2, y2 = box.tolist()
                dt_results.append(
                    {
                        "image_id": image_id,
                        "category_id": int(cls_id),
                        "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                        "score": float(score),
                    }
                )

        categories = [{"id": cls_id, "name": name} for cls_id, name in class_names.items()]
        coco_metrics = _compute_coco_metrics(gt_images, gt_annotations, dt_results, categories)
        precision, recall, map50, map50_95, per_class = coco_metrics
        macro_f1 = float(getattr(coco_metrics, "macro_f1", 0.0))

        results_dict = {
            "metrics/precision(B)": float(precision),
            "metrics/recall(B)": float(recall),
            "metrics/mAP50(B)": float(map50),
            "metrics/mAP50-95(B)": float(map50_95),
            "metrics/f1(B)": float(macro_f1),
        }
        fitness = 0.1 * float(map50) + 0.9 * float(map50_95)
        avg_inference = total_inference_ms / max(image_id, 1)
        return _ValMetrics(
            results_dict=results_dict,
            fitness=fitness,
            speed={"preprocess": 0.0, "inference": avg_inference, "postprocess": 0.0},
            per_class=per_class,
        )
