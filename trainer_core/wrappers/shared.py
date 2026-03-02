from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml


@dataclass
class SingleBox:
    xyxy: np.ndarray
    xywhn: np.ndarray
    conf: float
    cls: float


@dataclass
class Boxes:
    xyxy: np.ndarray
    xywhn: np.ndarray
    conf: np.ndarray
    cls: np.ndarray

    def __len__(self) -> int:
        return len(self.cls)

    def __iter__(self):
        for i in range(len(self)):
            yield SingleBox(
                xyxy=self.xyxy[i : i + 1],
                xywhn=self.xywhn[i : i + 1],
                conf=float(self.conf[i]),
                cls=float(self.cls[i]),
            )


def _class_color(cls_id: int) -> tuple[int, int, int]:
    rng = np.random.RandomState(cls_id + 1)
    return tuple(int(c) for c in rng.randint(60, 255, size=3))


@dataclass
class PredictionResult:
    boxes: Boxes
    orig_img: np.ndarray
    names: dict[int, str]

    def plot(
        self,
        line_width: int = 2,
        font_scale: float = 0.5,
        font_thickness: int = 1,
    ) -> np.ndarray:
        img = self.orig_img.copy()
        for i in range(len(self.boxes)):
            x1, y1, x2, y2 = self.boxes.xyxy[i].astype(int)
            cls_id = int(self.boxes.cls[i])
            conf = float(self.boxes.conf[i])
            label = self.names.get(cls_id, str(cls_id))
            text = f"{label} {conf:.2f}"
            color = _class_color(cls_id)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, line_width)
            (tw, th), _ = cv2.getTextSize(
                text,
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                font_thickness,
            )
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


@dataclass
class ValMetrics:
    results_dict: dict[str, float]
    fitness: float
    speed: dict[str, float]
    per_class: dict[str, dict[str, float]] = field(default_factory=dict)


@dataclass
class CocoEvalResults:
    precision: float
    recall: float
    map50: float
    map50_95: float
    per_class: dict[str, dict[str, float]]
    macro_f1: float = 0.0

    def __iter__(self):
        yield self.precision
        yield self.recall
        yield self.map50
        yield self.map50_95
        yield self.per_class


def load_image(source: str | Path | np.ndarray) -> np.ndarray:
    if isinstance(source, (str, Path)):
        img = cv2.imread(str(source))
        if img is None:
            raise FileNotFoundError(f"Could not read image: {source}")
        return img
    return source


def build_prediction_result(
    *,
    xyxy: Any,
    scores: Any,
    labels: Any,
    img: np.ndarray,
    class_names: dict[int, str],
) -> PredictionResult:
    xyxy_arr = np.asarray(xyxy, dtype=np.float32)
    if xyxy_arr.size == 0:
        xyxy_arr = np.empty((0, 4), dtype=np.float32)
    elif xyxy_arr.ndim == 1:
        xyxy_arr = xyxy_arr.reshape(-1, 4)

    scores_arr = np.asarray(scores, dtype=np.float32)
    labels_arr = np.asarray(labels, dtype=np.float32)

    if labels_arr.size == 0:
        empty = np.empty((0, 4), dtype=np.float32)
        return PredictionResult(
            boxes=Boxes(
                xyxy=empty,
                xywhn=empty,
                conf=np.empty(0, dtype=np.float32),
                cls=np.empty(0, dtype=np.float32),
            ),
            orig_img=img,
            names=class_names,
        )

    height, width = img.shape[:2]
    cx = ((xyxy_arr[:, 0] + xyxy_arr[:, 2]) / 2.0) / float(width)
    cy = ((xyxy_arr[:, 1] + xyxy_arr[:, 3]) / 2.0) / float(height)
    bw = (xyxy_arr[:, 2] - xyxy_arr[:, 0]) / float(width)
    bh = (xyxy_arr[:, 3] - xyxy_arr[:, 1]) / float(height)
    xywhn = np.stack([cx, cy, bw, bh], axis=1).astype(np.float32)

    return PredictionResult(
        boxes=Boxes(
            xyxy=xyxy_arr,
            xywhn=xywhn,
            conf=scores_arr,
            cls=labels_arr,
        ),
        orig_img=img,
        names=class_names,
    )


def load_dataset_yaml(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def parse_class_names(ds_cfg: dict) -> dict[int, str]:
    names = ds_cfg.get("names", {})
    if isinstance(names, list):
        return {i: str(name) for i, name in enumerate(names)}
    if isinstance(names, dict):
        parsed: dict[int, str] = {}
        for raw_key, raw_name in names.items():
            try:
                cls_id = int(raw_key)
            except (TypeError, ValueError) as e:
                raise ValueError(
                    f"Invalid class id in dataset.yaml names mapping: {raw_key!r}. "
                    "Expected integer keys (e.g. 0, 1) or numeric strings (e.g. '0', '1')."
                ) from e
            parsed[cls_id] = str(raw_name)
        return {k: parsed[k] for k in sorted(parsed)}
    return {}
