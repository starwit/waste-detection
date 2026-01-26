#!/usr/bin/env python3
"""
Prelabel material subclasses for existing YOLO bboxes in a raw-data folder using a VLM (Qwen3-VL).

Input folder structure (example):
  raw_data/train/cw33/
    data.yaml            # optional; used to infer "waste" class id by name
    images/...
    labels/...
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import torch
import yaml
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont, ImageOps
from tqdm import tqdm

IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}

TARGET_CLASSES = ["paper", "plastic", "bottle", "can", "other"]
CACHE_VERSION = 1

# Defaults (keep CLI minimal)
MODEL_ID_BY_SIZE = {
    "4b": "Qwen/Qwen3-VL-4B-Instruct",
    "8b": "Qwen/Qwen3-VL-8B-Instruct",
}
MODEL_SIZE_DEFAULT = "4b"
TORCH_DTYPE_DEFAULT = "auto"
DEVICE_MAP_DEFAULT = "auto"
ATTN_IMPL_DEFAULT: str | None = None

MIN_CROP_DEFAULT = 160
CLOSE_PAD_FRAC_DEFAULT = 0.15
PAD_FRAC_DEFAULT = 0.45
CONTEXT_PAD_FRAC_DEFAULT = 1.20
DARKEN_OUTSIDE_DEFAULT = 0.60
BLUR_RADIUS_DEFAULT = 6.0
BORDER_PX_DEFAULT = 6
INCLUDE_FULL_VIEW_DEFAULT = True
VIEW_MAX_PIXELS_DEFAULT = 262144
VIEW_MAX_PIXELS_8B_DEFAULT = 131072
FULL_VIEW_MAX_PIXELS_DEFAULT = 262144
FULL_VIEW_MAX_PIXELS_8B_DEFAULT = 131072

TIE_BREAK_RUNS_DEFAULT = 1
MAX_NEW_TOKENS_DEFAULT = 64
MAX_NEW_TOKENS_8B_DEFAULT = 32
MAX_ATTEMPTS_DEFAULT = 2
CONFIDENCE_THRESHOLD_DEFAULT = 0.60

QUALITY_PRESETS = ["safe", "max"]

CLASS_DEFINITIONS = {
    "paper": (
        "Paper/cardboard/tissue/receipts; paperboard. Includes beverage cartons / tetrapak-like cartons."
    ),
    "plastic": (
        "Any plastic packaging: plastic film/bags/wrappers AND rigid plastics (cups, tubs, lids/caps, trays). "
        "NOT bottles. NOT metal cans."
    ),
    "bottle": (
        "Bottle-shaped container (plastic or glass): neck + cylindrical body silhouette; includes fragments/neck."
    ),
    "can": (
        "Metal can (aluminum/steel). Cylindrical; metallic sheen; pull-tab top or sealed lid; may be crushed."
    ),
    "other": (
        "Everything else: cigarette packs, glass shards, wood/stone/food, mixed/unclear items, or not visible."
    ),
}

LABEL_JSON_RE = re.compile(r'"label"\s*:\s*"([^"]+)"', re.IGNORECASE)


@dataclass(frozen=True)
class BBox:
    cls: int
    xc: float
    yc: float
    w: float
    h: float


@dataclass
class ClassificationResult:
    final_label: str
    votes: list[str] = field(default_factory=list)
    confidence: float = 0.0
    margin: float = 0.0
    tie_break_used: bool = False
    raw_outputs: dict[str, str] = field(default_factory=dict)


@dataclass
class Stats:
    model_id: str = ""
    source_class_ids: list[int] = field(default_factory=list)
    target_classes: list[str] = field(default_factory=lambda: list(TARGET_CLASSES))

    total_images: int = 0
    total_reclassified: int = 0
    skipped_cached: int = 0
    tie_breaks: int = 0
    uncertain_count: int = 0
    errors: int = 0
    gated_small: int = 0

    counts: dict[str, int] = field(default_factory=lambda: {c: 0 for c in TARGET_CLASSES})

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "source_class_ids": self.source_class_ids,
            "target_classes": self.target_classes,
            "total_images": self.total_images,
            "total_reclassified": self.total_reclassified,
            "skipped_cached": self.skipped_cached,
            "tie_breaks": self.tie_breaks,
            "uncertain_count": self.uncertain_count,
            "errors": self.errors,
            "gated_small": self.gated_small,
            "counts": self.counts,
        }


def list_images(images_dir: Path) -> list[Path]:
    imgs = [p for p in images_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTENSIONS]
    imgs.sort()
    return imgs


def read_yolo_labels(label_path: Path) -> list[BBox]:
    if not label_path.exists():
        return []
    txt = label_path.read_text(encoding="utf-8").strip()
    if not txt:
        return []
    out: list[BBox] = []
    for line in txt.splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cls = int(float(parts[0]))
        xc, yc, w, h = map(float, parts[1:5])
        out.append(BBox(cls=cls, xc=xc, yc=yc, w=w, h=h))
    return out


def write_yolo_labels(label_path: Path, bboxes: list[BBox]) -> None:
    label_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{b.cls} {b.xc:.6f} {b.yc:.6f} {b.w:.6f} {b.h:.6f}" for b in bboxes]
    label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def yolo_to_xyxy(b: BBox, W: int, H: int) -> tuple[int, int, int, int]:
    x1 = int(round((b.xc - b.w / 2.0) * W))
    y1 = int(round((b.yc - b.h / 2.0) * H))
    x2 = int(round((b.xc + b.w / 2.0) * W))
    y2 = int(round((b.yc + b.h / 2.0) * H))
    x1 = int(clamp(x1, 0, W - 1))
    y1 = int(clamp(y1, 0, H - 1))
    x2 = int(clamp(x2, 0, W))
    y2 = int(clamp(y2, 0, H))
    if x2 <= x1:
        x2 = min(W, x1 + 1)
    if y2 <= y1:
        y2 = min(H, y1 + 1)
    return x1, y1, x2, y2


def bbox_norm_size(b: BBox) -> tuple[float, float, float]:
    bw = float(b.w)
    bh = float(b.h)
    return bw, bh, bw * bh


def enforce_min_crop(cx1: int, cy1: int, cx2: int, cy2: int, W: int, H: int, min_size: int) -> tuple[int, int, int, int]:
    bw = cx2 - cx1
    bh = cy2 - cy1
    if bw >= min_size and bh >= min_size:
        return cx1, cy1, cx2, cy2

    mx = (cx1 + cx2) / 2.0
    my = (cy1 + cy2) / 2.0
    half = min_size / 2.0

    nx1 = int(round(mx - half))
    ny1 = int(round(my - half))
    nx2 = int(round(mx + half))
    ny2 = int(round(my + half))

    nx1 = int(clamp(nx1, 0, W - 1))
    ny1 = int(clamp(ny1, 0, H - 1))
    nx2 = int(clamp(nx2, 0, W))
    ny2 = int(clamp(ny2, 0, H))

    if nx2 <= nx1:
        nx2 = min(W, nx1 + 1)
    if ny2 <= ny1:
        ny2 = min(H, ny1 + 1)

    return nx1, ny1, nx2, ny2


def make_focus_view(
    img: Image.Image,
    xyxy: tuple[int, int, int, int],
    pad_frac: float,
    min_crop: int,
    darken_outside: float,
    blur_radius: float,
    border_px: int,
) -> Image.Image:
    """
    Crop around bbox with padding while forcing attention:
    - outside bbox is darkened + blurred
    - inside bbox stays crisp
    - a red border is drawn around the bbox
    """
    W, H = img.size
    x1, y1, x2, y2 = xyxy
    bw = x2 - x1
    bh = y2 - y1

    pad_x = int(round(bw * pad_frac))
    pad_y = int(round(bh * pad_frac))

    cx1 = max(0, x1 - pad_x)
    cy1 = max(0, y1 - pad_y)
    cx2 = min(W, x2 + pad_x)
    cy2 = min(H, y2 + pad_y)

    cx1, cy1, cx2, cy2 = enforce_min_crop(cx1, cy1, cx2, cy2, W, H, min_crop)

    crop = img.crop((cx1, cy1, cx2, cy2)).convert("RGBA")
    cw, ch = crop.size

    rx1 = x1 - cx1
    ry1 = y1 - cy1
    rx2 = x2 - cx1
    ry2 = y2 - cy1

    outside = Image.new("L", (cw, ch), 255)
    mdraw = ImageDraw.Draw(outside)
    mdraw.rectangle([rx1, ry1, rx2, ry2], fill=0)

    dark = ImageEnhance.Brightness(crop).enhance(1.0 - float(darken_outside))
    dark_blur = dark.filter(ImageFilter.GaussianBlur(radius=float(blur_radius)))
    out = Image.composite(dark_blur, crop, outside)

    draw = ImageDraw.Draw(out)
    for i in range(int(border_px)):
        draw.rectangle([rx1 - i, ry1 - i, rx2 + i, ry2 + i], outline=(255, 0, 0, 255))

    return out.convert("RGB")


def bbox_key(rel_image_path: Path, bbox: BBox) -> str:
    s = f"{rel_image_path.as_posix()}|{bbox.cls}|{bbox.xc:.6f}|{bbox.yc:.6f}|{bbox.w:.6f}|{bbox.h:.6f}"
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def compute_confidence(votes: list[str]) -> tuple[float, float, str]:
    counts = Counter(votes)
    total = len(votes)
    if total == 0:
        return 0.0, 0.0, "other"
    sorted_counts = counts.most_common()
    winner, win_count = sorted_counts[0]
    conf = win_count / total
    if len(sorted_counts) > 1:
        margin = (win_count - sorted_counts[1][1]) / total
    else:
        margin = 1.0
    return conf, margin, winner


def majority_needed(n_votes: int) -> int:
    return (n_votes // 2) + 1


def build_prompt(view_kind: str) -> str:
    focus = (
        "The target object is inside the red bounding box.\n"
        "You MAY use the surrounding scene (outside the box) as context, but classify ONLY the object in the box.\n"
        if view_kind in {"context", "full"}
        else "The image is a crop around ONE object. Classify the object.\n"
    )
    defs_lines = "\n".join([f"- {c}: {CLASS_DEFINITIONS[c]}" for c in TARGET_CLASSES])
    return f"""You are labeling a single litter item.

{focus}
Choose EXACTLY ONE label from this closed set:
[{", ".join(TARGET_CLASSES)}]

Label definitions:
{defs_lines}

Rules:
- Output MUST be valid JSON with exactly one key: "label"
- The value MUST be one of: {TARGET_CLASSES}
- If it's a plastic or glass bottle: choose "bottle" (NOT "plastic").
- If it's a metal can: choose "can" (NOT "plastic").
- If it's a beverage carton / tetrapak: choose "paper".
- Paper vs plastic hint:
  - paper: matte/fibrous texture, torn edges, crumpled tissue/napkin, printed paper/receipts.
  - plastic: glossy/specular highlights, smooth uniform surface, translucent film, rigid packaging pieces.
- Avoid choosing "other" by default. If it looks like paper vs plastic, pick the best match.
- Choose "other" ONLY if the object is clearly none of paper/plastic/bottle/can, or it is truly not visible.

Return JSON now."""


def normalize_label(raw: str) -> Optional[str]:
    if not raw:
        return None
    s = raw.strip().lower()
    s = re.sub(r"[^a-z_]+", " ", s).strip().replace(" ", "_")

    # Direct hits
    if s in TARGET_CLASSES:
        return s

    # Map old taxonomy / common synonyms into the 5-class set
    if s in {"plastic_film", "rigid_plastic", "plastic_wrapper", "wrapper", "bag", "film"}:
        return "plastic"
    if s in {"beverage_carton", "carton", "tetrapak"}:
        return "paper"
    if s in {"cigarette", "cigarette_butt", "butt"}:
        return "other"
    if s in {"none", "unknown", "uncertain"}:
        return "other"

    # Keyword mapping (robust to messy outputs)
    if any(k in s for k in ["bottle", "flasche", "neck"]):
        return "bottle"
    if any(k in s for k in ["can", "aluminum", "aluminium", "tin", "dose"]):
        return "can"
    if any(k in s for k in ["paper", "cardboard", "receipt", "napkin", "tissue", "newspaper", "carton", "tetra"]):
        return "paper"
    if any(k in s for k in ["plastic", "wrapper", "film", "bag", "cup", "lid", "cap", "tray", "tub", "packaging"]):
        return "plastic"
    if any(k in s for k in ["glass", "wood", "stone", "food", "cig", "butt", "other", "unknown"]):
        return "other"
    return None


def extract_label(text: str) -> Optional[str]:
    if not text:
        return None
    m = LABEL_JSON_RE.search(text)
    if m:
        return normalize_label(m.group(1))

    low = text.lower()
    for c in TARGET_CLASSES:
        if c in low:
            return c
    return normalize_label(text)


def read_data_yaml_names(data_yaml_path: Path) -> dict[int, str]:
    if not data_yaml_path.exists():
        return {}
    try:
        data = yaml.safe_load(data_yaml_path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}
    names = data.get("names")
    if isinstance(names, dict):
        out: dict[int, str] = {}
        for k, v in names.items():
            try:
                out[int(k)] = str(v)
            except Exception:
                continue
        return out
    if isinstance(names, list):
        return {i: str(v) for i, v in enumerate(names)}
    return {}


def infer_source_class_ids(raw_data_dir: Path, fallback: list[int]) -> list[int]:
    names = read_data_yaml_names(raw_data_dir / "data.yaml")
    for i, name in names.items():
        if str(name).strip().lower() == "waste":
            return [i]
    return fallback


def build_output_class_map(raw_data_dir: Path, source_ids: set[int]) -> tuple[list[str], dict[int, int], dict[int, str]]:
    """
    Build an output class list + mapping so non-source classes (e.g. cigarette/leaf)
    remain distinct and do not collide with the 0..4 waste-subclass ids.

    Output class ids:
      0..4: TARGET_CLASSES (subclassed waste)
      5.. : original non-source classes in ascending original id order
    """
    names_by_id = read_data_yaml_names(raw_data_dir / "data.yaml")
    reserved = {c.lower() for c in TARGET_CLASSES}

    def normalize_name(name: str) -> str:
        s = name.strip().lower()
        s = re.sub(r"\s+", "_", s)
        s = re.sub(r"[^a-z0-9_]+", "", s)
        return s or "class"

    # Use stable order for non-source classes.
    non_source_ids = sorted([i for i in names_by_id.keys() if i not in source_ids])
    non_source_names: list[str] = []
    for i in non_source_ids:
        nm_raw = str(names_by_id.get(i, f"class_{i}"))
        nm = normalize_name(nm_raw)
        if nm in reserved:
            nm = f"orig_{nm}"
        non_source_names.append(nm)

    out_names = list(TARGET_CLASSES) + non_source_names
    old_to_new: dict[int, int] = {}
    for idx, old_id in enumerate(non_source_ids):
        old_to_new[old_id] = len(TARGET_CLASSES) + idx

    return out_names, old_to_new, names_by_id


class Qwen3VLClassifier:
    def __init__(
        self,
        model_id: str,
        torch_dtype: str,
        device_map: str,
        attn_impl: str | None,
        min_pixels: int | None,
        max_pixels: int | None,
    ):
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        from qwen_vl_utils import process_vision_info

        self.process_vision_info = process_vision_info

        dtype_map: dict[str, Any] = {
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
            "fp32": torch.float32,
            "auto": "auto",
        }
        if torch_dtype not in dtype_map:
            raise ValueError("--torch-dtype must be one of: auto, bf16, fp16, fp32")
        td = dtype_map[torch_dtype]

        proc_kwargs: dict[str, Any] = {}
        if min_pixels is not None:
            proc_kwargs["min_pixels"] = int(min_pixels)
        if max_pixels is not None:
            proc_kwargs["max_pixels"] = int(max_pixels)

        self.processor = AutoProcessor.from_pretrained(model_id, **proc_kwargs)

        model_kwargs: dict[str, Any] = {"device_map": device_map}
        if td != "auto":
            model_kwargs["torch_dtype"] = td
        if attn_impl:
            model_kwargs["attn_implementation"] = attn_impl

        self.model = Qwen3VLForConditionalGeneration.from_pretrained(model_id, **model_kwargs)
        self.model.eval()

    @torch.inference_mode()
    def generate(self, pil_image: Image.Image, prompt: str, max_new_tokens: int) -> str:
        messages = [{
            "role": "user",
            "content": [{"type": "image", "image": pil_image}, {"type": "text", "text": prompt}],
        }]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = self.process_vision_info(messages)

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)

        generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        out = self.processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        return out.strip()

    def classify_json(
        self,
        pil_image: Image.Image,
        prompt: str,
        max_attempts: int,
        max_new_tokens: int,
    ) -> tuple[str, str]:
        strict = prompt
        raw = ""
        for _ in range(max_attempts):
            raw = self.generate(pil_image, strict, max_new_tokens=max_new_tokens)
            lbl = extract_label(raw)
            if lbl in TARGET_CLASSES:
                return lbl, raw
            strict = prompt + f"\n\nREMINDER: Output ONLY JSON like {{\"label\":\"{TARGET_CLASSES[0]}\"}}."
        return "other", raw


def _load_font(size: int) -> ImageFont.ImageFont:
    for p in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ]:
        try:
            return ImageFont.truetype(p, size)
        except Exception:
            continue
    return ImageFont.load_default()


def make_full_view(img: Image.Image, xyxy: tuple[int, int, int, int], border_px: int) -> Image.Image:
    """Full image with a red bbox highlight (no darken/blur)."""
    out = img.copy().convert("RGBA")
    draw = ImageDraw.Draw(out)
    x1, y1, x2, y2 = xyxy
    for i in range(int(border_px)):
        draw.rectangle([x1 - i, y1 - i, x2 + i, y2 + i], outline=(255, 0, 0, 255))
    return out.convert("RGB")


def resize_to_max_pixels(img: Image.Image, max_pixels: int) -> Image.Image:
    if max_pixels <= 0:
        return img
    w, h = img.size
    cur = w * h
    if cur <= max_pixels:
        return img
    scale = (max_pixels / float(cur)) ** 0.5
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    return img.resize((nw, nh), Image.Resampling.LANCZOS)


def make_views(img: Image.Image, bbox: BBox, args: argparse.Namespace) -> dict[str, Image.Image]:
    W, H = img.size
    xyxy = yolo_to_xyxy(bbox, W, H)
    # Make close/padded/context meaningfully different, especially for tiny bboxes.
    close_min = int(getattr(args, "min_crop", MIN_CROP_DEFAULT))
    padded_min = max(close_min, int(round(close_min * 1.6)))
    context_min = max(close_min, int(round(close_min * 3.0)))
    close_pad = float(getattr(args, "close_pad_frac", CLOSE_PAD_FRAC_DEFAULT))
    pad = float(getattr(args, "pad_frac", PAD_FRAC_DEFAULT))
    context_pad = float(getattr(args, "context_pad_frac", CONTEXT_PAD_FRAC_DEFAULT))
    darken = float(getattr(args, "darken_outside", DARKEN_OUTSIDE_DEFAULT))
    blur = float(getattr(args, "blur_radius", BLUR_RADIUS_DEFAULT))
    border = int(getattr(args, "border_px", BORDER_PX_DEFAULT))
    views: dict[str, Image.Image] = {
        "close": make_focus_view(
            img, xyxy,
            pad_frac=close_pad,
            min_crop=close_min,
            darken_outside=darken,
            blur_radius=blur,
            border_px=border,
        ),
        "padded": make_focus_view(
            img, xyxy,
            pad_frac=pad,
            min_crop=padded_min,
            darken_outside=darken,
            blur_radius=blur,
            border_px=border,
        ),
        "context": make_focus_view(
            img, xyxy,
            pad_frac=context_pad,
            min_crop=context_min,
            # For context, do NOT destroy the surroundings: keep outside unblurred/undarkened.
            darken_outside=0.0,
            blur_radius=0.0,
            border_px=border,
        ),
    }
    if bool(getattr(args, "include_full_view", INCLUDE_FULL_VIEW_DEFAULT)):
        views["full"] = make_full_view(img, xyxy, border_px=border)
    # Bound memory by resizing views before sending to the VLM.
    # For "max" quality, we keep crops unscaled but still cap the FULL view (context-only) to avoid OOM.
    crop_cap = int(getattr(args, "view_max_pixels", VIEW_MAX_PIXELS_DEFAULT) or 0)
    full_cap = int(getattr(args, "full_view_max_pixels", FULL_VIEW_MAX_PIXELS_DEFAULT) or 0)
    if crop_cap or full_cap:
        for k in list(views.keys()):
            cap = full_cap if k == "full" else crop_cap
            if cap:
                views[k] = resize_to_max_pixels(views[k], cap)
    return views


def save_viz_montage(
    out_path: Path,
    views: dict[str, Image.Image],
    result: ClassificationResult,
    rel_image_path: Path,
    bbox_index: int,
    bbox: BBox,
) -> None:
    size = 330
    gutter = 12
    bar_h = 110
    order = ["close", "padded", "context"] + (["full"] if "full" in views else [])

    def _thumb(im: Image.Image) -> Image.Image:
        t = ImageOps.contain(im, (size, size), method=Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (size, size), "white")
        x = (size - t.size[0]) // 2
        y = (size - t.size[1]) // 2
        canvas.paste(t, (x, y))
        return canvas

    imgs = [(k, _thumb(views[k])) for k in order]
    cols = 2 if len(imgs) == 4 else len(imgs)
    rows = 2 if len(imgs) == 4 else 1
    grid_w = cols * size + (cols - 1) * gutter
    grid_h = rows * size + (rows - 1) * gutter
    W = grid_w
    H = grid_h + bar_h

    canvas = Image.new("RGB", (W, H), "white")

    draw = ImageDraw.Draw(canvas)
    font_title = _load_font(18)
    font_small = _load_font(14)

    for idx, (name, im) in enumerate(imgs):
        r = idx // cols
        c = idx % cols
        x = c * (size + gutter)
        y0 = r * (size + gutter)
        canvas.paste(im, (x, y0))
        draw.text(
            (x + 10, y0 + 10),
            name.upper(),
            font=font_small,
            fill="white",
            stroke_width=2,
            stroke_fill="black",
        )

    label = result.final_label
    votes = ", ".join(result.votes)
    meta1 = f"{rel_image_path.as_posix()}  bbox#{bbox_index}"
    meta2 = f"pred={label}  conf={result.confidence:.1%}  margin={result.margin:.1%}  tie_break={result.tie_break_used}"
    meta3 = f"votes=[{votes}]"
    area = float(bbox.w) * float(bbox.h)
    min_side = min(float(bbox.w), float(bbox.h))
    meta4 = f"bbox_norm: src_cls={bbox.cls}  xc={bbox.xc:.3f}  yc={bbox.yc:.3f}  w={bbox.w:.3f}  h={bbox.h:.3f}  area={area:.3%}  min_side={min_side:.3%}"

    y = grid_h + 8
    draw.text((8, y), meta1, font=font_title, fill="black")
    y += 24
    draw.text((8, y), meta2, font=font_small, fill="black")
    y += 20
    draw.text((8, y), meta3, font=font_small, fill="black")
    y += 20
    draw.text((8, y), meta4, font=font_small, fill="gray")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, quality=95)


def classify_bbox(
    clf: Qwen3VLClassifier,
    img: Image.Image,
    bbox: BBox,
    args: argparse.Namespace,
) -> tuple[ClassificationResult, dict[str, Image.Image]]:
    views = make_views(img, bbox, args)

    raw_outputs: dict[str, str] = {}
    votes: list[str] = []

    prompt_crop = build_prompt("crop")
    prompt_context = build_prompt("context")
    prompt_full = build_prompt("full")

    vote_plan: list[tuple[str, str]] = [("close", "crop"), ("padded", "crop"), ("context", "context")]
    if bool(getattr(args, "include_full_view", INCLUDE_FULL_VIEW_DEFAULT)):
        vote_plan.append(("full", "full"))

    for tag, view_kind in vote_plan:
        lbl, raw = clf.classify_json(
            views[tag],
            prompt_full if view_kind == "full" else (prompt_context if view_kind == "context" else prompt_crop),
            max_attempts=int(getattr(args, "max_attempts", MAX_ATTEMPTS_DEFAULT)),
            max_new_tokens=int(getattr(args, "max_new_tokens", MAX_NEW_TOKENS_DEFAULT)),
        )
        votes.append(lbl)
        if bool(getattr(args, "save_raw", False)):
            raw_outputs[tag] = raw

    conf, margin, winner = compute_confidence(votes)
    tie_break_used = False
    needed = majority_needed(len(votes))
    win_count = Counter(votes).most_common(1)[0][1] if votes else 0

    # If no majority -> do tie-break votes (use context view; full view is useful as one vote, but is
    # often too zoomed-out/noisy to repeat).
    tie_break_runs = int(getattr(args, "tie_break_runs", TIE_BREAK_RUNS_DEFAULT))
    if win_count < needed and tie_break_runs > 0:
        tie_break_used = True
        tie_tag = "context"
        tie_prompt = prompt_context
        for i in range(tie_break_runs):
            lbl, raw = clf.classify_json(
                views[tie_tag],
                tie_prompt,
                max_attempts=int(getattr(args, "max_attempts", MAX_ATTEMPTS_DEFAULT)),
                max_new_tokens=int(getattr(args, "max_new_tokens", MAX_NEW_TOKENS_DEFAULT)),
            )
            votes.append(lbl)
            if bool(getattr(args, "save_raw", False)):
                raw_outputs[f"tiebreak_{i+1}"] = raw
        conf, margin, winner = compute_confidence(votes)
        needed = majority_needed(len(votes))
        win_count = Counter(votes).most_common(1)[0][1] if votes else 0

    # If we still don't have a strict majority after tie-break votes, keep the plurality
    # winner. The confidence/margin will reflect uncertainty, and `--viz uncertain` can
    # be used to focus review.

    return (
        ClassificationResult(
            final_label=winner,
            votes=votes,
            confidence=conf,
            margin=margin,
            tie_break_used=tie_break_used,
            raw_outputs=raw_outputs,
        ),
        views,
    )


def load_cache(path: Path) -> tuple[Optional[str], dict[str, dict]]:
    if not path.exists():
        return None, {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None, {}
    if isinstance(data, dict) and "meta" in data and "items" in data:
        meta = data.get("meta")
        ns = meta.get("namespace") if isinstance(meta, dict) else None
        items = data.get("items")
        return (ns, items) if isinstance(items, dict) else (ns, {})
    return None, data if isinstance(data, dict) else {}


def save_cache(path: Path, cache: dict[str, dict], namespace: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta": {"cache_version": CACHE_VERSION, "namespace": namespace},
        "items": cache,
    }
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prelabel 5-class material subclasses for YOLO bboxes using Qwen3-VL",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--raw-data", type=str, required=True, help="Raw-data dataset folder containing images/ and labels/")
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output folder (default: tools/preannotate_subclasses/output/<raw_data_name>/)",
    )
    parser.add_argument("--model", type=str, choices=sorted(MODEL_ID_BY_SIZE.keys()), default=MODEL_SIZE_DEFAULT, help="Qwen3-VL model size")
    parser.add_argument("--quality", type=str, choices=QUALITY_PRESETS, default="safe", help="Quality preset (max disables downscaling; may use more VRAM)")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of images (0 = all)")
    parser.add_argument("--offset", type=int, default=0, help="Skip the first N images")
    parser.add_argument("--image-list", type=str, default=None, help="Optional text file with relative image paths to process (one per line, relative to images/)")

    parser.add_argument("--min-box-side-norm", type=float, default=0.0, help="Skip waste boxes smaller than this (normalized min side, 0..1; 0 = disabled)")
    parser.add_argument("--drop-gated", action="store_true", help="Drop gated waste boxes from output labels (instead of labeling them as other)")

    parser.add_argument("--viz", type=str, choices=["none", "uncertain", "all"], default="all", help="Save visualization montages")
    parser.add_argument("--viz-max", type=int, default=0, help="Max number of viz images to save (0 = unlimited)")
    parser.add_argument("--dry-run", action="store_true", help="Scan inputs and exit without running the model")

    args = parser.parse_args()

    model_id = MODEL_ID_BY_SIZE[str(args.model)]
    is_8b = str(args.model) == "8b"
    if str(args.quality) == "max":
        view_max_pixels = 0
        full_view_max_pixels = FULL_VIEW_MAX_PIXELS_8B_DEFAULT if is_8b else FULL_VIEW_MAX_PIXELS_DEFAULT
    else:
        view_max_pixels = VIEW_MAX_PIXELS_8B_DEFAULT if is_8b else VIEW_MAX_PIXELS_DEFAULT
        full_view_max_pixels = FULL_VIEW_MAX_PIXELS_8B_DEFAULT if is_8b else FULL_VIEW_MAX_PIXELS_DEFAULT
    max_new_tokens = MAX_NEW_TOKENS_8B_DEFAULT if is_8b else MAX_NEW_TOKENS_DEFAULT
    # Use fp16 on GPU to fit comfortably; this does not downscale image quality.
    torch_dtype = "fp16" if torch.cuda.is_available() else TORCH_DTYPE_DEFAULT
    device_map = "auto" if is_8b else ("cuda" if torch.cuda.is_available() else DEVICE_MAP_DEFAULT)

    raw_data_dir = Path(args.raw_data)
    images_dir = raw_data_dir / "images"
    labels_dir = raw_data_dir / "labels"
    if not images_dir.exists():
        raise SystemExit(f"Missing: {images_dir}")
    if not labels_dir.exists():
        raise SystemExit(f"Missing: {labels_dir}")

    tool_dir = Path(__file__).resolve().parent
    out_dir = Path(args.out) if args.out else (tool_dir / "output" / raw_data_dir.name)
    raw_root = raw_data_dir.resolve()
    out_root = out_dir.resolve()
    if out_root == raw_root or out_root.is_relative_to(raw_root):
        raise SystemExit(f"Refusing to write outputs inside --raw-data: {out_dir}\nChoose an --out folder outside {raw_data_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = out_dir / "subclass_cache.json"

    source_ids = set(infer_source_class_ids(raw_data_dir, fallback=[0]))
    if not source_ids:
        raise SystemExit("No source class ids configured.")

    out_class_names, non_source_old_to_new, data_yaml_names = build_output_class_map(raw_data_dir, source_ids)

    # Cache namespace: prompt/view/model settings should not reuse older cached labels.
    ns_payload = {
        "cache_version": CACHE_VERSION,
        "model_id": model_id,
        "target_classes": TARGET_CLASSES,
        "class_definitions": CLASS_DEFINITIONS,
        "source_class_ids": sorted(source_ids),
        "view": {
            "min_crop": MIN_CROP_DEFAULT,
            "close_pad_frac": CLOSE_PAD_FRAC_DEFAULT,
            "pad_frac": PAD_FRAC_DEFAULT,
            "context_pad_frac": CONTEXT_PAD_FRAC_DEFAULT,
            "darken_outside": DARKEN_OUTSIDE_DEFAULT,
            "blur_radius": BLUR_RADIUS_DEFAULT,
            "border_px": BORDER_PX_DEFAULT,
            "include_full_view": INCLUDE_FULL_VIEW_DEFAULT,
            "view_max_pixels": view_max_pixels,
            "full_view_max_pixels": full_view_max_pixels,
        },
        "gates": {
            "min_box_side_norm": float(args.min_box_side_norm),
            "drop_gated": bool(args.drop_gated),
        },
        "generation": {
            "max_new_tokens": max_new_tokens,
            "tie_break_runs": TIE_BREAK_RUNS_DEFAULT,
        },
    }
    cache_namespace = hashlib.sha1(json.dumps(ns_payload, sort_keys=True).encode("utf-8")).hexdigest()

    (out_dir / "classes.txt").write_text("\n".join(out_class_names) + "\n", encoding="utf-8")
    existing_ns, cache = load_cache(cache_path)
    if cache and existing_ns is None:
        print("Legacy cache file (no namespace). Ignoring to avoid stale results after prompt/view changes.")
        cache = {}
    elif existing_ns is not None and existing_ns != cache_namespace:
        print("Cache namespace mismatch (prompts/views/model changed). Ignoring existing cache file.")
        cache = {}

    # Internal knobs (not exposed via CLI).
    internal = argparse.Namespace(
        min_crop=MIN_CROP_DEFAULT,
        close_pad_frac=CLOSE_PAD_FRAC_DEFAULT,
        pad_frac=PAD_FRAC_DEFAULT,
        context_pad_frac=CONTEXT_PAD_FRAC_DEFAULT,
        darken_outside=DARKEN_OUTSIDE_DEFAULT,
        blur_radius=BLUR_RADIUS_DEFAULT,
        border_px=BORDER_PX_DEFAULT,
        include_full_view=INCLUDE_FULL_VIEW_DEFAULT,
        view_max_pixels=view_max_pixels,
        full_view_max_pixels=full_view_max_pixels,
        tie_break_runs=TIE_BREAK_RUNS_DEFAULT,
        max_new_tokens=max_new_tokens,
        max_attempts=MAX_ATTEMPTS_DEFAULT,
        save_raw=False,
    )

    stats = Stats(
        model_id=model_id,
        source_class_ids=sorted(source_ids),
        target_classes=list(TARGET_CLASSES),
    )

    torch.manual_seed(0)
    random.seed(0)

    images = list_images(images_dir)
    if args.image_list:
        listed: list[Path] = []
        for line in Path(args.image_list).read_text(encoding="utf-8").splitlines():
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            p = (images_dir / s)
            if p.exists() and p.is_file():
                listed.append(p)
        images = listed
    if args.offset and args.offset > 0:
        images = images[args.offset:]
    if args.limit and args.limit > 0:
        images = images[:args.limit]
    stats.total_images = len(images)

    if args.dry_run:
        src_boxes = 0
        other_boxes = 0
        other_ids: set[int] = set()
        for img_path in images:
            rel = img_path.relative_to(images_dir)
            label_path = (labels_dir / rel).with_suffix(".txt")
            for b in read_yolo_labels(label_path):
                if b.cls in source_ids:
                    src_boxes += 1
                else:
                    other_boxes += 1
                    other_ids.add(b.cls)
        print(f"Images: {stats.total_images}")
        print(f"Source boxes (will classify): {src_boxes}")
        print(f"Non-source boxes: {other_boxes} (kept; class ids appended after the 5 subclasses)")
        if other_ids:
            preview = ", ".join([f"{i}:{data_yaml_names.get(i, f'class_{i}')}" for i in sorted(list(other_ids))[:20]])
            print(f"Non-source ids (preview): {preview}")
            print(f"Output classes: {len(out_class_names)} (0..4 are waste subclasses; others appended)")
        print(f"Output: {out_dir}")
        return

    clf: Optional[Qwen3VLClassifier] = None

    def get_clf() -> Qwen3VLClassifier:
        nonlocal clf
        if clf is None:
            clf = Qwen3VLClassifier(
                model_id=model_id,
                torch_dtype=torch_dtype,
                device_map=device_map,
                attn_impl=ATTN_IMPL_DEFAULT,
                min_pixels=None,
                max_pixels=None,
            )
        return clf

    uncertain_rows: list[dict[str, Any]] = []
    viz_dir = out_dir / "viz"
    viz_saved = 0

    for img_path in tqdm(images, desc="Preannotating"):
        rel = img_path.relative_to(images_dir)
        label_path = (labels_dir / rel).with_suffix(".txt")
        bboxes = read_yolo_labels(label_path)
        if not bboxes:
            continue
        # Always process images that have any labels; we only reclassify boxes whose class id is "waste".

        try:
            img = Image.open(img_path)
            img = ImageOps.exif_transpose(img).convert("RGB")
        except Exception as e:
            stats.errors += 1
            print(f"Error loading {img_path}: {e}")
            continue

        out_bboxes: list[BBox] = []

        for bbox_idx, bbox in enumerate(bboxes):
            if bbox.cls not in source_ids:
                mapped = non_source_old_to_new.get(bbox.cls)
                if mapped is None:
                    # Class id not in data.yaml names: append it on-the-fly deterministically.
                    mapped = len(out_class_names)
                    out_class_names.append(f"class_{bbox.cls}")
                    non_source_old_to_new[bbox.cls] = mapped
                    (out_dir / "classes.txt").write_text("\n".join(out_class_names) + "\n", encoding="utf-8")
                out_bboxes.append(BBox(cls=mapped, xc=bbox.xc, yc=bbox.yc, w=bbox.w, h=bbox.h))
                continue

            k = hashlib.sha1((cache_namespace + "|" + bbox_key(rel, bbox)).encode("utf-8")).hexdigest()
            if k in cache and "error" not in cache[k]:
                pred_label = cache[k].get("label", "other")
                conf = float(cache[k].get("confidence", 0.0))
                votes = cache[k].get("votes", [])
                margin = float(cache[k].get("margin", 0.0))
                tie_break_used = bool(cache[k].get("tie_break", False))
                stats.skipped_cached += 1
                views: Optional[dict[str, Image.Image]] = None
                gated_reason: Optional[str] = cache[k].get("gated")
            else:
                try:
                    nw, nh, narea = bbox_norm_size(bbox)
                    nmin_side = float(min(nw, nh))
                    gated_reason = None
                    raw_for_cache: Optional[dict[str, str]] = None

                    if args.min_box_side_norm and nmin_side < float(args.min_box_side_norm):
                        gated_reason = "small_box"

                    if gated_reason is not None:
                        pred_label = "other"
                        conf = 0.0
                        votes = []
                        margin = 0.0
                        tie_break_used = False
                        views = None
                    else:
                        res, views = classify_bbox(get_clf(), img, bbox, internal)
                        pred_label = res.final_label
                        conf = float(res.confidence)
                        votes = list(res.votes)
                        margin = float(res.margin)
                        tie_break_used = bool(res.tie_break_used)
                        raw_for_cache = res.raw_outputs

                    entry: dict[str, Any] = {
                        "label": pred_label,
                        "confidence": conf,
                        "margin": margin,
                        "votes": votes,
                        "tie_break": tie_break_used,
                    }
                    if gated_reason is not None:
                        entry["gated"] = gated_reason
                    if bool(getattr(args, "save_raw", False)) and raw_for_cache is not None:
                        entry["raw"] = raw_for_cache
                    cache[k] = entry
                    if tie_break_used:
                        stats.tie_breaks += 1
                except Exception as e:
                    stats.errors += 1
                    pred_label = "other"
                    conf = 0.0
                    votes = []
                    margin = 0.0
                    tie_break_used = False
                    views = None
                    gated_reason = None
                    cache[k] = {"label": pred_label, "confidence": conf, "error": str(e)}

                if len(cache) % 50 == 0:
                    save_cache(cache_path, cache, cache_namespace)

            if pred_label not in TARGET_CLASSES:
                pred_label = "other"

            if gated_reason == "small_box":
                stats.gated_small += 1

            if gated_reason is not None and args.drop_gated:
                # Keep stats + review rows, but don't write this box to output labels.
                continue
            out_bboxes.append(BBox(cls=TARGET_CLASSES.index(pred_label), xc=bbox.xc, yc=bbox.yc, w=bbox.w, h=bbox.h))

            stats.counts[pred_label] = stats.counts.get(pred_label, 0) + 1
            stats.total_reclassified += 1

            is_uncertain = conf < CONFIDENCE_THRESHOLD_DEFAULT
            if is_uncertain:
                stats.uncertain_count += 1
                uncertain_rows.append({
                    "image": str(rel),
                    "bbox_index": bbox_idx,
                    "bbox": f"{bbox.xc:.4f},{bbox.yc:.4f},{bbox.w:.4f},{bbox.h:.4f}",
                    "label": pred_label,
                    "confidence": conf,
                    "votes": ",".join(votes),
                    "from_cache": k in cache and "error" not in cache[k],
                    "gated": gated_reason or "",
                })
            if args.viz != "none":
                want_viz = (args.viz == "all") or (args.viz == "uncertain" and is_uncertain)
                if want_viz and (args.viz_max == 0 or viz_saved < args.viz_max):
                    if views is None:
                        # If we hit cache, we didn't compute views yet. Compute views only (no model call).
                        views = make_views(img, bbox, internal)
                        res_for_viz = ClassificationResult(
                            final_label=pred_label,
                            votes=votes,
                            confidence=conf,
                            margin=margin,
                            tie_break_used=tie_break_used,
                            raw_outputs={},
                        )
                    else:
                        res_for_viz = ClassificationResult(
                            final_label=pred_label,
                            votes=votes,
                            confidence=conf,
                            margin=margin,
                            tie_break_used=tie_break_used,
                            raw_outputs={},
                        )

                    viz_name = f"{rel.as_posix().replace('/', '__')}__bbox{bbox_idx:02d}__{pred_label}__{conf:.2f}.jpg"
                    viz_path = viz_dir / viz_name
                    save_viz_montage(viz_path, views, res_for_viz, rel, bbox_idx, bbox)
                    viz_saved += 1

        out_label_path = (out_dir / rel).with_suffix(".txt")
        write_yolo_labels(out_label_path, out_bboxes)

    save_cache(cache_path, cache, cache_namespace)

    (out_dir / "report.json").write_text(json.dumps(stats.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")

    if uncertain_rows:
        csv_path = out_dir / "review_uncertain.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["image", "bbox_index", "bbox", "label", "confidence", "votes", "from_cache", "gated"])
            writer.writeheader()
            writer.writerows(uncertain_rows)

    print("\nDone.")
    print(f"Output labels: {out_dir}")
    print(f"Cache: {cache_path}")
    print(f"Report: {out_dir / 'report.json'}")
    if uncertain_rows:
        print(f"Review CSV: {out_dir / 'review_uncertain.csv'}")
    if args.viz != "none":
        print(f"Viz dir: {viz_dir} (saved {viz_saved})")


if __name__ == "__main__":
    main()
