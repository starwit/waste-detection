import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image
import imagehash
from tqdm import tqdm

import torch

from ultralytics import YOLO

# Linux X11/SSH support: ensure Qt-based backends use X11 (xcb) when DISPLAY is set.
if sys.platform.startswith("linux") and os.environ.get("DISPLAY"):
    os.environ.setdefault("QT_QPA_PLATFORM", "xcb")


# Optional GUI deps (available on local laptop). On headless epic gpu, these may be missing.
try:
    import tkinter as tk  # type: ignore
    from tkinter import filedialog, ttk, messagebox  # type: ignore
except Exception:
    tk = None  # type: ignore
    ttk = None  # type: ignore
    filedialog = None  # type: ignore
    messagebox = None  # type: ignore


VIDEO_EXTS = (".mp4", ".mov", ".avi", ".mkv")


@dataclass
class Selection:
    """User selections and runtime options from the GUI.

    - video_path: chosen video snippet
    - weights_path: YOLO `.pt` file path
    - output_dir: where saved frames and JSON sidecars go
    - skip_after_save: frames to jump ahead after saving
    """
    video_path: Path
    weights_path: Path
    output_dir: Path
    skip_after_save: int


def human_size(num_bytes: int) -> str:
    """Convert byte count to a human-friendly size label (e.g., '12.3 MB')."""
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(num_bytes)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def list_videos_sorted_by_size(folder: Path) -> List[Path]:
    """Return all video files in a folder sorted by size descending."""
    videos = [p for p in folder.iterdir() if p.suffix.lower() in VIDEO_EXTS and p.is_file()]
    videos.sort(key=lambda p: p.stat().st_size, reverse=True)
    return videos


def get_video_meta(path: Path) -> Tuple[Optional[int], Optional[float]]:
    """Return (total_frames, fps) probed via OpenCV; None if unavailable."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None, None
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return total if total > 0 else None, fps if fps and fps > 0 else None


class SelectorUI:
    """Minimal Tkinter UI to select inputs and options before playback."""
    def __init__(self, root, prefill_snippets_dir: Optional[str] = None, prefill_weights: Optional[str] = None, prefill_output_dir: Optional[str] = None):
        self.root = root
        self.root.title("Mine Frames from Video - Setup")
        self.root.geometry("900x560")

        self.snippets_dir_var = tk.StringVar(value=prefill_snippets_dir or "")
        self.weights_var = tk.StringVar(value=prefill_weights or "")
        self.output_dir_var = tk.StringVar(value=prefill_output_dir or "")
        self.skip_after_var = tk.IntVar(value=0)

        self._build()
        self.video_paths: List[Path] = []
        self.selected_video: Optional[Path] = None

        # If prefilled, auto-load videos
        if prefill_snippets_dir:
            try:
                self._load_videos(Path(prefill_snippets_dir))
            except Exception:
                pass

    def _build(self):
        pad = {"padx": 8, "pady": 6}
        frm = ttk.Frame(self.root)
        frm.pack(fill=tk.BOTH, expand=True)

        # Snippets directory chooser
        dir_row = ttk.Frame(frm)
        dir_row.pack(fill=tk.X, **pad)
        ttk.Label(dir_row, text="Snippets folder:").pack(side=tk.LEFT)
        ttk.Entry(dir_row, textvariable=self.snippets_dir_var).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=6)
        ttk.Button(dir_row, text="Browse", command=self._choose_snippets_dir).pack(side=tk.LEFT)

        # Video table
        table = ttk.Treeview(frm, columns=("name", "size", "duration"), show="headings", height=12)
        table.heading("name", text="Video")
        table.heading("size", text="Size")
        table.heading("duration", text="Duration")
        table.column("name", width=560)
        table.column("size", width=120, anchor=tk.E)
        table.column("duration", width=120, anchor=tk.CENTER)
        table.pack(fill=tk.BOTH, expand=True, **pad)
        self.video_table = table
        table.bind("<<TreeviewSelect>>", self._on_select_video)

        # Weights chooser
        w_row = ttk.Frame(frm)
        w_row.pack(fill=tk.X, **pad)
        ttk.Label(w_row, text="YOLO weights (.pt):").pack(side=tk.LEFT)
        ttk.Entry(w_row, textvariable=self.weights_var).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=6)
        ttk.Button(w_row, text="Browse", command=self._choose_weights).pack(side=tk.LEFT)

        # Output + options
        d_row = ttk.Frame(frm)
        d_row.pack(fill=tk.X, **pad)
        ttk.Label(d_row, text="Output folder:").pack(side=tk.LEFT)
        ttk.Entry(d_row, textvariable=self.output_dir_var).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=6)
        ttk.Button(d_row, text="Browse", command=self._choose_output_dir).pack(side=tk.LEFT)

        o_row = ttk.Frame(frm)
        o_row.pack(fill=tk.X, **pad)
        ttk.Label(o_row, text="Skip frames after save:").pack(side=tk.LEFT)
        ttk.Spinbox(o_row, from_=0, to=10000, textvariable=self.skip_after_var, width=8).pack(side=tk.LEFT, padx=(6, 16))

        # Start button
        btn_row = ttk.Frame(frm)
        btn_row.pack(fill=tk.X, **pad)
        ttk.Button(btn_row, text="Start", command=self._start).pack(side=tk.RIGHT)

        hint = ttk.Label(frm, text="Player shows overlays. Press H for help inside the player window.")
        hint.pack(anchor=tk.W, padx=12, pady=(0, 8))

    def _choose_snippets_dir(self):
        path = filedialog.askdirectory(title="Select snippets folder")
        if not path:
            return
        self.snippets_dir_var.set(path)
        self._load_videos(Path(path))

    def _load_videos(self, folder: Path):
        """Fill the table with file name, size and rough duration."""
        self.video_table.delete(*self.video_table.get_children())
        self.video_paths = list_videos_sorted_by_size(folder)
        for p in self.video_paths:
            size = human_size(p.stat().st_size)
            total, fps = get_video_meta(p)
            dur = None
            if total is not None and fps is not None and fps > 0:
                seconds = total / fps
                m = int(seconds // 60)
                s = int(seconds % 60)
                dur = f"{m:02d}:{s:02d}"
            self.video_table.insert("", tk.END, iid=str(p), values=(p.name, size, dur or "?"))

    def _on_select_video(self, _evt=None):
        sel = self.video_table.selection()
        if sel:
            self.selected_video = Path(sel[0])

    def _choose_weights(self):
        path = filedialog.askopenfilename(title="Select YOLO weights .pt", filetypes=[("PyTorch weights", "*.pt")])
        if path:
            self.weights_var.set(path)

    def _choose_output_dir(self):
        path = filedialog.askdirectory(title="Select output folder")
        if path:
            self.output_dir_var.set(path)

    def _start(self):
        """Validate inputs and close the dialog to continue to player."""
        if not self.selected_video:
            messagebox.showerror("Missing input", "Please select a video from the list.")
            return
        if not self.weights_var.get():
            messagebox.showerror("Missing weights", "Please select a YOLO .pt weights file.")
            return
        if not self.output_dir_var.get():
            messagebox.showerror("Missing output", "Please select an output folder.")
            return
        self.root.quit()

    def get_selection(self) -> Optional[Selection]:
        if not (self.selected_video and self.weights_var.get() and self.output_dir_var.get()):
            return None
        return Selection(
            video_path=self.selected_video,
            weights_path=Path(self.weights_var.get()),
            output_dir=Path(self.output_dir_var.get()),
            skip_after_save=int(self.skip_after_var.get()),
        )


class Deduper:
    """Perceptual-hash deduplication to skip near-identical saved frames."""
    def __init__(self, max_hamming: int = 4):
        self.max_hamming = max_hamming
        self._hashes: List[imagehash.ImageHash] = []

    def is_duplicate(self, img_bgr: np.ndarray) -> bool:
        """Return True if image matches a prior one within Hamming threshold."""
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        ph = imagehash.phash(Image.fromarray(img_rgb))
        for h in self._hashes:
            if h - ph <= self.max_hamming:
                return True
        self._hashes.append(ph)
        return False


class DiskCache:
    """Compact on-disk cache for per-frame detections.

    Files (stored next to the video):
    - <video_stem>__<weights_stem>.preds.npz
        boxes: float16, shape (M, 4), normalized xyxy in [0,1]
        confs: float16, shape (M,)
        clss:  uint16,  shape (M,)
        indptr: int64,  shape (total_frames + 1,) cumulative index per frame
    - <video_stem>__<weights_stem>.meta.json
        { version, video_basename, width, height, fps, total_frames,
          model_weights_basename, class_names }
    """

    def __init__(self, video_path: Path, weights_path: Path):
        """Derive cache file paths from video and weights basenames."""
        base = f"{video_path.stem}__{weights_path.stem}"
        self.npz_path = video_path.parent / f"{base}.preds.npz"
        self.meta_path = video_path.parent / f"{base}.meta.json"
        self._meta = None
        # In-memory arrays (Option A)
        self._boxes: Optional[np.ndarray] = None
        self._confs: Optional[np.ndarray] = None
        self._clss: Optional[np.ndarray] = None
        self._indptr: Optional[np.ndarray] = None

    def exists(self) -> bool:
        """True if both `.npz` and `.meta.json` exist."""
        return self.npz_path.exists() and self.meta_path.exists()

    def load(self):
        """Load compressed arrays + metadata ONCE into RAM; returns self."""
        # Read arrays one time to avoid per-frame unzip cost.
        with np.load(self.npz_path, allow_pickle=False) as z:
            self._boxes = z["boxes"]
            self._confs = z["confs"]
            self._clss = z["clss"]
            self._indptr = z["indptr"]
        with open(self.meta_path, "r") as f:
            self._meta = json.load(f)
        return self

    def save(self, width: int, height: int, fps: float, total: int, class_names: Dict[int, str], frames_preds: List[List[Dict]]):
        """Write a compact cache from per-frame detection dicts.

        Stores normalized float16 boxes and confidences with an indptr array
        to map frames → slice ranges, reducing file size while preserving info.
        """
        boxes_all: List[List[float]] = []
        confs_all: List[float] = []
        clss_all: List[int] = []
        indptr: List[int] = [0]

        W = float(max(1, width))
        H = float(max(1, height))

        for preds in frames_preds:
            if preds:
                for det in preds:
                    x1, y1, x2, y2 = det["xyxy"]
                    conf = float(det.get("confidence", 0.0))
                    cid = int(det.get("class_id", -1))
                    bx = [
                        max(0.0, min(1.0, x1 / W)),
                        max(0.0, min(1.0, y1 / H)),
                        max(0.0, min(1.0, x2 / W)),
                        max(0.0, min(1.0, y2 / H)),
                    ]
                    boxes_all.append(bx)
                    confs_all.append(conf)
                    clss_all.append(cid)
            indptr.append(len(confs_all))

        boxes_arr = np.asarray(boxes_all, dtype=np.float16) if boxes_all else np.zeros((0, 4), dtype=np.float16)
        confs_arr = np.asarray(confs_all, dtype=np.float16) if confs_all else np.zeros((0,), dtype=np.float16)
        clss_arr = np.asarray(clss_all, dtype=np.uint16) if clss_all else np.zeros((0,), dtype=np.uint16)
        indptr_arr = np.asarray(indptr, dtype=np.int64)

        np.savez_compressed(self.npz_path, boxes=boxes_arr, confs=confs_arr, clss=clss_arr, indptr=indptr_arr)

        base_parts = self.npz_path.stem.split("__")
        meta = {
            "version": 1,
            "video_basename": base_parts[0] if base_parts else self.npz_path.stem,
            "fps": float(fps),
            "total_frames": int(total),
            "width": int(width),
            "height": int(height),
            "model_weights_basename": base_parts[1] if len(base_parts) > 1 else "",
            "class_names": [class_names[i] for i in sorted(class_names.keys())] if isinstance(class_names, dict) else [],
        }
        with open(self.meta_path, "w") as f:
            json.dump(meta, f)

    def get_frame_dets(self, frame_idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (boxes, confs, clss) for the given frame index from in-memory arrays."""
        if self._boxes is None or self._indptr is None or self._confs is None or self._clss is None:
            raise RuntimeError("Cache not loaded")
        if frame_idx < 0 or frame_idx + 1 >= self._indptr.shape[0]:
            return self._boxes[:0], self._confs[:0], self._clss[:0]
        start = int(self._indptr[frame_idx])
        end = int(self._indptr[frame_idx + 1])
        return self._boxes[start:end], self._confs[start:end], self._clss[start:end]

    @property
    def meta(self):
        """Loaded metadata dictionary (or None if not loaded)."""
        return self._meta


class Player:
    """Video player that overlays YOLO results (live or cached) and saves frames."""
    def __init__(self, selection: Selection):
        if YOLO is None:
            raise RuntimeError("Ultralytics is not installed. Please install ultralytics.")

        self.video_path = selection.video_path
        self.output_dir = selection.output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.weights_path = selection.weights_path
        # Auto-select device: CUDA if available, else CPU
        self.device = "cuda:0" if (torch is not None and hasattr(torch, "cuda") and torch.cuda.is_available()) else "cpu"
        self.model = YOLO(str(self.weights_path))
        try:
            self.model.to(self.device)
        except Exception:
            pass

        self.class_names = self.model.names if hasattr(self.model, "names") else {}

        self.cap = cv2.VideoCapture(str(self.video_path))
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video: {self.video_path}")

        self.total = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 0.0) or 30.0
        # Cache width/height once to avoid per-frame queries
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.paused = False
        self.overlay = True
        self.playback_speed = 1.0
        self.conf = 0.25
        self.iou = 0.45
        self._seeking = False
        self._last_raw: Optional[np.ndarray] = None
        self._last_preds: List[Dict] = []
        self._deduper = Deduper(max_hamming=4)
        self.skip_after_save = int(selection.skip_after_save)
        self.cache: Optional[DiskCache] = None
        self.show_help = False
        self.current_tag: Optional[str] = None

        self.window = "Review Player"
        cv2.namedWindow(self.window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window, 1280, 720)
        # Trackbars
        cv2.createTrackbar("conf", self.window, int(self.conf * 100), 100, self._on_conf)
        cv2.createTrackbar("iou", self.window, int(self.iou * 100), 100, self._on_iou)
        if self.total > 0:
            cv2.createTrackbar("position", self.window, 0, self.total - 1, self._on_seek)

        cache_probe = DiskCache(self.video_path, self.weights_path)
        if cache_probe.exists():
            print("Found existing cache on disk; using it for playback …")
            self.cache = cache_probe.load()
            # Prefer class names from cache meta if present
            if self.cache.meta and isinstance(self.cache.meta.get("class_names", None), list):
                self.class_names = {i: n for i, n in enumerate(self.cache.meta["class_names"]) }

    def _ensure_cache(self):
        """Load existing cache or precompute once and save to disk."""
        cache = DiskCache(self.video_path, self.weights_path)
        if cache.exists():
            print("Loading cached detections from disk …")
            self.cache = cache.load()
            return
        print("Precomputing detections and saving cache to disk …")
        width = self.width
        height = self.height
        fps = self.fps
        total = self.total
        # Use very low conf to retain candidates during precompute
        orig_conf = self.conf
        orig_iou = self.iou
        self.conf = 0.001
        self.iou = 0.45
        frames_preds: List[List[Dict]] = []
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        with tqdm(total=total if total and total > 0 else None, unit="frame") as pbar:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break
                preds = self._infer(frame)
                frames_preds.append(preds)
                pbar.update(1)
        self.conf = orig_conf
        self.iou = orig_iou
        cache.save(width, height, fps, total, self.class_names if isinstance(self.class_names, dict) else {}, frames_preds)
        self.cache = cache.load()
        if self.cache.meta and isinstance(self.cache.meta.get("class_names", None), list):
            self.class_names = {i: n for i, n in enumerate(self.cache.meta["class_names"]) }

    def _on_conf(self, v: int):
        """Update confidence threshold via trackbar (0–1)."""
        self.conf = max(0.0, min(1.0, v / 100.0))

    def _on_iou(self, v: int):
        """Update IoU (NMS) threshold via trackbar (0–1)."""
        self.iou = max(0.0, min(1.0, v / 100.0))

    def _on_seek(self, pos: int):
        """Seek to an absolute frame index from the trackbar."""
        if self._seeking:
            return
        self._seeking = True
        try:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
        finally:
            self._seeking = False

    def _draw_overlays(self, frame: np.ndarray, preds: List[Dict]) -> np.ndarray:
        """Draw rectangles + labels onto a copy of the frame and return it."""
        out = frame.copy()
        for det in preds:
            x1, y1, x2, y2 = det["xyxy"]
            cls_id = det["class_id"]
            conf = det["confidence"]
            name = self.class_names.get(cls_id, str(cls_id)) if isinstance(self.class_names, dict) else str(cls_id)
            color = (0, 255, 0)
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            label = f"{name} {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(out, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
            cv2.putText(out, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
        return out

    def _infer(self, frame_bgr: np.ndarray) -> List[Dict]:
        """Run YOLO on a frame and return a list of detection dicts."""
        results = self.model.predict(source=frame_bgr, conf=self.conf, iou=self.iou, device=self.device, verbose=False)
        preds: List[Dict] = []
        if not results:
            return preds
        res = results[0]
        if res.boxes is None:
            return preds
        boxes = res.boxes
        xyxy = boxes.xyxy.cpu().numpy().astype(int)
        confs = boxes.conf.cpu().numpy().tolist() if boxes.conf is not None else []
        clss = boxes.cls.cpu().numpy().astype(int).tolist() if boxes.cls is not None else []
        for i, (x1, y1, x2, y2) in enumerate(xyxy):
            conf = float(confs[i]) if i < len(confs) else 0.0
            cid = int(clss[i]) if i < len(clss) else -1
            preds.append({
                "xyxy": [int(x1), int(y1), int(x2), int(y2)],
                "confidence": conf,
                "class_id": cid,
                "class_name": self.class_names.get(cid, str(cid)) if isinstance(self.class_names, dict) else str(cid),
            })
        return preds

    def _preds_from_cache(self, frame_idx: int) -> List[Dict]:
        """Reconstruct pixel-space detections for a frame from the disk cache."""
        if self.cache is None:
            return []
        boxes_n, confs_n, clss_n = self.cache.get_frame_dets(frame_idx)
        if boxes_n.shape[0] == 0:
            return []
        # Convert normalized to pixel coords using cached width/height
        w = self.width
        h = self.height
        preds: List[Dict] = []
        for i in range(boxes_n.shape[0]):
            if float(confs_n[i]) < self.conf:
                continue
            x1 = int(boxes_n[i, 0] * w)
            y1 = int(boxes_n[i, 1] * h)
            x2 = int(boxes_n[i, 2] * w)
            y2 = int(boxes_n[i, 3] * h)
            cid = int(clss_n[i])
            preds.append({
                "xyxy": [x1, y1, x2, y2],
                "confidence": float(confs_n[i]),
                "class_id": cid,
                "class_name": self.class_names.get(cid, str(cid)) if isinstance(self.class_names, dict) else str(cid),
            })
        return preds

    def _get_predictions(self, frame_idx: int, frame_bgr: np.ndarray) -> List[Dict]:
        """Return cached or live predictions based on cache availability."""
        if self.cache is not None:
            return self._preds_from_cache(frame_idx)
        return self._infer(frame_bgr)

    def _save_current(self, frame_bgr: np.ndarray, frame_idx: int):
        """Write the raw frame as JPEG and a JSON sidecar with predictions/tag."""
        # Dedup by perceptual hash
        if self._deduper.is_duplicate(frame_bgr):
            print("[dedup] Skipped near-duplicate frame")
            return

        ts_sec = frame_idx / self.fps if self.fps > 0 else 0.0
        base = f"{self.video_path.stem}_f{frame_idx:06d}"
        img_path = self.output_dir / f"{base}.jpg"
        json_path = self.output_dir / f"{base}.json"

        # Ensure unique filename if exists
        idx = 1
        while img_path.exists() or json_path.exists():
            base2 = f"{base}_{idx}"
            img_path = self.output_dir / f"{base2}.jpg"
            json_path = self.output_dir / f"{base2}.json"
            idx += 1

        # Save image (raw, full-res, no overlays)
        cv2.imwrite(str(img_path), frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])

        # Metadata sidecar
        meta = {
            "video_path": str(self.video_path.resolve()),
            "frame_index": frame_idx,
            "timestamp_sec": ts_sec,
            "model_weights": str(self.weights_path.resolve()),
            "predictions": self._last_preds,
        }
        if self.current_tag:
            meta["tag"] = self.current_tag
        with open(json_path, "w") as f:
            json.dump(meta, f, indent=2)
        print(f"Saved {img_path.name} and metadata")

        # Optional skip ahead to avoid near-identical scenes after save
        if self.skip_after_save > 0 and self.total > 0:
            target = min(self.total - 1, frame_idx + self.skip_after_save)
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, target)
            ret, frame = self.cap.read()
            if ret:
                new_idx = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
                self._last_raw = frame.copy()
                self._last_preds = self._get_predictions(new_idx, frame)
                self._update_pos_trackbar(new_idx)
                self.paused = True

    def _update_pos_trackbar(self, frame_idx: int):
        """Update the position trackbar safely without re-entrant callbacks."""
        if self.total <= 0:
            return
        try:
            self._seeking = True
            cv2.setTrackbarPos("position", self.window, max(0, min(self.total - 1, frame_idx)))
        finally:
            self._seeking = False

    def _draw_help(self, img: np.ndarray) -> np.ndarray:
        """Overlay a larger hotkey reference and current tag on the image."""
        lines = [
            "Controls:",
            "Space: pause/resume    O: overlay on/off",
            "S: save frame          H: toggle help",
            "F: tag false-positive  N: tag false-negative  U: uncertain  C: clear tag",
            "Left/Right or A/D: step when paused",
            "+/- or 1/2/3: speed    Seek: position trackbar",
            "Note: With cache, IoU changes won't re-run NMS",
        ]
        out = img.copy()
        # Larger font and spacing
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.2
        thickness_bg = 5
        thickness_fg = 3
        color_bg = (30, 30, 30)
        color_fg = (255, 255, 255)
        x, y = 30, 100
        # Show current tag mode at the top
        tag_txt = f"Current tag: {self.current_tag if self.current_tag else '-'}"
        cv2.putText(out, tag_txt, (x, y - 50), font, font_scale, (0, 128, 255), thickness_bg + 1, cv2.LINE_AA)
        cv2.putText(out, tag_txt, (x, y - 50), font, font_scale, (255, 255, 255), thickness_fg, cv2.LINE_AA)
        for ln in lines:
            cv2.putText(out, ln, (x, y), font, font_scale, color_bg, thickness_bg, cv2.LINE_AA)
            cv2.putText(out, ln, (x, y), font, font_scale, color_fg, thickness_fg, cv2.LINE_AA)
            y += int(48 * font_scale)
        return out

    def run(self):
        """Main loop: read frames, get predictions, render, and handle hotkeys. Includes timing debug prints."""
        import time
        frame_idx = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
        delay_ms = int(max(1, 1000 / max(1e-6, self.fps) / self.playback_speed))

        while True:
            t0 = time.time()
            if not self.paused:
                t_read0 = time.time()
                ret, frame = self.cap.read()
                t_read1 = time.time()
                if not ret:
                    break
                frame_idx = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
                self._last_raw = frame.copy()
                t_pred0 = time.time()
                preds = self._get_predictions(frame_idx, frame)
                t_pred1 = time.time()
                self._last_preds = preds
                t_overlay0 = time.time()
                disp = self._draw_overlays(frame, preds) if self.overlay else frame
                t_overlay1 = time.time()
                self._update_pos_trackbar(frame_idx)
            else:
                if self._last_raw is None:
                    t_read0 = time.time()
                    ret, frame = self.cap.read()
                    t_read1 = time.time()
                    if not ret:
                        break
                    frame_idx = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
                    self._last_raw = frame.copy()
                    t_pred0 = time.time()
                    self._last_preds = self._get_predictions(frame_idx, frame)
                    t_pred1 = time.time()
                else:
                    t_read0 = t_read1 = t_pred0 = t_pred1 = time.time()
                t_overlay0 = time.time()
                disp = self._draw_overlays(self._last_raw, self._last_preds) if self.overlay else self._last_raw
                t_overlay1 = time.time()

            # HUD
            t_hud0 = time.time()
            hud = disp.copy()
            cache_txt = "CACHE" if self.cache is not None else "LIVE"
            tag_txt = f"TAG: {self.current_tag}" if self.current_tag else "TAG: -"
            status = f"{Path(self.video_path).name}  f={frame_idx}/{self.total}  {cache_txt}  conf={self.conf:.2f}  iou={self.iou:.2f}  {('PAUSED' if self.paused else f'{self.playback_speed:.2f}x')}  {tag_txt}"
            cv2.putText(hud, status, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 50, 50), 3, cv2.LINE_AA)
            cv2.putText(hud, status, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(hud, "Press H for help", (10, hud.shape[0] - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(hud, "Press H for help", (10, hud.shape[0] - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            if self.show_help:
                hud = self._draw_help(hud)
            t_hud1 = time.time()
            t_disp0 = time.time()
            cv2.imshow(self.window, hud)
            t_disp1 = time.time()

            # Print timing debug info
            print(f"[DEBUG] frame={frame_idx} read={t_read1-t_read0:.4f}s pred={t_pred1-t_pred0:.4f}s overlay={t_overlay1-t_overlay0:.4f}s hud={t_hud1-t_hud0:.4f}s disp={t_disp1-t_disp0:.4f}s total={t_disp1-t0:.4f}s")

            key = cv2.waitKey(delay_ms if not self.paused else 30) & 0xFF
            if key == 27:  # ESC
                break
            elif key == ord(' '):
                self.paused = not self.paused
            elif key == ord('o'):
                self.overlay = not self.overlay
            elif key == ord('h'):
                self.show_help = not self.show_help
            elif key == ord('f'):
                self.current_tag = "false_positive"
            elif key == ord('n'):
                self.current_tag = "false_negative"
            elif key == ord('u'):
                self.current_tag = "uncertain"
            elif key == ord('c'):
                self.current_tag = None
            elif key == ord('s'):
                if self._last_raw is not None:
                    self._save_current(self._last_raw, frame_idx)
            elif key == ord('+') or key == ord('='):
                self.playback_speed = min(4.0, self.playback_speed + 0.25)
                delay_ms = int(max(1, 1000 / max(1e-6, self.fps) / self.playback_speed))
            elif key == ord('-') or key == ord('_'):
                self.playback_speed = max(0.25, self.playback_speed - 0.25)
                delay_ms = int(max(1, 1000 / max(1e-6, self.fps) / self.playback_speed))
            elif key == ord('1'):
                self.playback_speed = 0.5
                delay_ms = int(max(1, 1000 / max(1e-6, self.fps) / self.playback_speed))
            elif key == ord('2'):
                self.playback_speed = 1.0
                delay_ms = int(max(1, 1000 / max(1e-6, self.fps) / self.playback_speed))
            elif key == ord('3'):
                self.playback_speed = 2.0
                delay_ms = int(max(1, 1000 / max(1e-6, self.fps) / self.playback_speed))
            elif key in (81, ord('a')):  # left arrow
                target = max(0, frame_idx - 1)
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, target)
                ret, frame = self.cap.read()
                if ret:
                    frame_idx = target
                    self._last_raw = frame.copy()
                    self._last_preds = self._get_predictions(frame_idx, frame)
                    self.paused = True
                self._update_pos_trackbar(frame_idx)
            elif key in (83, ord('d')):  # right arrow
                target = min(self.total - 1, frame_idx + 1) if self.total > 0 else frame_idx + 1
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, target)
                ret, frame = self.cap.read()
                if ret:
                    frame_idx = target
                    self._last_raw = frame.copy()
                    self._last_preds = self._get_predictions(frame_idx, frame)
                    self.paused = True
                self._update_pos_trackbar(frame_idx)
            elif key == 36:  # Home
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self.paused = True
            elif key == 35:  # End
                if self.total > 0:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.total - 1)
                    self.paused = True

        self.cap.release()
        cv2.destroyAllWindows()


def run_gui_and_play(prefill_snippets_dir: Optional[str] = None, prefill_weights: Optional[str] = None, prefill_output_dir: Optional[str] = None):
    """Launch the Tkinter selector (if available) and start the player. Optionally prefill paths."""
    if tk is None:
        print("GUI libraries not available in this environment.")
        return
    try:
        root = tk.Tk()
    except Exception as e:
        # Helpful hint for SSH/X11 users
        if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
            print("Error: No DISPLAY found. On Linux over SSH, enable X11 forwarding with 'ssh -X' or 'ssh -Y'.")
        else:
            print(f"Error initializing Tkinter: {e}")
        return
    ui = SelectorUI(root, prefill_snippets_dir=prefill_snippets_dir, prefill_weights=prefill_weights, prefill_output_dir=prefill_output_dir)
    root.mainloop()
    sel = ui.get_selection()
    root.destroy()
    if not sel:
        print("Canceled or incomplete selection.")
        return
    player = Player(sel)
    player.run()


def _auto_device() -> str:
    """Best-effort device string: 'cuda:0' if available, else 'cpu'."""
    return "cuda:0" if (torch is not None and hasattr(torch, "cuda") and torch.cuda.is_available()) else "cpu"


def _infer_frame(model: YOLO, frame_bgr: np.ndarray, class_names: Dict[int, str], conf: float, iou: float, device: str) -> List[Dict]:
    """Helper for batch mode: run YOLO on a frame and return detection dicts."""
    results = model.predict(source=frame_bgr, conf=conf, iou=iou, device=device, verbose=False)
    preds: List[Dict] = []
    if not results:
        return preds
    res = results[0]
    if res.boxes is None:
        return preds
    boxes = res.boxes
    xyxy = boxes.xyxy.cpu().numpy().astype(int)
    confs = boxes.conf.cpu().numpy().tolist() if boxes.conf is not None else []
    clss = boxes.cls.cpu().numpy().astype(int).tolist() if boxes.cls is not None else []
    for i, (x1, y1, x2, y2) in enumerate(xyxy):
        conf_i = float(confs[i]) if i < len(confs) else 0.0
        cid = int(clss[i]) if i < len(clss) else -1
        preds.append({
            "xyxy": [int(x1), int(y1), int(x2), int(y2)],
            "confidence": conf_i,
            "class_id": cid,
            "class_name": class_names.get(cid, str(cid)) if isinstance(class_names, dict) else str(cid),
        })
    return preds


def batch_precompute(input_dir: Path, pattern: str, weights_path: Path, force: bool = False, output_dir: Optional[Path] = None, max_videos: Optional[int] = None, min_frames: Optional[int] = None):
    """Headless precompute over input_dir + pattern; writes caches to output_dir and optionally copies videos. Only processes videos with at least min_frames frames if specified."""
    import shutil
    if YOLO is None:
        raise RuntimeError("Ultralytics is not installed. Please install ultralytics.")
    device = _auto_device()
    print(f"Loading model on {device}…")
    model = YOLO(str(weights_path))
    try:
        model.to(device)
    except Exception:
        pass
    class_names = model.names if hasattr(model, "names") else {}

    # Gather videos
    all_videos = [p for p in input_dir.glob(pattern) if p.suffix.lower() in VIDEO_EXTS and p.is_file()]
    filtered_videos = []
    for p in all_videos:
        cap = cv2.VideoCapture(str(p))
        if not cap.isOpened():
            continue
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        if min_frames is not None and total < min_frames:
            continue
        filtered_videos.append(p)
    videos = filtered_videos
    if not videos:
        print("No videos found for pattern (or all below min-frames).")
        return
    if max_videos is not None:
        videos = videos[:max_videos]
    print(f"Found {len(videos)} videos to precompute.")

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    for vp in videos:
        # Set up output paths
        out_dir = output_dir if output_dir is not None else vp.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        # Copy video to output_dir if needed
        out_video_path = out_dir / vp.name
        if output_dir is not None and not out_video_path.exists():
            try:
                shutil.copy2(vp, out_video_path)
                print(f"[copy] Video copied to: {out_video_path}")
            except Exception as e:
                print(f"[warn] Could not copy video: {e}")
        # Use output_dir for cache files
        cache = DiskCache(out_video_path, weights_path) if output_dir is not None else DiskCache(vp, weights_path)
        if cache.exists() and not force:
            print(f"[skip] Cache exists: {out_video_path.name if output_dir is not None else vp.name}")
            continue
        print(f"[precompute] {out_video_path if output_dir is not None else vp}")
        cap = cv2.VideoCapture(str(out_video_path if output_dir is not None else vp))
        if not cap.isOpened():
            print(f"[warn] Cannot open video: {out_video_path if output_dir is not None else vp}")
            continue
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        frames_preds: List[List[Dict]] = []
        conf = 0.001
        iou = 0.45
        with tqdm(total=total if total and total > 0 else None, unit="frame") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                preds = _infer_frame(model, frame, class_names, conf, iou, device)
                frames_preds.append(preds)
                pbar.update(1)
        cap.release()
        cache.save(width, height, fps, total, class_names if isinstance(class_names, dict) else {}, frames_preds)
        print(f"[done] Cached: {cache.npz_path.name}")


def main():
    """CLI entry point: batch-precompute mode or interactive review via GUI."""

    parser = argparse.ArgumentParser(description="Review or precompute detections for hard-example mining")
    parser.add_argument("--batch-precompute", action="store_true", help="Run headless precompute over a folder of videos")
    parser.add_argument("--input-dir", type=Path, help="Input directory with videos for precompute")
    parser.add_argument("--pattern", type=str, default="*.mp4", help="Globbing pattern for videos, e.g., *.mp4 or **/*.mp4")
    parser.add_argument("--weights", type=Path, help="YOLO weights .pt path (required for precompute)")
    parser.add_argument("--force", action="store_true", help="Recompute caches even if they already exist")
    parser.add_argument("--output-dir", type=Path, help="Directory to save output videos and label files (for download)")
    parser.add_argument("--max-videos", type=int, help="Maximum number of videos to process in batch mode")
    parser.add_argument("--min-frames", type=int, help="Minimum number of frames a video must have to be processed in batch mode")
    args = parser.parse_args()

    if args.batch_precompute:
        if not args.input_dir or not args.weights:
            raise SystemExit("--batch-precompute requires --input-dir and --weights")
        batch_precompute(
            args.input_dir,
            args.pattern,
            args.weights,
            force=args.force,
            output_dir=args.output_dir,
            max_videos=args.max_videos,
            min_frames=args.min_frames
        )
        return

    # GUI mode, optionally prefill weights and input-dir
    prefill_snippets_dir = str(args.input_dir) if args.input_dir else None
    prefill_weights = str(args.weights) if args.weights else None
    prefill_output_dir = str(args.output_dir) if args.output_dir else None
    run_gui_and_play(
        prefill_snippets_dir=prefill_snippets_dir,
        prefill_weights=prefill_weights,
        prefill_output_dir=prefill_output_dir
    )


if __name__ == "__main__":
    main()
