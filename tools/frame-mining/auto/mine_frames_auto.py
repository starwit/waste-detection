"""
Lean Intelligent Keyframe Mining (single file)

Features
- FrameStore: one-pass per video to cache resized-frame detections (low conf), sharpness, brightness, pHash,
  and (optionally) thumbnails. Store sampling stride is configurable and defaults to runtime stride.
- Vectorized IoU tracker.
- Logic tuned for recall on imperfect detectors (AUTO guarantee, capped s_peak term, conservative pHash dedupe with per-track bypass,
  adaptive quotas, improved keyframe policy).
- Correct box scaling between resized-detection space and original image space.
- Optional annotated export (--save_annotated).
"""

from __future__ import annotations
import os
import cv2
import glob
import json
import pickle
import argparse
import warnings
import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Iterable
from collections import defaultdict, Counter

import numpy as np
from tqdm import tqdm

# =============================== CONFIG ===============================

@dataclass(frozen=True)
class Config:
    # Budgets
    B_TRACKS: int = 400
    B_FRAMES: int = 800

    # Selection weights (S = αU + βD + γR + δN + ε s_peak_cap)
    ALPHA: float = 0.25
    BETA: float = 0.25
    GAMMA: float = 0.20
    DELTA: float = 0.15
    EPSIL: float = 0.15
    S_PEAK_CAP: float = 0.70

    # Quotas
    Q_FLOOR_FRAC: float = 0.15
    BUCKETS: Tuple[str, ...] = ("tiny_small", "night", "standard")

    # Tracker / gates
    ASSOC_IOU: float = 0.20
    TRACKER_MAX_AGE_M: int = 20
    TRACKER_MAX_TRACKS: int = 1500
    TRACKER_MAX_LENGTH: int = 100
    TAU_D: float = 0.25

    # Quality floors
    SMALL_AREA: float = 0.008
    ABS_MIN_SIDE_PX: int = 12
    MIN_SHARP_LABEL: float = 10.0

    # Dedupe / thinning
    TEMPORAL_THIN_FR: int = 2
    PHASH_HAMMING_MAX: int = 6
    ALLOW_PER_TRACK_PHASH_BYPASS: bool = True

    # Inference / store
    INFER_IMG_LONG_EDGE: int = 960
    INFER_BASE_CONF: float = 0.05
    DEFAULT_CONF: float = 0.10
    BATCH_SIZE: int = 8
    THUMB_QUALITY: int = 90

    # Night heuristic
    NIGHT_BRIGHTNESS: float = 80.0

    # Novelty memory bank
    MAX_MEM_BANK: int = 100_000

    # DROP analysis (optional)
    DROP_RATIO: float = 0.15
    DROP_ANALYSIS_BUDGET: int = 50

    # Debug logging
    DEBUG: bool = True

cfg = Config()

# =====================================================================

def load_classes_from_params(base_dir: Optional[str] = None) -> List[str]:
    """
    Load class names from params.yaml in the workspace.
    
    Args:
        base_dir: Optional base directory to search from. If None, uses current working directory.
        
    Returns:
        List of class names, defaults to ["waste", "cigarette"] if not found.
    """
    # Find params.yaml in workspace
    if base_dir:
        workspace_root = Path(base_dir).resolve()
    else:
        workspace_root = Path.cwd().resolve()
        
    while workspace_root.parent != workspace_root:
        params_path = workspace_root / "params.yaml"
        if params_path.exists():
            break
        workspace_root = workspace_root.parent
    else:
        # Default classes if no params.yaml found
        return ["waste", "cigarette"]

    # Load classes from params.yaml
    try:
        with open(params_path, 'r') as f:
            params = yaml.safe_load(f)
        custom_classes = params.get('data', {}).get('custom_classes', [])
        if custom_classes and isinstance(custom_classes, list):
            # Filter out empty/None values and use only valid class names
            valid_classes = [cls for cls in custom_classes if cls and isinstance(cls, str)]
            if valid_classes:
                return valid_classes
    except Exception as e:
        if cfg.DEBUG:
            print(f"Warning: Could not read params.yaml: {e}")
    
    # Fallback to default classes
    return ["waste", "cigarette"]

# ------------------------------- Utils -------------------------------

def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    inter = max(0.0, x2-x1) * max(0.0, y2-y1)
    if inter <= 0: return 0.0
    area_a = max(0.0, a[2]-a[0]) * max(0.0, a[3]-a[1])
    area_b = max(0.0, b[2]-b[0]) * max(0.0, b[3]-b[1])
    return inter / (area_a + area_b - inter + 1e-6)

def iou_matrix(tracks_last: np.ndarray, dets: np.ndarray) -> np.ndarray:
    if tracks_last.size == 0 or dets.size == 0:
        return np.zeros((tracks_last.shape[0], dets.shape[0]), dtype=np.float32)
    tl = np.maximum(tracks_last[:, None, :2], dets[None, :, :2])
    br = np.minimum(tracks_last[:, None, 2:], dets[None, :, 2:])
    wh = np.clip(br - tl, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    area_t = (tracks_last[:, 2] - tracks_last[:, 0]) * (tracks_last[:, 3] - tracks_last[:, 1])
    area_d = (dets[:, 2] - dets[:, 0]) * (dets[:, 3] - dets[:, 1])
    union = area_t[:, None] + area_d[None, :] - inter + 1e-6
    return (inter / union).astype(np.float32)

def lap_var(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())

def brightness(bgr: np.ndarray) -> float:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    return float(hsv[..., 2].mean())

def hsv_hist(bgr: np.ndarray, bins: int = 16) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv],[0,1,2],None,[bins,bins,bins],[0,180,0,256,0,256])
    hist = cv2.normalize(hist, None).flatten().astype(np.float32)
    n = np.linalg.norm(hist) + 1e-9
    return hist / n

def normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v) + 1e-9
    return v / n

def phash64(gray: np.ndarray) -> int:
    small = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    dct = cv2.dct(small)
    dct8 = dct[:8,:8]
    med = np.median(dct8[1:,:])
    bits = (dct8 > med).astype(np.uint8)
    acc = 0
    for i in range(8):
        for j in range(8):
            acc = (acc << 1) | int(bits[i,j])
    return int(acc)

def hamming64(a: int, b: int) -> int:
    return bin(a ^ b).count("1")

def yolo_line(xyxy: Iterable[float], W: int, H: int, cls_id: int = 0) -> str:
    x1,y1,x2,y2 = xyxy
    w = max(0.0, x2-x1); h = max(0.0, y2-y1)
    cx = x1 + w/2.0; cy = y1 + h/2.0
    return f"{cls_id} {cx/W:.6f} {cy/H:.6f} {w/W:.6f} {h/H:.6f}"

class SuppressStderr:
    def __enter__(self):
        self._orig = os.dup(2)
        self._devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(self._devnull, 2)
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        os.dup2(self._orig, 2)
        os.close(self._devnull)

# ---------------------------- Data classes ---------------------------

@dataclass
class TrkDet:
    frame_idx: int
    box: np.ndarray
    score: float
    sharp: float

@dataclass
class Track:
    id: int
    video_path: str
    route_id: str
    W: int
    H: int
    dets: List[TrkDet] = field(default_factory=list)
    # metrics
    c: float=0.0; s_min: float=0.0; s_mean: float=0.0; s_var: float=0.0
    J: float=0.0; a_med: float=0.0; L_med: float=0.0
    night: bool=False
    emb: Optional[np.ndarray]=None
    s_peak: float=0.0

# --------------------------- Detector wrapper ------------------------

def load_detector(model_path: str):
    from ultralytics import YOLO
    import torch
    os.environ['OPENCV_LOG_LEVEL'] = 'SILENT'
    model = YOLO(model_path)
    if torch.cuda.is_available():
        model.to('cuda')
        torch.backends.cudnn.benchmark = True
    return model


def detect_batch(model, images: List[np.ndarray], imgsz: int, conf: float):
    import torch
    if not images:
        return []
    use_half = torch.cuda.is_available()
    try:
        res_list = model.predict(images, imgsz=imgsz, conf=conf, verbose=False, half=use_half)
    except Exception:
        res_list = model.predict(images, imgsz=imgsz, conf=conf, verbose=False)

    outputs = []
    for r in res_list:
        frame_out = []
        if r.boxes is not None:
            for b in r.boxes:
                xyxy = b.xyxy.cpu().numpy().reshape(-1).astype(np.float32)
                score = float(b.conf.cpu().numpy().reshape(-1)[0])
                frame_out.append((xyxy, score))
        outputs.append(frame_out)
    return outputs

# --------------------------- FrameStore build ------------------------

def thumb_path(store_root: str, stem: str, frame_idx: int) -> str:
    return os.path.join(store_root, stem, "thumbs", f"thumb_f{frame_idx:06d}.jpg")


def build_framestore(videos: List[str], out_logs: str, model_path: str,
                     store_stride: int, write_thumbs: bool) -> str:
    model = load_detector(model_path)
    store_root = os.path.join(out_logs, "frame_store")
    ensure_dir(store_root)

    for vp in videos:
        stem = os.path.splitext(os.path.basename(vp))[0]
        vdir = os.path.join(store_root, stem)
        idx_path = os.path.join(vdir, "index.pkl")
        thumbs_dir = os.path.join(vdir, "thumbs")
        if os.path.exists(idx_path):
            if cfg.DEBUG: print(f"[FrameStore] Exists for {stem}, skipping.")
            continue

        ensure_dir(vdir)
        if write_thumbs:
            ensure_dir(thumbs_dir)

        with SuppressStderr():
            cap = cv2.VideoCapture(vp)
        if not cap.isOpened():
            print(f"[WARN] cannot open {vp}, skipping.")
            continue

        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        FPS = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        frames = []; sharps = []; brs = []; phs = []; dets: List[Tuple[np.ndarray, np.ndarray]] = []
        thumb_W = None; thumb_H = None

        pbar = tqdm(total=total, desc=f"Store {stem}", unit="fr", leave=False)
        batch_imgs = []; batch_meta = []

        def flush():
            nonlocal batch_imgs, batch_meta, thumb_W, thumb_H
            if not batch_imgs: return
            res_list = detect_batch(model, batch_imgs, imgsz=min(max(H,W), cfg.INFER_IMG_LONG_EDGE), conf=cfg.INFER_BASE_CONF)
            for (fr_idx, thumb_bgr, sharp_i, bright_i, ph_i), dets_i in zip(batch_meta, res_list):
                if write_thumbs:
                    outp = os.path.join(thumbs_dir, f"thumb_f{fr_idx:06d}.jpg")
                    cv2.imwrite(outp, thumb_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), cfg.THUMB_QUALITY])

                if thumb_W is None or thumb_H is None:
                    h_, w_ = thumb_bgr.shape[:2]
                    thumb_W, thumb_H = int(w_), int(h_)

                if dets_i:
                    boxes = np.stack([d[0] for d in dets_i], axis=0).astype(np.float32)
                    scores = np.array([d[1] for d in dets_i], dtype=np.float32)
                else:
                    boxes = np.zeros((0,4), dtype=np.float32)
                    scores = np.zeros((0,), dtype=np.float32)

                frames.append(fr_idx)
                sharps.append(float(sharp_i))
                brs.append(float(bright_i))
                phs.append(int(ph_i))
                dets.append((boxes, scores))

            batch_imgs = []; batch_meta = []

        cur = 0
        while True:
            with SuppressStderr():
                ok, img = cap.read()
            if not ok: break
            if (cur % store_stride) == 0:
                H0,W0 = img.shape[:2]
                scale = cfg.INFER_IMG_LONG_EDGE / max(H0,W0)
                if scale < 1.0:
                    new_W = int(round(W0 * scale)); new_H = int(round(H0 * scale))
                    thumb = cv2.resize(img, (new_W, new_H), interpolation=cv2.INTER_AREA)
                else:
                    thumb = img.copy()
                gray = cv2.cvtColor(thumb, cv2.COLOR_BGR2GRAY)
                sharp_i = lap_var(gray)
                bright_i = brightness(thumb)
                ph_i = phash64(gray)
                batch_imgs.append(thumb)
                batch_meta.append((cur, thumb, sharp_i, bright_i, ph_i))
                if len(batch_imgs) >= cfg.BATCH_SIZE:
                    flush()
            cur += 1; pbar.update(1)

        flush(); pbar.close(); cap.release()

        if not frames:
            print(f"[WARN] No frames stored for {stem}")
            continue

        if thumb_W is None or thumb_H is None:
            scale = cfg.INFER_IMG_LONG_EDGE / max(H, W)
            if scale < 1.0:
                thumb_W = int(round(W * scale)); thumb_H = int(round(H * scale))
            else:
                thumb_W = W; thumb_H = H

        index = dict(
            video_path=vp, stem=stem, W=W, H=H, fps=FPS,
            frames=frames, sharps=sharps, brs=brs, phash=phs, dets=dets,
            imgsz=cfg.INFER_IMG_LONG_EDGE, base_conf=cfg.INFER_BASE_CONF,
            store_stride=store_stride, write_thumbs=bool(write_thumbs),
            thumb_W=int(thumb_W), thumb_H=int(thumb_H)
        )
        with open(idx_path, "wb") as f: pickle.dump(index, f)
        if cfg.DEBUG:
            print(f"[FrameStore] Built for {stem}: {len(frames)} frames @ stride {store_stride} (thumb {thumb_W}x{thumb_H})")

    return os.path.join(out_logs, "frame_store")


def load_store(store_root: str, video_stem: str) -> Dict:
    idx_path = os.path.join(store_root, video_stem, "index.pkl")
    with open(idx_path, "rb") as f:
        return pickle.load(f)

# --------------------------- Metrics & triage -------------------------

def compute_metrics(T: Track, tau_d: float = cfg.TAU_D) -> None:
    N = len(T.dets)
    scores = np.array([d.score for d in T.dets], np.float32) if N else np.zeros(0)
    boxes  = np.array([d.box   for d in T.dets], np.float32) if N else np.zeros((0,4))
    T.c     = float((scores >= tau_d).mean()) if N else 0.0
    T.s_min = float(scores.min())  if N else 0.0
    T.s_mean= float(scores.mean()) if N else 0.0
    T.s_var = float(np.var(scores))if N else 0.0
    T.s_peak= float(scores.max())  if N else 0.0
    if N>=2:
        ious = [iou_xyxy(boxes[i-1], boxes[i]) for i in range(1,N)]
        T.J = float(np.median(ious)) if ious else 0.0
    else:
        T.J = 0.0
    areas = (boxes[:,2]-boxes[:,0]) * (boxes[:,3]-boxes[:,1]) if N else np.array([0.0])
    a_rel = areas / float(T.W*T.H + 1e-9)
    T.a_med = float(np.median(a_rel)) if N else 0.0
    T.L_med = float(np.median([d.sharp for d in T.dets])) if N else 0.0

def is_night(mean_brightnesses: List[float], thr: float = cfg.NIGHT_BRIGHTNESS) -> bool:
    return (np.mean(mean_brightnesses) < thr)


def triage(T: Track) -> str:
    N = len(T.dets)
    boxes = np.array([d.box for d in T.dets], np.float32) if N else np.zeros((0,4))
    if N:
        wmed = float(np.median(boxes[:,2]-boxes[:,0])); hmed = float(np.median(boxes[:,3]-boxes[:,1]))
        if min(wmed, hmed) < cfg.ABS_MIN_SIDE_PX:
            return "DROP"

    if T.L_med < cfg.MIN_SHARP_LABEL:
        return "DROP"

    if (T.c >= 0.40 and T.s_min >= 0.22 and T.J >= 0.12 and N >= 2 and T.L_med >= 15.0):
        return "AUTO"

    if (T.a_med < cfg.SMALL_AREA) or (T.s_min < 0.40) or (T.J < 0.35):
        return "LABEL"

    return "DROP"

# --------------------------- Keyframe logic --------------------------

def keyframe_indices(T: Track) -> List[int]:
    if not T.dets: return []
    scores = np.array([d.score for d in T.dets], np.float32)
    boxes  = np.array([d.box   for d in T.dets], np.float32)
    areas  = (boxes[:,2]-boxes[:,0]) * (boxes[:,3]-boxes[:,1])

    peak_idx_local = int(np.argmax(scores))
    n = len(T.dets)

    def segment_max_area(lo: int, hi: int) -> int:
        if lo >= hi: return lo
        seg = np.argmax(areas[lo:hi]) + lo
        return int(seg)

    start_end_third = max(1, n // 3)
    birth_local  = segment_max_area(0, start_end_third)
    death_local  = segment_max_area(n - start_end_third, n)

    candidates_local = {peak_idx_local, birth_local, death_local}
    if n >= 2:
        if peak_idx_local - 1 >= 0: candidates_local.add(peak_idx_local - 1)
        if peak_idx_local + 1 < n:  candidates_local.add(peak_idx_local + 1)

    frames = sorted({T.dets[i].frame_idx for i in candidates_local if 0 <= i < n})
    return frames


def assign_bucket(T: Track) -> str:
    if T.a_med < cfg.SMALL_AREA: return "tiny_small"
    if T.night:              return "night"
    return "standard"

# ------------------------- Vectorized tracker ------------------------

class VecTracker:
    def __init__(self, assoc_iou: float, max_age: int, max_tracks: int):
        self.assoc_iou = assoc_iou
        self.max_age   = max_age
        self.max_tracks= max_tracks
        self.next_id   = 1
        self.last_boxes = np.zeros((0,4), dtype=np.float32)
        self.ages       = np.zeros((0,), dtype=np.int32)
        self.ids        = np.zeros((0,), dtype=np.int32)
        self.hist: Dict[int, List[TrkDet]] = {}

    def prune_too_old(self) -> None:
        keep = self.ages <= self.max_age
        self.last_boxes = self.last_boxes[keep]
        self.ages       = self.ages[keep]
        self.ids        = self.ids[keep]

    def update(self, frame_idx: int, dets: List[Tuple[np.ndarray, float, float]]):
        self.ages += 1
        self.prune_too_old()

        if len(dets) == 0:
            return

        det_boxes = np.stack([d[0] for d in dets], axis=0).astype(np.float32)
        det_scores= np.array([d[1] for d in dets], dtype=np.float32)
        det_sharp = np.array([d[2] for d in dets], dtype=np.float32)

        Tn = self.last_boxes.shape[0]; Dn = det_boxes.shape[0]
        assigned_det = np.full(Dn, -1, dtype=np.int32)
        assigned_trk = np.full(Tn, -1, dtype=np.int32)

        if Tn > 0:
            ious = iou_matrix(self.last_boxes, det_boxes)
            while True:
                ti, di = np.unravel_index(np.argmax(ious), ious.shape)
                best = ious[ti, di]
                if best < self.assoc_iou: break
                assigned_trk[ti] = di
                assigned_det[di] = self.ids[ti]
                ious[ti, :] = -1.0
                ious[:, di] = -1.0

        for ti in range(Tn):
            didx = assigned_trk[ti]
            if didx >= 0:
                tid = self.ids[ti]
                box = det_boxes[didx]; score = float(det_scores[didx]); sharp = float(det_sharp[didx])
                self.hist.setdefault(tid, []).append(TrkDet(frame_idx, box.copy(), score, sharp))
                self.last_boxes[ti] = box
                self.ages[ti] = 0

        for di in range(Dn):
            if assigned_det[di] >= 0: continue
            if self.ids.shape[0] >= self.max_tracks:
                continue
            tid = self.next_id; self.next_id += 1
            self.hist[tid] = [TrkDet(frame_idx, det_boxes[di].copy(), float(det_scores[di]), float(det_sharp[di]))]
            self.last_boxes = np.vstack([self.last_boxes, det_boxes[di][None, :]]) if self.last_boxes.size else det_boxes[di][None, :]
            self.ages       = np.append(self.ages, 0)
            self.ids        = np.append(self.ids, tid)

        to_del = []
        for tid, detlist in self.hist.items():
            if len(detlist) > cfg.TRACKER_MAX_LENGTH:
                to_del.append(tid)
        if to_del:
            keep = np.array([tid not in to_del for tid in self.ids], dtype=bool)
            self.last_boxes = self.last_boxes[keep]
            self.ages       = self.ages[keep]
            self.ids        = self.ids[keep]
            for tid in to_del: self.hist.pop(tid, None)

    def finalize(self) -> Dict[int, List[TrkDet]]:
        return {tid: dets for tid, dets in self.hist.items() if len(dets) > 0}

# ---------------------------- Pipeline steps -------------------------

def setup_dirs(out_dir: str, save_annotated: bool) -> Dict[str,str]:
    ensure_dir(out_dir)
    images_dir = os.path.join(out_dir, "images"); ensure_dir(images_dir)
    labels_dir = os.path.join(out_dir, "labels"); ensure_dir(labels_dir)
    logs_dir   = os.path.join(out_dir, "logs");   ensure_dir(logs_dir)
    annotated_dir = os.path.join(out_dir, "annotated")
    if save_annotated:
        ensure_dir(annotated_dir)
    return dict(images=images_dir, labels=labels_dir, logs=logs_dir, annotated=annotated_dir)


def collect_videos(videos_dir: str) -> List[str]:
    vids = sorted([p for p in glob.glob(os.path.join(videos_dir, "*.*"))
                   if os.path.splitext(p)[1].lower() in [".mp4",".avi",".mov",".mkv"]])
    if not vids:
        raise SystemExit(f"No videos in {videos_dir}")
    return vids


def track_all_videos(vids: List[str], store_root: str, stride_k: int,
                     eff_conf: float) -> Tuple[List[Track], Dict[Tuple[str,int], float], Dict[str, Tuple[int,int]]]:
    all_tracks: List[Track] = []
    frame_brightness: Dict[Tuple[str,int], float] = {}
    video_thumb_sizes: Dict[str, Tuple[int,int]] = {}

    for vp in vids:
        stem = os.path.splitext(os.path.basename(vp))[0]
        store = load_store(store_root, stem)
        W, H, FPS = store["W"], store["H"], store["fps"]
        frames = store["frames"]; sharps = store["sharps"]; brs = store["brs"]
        dets_list = store["dets"]
        thumb_W, thumb_H = int(store["thumb_W"]), int(store["thumb_H"])
        video_thumb_sizes[vp] = (thumb_W, thumb_H)

        route_id = stem
        tracker = VecTracker(cfg.ASSOC_IOU, cfg.TRACKER_MAX_AGE_M * stride_k, cfg.TRACKER_MAX_TRACKS)

        for fr, sharp_i, bright_i, (boxes, scores) in zip(frames, sharps, brs, dets_list):
            if (fr % stride_k) != 0: continue
            keep = scores >= eff_conf
            dets = [(boxes[i], float(scores[i]), float(sharp_i)) for i in np.where(keep)[0]] if np.any(keep) else []
            tracker.update(fr, dets)
            frame_brightness[(vp, fr)] = float(bright_i)

        final_tracks = tracker.finalize()
        for tid, dets in final_tracks.items():
            T = Track(tid, vp, route_id, W, H, dets)
            compute_metrics(T)
            brs_track = [frame_brightness.get((vp, d.frame_idx), 255.0) for d in dets]
            T.night = is_night(brs_track)
            all_tracks.append(T)

        if cfg.DEBUG:
            lens = [len(d) for d in final_tracks.values()]
            if lens:
                print(f"[Tracks] {stem}: {len(final_tracks)} tracks, N(min/avg/max) = {min(lens)}/{np.mean(lens):.1f}/{max(lens)}")

    return all_tracks, frame_brightness, video_thumb_sizes


def split_triage(tracks: List[Track]) -> Tuple[List[Track], List[Track], List[Track]]:
    AUTO: List[Track] = []
    LABEL: List[Track] = []
    DROP: List[Track] = []
    for T in tracks:
        d = triage(T)
        (AUTO if d=="AUTO" else LABEL if d=="LABEL" else DROP).append(T)
    if cfg.DEBUG:
        print(f"[Triage] AUTO={len(AUTO)} LABEL={len(LABEL)} DROP={len(DROP)}")
    return AUTO, LABEL, DROP


def compute_label_embeddings(LABEL: List[Track], store_root: str,
                             video_thumb_sizes: Dict[str, Tuple[int,int]]) -> None:
    for T in LABEL:
        if T.emb is not None: continue
        peak_local = int(np.argmax([d.score for d in T.dets]))
        fr = T.dets[peak_local].frame_idx
        stem = os.path.splitext(os.path.basename(T.video_path))[0]
        tp = thumb_path(store_root, stem, fr)
        img = cv2.imread(tp, cv2.IMREAD_COLOR)
        if img is None:
            with SuppressStderr():
                cap = cv2.VideoCapture(T.video_path)
            if cap.isOpened():
                cur = 0; ok = True; frame_img = None
                while cur <= fr and ok:
                    ok, frame_img = cap.read(); cur += 1
                cap.release(); img = frame_img
        if img is None:
            continue
        x1,y1,x2,y2 = T.dets[peak_local].box.astype(int)
        thumb_W, thumb_H = video_thumb_sizes[T.video_path]
        if img.shape[1] != thumb_W or img.shape[0] != thumb_H:
            sx = img.shape[1] / float(thumb_W); sy = img.shape[0] / float(thumb_H)
            x1 = int(round(x1 * sx)); x2 = int(round(x2 * sx))
            y1 = int(round(y1 * sy)); y2 = int(round(y2 * sy))
        Ht,Wt = img.shape[:2]; bw, bh = x2-x1, y2-y1
        x1p = max(0, int(x1 - 0.10*bw)); y1p = max(0, int(y1 - 0.10*bh))
        x2p = min(Wt-1, int(x2 + 0.10*bw)); y2p = min(Ht-1, int(y2 + 0.10*bh))
        crop = img[y1p:y2p, x1p:x2p] if (x2p>x1p and y2p>y1p) else img
        T.emb = hsv_hist(crop)


def diversity_and_novelty(LABEL: List[Track], mem_bank: Optional[np.ndarray]) -> Tuple[np.ndarray,np.ndarray]:
    label_embs = [T.emb for T in LABEL if T.emb is not None]
    if not label_embs:
        return np.zeros(len(LABEL), dtype=np.float32), np.ones(len(LABEL), dtype=np.float32)

    F = np.stack(label_embs, axis=0).astype(np.float32)
    F = F / (np.linalg.norm(F, axis=1, keepdims=True) + 1e-9)
    sims = F @ F.T
    num_seeds = min(10, F.shape[0])
    seed_ids = list(range(num_seeds))
    div_vals = 1.0 - np.max(sims[:, seed_ids], axis=1)

    D_current = np.zeros(len(LABEL), dtype=np.float32)
    j = 0
    for i, T in enumerate(LABEL):
        if T.emb is not None:
            D_current[i] = float(div_vals[j]); j += 1

    N_hist = np.ones(len(LABEL), dtype=np.float32)
    if mem_bank is not None and isinstance(mem_bank, np.ndarray) and mem_bank.size > 0:
        MB = mem_bank
        MB = MB / (np.linalg.norm(MB, axis=1, keepdims=True) + 1e-9)
        sims_hist = MB @ F.T
        nov = 1.0 - np.max(sims_hist, axis=0)
        j = 0
        for i, T in enumerate(LABEL):
            if T.emb is not None:
                N_hist[i] = float(nov[j]); j += 1
    return D_current, N_hist


def score_and_pick_tracks(LABEL: List[Track], AUTO: List[Track],
                          D_current: np.ndarray, N_hist: np.ndarray) -> List[Track]:
    U = np.array([1.0 - np.mean([d.score for d in T.dets]) if T.dets else 1.0 for T in LABEL], dtype=np.float32)
    R = np.array([
        min(2.0, (cfg.SMALL_AREA / (T.a_med + 1e-6))) + (1.0 if T.J < 0.35 else 0.0)
        for T in LABEL
    ], dtype=np.float32)
    S_peak = np.array([min(cfg.S_PEAK_CAP, T.s_peak) for T in LABEL], dtype=np.float32)
    S = cfg.ALPHA*U + cfg.BETA*D_current + cfg.GAMMA*R + cfg.DELTA*N_hist + cfg.EPSIL*S_peak
    order = np.argsort(-S)
    LABEL_sorted = [LABEL[i] for i in order]

    K_seed = min(100, max(10, cfg.B_TRACKS // 5))
    seed_by_speak = sorted(LABEL, key=lambda t: t.s_peak, reverse=True)[:K_seed]
    seed_ids = {id(t) for t in seed_by_speak}

    counts = Counter(assign_bucket(t) for t in LABEL_sorted)
    total_lbl = sum(counts.values()) or 1
    floors_per_bucket = int(round((cfg.Q_FLOOR_FRAC * cfg.B_TRACKS) / len(cfg.BUCKETS)))
    limits = {}
    for b in cfg.BUCKETS:
        share = counts[b] / total_lbl
        limits[b] = max(floors_per_bucket, int(round(share * cfg.B_TRACKS)))
    adjust = cfg.B_TRACKS - sum(limits.values())
    if adjust != 0:
        biggest = max(cfg.BUCKETS, key=lambda k: limits[k])
        limits[biggest] = max(0, limits[biggest] + adjust)

    picked: List[Track] = []
    used = {b:0 for b in cfg.BUCKETS}
    seen_stems = defaultdict(int)
    seen_routes= defaultdict(int)

    for T in LABEL_sorted:
        b = assign_bucket(T)
        if id(T) in seed_ids:
            picked.append(T); used[b]+=1
            continue
        if used[b] >= limits[b]: continue
        stem = os.path.basename(T.video_path)
        if seen_stems[stem] >= int(0.30 * cfg.B_TRACKS): continue
        if seen_routes[T.route_id] >= int(0.35 * cfg.B_TRACKS): continue
        picked.append(T)
        used[b]+=1; seen_stems[stem]+=1; seen_routes[T.route_id]+=1
        if len(picked) >= cfg.B_TRACKS: break

    return picked


def gather_keyframe_candidates(contrib: List[Track]) -> List[Tuple[str,int,Track,float]]:
    candidates: List[Tuple[str,int,Track,float]] = []
    for T in contrib:
        kfs = keyframe_indices(T)
        for fr in kfs:
            w = 1.0
            if T.a_med < cfg.SMALL_AREA: w *= 1.2
            if T.J < 0.35: w *= 1.1
            w *= (1.5 if T in contrib[:len(contrib)] else 1.0)
            candidates.append((T.video_path, fr, T, w))
    candidates.sort(key=lambda x: x[3], reverse=True)
    return candidates


def build_phash_map(vids: List[str], store_root: str) -> Dict[Tuple[str,int], int]:
    phash_map: Dict[Tuple[str,int], int] = {}
    for vp in vids:
        stem = os.path.splitext(os.path.basename(vp))[0]
        store = load_store(store_root, stem)
        for fr, ph in zip(store["frames"], store["phash"]):
            phash_map[(vp, fr)] = int(ph)
    return phash_map


def select_keyframes(vids: List[str], candidates: List[Tuple[str,int,Track,float]],
                     AUTO: List[Track], phash_map: Dict[Tuple[str,int], int]) -> List[Tuple[str,int]]:
    selected: List[Tuple[str,int]] = []
    per_video_ph: Dict[str, List[Tuple[int,int]]] = defaultdict(list)
    per_track_selected: Dict[int, int] = defaultdict(int)

    def ok_temporal(vp: str, fr: int) -> bool:
        return all((vp != v) or (abs(fr-f) > cfg.TEMPORAL_THIN_FR) for (v,f) in selected)

    def ok_phash(vp: str, fr: int, track_id: int) -> bool:
        if cfg.ALLOW_PER_TRACK_PHASH_BYPASS and per_track_selected[track_id] == 0:
            return True
        hval = phash_map.get((vp, fr), None)
        if hval is None:
            return True
        for (_, hp) in per_video_ph[vp]:
            if hamming64(hval, hp) <= cfg.PHASH_HAMMING_MAX:
                return False
        return True

    for (vp, fr, T, w) in candidates:
        if len(selected) >= cfg.B_FRAMES: break
        if not ok_temporal(vp, fr): continue
        if not ok_phash(vp, fr, T.id): continue
        selected.append((vp, fr))
        hval = phash_map.get((vp, fr), None)
        if hval is not None:
            per_video_ph[vp].append((fr, hval))
        per_track_selected[T.id] += 1

    for T in AUTO:
        if len(selected) >= cfg.B_FRAMES: break
        peak_fr = T.dets[int(np.argmax([d.score for d in T.dets]))].frame_idx
        if (T.video_path, peak_fr) not in selected and ok_temporal(T.video_path, peak_fr):
            selected.append((T.video_path, peak_fr))

    if cfg.DEBUG:
        print(f"[Greedy] Selected frames: {len(selected)} (budget {cfg.B_FRAMES})")

    return selected


def index_boxes_for_selected(contrib: List[Track]) -> Dict[Tuple[str,int], List[np.ndarray]]:
    frame_boxes_resized = defaultdict(list)
    for T in contrib:
        kfs_set = set(keyframe_indices(T))
        for d in T.dets:
            if d.frame_idx in kfs_set:
                frame_boxes_resized[(T.video_path, d.frame_idx)].append(d.box.copy())
    return frame_boxes_resized


def export_selected(by_video: Dict[str, List[int]], images_dir: str, labels_dir: str, annotated_dir: str,
                    video_thumb_sizes: Dict[str, Tuple[int,int]], frame_boxes_resized: Dict[Tuple[str,int], List[np.ndarray]],
                    save_annotated: bool) -> Tuple[int,int]:
    written = 0; skipped = 0
    for vp, frames in tqdm(by_video.items(), desc="Export", unit="vid"):
        with SuppressStderr():
            cap = cv2.VideoCapture(vp)
        if not cap.isOpened():
            print(f"[WARN] Cannot open {vp} for export.")
            skipped += len(frames)
            continue
        want = set(frames); cur = 0; cache = {}
        while True:
            ok, img = cap.read()
            if not ok: break
            if cur in want:
                cache[cur] = img.copy()
                if len(cache) == len(want): break
            cur += 1
        cap.release()

        stem = os.path.splitext(os.path.basename(vp))[0]
        thumb_W, thumb_H = video_thumb_sizes[vp]

        for fr in frames:
            img = cache.get(fr, None)
            if img is None:
                skipped += 1
                continue
            H0,W0 = img.shape[:2]
            sx = W0 / float(thumb_W)
            sy = H0 / float(thumb_H)

            out_stem = f"{stem}_f{fr:06d}"
            ip = os.path.join(images_dir, f"{out_stem}.jpg")
            lp = os.path.join(labels_dir, f"{out_stem}.txt")

            ok_w = cv2.imwrite(ip, img)
            if not ok_w:
                skipped += 1
                continue

            with open(lp, "w") as f:
                for box in frame_boxes_resized.get((vp, fr), []):
                    x1,y1,x2,y2 = box
                    box_scaled = np.array([x1*sx, y1*sy, x2*sx, y2*sy], dtype=np.float32)
                    f.write(yolo_line(box_scaled, W0, H0, cls_id=0) + "\n")

            if save_annotated:
                ann = img.copy()
                for box in frame_boxes_resized.get((vp, fr), []):
                    x1,y1,x2,y2 = box
                    x1 = int(round(x1*sx)); y1 = int(round(y1*sy))
                    x2 = int(round(x2*sx)); y2 = int(round(y2*sy))
                    cv2.rectangle(ann, (x1,y1), (x2,y2), (0,255,0), 2)
                ap = os.path.join(annotated_dir, f"{out_stem}.jpg")
                cv2.imwrite(ap, ann)

            written += 1

    if cfg.DEBUG:
        print(f"[Export] {written} images written, {skipped} skipped")
    return written, skipped


def update_memory_bank(logs_dir: str, picked: List[Track],
                       store_root: str, video_thumb_sizes: Dict[str, Tuple[int,int]],
                       mem_bank: Optional[np.ndarray]) -> None:
    new_embs = []
    for T in picked:
        if T.emb is not None:
            new_embs.append(T.emb); continue
        peak_local = int(np.argmax([d.score for d in T.dets]))
        fr = T.dets[peak_local].frame_idx
        stem = os.path.splitext(os.path.basename(T.video_path))[0]
        tp = thumb_path(store_root, stem, fr)
        img = cv2.imread(tp, cv2.IMREAD_COLOR)
        if img is None:
            with SuppressStderr():
                cap = cv2.VideoCapture(T.video_path)
            if cap.isOpened():
                cur = 0; ok = True; frame_img = None
                while cur <= fr and ok:
                    ok, frame_img = cap.read(); cur += 1
                cap.release(); img = frame_img
        if img is None:
            continue
        x1,y1,x2,y2 = T.dets[peak_local].box.astype(int)
        thumb_W, thumb_H = video_thumb_sizes[T.video_path]
        if img.shape[1] != thumb_W or img.shape[0] != thumb_H:
            sx = img.shape[1] / float(thumb_W)
            sy = img.shape[0] / float(thumb_H)
            x1 = int(round(x1 * sx)); x2 = int(round(x2 * sx))
            y1 = int(round(y1 * sy)); y2 = int(round(y2 * sy))
        Ht,Wt = img.shape[:2]
        bw, bh = x2-x1, y2-y1
        x1p = max(0, int(x1 - 0.10*bw)); y1p = max(0, int(y1 - 0.10*bh))
        x2p = min(Wt-1, int(x2 + 0.10*bw)); y2p = min(Ht-1, int(y2 + 0.10*bh))
        crop = img[y1p:y2p, x1p:x2p] if (x2p>x1p and y2p>y1p) else img
        new_embs.append(hsv_hist(crop))

    mem_path = os.path.join(logs_dir, "memory_bank.npy")
    if new_embs:
        nb = np.stack([normalize(e) for e in new_embs], axis=0).astype(np.float32)
        if mem_bank is None or not isinstance(mem_bank, np.ndarray) or mem_bank.size == 0:
            np.save(mem_path, nb)
        else:
            mb = np.concatenate([mem_bank, nb], axis=0)
            if len(mb) > cfg.MAX_MEM_BANK:
                mb = mb[-cfg.MAX_MEM_BANK:]
            np.save(mem_path, mb)


def write_logs(logs_dir: str, summary: Dict, AUTO: List[Track], LABEL: List[Track], DROP: List[Track], selected: List[Tuple[str,int]]) -> None:
    def to_dict(T: Track):
        return dict(id=T.id, video=T.video_path, route=T.route_id, W=T.W, H=T.H, N=len(T.dets),
                    c=T.c, s_min=T.s_min, s_mean=T.s_mean, s_var=T.s_var, J=T.J,
                    a_med=T.a_med, L_med=T.L_med, night=bool(T.night), s_peak=T.s_peak)
    ensure_dir(logs_dir)
    with open(os.path.join(logs_dir, "summary.json"), "w") as f: json.dump(summary, f, indent=2)
    with open(os.path.join(logs_dir, "tracks_AUTO.json"), "w") as f: json.dump([to_dict(t) for t in AUTO], f, indent=2)
    with open(os.path.join(logs_dir, "tracks_LABEL.json"), "w") as f: json.dump([to_dict(t) for t in LABEL], f, indent=2)
    with open(os.path.join(logs_dir, "tracks_DROP.json"), "w") as f: json.dump([to_dict(t) for t in DROP], f, indent=2)
    with open(os.path.join(logs_dir, "selected_frames.json"), "w") as f: json.dump([{"video":vp,"frame":fr} for (vp,fr) in selected], f, indent=2)


def create_cvat_export(out_dir: str, logs_dir: str) -> Optional[str]:
    """Create CVAT-compatible dataset export with classes from params.yaml"""
    import subprocess
    import sys

    # Load classes from params.yaml
    classes = load_classes_from_params(out_dir)
    if cfg.DEBUG:
        print(f"[CVAT Export] Using classes: {classes}")

    try:
        script_path = os.path.join(os.path.dirname(__file__), "make_cvat_folder.py")
        classes_arg = ",".join(classes)

        result = subprocess.run([
            sys.executable, script_path, out_dir,
            "--classes", classes_arg
        ], capture_output=True, text=True)

        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            zip_path = None
            for line in lines:
                if "Zipped dataset to:" in line:
                    zip_path = line.split("Zipped dataset to: ")[-1]
                    break
            if zip_path:
                print(f"[CVAT Export] Created: {zip_path}")
                return zip_path
            else:
                print("[CVAT Export] Completed but couldn't find zip path in output")
                if cfg.DEBUG:
                    print(f"[CVAT Export] Output: {result.stdout}")
        else:
            print(f"[CVAT Export] Failed with return code {result.returncode}")
            if cfg.DEBUG:
                print(f"[CVAT Export] Error: {result.stderr}")
            return None

    except FileNotFoundError:
        print("[CVAT Export] Error: make_cvat_folder.py script not found")
        return None
    except Exception as e:
        print(f"[CVAT Export] Error: {e}")
        return None

# ---------------------------- Main process ---------------------------

def process(videos_dir: str, out_dir: str, model_path: str,
            stride_k: int, conf_thresh: Optional[float],
            store_stride: Optional[int] = None, write_thumbs: bool = True,
            save_annotated: bool = False, create_cvat: bool = False):

    dirs = setup_dirs(out_dir, save_annotated)
    vids = collect_videos(videos_dir)

    effective_store_stride = max(1, store_stride if store_stride is not None else stride_k)
    store_root = build_framestore(vids, dirs['logs'], model_path,
                                  store_stride=effective_store_stride,
                                  write_thumbs=write_thumbs)

    mem_path = os.path.join(dirs['logs'], "memory_bank.npy")
    mem_bank = None
    if os.path.exists(mem_path):
        try:
            mem_bank = np.load(mem_path)
            if mem_bank.ndim == 1:
                mem_bank = mem_bank.reshape(1,-1)
        except Exception:
            mem_bank = None

    eff_conf = conf_thresh if conf_thresh is not None else cfg.DEFAULT_CONF

    all_tracks, frame_brightness, video_thumb_sizes = track_all_videos(vids, store_root, stride_k, eff_conf)

    AUTO, LABEL, DROP = split_triage(all_tracks)

    compute_label_embeddings(LABEL, store_root, video_thumb_sizes)

    D_current, N_hist = diversity_and_novelty(LABEL, mem_bank)

    picked = score_and_pick_tracks(LABEL, AUTO, D_current, N_hist)

    contrib = list(AUTO) + picked

    candidates = gather_keyframe_candidates(contrib)

    phash_map = build_phash_map(vids, store_root)

    selected = select_keyframes(vids, candidates, AUTO, phash_map)

    by_video = defaultdict(list)
    for (vp, fr) in selected:
        by_video[vp].append(fr)
    by_video = {vp: sorted(set(frs)) for vp, frs in by_video.items()}

    frame_boxes_resized = index_boxes_for_selected(contrib)

    written, skipped = export_selected(by_video, dirs['images'], dirs['labels'], dirs['annotated'],
                                       video_thumb_sizes, frame_boxes_resized, save_annotated)

    drop_target = max(1, int(round(cfg.DROP_RATIO * written)))
    drop_sorted = sorted(DROP, key=lambda T: (
        (T.s_min > 0.15) * 2.0 +
        (T.s_var > 0.1) * 1.5 +
        (3 <= len(T.dets) <= 8) * 1.0 +
        (T.night) * 1.0 +
        (T.a_med < cfg.SMALL_AREA*2) * 1.0 +
        (0.1 < T.J < 0.4) * 0.5
    ), reverse=True)
    drop_picked = drop_sorted[:min(cfg.DROP_ANALYSIS_BUDGET, len(drop_sorted))]

    update_memory_bank(dirs['logs'], picked, store_root, video_thumb_sizes, mem_bank)

    summary = dict(
        budgets=dict(B_TRACKS=cfg.B_TRACKS, B_FRAMES=cfg.B_FRAMES),
        totals=dict(tracks=len(all_tracks), auto=len(AUTO), label_candidates=len(LABEL), drop=len(DROP)),
        picked_tracks=len(picked),
        keyframes_written=written,
        drop_analysis_target=drop_target,
        params=dict(
            stride_k=stride_k, conf_thresh=eff_conf, assoc_iou=cfg.ASSOC_IOU,
            small_area=cfg.SMALL_AREA, abs_min_side_px=cfg.ABS_MIN_SIDE_PX,
            temporal_thin=cfg.TEMPORAL_THIN_FR, phash_hamming_max=cfg.PHASH_HAMMING_MAX,
            store_stride=effective_store_stride, write_thumbs=write_thumbs,
            save_annotated=save_annotated
        )
    )

    write_logs(dirs['logs'], summary, AUTO, LABEL, DROP, selected)

    if create_cvat:
        zip_path = create_cvat_export(out_dir, dirs['logs'])
        if zip_path:
            summary["cvat_export"] = zip_path

    print(json.dumps(summary, indent=2))
    print("Done.")

# ------------------------------- CLI ---------------------------------

def main():
    ap = argparse.ArgumentParser(description="Lean keyframe miner with FrameStore + vectorized tracking.")
    ap.add_argument("--videos_dir", required=True, help="Directory with video files")
    ap.add_argument("--out_dir", required=True, help="Output directory (images, labels, logs)")
    ap.add_argument("--model", default="yolov8n.pt", help="YOLO model path")
    ap.add_argument("--stride_k", type=int, default=3, help="Process every K-th frame (uses cached store frames)")
    ap.add_argument("--conf_thresh", type=float, default=None, help="Detection confidence threshold (filters cached dets)")
    ap.add_argument("--store_stride", type=int, default=None,
                    help="Stride for FrameStore build. Default: use --stride_k (match old sampling).")
    ap.add_argument("--no_thumbs", action="store_true",
                    help="Do not save thumbnails in the FrameStore (smaller/faster build).")
    ap.add_argument("--save_annotated", action="store_true",
                    help="Also save annotated images (with bounding boxes) for final selected frames.")
    ap.add_argument("--create_cvat", action="store_true",
                    help="Create CVAT-compatible dataset export (data.yaml, train.txt, and zip file).")
    args = ap.parse_args()

    warnings.filterwarnings("ignore", category=UserWarning)

    process(args.videos_dir, args.out_dir, args.model,
            args.stride_k, args.conf_thresh,
            store_stride=args.store_stride, write_thumbs=(not args.no_thumbs),
            save_annotated=args.save_annotated, create_cvat=args.create_cvat)

if __name__ == "__main__":
    main()
