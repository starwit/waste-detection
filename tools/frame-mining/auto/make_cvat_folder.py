#!/usr/bin/env python3
"""
build_train_and_yaml.py — CVAT/Ultralytics YOLO compatible

- Expects: <root>/images/, <root>/labels/ (any nesting)
- Creates/overwrites: train.txt (with images/train/... paths), data.yaml
- Ensures every image has a label file (creates empty if missing)
- Zips the dataset into <root>/<rootname>_dataset_YYYYmmdd_HHMMSS.zip with:
    data.yaml
    train.txt
    images/train/...
    labels/train/...

This matches CVAT's required layout for Ultralytics YOLO import.
"""

from __future__ import annotations
import argparse
from pathlib import Path
from datetime import datetime
import zipfile

# === EDIT YOUR DEFAULT CLASSES HERE ===
DEFAULT_CLASSES = ["waste", "cigarette"]

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
SUBSET = "train"  # fixed single-subset layout for CVAT

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create CVAT-ready Ultralytics YOLO dataset (train.txt, data.yaml, zip).")
    p.add_argument("root", nargs="?", default=".", help="Dataset root containing images/ and labels/ (default: current dir).")
    p.add_argument("--classes", default=None, help="Comma-separated class names to override defaults.")
    return p.parse_args()

def load_classes(arg: str | None) -> list[str]:
    if arg is None:
        return list(DEFAULT_CLASSES)
    names = [c.strip() for c in arg.split(",")]
    names = [c for c in names if c]
    if not names:
        raise SystemExit("No classes provided after parsing --classes.")
    return names

def find_images(images_dir: Path) -> list[Path]:
    files = []
    for ext in IMAGE_EXTS:
        files.extend(images_dir.rglob(f"*{ext}"))
    return sorted(set(files))

def write_yaml(path: Path, classes: list[str]) -> None:
    # Quote class names for YAML safety
    lines = [
        f"# Auto-generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "names:",
    ]
    for i, name in enumerate(classes):
        safe = name.replace("'", "''")
        lines.append(f"  {i}: '{safe}'")
    lines += [
        "path: .",
        "train: train.txt",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

def main():
    args = parse_args()
    root = Path(args.root).resolve()
    images_dir = root / "images"
    labels_dir = root / "labels"
    train_file = root / "train.txt"
    data_yaml = root / "data.yaml"

    classes = load_classes(args.classes)

    if not images_dir.is_dir():
        raise SystemExit(f"Images folder not found: {images_dir}")
    labels_dir.mkdir(parents=True, exist_ok=True)

    imgs = find_images(images_dir)
    if not imgs:
        raise SystemExit(f"No images found under {images_dir} with extensions: {sorted(IMAGE_EXTS)}")

    # Ensure label exists for every image; collect relative paths
    created_labels = 0
    rel_image_paths = []  # relative to <root>/images
    for img in imgs:
        rel_from_images = img.relative_to(images_dir)
        lbl_path = (labels_dir / rel_from_images).with_suffix(".txt")
        if not lbl_path.exists():
            lbl_path.parent.mkdir(parents=True, exist_ok=True)
            lbl_path.write_text("", encoding="utf-8")
            created_labels += 1
        rel_image_paths.append(str(rel_from_images).replace("\\", "/"))

    # Write train.txt with the subset prefix (CVAT expects images/<subset>/...)
    # Even if your on-disk files are under images/, the ZIP will remap to images/train/
    train_lines = [f"images/{SUBSET}/{p}" for p in rel_image_paths]
    train_file.write_text("\n".join(train_lines) + "\n", encoding="utf-8")

    # Write data.yaml (includes names + path + train)
    write_yaml(data_yaml, classes)

    # Create ZIP with remapped arcnames to include the subset level
    # ZIP structure will be:
    # - data.yaml (root level)
    # - train.txt (root level)  
    # - images/train/... (all image files)
    # - labels/train/... (all label .txt files)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_path = root / f"{root.name}_dataset_{ts}.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        # Add data.yaml and train.txt at archive root
        zf.write(data_yaml, arcname="data.yaml")
        zf.write(train_file, arcname="train.txt")

        # Add images -> images/train/...
        for img in imgs:
            rel_from_images = str(img.relative_to(images_dir)).replace("\\", "/")
            arc = f"images/{SUBSET}/{rel_from_images}"
            zf.write(img, arcname=arc)

        # Add labels -> labels/train/... (only .txt files)
        for lbl in labels_dir.rglob("*.txt"):
            rel_from_labels = str(lbl.relative_to(labels_dir)).replace("\\", "/")
            arc = f"labels/{SUBSET}/{rel_from_labels}"
            zf.write(lbl, arcname=arc)

    print(f"Images found: {len(imgs)}")
    print(f"Empty labels created: {created_labels}")
    print(f"Wrote: {train_file}")
    print(f"Wrote: {data_yaml}")
    print(f"Zipped dataset to: {zip_path}")
    
    # Clean up temporary files after successful zip creation
    try:
        train_file.unlink()  # Remove train.txt
        data_yaml.unlink()   # Remove data.yaml
        print(f"Cleaned up temporary files: {train_file.name}, {data_yaml.name}")
    except Exception as e:
        print(f"Warning: Could not clean up temporary files: {e}")

if __name__ == "__main__":
    main()
