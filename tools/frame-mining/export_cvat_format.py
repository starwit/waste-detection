#!/usr/bin/env python3
"""
Export video mining output folders to CVAT-compatible ZIP format.

This script creates a ZIP file that can be imported into CVAT,
containing data.yaml, images/, and labels/ in the YOLO 1.1 format.

Class mapping (from mine_videos_sam3_yolo.py):
    0: waste
    1: cigarette
    2: leaf_pile
    3: leaf_region
"""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path
from textwrap import dedent


# Class names used in mine_videos_sam3_yolo.py
CLASS_NAMES = {
    0: "waste",
    1: "cigarette",
    2: "leaf_pile",
    3: "leaf_region_dense",
}


def create_data_yaml_content(class_names: dict[int, str]) -> str:
    """Create data.yaml content for YOLO/CVAT compatibility."""
    yaml_content = dedent("""\
        path: .
        train: images/train

        names:
    """)
    for idx in sorted(class_names.keys()):
        yaml_content += f"    {idx}: {class_names[idx]}\n"
    return yaml_content


def create_train_txt_content(image_files: list[Path]) -> str:
    """Create train.txt content listing all images."""
    lines = []
    for img_file in sorted(image_files):
        lines.append(f"images/train/{img_file.name}")
    return "\n".join(lines) + "\n"


def export_folder_to_zip(folder_path: Path, class_names: dict[int, str]) -> Path | None:
    """Export a single folder to CVAT-compatible ZIP format."""
    print(f"\nProcessing: {folder_path}")

    if not folder_path.exists():
        print(f"Error: Folder does not exist: {folder_path}")
        return None

    images_dir = folder_path / "images"
    labels_dir = folder_path / "labels"

    if not images_dir.exists():
        print(f"Warning: No images folder found in {folder_path}")
        return None

    if not labels_dir.exists():
        print(f"Warning: No labels folder found in {folder_path}")

    # Count files
    image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
    label_files = list(labels_dir.glob("*.txt")) if labels_dir.exists() else []
    
    if not image_files:
        print(f"Warning: No images found in {images_dir}")
        return None

    print(f"Found {len(image_files)} images, {len(label_files)} labels")

    # Create ZIP file
    zip_path = folder_path / "dataset.zip"
    data_yaml_content = create_data_yaml_content(class_names)
    train_txt_content = create_train_txt_content(image_files)

    with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_STORED) as zipf:
        # Add data.yaml
        zipf.writestr('data.yaml', data_yaml_content)

        # Add train.txt
        zipf.writestr('train.txt', train_txt_content)

        # Add images to images/train/ (CVAT YOLO format)
        for img_file in sorted(image_files):
            arcname = f"images/train/{img_file.name}"
            zipf.write(img_file, arcname=arcname)

        # Add labels to labels/train/
        if labels_dir.exists():
            for label_file in sorted(label_files):
                arcname = f"labels/train/{label_file.name}"
                zipf.write(label_file, arcname=arcname)

    print(f"Created ZIP archive: {zip_path}")
    print(f"  - {len(image_files)} images in images/train/")
    print(f"  - {len(label_files)} labels in labels/train/")
    print(f"  - data.yaml with classes: {list(class_names.values())}")
    print(f"  - train.txt with {len(image_files)} entries")

    return zip_path


def main():
    parser = argparse.ArgumentParser(
        description="Export video mining output folders to CVAT-compatible ZIP format",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent / "video_mining_output",
        help="Base output directory containing sam3_gated and false_positives folders",
    )
    parser.add_argument(
        "--folders",
        nargs="+",
        default=["sam3_gated", "false_positives"],
        help="Subfolders to process",
    )
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()

    if not output_dir.exists():
        print(f"Error: Output directory does not exist: {output_dir}")
        return

    print(f"Base directory: {output_dir}")
    print(f"Class names: {CLASS_NAMES}")

    created_zips = []
    for folder_name in args.folders:
        folder_path = output_dir / folder_name
        zip_path = export_folder_to_zip(folder_path, CLASS_NAMES)
        if zip_path:
            created_zips.append(zip_path)

    print(f"\n{'='*60}")
    print("Done! Created CVAT-compatible ZIP files:")
    for zp in created_zips:
        print(f"  - {zp}")
    print("\nYou can import these ZIP files directly into CVAT using:")
    print("  'Create from backup' or 'Upload annotations' with format 'YOLO 1.1'")


if __name__ == "__main__":
    main()
