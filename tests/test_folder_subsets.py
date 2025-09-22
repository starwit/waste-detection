"""Tests for per-folder subset configuration (under/oversampling).
This suite verifies that `process_single_images` honors the `folder_subsets`
ratios per source folder:
- Subsampling (ratio < 1.0) reduces the number of pairs taken from a folder
- Oversampling (ratio > 1.0) duplicates pairs in-memory (not files on disk)

The images include simple geometric patterns to avoid false-positive duplicate
removal by the smart dedup stage.
"""

from pathlib import Path
import cv2
import numpy as np

from yolov8_training.utils.data_utils import process_single_images


def test_over_under_sampling(tmp_path: Path):
    # Build minimal input structure: raw_data/train/{a,b}/{images,labels}
    # Folder 'a' has 4 images, folder 'b' has 2 images.
    # Each image gets a distinct pattern so the dedup stage will not remove it.
    input_root = tmp_path / "raw_data" / "train"
    for folder, count, base_val in [("a", 4, 25), ("b", 2, 200)]:
        images = input_root / folder / "images"
        labels = input_root / folder / "labels"
        images.mkdir(parents=True, exist_ok=True)
        labels.mkdir(parents=True, exist_ok=True)
        for i in range(count):
            img = np.full((64, 64, 3), base_val + i, dtype=np.uint8)
            # Add distinct patterns to avoid near-duplicate detection
            if folder == "a":
                cv2.rectangle(img, (5 + i, 5 + i), (30 + i, 30 + i), (255, 255, 255), -1)
            else:
                cv2.circle(img, (32, 32), 10 + i, (255, 255, 255), -1)
            cv2.imwrite(str(images / f"img_{i}.jpg"), img)
            with open(labels / f"img_{i}.txt", "w") as f:
                f.write("0 0.5 0.5 0.2 0.2\n")

    # Output roots
    train_out = tmp_path / "dataset" / "train"
    test_out = tmp_path / "dataset" / "test"

    # Subsample folder 'a' and oversample folder 'b', verify counts.
    folder_subsets = {"a": 0.5, "b": 2.0}

    train_count, val_count, test_count = process_single_images(
        input_path=input_root,
        train_output_path=train_out,
        test_output_path=test_out,
        val_split=0.0,
        test_split=0.0,
        augment_multiplier=1,
        folder_subsets=folder_subsets,
    )

    assert val_count == 0
    assert test_count == 0
    # Expect exactly 6 pairs in train after subset logic
    assert train_count == 6
