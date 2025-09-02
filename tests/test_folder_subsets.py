"""Tests for per-folder subset configuration (under/oversampling).

This suite verifies that `process_single_images` honors the `folder_subsets`
ratios per source folder:
- Subsampling (ratio < 1.0) reduces the number of pairs taken from a folder
- Oversampling (ratio > 1.0) duplicates pairs in-memory (not files on disk)

The images include simple geometric patterns to avoid false-positive duplicate
removal by the smart dedup stage.
"""

import tempfile
import unittest
from pathlib import Path
import cv2
import numpy as np

from yolov8_training.utils.data_utils import process_single_images


class FolderSubsetsTestCase(unittest.TestCase):
    """Builds two folders (a, b) and checks subset math on returned counts."""
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

        # Build minimal input structure: raw_data/train/{a,b}/{images,labels}
        # Folder 'a' has 4 images, folder 'b' has 2 images.
        # Each image gets a distinct pattern so the dedup stage will not remove it.
        self.input_root = self.temp_path / "raw_data" / "train"
        for folder, count, base_val in [("a", 4, 25), ("b", 2, 200)]:
            images = self.input_root / folder / "images"
            labels = self.input_root / folder / "labels"
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
        self.train_out = self.temp_path / "dataset" / "train"
        self.test_out = self.temp_path / "dataset" / "test"

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_over_under_sampling(self):
        """Subsample folder 'a' and oversample folder 'b', verify counts.

        Expectations:
        - folder a: 4 images with ratio 0.5 -> 2 pairs kept
        - folder b: 2 images with ratio 2.0 -> 4 pairs used (oversampled)
        - No val/test split and no augmentation -> total train_count = 6
        Note: oversampling duplicates pairs logically; it does not create extra
        files on disk with new filenames.
        """
        folder_subsets = {"a": 0.5, "b": 2.0}

        train_count, val_count, test_count = process_single_images(
            input_path=self.input_root,
            train_output_path=self.train_out,
            test_output_path=self.test_out,
            val_split=0.0,
            test_split=0.0,
            augment_multiplier=1,
            folder_subsets=folder_subsets,
        )

        self.assertEqual(val_count, 0)
        self.assertEqual(test_count, 0)
        # Expect exactly 6 pairs in train after subset logic
        self.assertEqual(train_count, 6)


if __name__ == "__main__":
    unittest.main()
