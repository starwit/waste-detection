"""Tests for CVAT split file path handling.

CVAT exports sometimes write split files (e.g., train.txt) where image paths
are recorded with a leading "data/" prefix, like "data/images/img1.jpg".
This suite verifies that `process_single_images` correctly resolves those paths
relative to the source folder, copies images into the YOLO train subset, and
creates empty label files when they are missing.
"""

import os
import tempfile
import unittest
from pathlib import Path
import cv2
import numpy as np

from yolov8_training.utils.data_utils import process_single_images, create_dataset_yaml


class CVATPathsTestCase(unittest.TestCase):
    """Builds a minimal CVAT-like export and checks path resolution."""
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

        # Simulate a CVAT export structure under raw_data/train/source
        self.input_root = self.temp_path / "raw_data" / "train" / "source_cvat"
        (self.input_root / "images").mkdir(parents=True, exist_ok=True)
        (self.input_root / "labels").mkdir(parents=True, exist_ok=True)

        # Create two images
        img1 = np.full((64, 64, 3), 128, dtype=np.uint8)
        img2 = np.full((64, 64, 3), 200, dtype=np.uint8)
        cv2.imwrite(str(self.input_root / "images" / "img1.jpg"), img1)
        cv2.imwrite(str(self.input_root / "images" / "img2.jpg"), img2)

        # CVAT-style train.txt listing with extra 'data' prefix
        train_txt = self.input_root / "train.txt"
        train_txt.write_text("\n".join([
            "data/images/img1.jpg",
            "data/images/img2.jpg",
        ]))

        # Output dataset folders
        self.train_out = self.temp_path / "dataset" / "train"
        self.test_out = self.temp_path / "dataset" / "test"
        (self.train_out / "train" / "images").mkdir(parents=True, exist_ok=True)
        (self.train_out / "train" / "labels").mkdir(parents=True, exist_ok=True)
        (self.train_out / "val" / "images").mkdir(parents=True, exist_ok=True)
        (self.train_out / "val" / "labels").mkdir(parents=True, exist_ok=True)
        (self.test_out / "val" / "images").mkdir(parents=True, exist_ok=True)
        (self.test_out / "val" / "labels").mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_cvat_train_txt_paths(self):
        """Resolve CVAT-style paths and ensure images+labels land in train.

        Expectations:
        - train.txt lines like "data/images/…" are resolved to existing files
        - two images are copied into train/images
        - corresponding label files exist (created empty if missing)
        - no val/test split applied
        """
        train_count, val_count, test_count = process_single_images(
            input_path=self.input_root.parent,  # one level up to include source_cvat
            train_output_path=self.train_out,
            test_output_path=self.test_out,
            val_split=0.0,
            test_split=0.0,
            augment_multiplier=1,
        )

        self.assertEqual(train_count, 2)
        self.assertEqual(val_count, 0)
        self.assertEqual(test_count, 0)

        # Check images and labels copied
        copied_imgs = list((self.train_out / "train" / "images").glob("*.jpg"))
        copied_lbls = list((self.train_out / "train" / "labels").glob("*.txt"))
        self.assertEqual(len(copied_imgs), 2)
        self.assertEqual(len(copied_lbls), 2)

        # Ensure label files exist and are possibly empty
        for img_path in copied_imgs:
            lbl = (self.train_out / "train" / "labels" / img_path.with_suffix(".txt").name)
            self.assertTrue(lbl.exists())


if __name__ == "__main__":
    unittest.main()
