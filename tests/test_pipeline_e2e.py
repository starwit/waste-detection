"""End-to-end pipeline smoke test.

Creates a tiny dataset in a temp working directory, runs the prepare stage to
materialize YOLO datasets (including a dedicated test folder to exercise scene
metrics), then runs the train+eval stage. Verifies that results are generated
and that scene metrics are present and numeric in the results CSV.
"""

import os
import tempfile
import unittest
from pathlib import Path
import shutil
import csv

import cv2
import numpy as np

from yolov8_training.train_pipeline import run_prepare_stage, run_train_eval_stage


class PipelineE2ETestCase(unittest.TestCase):
    """Exercises run_prepare_stage and run_train_eval_stage with minimal data."""
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.cwd = Path.cwd()
        self.temp_path = Path(self.temp_dir.name)
        os.chdir(self.temp_path)

        # Build raw_data/train/source1 with a couple of images
        # Training folder 1
        images_dir = self.temp_path / "raw_data" / "train" / "source1" / "images"
        labels_dir = self.temp_path / "raw_data" / "train" / "source1" / "labels"
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)

        # One image with a box
        img1 = np.full((128, 128, 3), 100, dtype=np.uint8)
        cv2.rectangle(img1, (32, 32), (96, 96), (255, 255, 255), -1)
        cv2.imwrite(str(images_dir / "img1.jpg"), img1)
        with open(labels_dir / "img1.txt", "w") as f:
            f.write("0 0.5 0.5 0.5 0.5\n")

        # Second image with labels, too (keep pipeline validation happy)
        img2 = np.full((128, 128, 3), 150, dtype=np.uint8)
        cv2.imwrite(str(images_dir / "img2.jpg"), img2)
        with open(labels_dir / "img2.txt", "w") as f:
            f.write("0 0.5 0.5 0.3 0.3\n")

        # Training folder 2 (ensure multiple folders are handled)
        images_dir2 = self.temp_path / "raw_data" / "train" / "source2" / "images"
        labels_dir2 = self.temp_path / "raw_data" / "train" / "source2" / "labels"
        images_dir2.mkdir(parents=True, exist_ok=True)
        labels_dir2.mkdir(parents=True, exist_ok=True)
        img3 = np.full((128, 128, 3), 60, dtype=np.uint8)
        cv2.line(img3, (0, 0), (127, 127), (255, 255, 255), 3)
        cv2.imwrite(str(images_dir2 / "img3.jpg"), img3)
        with open(labels_dir2 / "img3.txt", "w") as f:
            f.write("0 0.5 0.5 0.2 0.2\n")

        # Also create a tiny dedicated test set (ensures scene metrics suffix)
        test_images_dir = self.temp_path / "raw_data" / "test" / "sourceT" / "images"
        test_labels_dir = self.temp_path / "raw_data" / "test" / "sourceT" / "labels"
        test_images_dir.mkdir(parents=True, exist_ok=True)
        test_labels_dir.mkdir(parents=True, exist_ok=True)
        imgT = np.full((128, 128, 3), 80, dtype=np.uint8)
        cv2.circle(imgT, (64, 64), 20, (255, 255, 255), -1)
        cv2.imwrite(str(test_images_dir / "test1.jpg"), imgT)
        with open(test_labels_dir / "test1.txt", "w") as f:
            f.write("0 0.5 0.5 0.25 0.25\n")

        # Create dataset root folders (created by pipeline anyway)
        (self.temp_path / "datasets").mkdir(exist_ok=True)

    def tearDown(self):
        os.chdir(self.cwd)
        self.temp_dir.cleanup()

    def test_pipeline_end_to_end(self):
        # Prepare args
        class Args:
            stage = None
            seed = 42
            dataset_name = "e2e_dataset"
            model_size = "n"
            image_size = 320
            epochs = 1
            batch_size = 1
            val_split = 0.5
            test_split = 0.0
            recreate_dataset = True
            augment_multiplier = 1
            folder_subset = None

        # Run prepare and then train/eval
        run_prepare_stage(Args)
        run_train_eval_stage(Args)

        # Validate outputs exist
        # 1) Datasets created
        self.assertTrue((self.temp_path / "datasets" / Args.dataset_name).exists())
        # 2) Results written by mean_table and include scene metric column
        results_csv = self.temp_path / "results_comparison" / "results.csv"
        self.assertTrue(results_csv.exists())
        scene_name = "sourceT"
        col_name = f"scene_{scene_name}_fitness"
        with open(results_csv, newline="") as f:
            reader = csv.DictReader(f)
            header = reader.fieldnames or []
            rows = list(reader)
        # Header must contain the scene metric column
        self.assertIn(col_name, header)
        # And values in that column should be numeric (>= 0)
        for row in rows:
            val_str = (row.get(col_name) or "").strip()
            # tolerate missing or '-' for base rows without metrics
            if val_str and val_str != "-":
                try:
                    val = float(val_str)
                    self.assertGreaterEqual(val, 0.0)
                except ValueError:
                    self.fail(f"{col_name} is not numeric: {val_str}")


if __name__ == "__main__":
    unittest.main()
