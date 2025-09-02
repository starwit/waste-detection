"""Duplicate detection smoke test.

Creates two identical images and one distinct image, runs DuplicateDetector
with reasonably strict thresholds, and verifies that duplicates are clustered
and the number of unique images is computed as expected.
"""

import tempfile
import unittest
from pathlib import Path
import shutil

import numpy as np
import cv2

from yolov8_training.utils.find_duplicates import DuplicateDetector


class DuplicateDetectorTestCase(unittest.TestCase):
    """Checks phash/SSIM clustering and unique selection on a tiny set."""
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)
        self.images_dir = self.temp_path / "images"
        self.images_dir.mkdir(parents=True, exist_ok=True)

        # Create two identical images and one different image
        img1 = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.rectangle(img1, (10, 10), (90, 90), (255, 0, 0), -1)
        img2 = img1.copy()  # duplicate
        img3 = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.circle(img3, (50, 50), 30, (0, 255, 0), -1)

        cv2.imwrite(str(self.images_dir / "a.jpg"), img1)
        cv2.imwrite(str(self.images_dir / "b.jpg"), img2)
        cv2.imwrite(str(self.images_dir / "c.jpg"), img3)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_detect_duplicates(self):
        """Detect duplicates and validate cluster size and unique count."""
        detector = DuplicateDetector(phash_threshold=2, ssim_threshold=0.95)
        image_paths = list(self.images_dir.glob("*.jpg"))
        clusters = detector.find_duplicates(image_paths)

        # Expect exactly one cluster containing 2 images (the duplicates)
        total_clustered = sum(len(v) for v in clusters.values())
        self.assertTrue(total_clustered >= 2)

        # Unique images should be 2 (one from duplicate pair + the distinct one)
        uniques = detector.get_unique_images(image_paths)
        self.assertEqual(len(uniques), 2)


if __name__ == "__main__":
    unittest.main()
