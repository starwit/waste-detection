import os
import shutil
import tempfile
import unittest
from pathlib import Path
import cv2
import numpy as np

from yolov8_training.utils.data_utils import process_single_images, create_dataset_yaml
from yolov8_training.train_pipeline import train_model


class PipelineTestCase(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

        # Create raw_data structure
        self.raw_data = self.temp_path / "raw_data"
        self.train_raw = self.raw_data / "train"
        self.train_raw.mkdir(parents=True, exist_ok=True)

        # Create source folder with images and labels
        self.source = self.train_raw / "source1"
        self.images_dir = self.source / "images"
        self.labels_dir = self.source / "labels"
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.labels_dir.mkdir(parents=True, exist_ok=True)

        # Create 5 unique dummy images with different patterns
        num_images = 5
        for i in range(1, num_images + 1):
            img_filename = f"image{i}.jpg"
            img_path = self.images_dir / img_filename
            # Create a unique pattern for each image
            np.random.seed(i * 100)  # Different seed for each image
            dummy_img = np.random.randint(0, 256, (640, 640, 3), dtype=np.uint8)
            # Add some distinct patterns to make images even more different
            cv2.rectangle(
                dummy_img,
                (50 * i, 50 * i),
                (200 + i * 20, 200 + i * 20),
                (i * 50, 255 - i * 40, i * 30),
                -1,
            )
            cv2.imwrite(str(img_path), dummy_img)

            if i in [1, 4, 5]:
                # Create a valid label file with one bounding box
                label_path = self.labels_dir / f"image{i}.txt"
                with open(label_path, "w") as f:
                    f.write("0 0.5 0.5 0.2 0.2\n")
            elif i == 3:
                # Create an empty label file
                label_path = self.labels_dir / f"image{i}.txt"
                with open(label_path, "w") as f:
                    f.write("")
            # For image2: no label file is created

        # Create output directories
        self.processed_output = self.temp_path / "processed_dataset"
        self.train_output_path = self.processed_output / "train"
        self.test_output_path = self.processed_output / "test"

        # Create the dataset directory that will be used for training
        self.dataset_dir = self.temp_path / "datasets" / "test_dataset"
        self.dataset_dir.mkdir(parents=True, exist_ok=True)

        # Create the required directory structure for YOLO training
        (self.dataset_dir / "train" / "images").mkdir(parents=True, exist_ok=True)
        (self.dataset_dir / "train" / "labels").mkdir(parents=True, exist_ok=True)
        (self.dataset_dir / "val" / "images").mkdir(parents=True, exist_ok=True)
        (self.dataset_dir / "val" / "labels").mkdir(parents=True, exist_ok=True)

        # Create a proper dataset YAML using the utility function
        create_dataset_yaml(self.dataset_dir)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_process_single_images(self):
        # Process images with no validation or test split
        train_count, val_count, test_count = process_single_images(
            input_path=self.train_raw,
            train_output_path=self.train_output_path,
            test_output_path=self.test_output_path,
            val_split=0.0,
            test_split=0.0,
            augment_multiplier=1
        )

        # We expect all 5 images to be processed
        self.assertEqual(train_count, 5)
        self.assertEqual(val_count, 0)
        self.assertEqual(test_count, 0)

        # Check that all images have corresponding label files
        train_images_dir = self.train_output_path / "train" / "images"
        train_labels_dir = self.train_output_path / "train" / "labels"
        train_images = list(train_images_dir.glob("*.jpg"))
        train_labels = list(train_labels_dir.glob("*.txt"))
        self.assertEqual(len(train_images), 5)
        self.assertEqual(len(train_labels), 5)

        # Verify content of label files
        for image_file in train_images:
            label_file = train_labels_dir / image_file.with_suffix(".txt").name
            with open(label_file, "r") as f:
                content = f.read().strip()
            if "image2" in image_file.name or "image3" in image_file.name:
                self.assertEqual(
                    content, "", f"Label for {image_file.name} should be empty"
                )
            else:
                self.assertNotEqual(
                    content, "", f"Label for {image_file.name} should not be empty"
                )

    def test_train_model_minimal(self):
        # First populate the dataset directory with some sample data
        # Copy a few images and labels to the train and val directories
        src_img = self.images_dir / "image1.jpg"
        src_label = self.labels_dir / "image1.txt"

        # Add one image with objects to train
        shutil.copy(src_img, self.dataset_dir / "train" / "images" / "image1.jpg")
        shutil.copy(src_label, self.dataset_dir / "train" / "labels" / "image1.txt")

        # Add one image without objects to train
        empty_img = self.images_dir / "image2.jpg"
        shutil.copy(empty_img, self.dataset_dir / "train" / "images" / "image2.jpg")
        # Create empty label file
        with open(self.dataset_dir / "train" / "labels" / "image2.txt", "w") as f:
            pass

        # Add an image to validation set as well
        shutil.copy(src_img, self.dataset_dir / "val" / "images" / "image1.jpg")
        shutil.copy(src_label, self.dataset_dir / "val" / "labels" / "image1.txt")

        try:
            # Try a minimal training run
            model, results, train_output_dir = train_model(
                dataset_path=self.dataset_dir,
                model_size="n",  # Use smallest model
                image_size=320,  # Use small image size for faster test
                batch_size=1,
                experiment_name="test_experiment",
                epochs=1
            )
            # If we get here, the training ran without error
            self.assertTrue(train_output_dir.exists())
        except Exception as e:
            self.fail(f"train_model failed with exception: {e}")


if __name__ == "__main__":
    unittest.main()
