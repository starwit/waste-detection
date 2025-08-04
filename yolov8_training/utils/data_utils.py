import json
import random
import os
from pathlib import Path
import cv2
import shutil
import numpy as np
import re
import yaml
import csv
from typing import List, Tuple

from yolov8_training.utils.find_duplicates import DuplicateDetector

from yolov8_training.utils.augmentation import YOLOAugmenter

COCO_CLASSES = {
    "person": 0,
    "bicycle": 1,
    "car": 2,
    "motorcycle": 3,
    "airplane": 4,
    "bus": 5,
    "train": 6,
    "truck": 7,
    "boat": 8,
    "traffic light": 9,
    "fire hydrant": 10,
    "stop sign": 11,
    "parking meter": 12,
    "bench": 13,
    "bird": 14,
    "cat": 15,
    "dog": 16,
    "horse": 17,
    "sheep": 18,
    "cow": 19,
    "elephant": 20,
    "bear": 21,
    "zebra": 22,
    "giraffe": 23,
    "backpack": 24,
    "umbrella": 25,
    "handbag": 26,
    "tie": 27,
    "suitcase": 28,
    "frisbee": 29,
    "skis": 30,
    "snowboard": 31,
    "sports ball": 32,
    "kite": 33,
    "baseball bat": 34,
    "baseball glove": 35,
    "skateboard": 36,
    "surfboard": 37,
    "tennis racket": 38,
    "bottle": 39,
    "wine glass": 40,
    "cup": 41,
    "fork": 42,
    "knife": 43,
    "spoon": 44,
    "bowl": 45,
    "banana": 46,
    "apple": 47,
    "sandwich": 48,
    "orange": 49,
    "broccoli": 50,
    "carrot": 51,
    "hot dog": 52,
    "pizza": 53,
    "donut": 54,
    "cake": 55,
    "chair": 56,
    "couch": 57,
    "potted plant": 58,
    "bed": 59,
    "dining table": 60,
    "toilet": 61,
    "tv": 62,
    "laptop": 63,
    "mouse": 64,
    "remote": 65,
    "keyboard": 66,
    "cell phone": 67,
    "microwave": 68,
    "oven": 69,
    "toaster": 70,
    "sink": 71,
    "refrigerator": 72,
    "book": 73,
    "clock": 74,
    "vase": 75,
    "scissors": 76,
    "teddy bear": 77,
    "hair drier": 78,
    "toothbrush": 79,
}

ALL_COCO_BY_ID = [name for name, _id in sorted(COCO_CLASSES.items(), key=lambda kv: kv[1])]
selected_classes = ALL_COCO_BY_ID
selected_coco_classes = {i: cls for i, cls in enumerate(selected_classes)}


def get_class_mapping(custom_classes=None, use_coco_classes=True):
    """
    Get the class mapping to use for the dataset.
    
    Args:
        custom_classes (list): List of custom class names
        use_coco_classes (bool): Whether to use COCO classes when custom_classes is empty
        
    Returns:
        dict: Class mapping {id: name}
    """
    if custom_classes:
        return {i: class_name for i, class_name in enumerate(custom_classes)}
    elif use_coco_classes:
        return selected_coco_classes
    else:
        return {}


def map_class_names_to_ids(class_names, target_mapping):
    """
    Map class names to target class IDs.
    
    Args:
        class_names (dict or list): Source class mapping {id: name} or list of names
        target_mapping (dict): Target class mapping {id: name}
        
    Returns:
        dict: Mapping from source IDs to target IDs
    """
    mapping = {}
    target_name_to_id = {name.lower(): id for id, name in target_mapping.items()}
    
    # Normalize class_names to dictionary format
    if isinstance(class_names, list):
        class_names_dict = {i: name for i, name in enumerate(class_names)}
    elif isinstance(class_names, dict):
        class_names_dict = class_names
    else:
        print(f"Warning: Unexpected format for class_names. Expected dict or list, got {type(class_names)}")
        return mapping
    
    for source_id, source_name in class_names_dict.items():
        source_name_lower = source_name.lower()
        if source_name_lower in target_name_to_id:
            mapping[int(source_id)] = target_name_to_id[source_name_lower]
            print(f"Mapped {source_name} ({source_id}) → {source_name} ({target_name_to_id[source_name_lower]})")
        else:
            print(f"Warning: No mapping found for class '{source_name}' (ID: {source_id})")
    
    return mapping


def sorted_iterdir(path):
    return sorted(path.iterdir(), key=lambda p: p.name)

def sorted_glob(pattern):
    return sorted(pattern, key=lambda p: p.name)
    
def check_for_test_images(test_image_input_path):
    test_exists = False
    if test_image_input_path.exists():
        for image_folder in sorted_iterdir(test_image_input_path):
            if image_folder.is_dir():
                test_exists = True
                break

    return test_exists


def remap_yaml_dataset_labels(dataset_dir: Path, target_class_mapping: dict) -> None:
    """
    Processes a dataset with data.yaml:
    Remaps labels to match target classes without moving files

    Args:
        dataset_dir: Path to the dataset directory containing data.yaml
        target_class_mapping: Target class mapping {id: name}
    """
    yaml_file = dataset_dir / "data.yaml"
    if not yaml_file.exists():
        return

    print(f"\nProcessing dataset in: {dataset_dir}")

    # Read yaml and create class mapping
    with open(yaml_file, "r") as f:
        dataset_config = yaml.safe_load(f)

    # Get mapping from source class names to target class IDs
    class_mapping = map_class_names_to_ids(dataset_config["names"], target_class_mapping)

    # Update all label files in the dataset directory recursively
    for label_file in dataset_dir.rglob("*.txt"):
        # Skip files that aren't in a 'labels' directory
        if "labels" not in str(label_file.parent):
            continue

        with open(label_file, "r") as f:
            lines = [line.strip().split() for line in f.readlines()]

        new_lines = []
        for parts in lines:
            if parts:
                orig_class_id = int(parts[0])
                if orig_class_id in class_mapping:
                    parts[0] = str(class_mapping[orig_class_id])
                    new_lines.append(" ".join(parts) + "\n")
                else:
                    # Skip labels for classes that couldn't be mapped
                    print(f"Skipping label with unmapped class ID: {orig_class_id}")

        with open(label_file, "w") as f:
            f.writelines(new_lines)


def create_dataset_yaml(dataset_path: Path, custom_classes=None, use_coco_classes=True):
    """
    Create dataset.yaml file with appropriate class mapping.
    
    Args:
        dataset_path: Path to the dataset directory
        custom_classes: List of custom class names
        use_coco_classes: Whether to use COCO classes when custom_classes is empty
    """
    # Get all parts of the path
    yaml_dataset_path = dataset_path.absolute()

    yaml_content = f"""
path: {yaml_dataset_path}
train: train/images
val: val/images

names:
"""
    
    class_mapping = get_class_mapping(custom_classes, use_coco_classes)
    for key, value in class_mapping.items():
        yaml_content += f"  {key}: {value}\n"

    with open(dataset_path / "dataset.yaml", "w") as f:
        f.write(yaml_content)
    
    print(f"Created dataset.yaml with {len(class_mapping)} classes: {list(class_mapping.values())}")


def ensure_equal_files(frames_path, labels_path):
    """
    Ensure there are equal numbers of frame and label files by removing excess files from the directory with more files.
    """
    frames_files = sorted(os.listdir(frames_path))
    labels_files = sorted(os.listdir(labels_path))

    excess_frames = frames_files[len(labels_files) :]
    excess_labels = labels_files[len(frames_files) :]

    for frame in excess_frames:
        os.remove(frames_path / frame)

    for label in excess_labels:
        os.remove(labels_path / label)


def move_images_to_output(
    train_images,
    train_labels,
    val_images,
    val_labels,
    train_output_path,
    val_output_path,
    video_name,
    nth_frame=1,
):
    """
    Handle moving of images and labels into training and validation directories,
    preserving original frame numbers in the file names. Only keeps every nth frame if specified.
    """

    def move_files(images, labels, output_path, nth_frame):
        for i, (image, label) in enumerate(zip(images, labels)):
            if i % nth_frame == 0:
                original_frame_number = int(image.stem.split("_")[-1])
                new_image_name = f"{video_name}_{original_frame_number:06d}.jpg"
                new_label_name = f"{video_name}_{original_frame_number:06d}.txt"
                shutil.move(image, output_path / "images" / new_image_name)
                shutil.move(label, output_path / "labels" / new_label_name)

    move_files(train_images, train_labels, train_output_path, nth_frame)
    move_files(val_images, val_labels, val_output_path, nth_frame)


def split_data(
    images: List[Path], labels: List[Path], val_split: float
) -> Tuple[List[Path], List[Path], List[Path], List[Path]]:
    """
    Split the data into training and validation sets using random sampling.

    Args:
        images: List of paths to image files
        labels: List of paths to label files
        val_split: Fraction of data to use for validation (0.0 to 1.0)

    Returns:
        Tuple of (train_images, train_labels, val_images, val_labels)
    """
    if val_split == 0:
        return images, labels, [], []

    if val_split == 1:
        return [], [], images, labels

    # Calculate validation set size
    val_size = int(len(images) * val_split)

    # Generate random indices for validation set
    val_indices = set(random.sample(range(len(images)), val_size))

    # Split images and labels
    train_images = [img for i, img in enumerate(images) if i not in val_indices]
    train_labels = [lbl for i, lbl in enumerate(labels) if i not in val_indices]
    val_images = [img for i, img in enumerate(images) if i in val_indices]
    val_labels = [lbl for i, lbl in enumerate(labels) if i in val_indices]

    return train_images, train_labels, val_images, val_labels


def augment(image_label_pairs, augment_multiplier=1):

    print(f"Applying augmentation with multiplier {augment_multiplier}...")
    augmenter = YOLOAugmenter(multiplier=augment_multiplier)
    augmented_pairs = []

    temp_folders = []

    # Only augment training data, not validation or test
    for img_path, label_path, scene_name in image_label_pairs:
        # Read image and labels
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        with open(label_path, "r") as f:
            labels = [list(map(float, line.strip().split())) for line in f.readlines()]

        # Get augmented versions
        augmented_results = augmenter.augment_image_and_labels(image, labels)

        # Save augmented versions
        for idx, (aug_image, aug_labels) in enumerate(augmented_results):
            if idx == 0:  # Original image
                continue

            aug_img_folder_path = (
                Path(f"{img_path.parent.parent}-aug") / img_path.parent.name
            )
            aug_label_folder_path = (
                Path(f"{label_path.parent.parent}-aug") / label_path.parent.name
            )

            if not os.path.exists(aug_img_folder_path):
                os.makedirs(aug_img_folder_path)

            if not os.path.exists(aug_label_folder_path):
                os.makedirs(aug_label_folder_path)

            temp_folders.append(aug_img_folder_path.parent)

            # Create new filenames for augmented data
            aug_img_path = (
                Path(f"{img_path.parent.parent}-aug")
                / img_path.parent.name
                / f"{img_path.stem}_aug{idx}{img_path.suffix}"
            )
            aug_label_path = (
                Path(f"{label_path.parent.parent}-aug")
                / label_path.parent.name
                / f"{label_path.stem}_aug{idx}{label_path.suffix}"
            )

            # Save augmented image
            aug_image_bgr = cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(aug_img_path), aug_image_bgr)

            # Save augmented labels
            with open(aug_label_path, "w") as f:
                for label in aug_labels:
                    f.write(" ".join(map(str, label)) + "\n")

            augmented_pairs.append((aug_img_path, aug_label_path, scene_name))

    # Add augmented pairs to original pairs
    print(f"Added {len(augmented_pairs)} augmented images")

    return augmented_pairs, list(set(temp_folders))


def process_single_images(
    input_path: Path,
    train_output_path: Path,
    test_output_path: Path,
    val_split: float,
    test_split: float,
    augment_multiplier: int,
    custom_classes=None,
    use_coco_classes=True,
):
    """
    Process single images and their labels, splitting them into train/val/test sets.
    Now handles both traditional folder structure and CVAT exports with train.txt
    
    Args:
        input_path: Path to input images
        train_output_path: Path to store training data
        test_output_path: Path to store test data
        val_split: Validation split ratio
        test_split: Test split ratio
        augment_multiplier: Augmentation multiplier
        custom_classes: List of custom class names
        use_coco_classes: Whether to use COCO classes when custom_classes is empty
    """
    # Create output directories
    train_img_output_path = train_output_path / "train" / "images"
    val_img_output_path = train_output_path / "val" / "images"
    test_img_output_path = test_output_path / "val" / "images"
    train_label_output_path = train_output_path / "train" / "labels"
    val_label_output_path = train_output_path / "val" / "labels"
    test_label_output_path = test_output_path / "val" / "labels"

    for path in [
        train_img_output_path,
        val_img_output_path,
        test_img_output_path,
        train_label_output_path,
        val_label_output_path,
        test_label_output_path,
    ]:
        os.makedirs(path, exist_ok=True)

    if not input_path.exists():
        print(f"Skipping {input_path} - not found.")
        return 0, 0, 0

    print(f"Processing images from {input_path}")

    # Collect all valid image-label pairs
    image_label_pairs = []
    temp_folders = []  # Keep track of temporary folders to clean up later
    empty_label_count = 0
    skip_count = 0

    # Track if we're processing test data
    is_test_data = (test_output_path == train_output_path) and (val_split == 1)
    
    # Get target class mapping for remapping
    target_class_mapping = get_class_mapping(custom_classes, use_coco_classes)

    for some_folder in sorted_iterdir(input_path):
        if not some_folder.is_dir():
            continue

        folder_to_process = some_folder
        scene_name = some_folder.name  # Extract scene name from folder name

        # Check if this is a CVAT export folder with train.txt
        train_txt = folder_to_process / "train.txt"
        if train_txt.exists():
            # Process CVAT export format
            with open(train_txt, "r") as f:
                image_paths = [line.strip() for line in f.readlines()]

            # Get the base directory for images and labels
            for image_rel_path in image_paths:
                # paths in the train.txt includes unnecessary 'data' folder prefix, which needs to be removed
                path = Path(image_rel_path)
                image_rel_path = Path(*path.parts[1:])

                # Convert relative path to absolute path
                image_path = folder_to_process / image_rel_path

                # Construct corresponding label path
                # Replace 'images' with 'labels' and change extension to .txt
                label_rel_path = Path("labels") / image_rel_path.relative_to(
                    "images"
                ).with_suffix(".txt")
                label_path = folder_to_process / label_rel_path

                 # Include image if it exists, even if label doesn't exist or is empty
                if image_path.exists():
                    if not label_path.exists() or label_path.stat().st_size == 0:
                        # Ensure parent directories exist
                        label_path.parent.mkdir(parents=True, exist_ok=True)
                        # Create empty label file if needed
                        with open(label_path, "w") as f:
                            pass
                        empty_label_count += 1

                    image_label_pairs.append((image_path, label_path, scene_name))
                else:
                    skip_count += 1

            # Remap labels if data.yaml exists
            if (folder_to_process / "data.yaml").exists():
                temp_folder = some_folder.parent / f"{some_folder.name}_temp"
                shutil.copytree(folder_to_process, temp_folder)
                remap_yaml_dataset_labels(temp_folder, target_class_mapping)
                temp_folders.append(temp_folder)

                # Update paths to use temp folder
                image_label_pairs = [
                    (
                        Path(
                            str(img_path).replace(
                                str(folder_to_process), str(temp_folder)
                            )
                        ),
                        Path(
                            str(lbl_path).replace(
                                str(folder_to_process), str(temp_folder)
                            )
                        ),
                        scene_name,
                    )
                    for img_path, lbl_path, scene_name in image_label_pairs
                ]
        else:
            # Manual dataset structure with images/ and labels/ folders
            images_folder = folder_to_process / "images"
            labels_folder = folder_to_process / "labels"

            if not images_folder.exists():
                print(f"Skipping {folder_to_process.name}: Missing 'images' folder.")
                continue

            # Create labels folder if it doesn't exist
            if not labels_folder.exists():
                labels_folder.mkdir(parents=True, exist_ok=True)

            # Collect valid image-label pairs
            temp_pairs = []
            for image_file in sorted_glob(images_folder.glob("*")):
                if image_file.is_file() and image_file.suffix.lower() in [
                    ".jpg",
                    ".jpeg",
                    ".png",
                    ".bmp",
                ]:
                    label_file = labels_folder / image_file.with_suffix(".txt").name

                    # Include all images, creating empty label files if needed
                    if not label_file.exists() or label_file.stat().st_size == 0:
                        with open(label_file, "w") as f:
                            pass
                        empty_label_count += 1

                    temp_pairs.append((image_file, label_file, scene_name))

            # Check if this manual structure has a data.yaml for class mapping
            data_yaml_path = folder_to_process / "data.yaml"
            if data_yaml_path.exists():
                print(f"Found data.yaml in manual structure: {folder_to_process.name}")
                
                # Validate the data.yaml has class names that can be mapped
                try:
                    with open(data_yaml_path, "r") as f:
                        yaml_config = yaml.safe_load(f)
                    
                    if "names" not in yaml_config:
                        raise ValueError("data.yaml missing 'names' section")
                    
                    # Check if we can map any classes
                    yaml_classes = yaml_config["names"]
                    mapping = map_class_names_to_ids(yaml_classes, target_class_mapping)
                    
                    if not mapping:
                        print(f"Warning: No classes in {data_yaml_path} can be mapped to target classes")
                        # Handle both dict and list formats for displaying source classes
                        if isinstance(yaml_classes, list):
                            source_classes = yaml_classes
                        else:
                            source_classes = list(yaml_classes.values())
                        print(f"Source classes: {source_classes}")
                        print(f"Target classes: {list(target_class_mapping.values())}")
                        # Still process the data but labels may be filtered out
                    
                    # Create temp folder and remap labels
                    temp_folder = some_folder.parent / f"{some_folder.name}_temp"
                    shutil.copytree(folder_to_process, temp_folder)
                    remap_yaml_dataset_labels(temp_folder, target_class_mapping)
                    temp_folders.append(temp_folder)

                    # Update paths to use temp folder
                    temp_pairs = [
                        (
                            Path(str(img_path).replace(str(folder_to_process), str(temp_folder))),
                            Path(str(lbl_path).replace(str(folder_to_process), str(temp_folder))),
                            scene_name,
                        )
                        for img_path, lbl_path, scene_name in temp_pairs
                    ]
                    
                except Exception as e:
                    print(f"Error processing data.yaml in {folder_to_process.name}: {e}")
                    print("Continuing without class mapping for this folder.")
            
            # Add all pairs from this folder
            image_label_pairs.extend(temp_pairs)

    print(f"Included {empty_label_count} images with empty labels (no objects).")
    print(f"Skipped {skip_count} images that couldn't be found.")

    if not image_label_pairs:
        print("No valid image-label pairs found.")
        # Clean up temp folders
        for temp_folder in temp_folders:
            shutil.rmtree(temp_folder)
        return 0, 0, 0

    print("Find duplicate images...")

    # Initialize duplicate detector
    detector = DuplicateDetector(phash_threshold=2, ssim_threshold=0.95)

    # Get all image paths
    image_paths = [img_path for img_path, _, _ in image_label_pairs]

    # Find and print duplicate clusters
    clusters = detector.find_duplicates(image_paths)

    detector.print_duplicate_clusters(clusters)

    # Get unique images
    unique_images = detector.get_unique_images(image_paths)

    # Filter image_label_pairs to keep only unique images
    filtered_pairs = [
        (img, lbl, scene)
        for img, lbl, scene in image_label_pairs
        if img in unique_images
    ]

    print(f"\nOriginal number of images: {len(image_label_pairs)}")
    print(f"Number of unique images after duplicate removal: {len(filtered_pairs)}")

    # Continue with the filtered pairs instead of original image_label_pairs
    image_label_pairs = filtered_pairs

    # add augmentation to the remaining images
    augmented_pairs, aug_temp_folders = augment(image_label_pairs, augment_multiplier)

    temp_folders.extend(aug_temp_folders)

    # Shuffle data
    random.shuffle(image_label_pairs)
    random.shuffle(augmented_pairs)

    # Calculate split indices
    total_images = len(image_label_pairs)
    test_size = int(total_images * test_split)
    val_size = int(total_images * val_split)

    # Split the data
    test_pairs = image_label_pairs[:test_size]
    val_pairs = image_label_pairs[test_size : test_size + val_size]
    train_pairs = image_label_pairs[test_size + val_size :]

    # only add augmentation to training data
    train_pairs.extend(augmented_pairs)

    # Helper function to copy files with scene information
    def copy_pairs(pairs, img_path, label_path):
        count = 0
        for img_file, label_file, scene_name in pairs:
            if is_test_data:
                # For test data, add scene suffix to filename
                new_img_name = f"{img_file.stem}__scene_{scene_name}{img_file.suffix}"
                new_label_name = (
                    f"{label_file.stem}__scene_{scene_name}{label_file.suffix}"
                )
            else:
                new_img_name = img_file.name
                new_label_name = label_file.name

            shutil.copy(img_file, img_path / new_img_name)
            shutil.copy(label_file, label_path / new_label_name)
            count += 1
        return count

    # Copy files to respective directories
    test_count = copy_pairs(test_pairs, test_img_output_path, test_label_output_path)
    val_count = copy_pairs(val_pairs, val_img_output_path, val_label_output_path)
    train_count = copy_pairs(
        train_pairs, train_img_output_path, train_label_output_path
    )

    # Clean up temp folders
    for temp_folder in temp_folders:
        if os.path.exists(temp_folder):
            shutil.rmtree(temp_folder)

    return train_count, val_count, test_count


def save_test_results(train_output_dir, test_results, metadata):
    # Save test results
    with open(train_output_dir / "test_metrics.json", "w") as f:
        json.dump(test_results, f, indent=4)

    # Prepare the data for CSV
    csv_data = {
        "MODEL": metadata["experiment_name"],
        **{
            k: v
            for k, v in test_results.items()
            if k
            in [
                "mp",
                "mr",
                "map",
                "map50",
                "map75",
                "fp",
                "fn",
                "tp",
                "fpr",
                "f1_score",
                "time",
            ]
        },
        "val_split": metadata["split_parameters"]["val_split"],
        "split_strategy": metadata["split_parameters"]["split_strategy"],
        "temporal_split_size": metadata["split_parameters"]["temporal_split_size"],
    }

    # Save results to CSV
    csv_path = train_output_dir / "test_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_data.keys())
        writer.writeheader()
        writer.writerow(csv_data)
    print(f"Test results saved to {train_output_dir / 'test_metrics.json'}")
    print(f"Results CSV saved to {csv_path}")


def reorganize_output(train_output_dir, training_path, test_path, metadata):
    # Save metadata
    with open(train_output_dir / "metadata.yaml", "w") as f:
        yaml.dump(metadata, f)

    # Copy dataset YAML files
    train_dataset_yaml_path = training_path / "dataset.yaml"
    test_dataset_yaml_path = test_path / "dataset.yaml"
    if train_dataset_yaml_path.exists():
        shutil.copy(train_dataset_yaml_path, train_output_dir / "train_dataset.yaml")
    if test_dataset_yaml_path.exists():
        shutil.copy(test_dataset_yaml_path, train_output_dir / "test_dataset.yaml")

    # Organize files after training
    plots_dir = train_output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    # Move plot files to a new plots directory
    for plot_file in list(sorted_glob(train_output_dir.glob("*.png"))) + list(
        sorted_glob(train_output_dir.glob("*.jpg"))
    ):
        shutil.move(str(plot_file), str(plots_dir / plot_file.name))

    print(f"Experiment data organized in {train_output_dir}")
    print(f"Plots saved to {plots_dir}")
