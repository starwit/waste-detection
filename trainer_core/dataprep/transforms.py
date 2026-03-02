import random
import os
from pathlib import Path
import cv2
import shutil
import numpy as np
import yaml
from typing import List, Dict, Set

from trainer_core.dataprep.types import (
    ImageLabelPair,
    ProcessedFolder,
)

from trainer_core.dataprep.augmentation import YOLOAugmenter
from trainer_core.dataprep.find_duplicates import DuplicateDetector

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


def apply_class_mapping_config(custom_classes, class_mapping_config):
    """
    Apply class mapping configuration to merge classes together.
    
    This function takes the original custom_classes list and a mapping configuration,
    then returns a new class list where source classes are mapped to target classes.
    
    Args:
        custom_classes (list): Original list of class names
        class_mapping_config (dict): Mapping of target_class -> [source_classes]
                                     Example: {"waste": ["waste", "cigarette"]}
    
    Returns:
        tuple: (mapped_classes, source_to_target_map)
            - mapped_classes: List of unique target class names
            - source_to_target_map: Dict mapping source class name -> target class name
    """
    if not class_mapping_config or not custom_classes:
        return custom_classes, {}
    
    # Build source -> target mapping
    source_to_target = {}
    for target_class, source_classes in class_mapping_config.items():
        if isinstance(source_classes, list):
            for source_class in source_classes:
                source_to_target[source_class] = target_class
        else:
            # Handle case where a single string is provided instead of a list
            source_to_target[source_classes] = target_class
    
    # Build the mapped class list (only unique target classes)
    mapped_classes = []
    seen_targets = set()
    
    for original_class in custom_classes:
        # Map to target class if mapping exists, otherwise keep original
        target_class = source_to_target.get(original_class, original_class)
        if target_class not in seen_targets:
            mapped_classes.append(target_class)
            seen_targets.add(target_class)
    
    print(f"\nClass mapping applied:")
    print(f"  Original classes: {custom_classes}")
    print(f"  Mapped classes: {mapped_classes}")
    print(f"  Mapping rules: {source_to_target}")
    
    return mapped_classes, source_to_target


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


def remap_labels_with_class_mapping(label_path: Path, source_to_target_map: dict, 
                                    original_class_list: list, final_class_list: list) -> None:
    """
    Remap class IDs in label files based on class mapping configuration.
    
    This function reads YOLO format labels and remaps class IDs when classes are merged.
    For example, if 'cigarette' (class 1) is mapped to 'waste' (class 0), all labels
    with class ID 1 will be changed to class ID 0.
    
    Args:
        label_path: Path to the label file
        source_to_target_map: Dict mapping source class name -> target class name
        original_class_list: Original list of class names (to map IDs to names)
        final_class_list: Final list of class names after mapping (to get target IDs)
    """
    if not label_path.exists() or label_path.stat().st_size == 0:
        return
    
    # Build ID mappings
    # original_class_list has the classes in order: [waste, cigarette]
    # final_class_list has the unique target classes: [waste]
    
    # Map: source_class_id -> target_class_id
    id_mapping = {}
    for source_id, source_class_name in enumerate(original_class_list):
        target_class_name = source_to_target_map.get(source_class_name, source_class_name)
        if target_class_name in final_class_list:
            target_id = final_class_list.index(target_class_name)
            id_mapping[source_id] = target_id
    
    # Read and remap labels
    with open(label_path, 'r') as f:
        lines = f.readlines()
    
    remapped_lines = []
    modified = False
    
    for line in lines:
        parts = line.strip().split()
        if not parts:
            continue
        
        original_class_id = int(parts[0])
        if original_class_id in id_mapping:
            new_class_id = id_mapping[original_class_id]
            if new_class_id != original_class_id:
                modified = True
            parts[0] = str(new_class_id)
        
        remapped_lines.append(' '.join(parts) + '\n')
    
    # Only write if something changed
    if modified:
        with open(label_path, 'w') as f:
            f.writelines(remapped_lines)


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


# ------------------------------- helpers ---------------------------------

def _apply_subset_sampling(
    folder_name: str,
    pairs: List[ImageLabelPair],
    subset_ratio
) -> List[ImageLabelPair]:
    """Apply per-folder sampling/oversampling.

    subset_ratio can be:
    - float >1.0   → oversample (handled later in _oversample_train_pairs)
    - float 0..1   → proportional subsample
    - int / numeric string → absolute count (deterministic with the set seed)
    """
    # Absolute count support ("200" or 200). Treat 1 as 100% (not absolute 1).
    if isinstance(subset_ratio, str) and subset_ratio.isdigit():
        count = int(subset_ratio)
        if count == 1:
            subset_ratio = 1.0
        elif count >= 2:
            if count <= 0:
                print(f"Warning: Invalid absolute count {subset_ratio} for '{folder_name}'.")
                return pairs
            pairs_copy = pairs.copy()
            random.shuffle(pairs_copy)
            take = min(count, len(pairs_copy))
            print(f"Folder '{folder_name}': Using exactly {take} images (absolute count)")
            return pairs_copy[:take]

    if isinstance(subset_ratio, int):
        count = int(subset_ratio)
        if count == 1:
            subset_ratio = 1.0
        elif count >= 2:
            pairs_copy = pairs.copy()
            random.shuffle(pairs_copy)
            take = min(count, len(pairs_copy))
            print(f"Folder '{folder_name}': Using exactly {take} images (absolute count)")
            return pairs_copy[:take]

    # Float logic (ratios)
    if isinstance(subset_ratio, float) and subset_ratio > 1:
        print(
            f"Folder '{folder_name}': Oversampling requested "
            f"({subset_ratio*100:.1f}%), applied to training split only"
        )
        return pairs

    elif isinstance(subset_ratio, float) and 0 < subset_ratio < 1:
        original_count = len(pairs)
        pairs_copy = pairs.copy()
        random.shuffle(pairs_copy)
        subset_count = int(len(pairs) * subset_ratio)
        print(
            f"Folder '{folder_name}': Using {subset_count}/"
            f"{original_count} images ({subset_ratio*100:.1f}%)"
        )
        return pairs_copy[:subset_count]

    elif subset_ratio == 1 or subset_ratio == 1.0:
        print(
            f"Folder '{folder_name}': Using all "
            f"{len(pairs)} images (100%)"
        )
        return pairs

    else:
        print(
            f"Warning: Invalid subset ratio {subset_ratio} "
            f"for folder '{folder_name}'. Must be > 0."
        )
        return pairs


def _process_cvat_folder(
    some_folder: Path,
    folder_to_process: Path,
    scene_name: str,
    target_class_mapping: Dict[int, str],
    folder_subsets: Dict[str, int | float],
    temp_folders: List[Path],
) -> ProcessedFolder:
    folder_pairs: List[ImageLabelPair] = []
    empty_label_count = 0
    skip_count = 0

    train_txt = folder_to_process / "train.txt"
    with open(train_txt, "r") as f:
        image_paths = [line.strip() for line in f.readlines()]

    for image_rel_path in image_paths:
        path = Path(image_rel_path)
        image_rel_path = Path(*path.parts[1:])  # remove leading 'data'

        image_path = folder_to_process / image_rel_path

        label_rel_path = Path("labels") / image_rel_path.relative_to("images").with_suffix(".txt")
        label_path = folder_to_process / label_rel_path

        if image_path.exists():
            if not label_path.exists() or label_path.stat().st_size == 0:
                label_path.parent.mkdir(parents=True, exist_ok=True)
                label_path.touch()
                empty_label_count += 1
            convert_polygons_to_bboxes_inplace(label_path)
            folder_pairs.append(ImageLabelPair(image_path, label_path, scene_name))
        else:
            skip_count += 1

    if (folder_to_process / "data.yaml").exists():
        temp_folder = some_folder.parent / f"{some_folder.name}_temp"
        if temp_folder.exists():
            shutil.rmtree(temp_folder)
        shutil.copytree(folder_to_process, temp_folder)
        try:
            remap_yaml_dataset_labels(temp_folder, target_class_mapping)
        except (KeyError, TypeError, ValueError, yaml.YAMLError) as e:
            shutil.rmtree(temp_folder, ignore_errors=True)
            raise ValueError(
                f"Failed to apply class mapping for CVAT folder '{folder_to_process.name}': {e}"
            ) from e

        temp_folders.append(temp_folder)
        folder_pairs = [
            ImageLabelPair(
                temp_folder.joinpath(p.image.relative_to(folder_to_process)),
                temp_folder.joinpath(p.label.relative_to(folder_to_process)),
                p.scene,
            )
            for p in folder_pairs
        ]

    folder_name = some_folder.name
    if folder_name in folder_subsets:
        folder_pairs = _apply_subset_sampling(
            folder_name, folder_pairs, folder_subsets[folder_name]
        )

    return ProcessedFolder(folder_pairs, temp_folders, empty_label_count, skip_count)


def _process_manual_folder(
    some_folder: Path,
    folder_to_process: Path,
    scene_name: str,
    target_class_mapping: Dict[int, str],
    folder_subsets: Dict[str, int | float],
    temp_folders: List[Path],
) -> ProcessedFolder:
    temp_pairs: List[ImageLabelPair] = []
    empty_label_count = 0

    images_folder = folder_to_process / "images"
    labels_folder = folder_to_process / "labels"
    if not images_folder.exists():
        print(f"Skipping {folder_to_process.name}: Missing 'images' folder.")
        return ProcessedFolder([], temp_folders, empty_label_count)

    if not labels_folder.exists():
        labels_folder.mkdir(parents=True, exist_ok=True)

    for image_file in sorted_glob(images_folder.glob("*")):
        if image_file.is_file() and image_file.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]:
            label_file = labels_folder / image_file.with_suffix(".txt").name
            if not label_file.exists() or label_file.stat().st_size == 0:
                label_file.touch()
                empty_label_count += 1
            convert_polygons_to_bboxes_inplace(label_file)
            temp_pairs.append(ImageLabelPair(image_file, label_file, scene_name))

    data_yaml_path = folder_to_process / "data.yaml"
    if data_yaml_path.exists():
        print(f"Found data.yaml in manual structure: {folder_to_process.name}")
        temp_folder = some_folder.parent / f"{some_folder.name}_temp"
        with open(data_yaml_path, "r") as f:
            yaml_config = yaml.safe_load(f) or {}
        if not isinstance(yaml_config, dict):
            raise ValueError(
                f"Invalid data.yaml in '{folder_to_process.name}': expected mapping, got {type(yaml_config)}"
            )
        if "names" not in yaml_config:
            raise ValueError(f"Invalid data.yaml in '{folder_to_process.name}': missing 'names' section")

        yaml_classes = yaml_config["names"]
        mapping = map_class_names_to_ids(yaml_classes, target_class_mapping)
        if not mapping:
            print(f"Warning: No classes in {data_yaml_path} can be mapped to target classes")
            if isinstance(yaml_classes, list):
                source_classes = yaml_classes
            else:
                source_classes = list(yaml_classes.values())
            print(f"Source classes: {source_classes}")
            print(f"Target classes: {list(target_class_mapping.values())}")

        if temp_folder.exists():
            shutil.rmtree(temp_folder)
        shutil.copytree(folder_to_process, temp_folder)
        try:
            remap_yaml_dataset_labels(temp_folder, target_class_mapping)
        except (KeyError, TypeError, ValueError, yaml.YAMLError) as e:
            shutil.rmtree(temp_folder, ignore_errors=True)
            raise ValueError(
                f"Failed to apply class mapping for folder '{folder_to_process.name}': {e}"
            ) from e

        temp_folders.append(temp_folder)
        temp_pairs = [
            ImageLabelPair(
                temp_folder.joinpath(p.image.relative_to(folder_to_process)),
                temp_folder.joinpath(p.label.relative_to(folder_to_process)),
                p.scene,
            )
            for p in temp_pairs
        ]

    folder_name = some_folder.name
    if folder_name in folder_subsets:
        temp_pairs = _apply_subset_sampling(
            folder_name, temp_pairs, folder_subsets[folder_name]
        )

    return ProcessedFolder(temp_pairs, temp_folders, empty_label_count)


def _dedupe_pairs(
    image_label_pairs: List[ImageLabelPair],
    folder_subsets: Dict[str, int | float]
) -> List[ImageLabelPair]:
    detector = DuplicateDetector(phash_threshold=2, ssim_threshold=0.95)
    folder_groups: Dict[str, List[ImageLabelPair]] = {}
    for img_path, lbl_path, scene_name in image_label_pairs:
        folder_groups.setdefault(scene_name, []).append(
            ImageLabelPair(img_path, lbl_path, scene_name)
        )

    def _is_oversample(v) -> bool:
        return isinstance(v, float) and v > 1.0

    oversampled_folders: Set[str] = {
        folder_name for folder_name, ratio in folder_subsets.items() if _is_oversample(ratio)
    }

    processed_pairs: List[ImageLabelPair] = []
    unique_images_per_folder: Dict[str, Set[Path]] = {}

    for folder_name, folder_pairs in folder_groups.items():
        folder_images = [img_path for img_path, _, _ in folder_pairs]
        if folder_name in oversampled_folders:
            print(f"Folder '{folder_name}': Keeping all {len(folder_pairs)} images (oversampled folder)")
            processed_pairs.extend(folder_pairs)
            unique_images_per_folder[folder_name] = set(detector.get_unique_images(folder_images))
        else:
            clusters = detector.find_duplicates(folder_images)
            if clusters:
                print(f"Found {len(clusters)} duplicate clusters in folder '{folder_name}':")
                detector.print_duplicate_clusters(clusters)
            unique_folder_images = set(detector.get_unique_images(folder_images))
            unique_folder_pairs = [
                ImageLabelPair(img, lbl, scene)
                for img, lbl, scene in folder_pairs
                if img in unique_folder_images
            ]
            print(f"Folder '{folder_name}': {len(unique_folder_pairs)}/{len(folder_pairs)} unique images after duplicate removal")
            processed_pairs.extend(unique_folder_pairs)
            unique_images_per_folder[folder_name] = unique_folder_images

    if len(folder_groups) > 1:
        print("\nChecking for duplicates between different folders...")
        all_unique_images: List[Path] = []
        for _folder_name, unique_images in unique_images_per_folder.items():
            all_unique_images.extend(list(unique_images))
        cross_clusters = detector.find_duplicates(all_unique_images)
        if cross_clusters:
            print(f"Found {len(cross_clusters)} duplicate clusters between folders:")
            detector.print_duplicate_clusters(cross_clusters)
            # Prefer keeping images from oversampled folders (e.g., new week, replay)
            # If none in the cluster, fall back to lexicographic first (as detector.get_unique_images)
            scene_by_path: Dict[Path, str] = {}
            for scene_name, folder_pairs in folder_groups.items():
                for img_path, _lbl, _scene in folder_pairs:
                    scene_by_path[img_path] = scene_name

            cross_keep: Set[Path] = set()
            for _root, imgs in cross_clusters.items():
                # rank candidates: replay > any oversampled scene > lexicographic
                def rank(p: Path) -> tuple[int, int, str]:
                    scene = scene_by_path.get(p, "")
                    is_replay = 1 if scene == "replay" or "/replay/" in str(p) else 0
                    is_oversampled = 1 if scene in oversampled_folders else 0
                    return (-is_replay, -is_oversampled, str(p))

                keep = sorted(imgs, key=rank)[0]
                cross_keep.add(keep)
            # Also keep all images that were not in any duplicate cluster
            in_any_cluster = {p for imgs in cross_clusters.values() for p in imgs}
            cross_unique_images = set(p for p in all_unique_images if p not in in_any_cluster) | cross_keep
            processed_pairs = [
                ImageLabelPair(img_path, lbl_path, scene_name)
                for img_path, lbl_path, scene_name in processed_pairs
                if img_path in cross_unique_images
            ]
        else:
            print("No duplicates found between different folders")

    return processed_pairs


def _oversample_train_pairs(
    train_pairs: List[ImageLabelPair],
    folder_subsets: Dict[str, int | float]
) -> List[ImageLabelPair]:
    if not folder_subsets:
        return train_pairs

    extended_pairs = list(train_pairs)
    for folder_name, ratio in folder_subsets.items():
        # oversample only if float > 1.0; integers are reserved for absolute counts
        if not (isinstance(ratio, float) and ratio > 1.0):
            continue

        folder_pairs = [pair for pair in train_pairs if pair.scene == folder_name]
        if not folder_pairs:
            continue

        oversample_factor = int(ratio)
        remainder = ratio - oversample_factor

        duplicates: List[ImageLabelPair] = []
        if oversample_factor > 1:
            duplicates.extend(folder_pairs * (oversample_factor - 1))

        if remainder > 0:
            folder_copy = folder_pairs.copy()
            random.shuffle(folder_copy)
            partial_count = int(len(folder_pairs) * remainder)
            duplicates.extend(folder_copy[:partial_count])

        if duplicates:
            print(
                f"Folder '{folder_name}': Added {len(duplicates)} extra training copies "
                f"({ratio*100:.1f}%)"
            )
            extended_pairs.extend(duplicates)

    return extended_pairs


def create_dataset_yaml(dataset_path: Path, custom_classes=None, use_coco_classes=True, class_mapping_config=None):
    """
    Create dataset.yaml file with appropriate class mapping.
    
    Args:
        dataset_path: Path to the dataset directory
        custom_classes: List of custom class names
        use_coco_classes: Whether to use COCO classes when custom_classes is empty
        class_mapping_config: Dictionary mapping target classes to source classes for merging
    """
    # Apply class mapping if configured
    final_classes = custom_classes
    if class_mapping_config and custom_classes:
        final_classes, _ = apply_class_mapping_config(custom_classes, class_mapping_config)
    
    class_mapping = get_class_mapping(final_classes, use_coco_classes)

    # Get all parts of the path
    yaml_dataset_path = dataset_path.absolute()

    yaml_content = f"""
path: {yaml_dataset_path}
train: train/images
val: val/images
nc: {len(class_mapping)} 

names:
"""
    for key, value in class_mapping.items():
        yaml_content += f"  {key}: {value}\n"

    with open(dataset_path / "dataset.yaml", "w") as f:
        f.write(yaml_content)
    
    print(f"Created dataset.yaml with {len(class_mapping)} classes: {list(class_mapping.values())}")


def _poly_to_bbox_row(row):
    """row = [cls, x1, y1, x2, y2, …]  (normalised)  →  [cls, xc, yc, w, h]"""
    cls, pts = int(row[0]), row[1:]
    xs, ys = pts[0::2], pts[1::2]
    x1, y1, x2, y2 = max(0, min(xs)), max(0, min(ys)), min(1, max(xs)), min(1, max(ys))
    xc, yc, w, h = (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1
    return [cls, xc, yc, w, h]


def convert_polygons_to_bboxes_inplace(label_path: Path) -> None:
    """
    Re-writes *label_path* if any row contains polygons (≥7 numbers).
    Works in-place and is idempotent.
    """
    with open(label_path) as f:
        rows = [line.strip().split() for line in f if line.strip()]

    changed = False
    out = []
    for r in rows:
        if len(r) > 5:                               # polygon row
            r = _poly_to_bbox_row(list(map(float, r)))
            changed = True
        else:                                        # already bbox row
            r = list(map(float, r))                  # Convert to floats
        out.append(" ".join(f"{v:.6f}" if i else str(int(v))
                            for i, v in enumerate(r)))

    if changed:                                      # overwrite only if we edited
        with open(label_path, "w") as f:
            f.write("\n".join(out) + "\n")


def augment(image_label_pairs: List[ImageLabelPair], augment_multiplier: int = 1):

    print(f"Applying augmentation with multiplier {augment_multiplier}...")
    augmenter = YOLOAugmenter(multiplier=augment_multiplier)
    augmented_pairs: List[ImageLabelPair] = []

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

            augmented_pairs.append(ImageLabelPair(aug_img_path, aug_label_path, scene_name))

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
    folder_subsets=None,
    class_mapping_config=None,
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
        folder_subsets: Dictionary mapping folder names to subset ratios (0.0-1.0)
        class_mapping_config: Dictionary mapping target classes to source classes for merging
    """
    if folder_subsets is None:
        folder_subsets = {}
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
    
    # Apply class mapping if configured
    original_class_list = custom_classes if custom_classes else []
    final_class_list = custom_classes if custom_classes else []
    source_to_target_map = {}
    
    if class_mapping_config and custom_classes:
        final_class_list, source_to_target_map = apply_class_mapping_config(
            custom_classes, class_mapping_config
        )
        print(f"\nClass mapping will be applied to all label files.")
        print(f"  Source-to-target mapping: {source_to_target_map}")

    for some_folder in sorted_iterdir(input_path):
        if not some_folder.is_dir():
            continue

        folder_to_process = some_folder
        scene_name = some_folder.name  # Extract scene name from folder name
        # Check if this is a CVAT export folder with train.txt
        train_txt = folder_to_process / "train.txt"
        if train_txt.exists():
            cvat_result = _process_cvat_folder(
                some_folder,
                folder_to_process,
                scene_name,
                target_class_mapping,
                folder_subsets,
                temp_folders,
            )
            empty_label_count += cvat_result.empty_label_count
            skip_count += cvat_result.skip_count
            image_label_pairs.extend(cvat_result.pairs)
        else:
            manual_result = _process_manual_folder(
                some_folder,
                folder_to_process,
                scene_name,
                target_class_mapping,
                folder_subsets,
                temp_folders,
            )
            empty_label_count += manual_result.empty_label_count
            image_label_pairs.extend(manual_result.pairs)

    print(f"Included {empty_label_count} images with empty labels (no objects).")
    print(f"Skipped {skip_count} images that couldn't be found.")

    if not image_label_pairs:
        print("No valid image-label pairs found.")
        # Clean up temp folders
        for temp_folder in temp_folders:
            shutil.rmtree(temp_folder)
        return 0, 0, 0

    print("Find duplicate images...")
    processed_pairs = _dedupe_pairs(image_label_pairs, folder_subsets)

    print(f"\nOriginal number of images: {len(image_label_pairs)}")
    print(f"Number of images after smart duplicate removal: {len(processed_pairs)}")
    
    # Continue with the processed pairs instead of original image_label_pairs
    image_label_pairs = processed_pairs

    # Shuffle data
    random.shuffle(image_label_pairs)

    # Calculate split indices
    total_images = len(image_label_pairs)
    test_size = int(total_images * test_split)
    val_size = int(total_images * val_split)

    # Split the data
    test_pairs = image_label_pairs[:test_size]
    val_pairs = image_label_pairs[test_size : test_size + val_size]
    train_pairs = image_label_pairs[test_size + val_size :]

    # Apply oversampling to training pairs only to avoid leakage
    train_pairs = _oversample_train_pairs(train_pairs, folder_subsets)

    # add augmentation after oversampling so duplicates receive augmentations too
    augmented_pairs, aug_temp_folders = augment(train_pairs, augment_multiplier)

    temp_folders.extend(aug_temp_folders)

    random.shuffle(augmented_pairs)

    # only add augmentation to training data
    train_pairs.extend(augmented_pairs)

    # Helper function to copy files with scene information
    def copy_pairs(pairs, img_path, label_path):
        count = 0

        def with_dup_suffix(name: str, duplicate_index: int) -> str:
            if duplicate_index == 0:
                return name
            stem, ext = os.path.splitext(name)
            return f"{stem}__dup{duplicate_index}{ext}"

        def link_or_copy(src: Path, dst: Path) -> None:
            try:
                os.link(src, dst)
            except OSError:
                shutil.copy2(src, dst)

        name_counts: Dict[str, int] = {}

        for img_file, label_file, scene_name in pairs:
            if is_test_data:
                base_img_name = f"{img_file.stem}__scene_{scene_name}{img_file.suffix}"
                base_label_name = (
                    f"{label_file.stem}__scene_{scene_name}{label_file.suffix}"
                )
            else:
                base_img_name = img_file.name
                base_label_name = label_file.name

            duplicate_index = name_counts.get(base_img_name, 0)
            name_counts[base_img_name] = duplicate_index + 1

            new_img_name = with_dup_suffix(base_img_name, duplicate_index)
            new_label_name = with_dup_suffix(base_label_name, duplicate_index)

            target_img = img_path / new_img_name
            target_label = label_path / new_label_name

            link_or_copy(img_file, target_img)
            link_or_copy(label_file, target_label)
            
            # Apply class mapping to the copied label file if configured
            if source_to_target_map:
                remap_labels_with_class_mapping(
                    target_label, 
                    source_to_target_map, 
                    original_class_list, 
                    final_class_list
                )

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


def resolve_folder_subsets(
    folder_subsets: Dict[str, int | float] | None, cli_overrides
) -> Dict[str, int | float]:
    resolved = dict(folder_subsets or {})
    if not cli_overrides:
        return resolved

    print("Overriding folder subset configuration with command-line arguments:")
    for folder_name, ratio_str in cli_overrides:
        try:
            ratio = float(ratio_str)
            if ratio <= 0:
                print(f"  Warning: Invalid ratio {ratio} for {folder_name}. Must be > 0.")
                continue
            resolved[folder_name] = ratio
            suffix = " (oversampling)" if ratio > 1 else ""
            print(f"  {folder_name}: {ratio*100:.1f}%{suffix}")
        except ValueError:
            print(f"  Warning: Invalid ratio '{ratio_str}' for {folder_name}. Must be a number.")
    return resolved


def create_dataset_from_raw(
    *,
    dataset_path: Path,
    training_path: Path,
    test_path: Path,
    train_image_input_path: Path,
    test_image_input_path: Path,
    val_split: float,
    test_split: float,
    augment_multiplier: int,
    custom_classes,
    use_coco_classes: bool,
    folder_subsets: Dict[str, int | float],
    class_mapping_config: dict,
    test_data_exists: bool,
    recreate_dataset: bool,
) -> tuple[int, int, int]:
    if dataset_path.exists() and recreate_dataset:
        print(f"Recreating dataset '{dataset_path.name}'...")
        shutil.rmtree(dataset_path)

    dataset_path.mkdir(parents=True, exist_ok=True)
    for path in (training_path, test_path):
        path.mkdir(parents=True, exist_ok=True)

    for path in (
        training_path / "train" / "images",
        training_path / "train" / "labels",
        training_path / "val" / "images",
        training_path / "val" / "labels",
        test_path / "val" / "images",
        test_path / "val" / "labels",
    ):
        path.mkdir(parents=True, exist_ok=True)

    total_train_frames, total_val_frames, total_test_frames = process_single_images(
        input_path=train_image_input_path,
        train_output_path=training_path,
        test_output_path=test_path,
        val_split=val_split,
        test_split=test_split,
        augment_multiplier=augment_multiplier,
        custom_classes=custom_classes,
        use_coco_classes=use_coco_classes,
        folder_subsets=folder_subsets,
        class_mapping_config=class_mapping_config,
    )
    if total_train_frames <= 0:
        raise ValueError(
            "Prepare stage produced 0 training frames. "
            f"(train={total_train_frames}, val={total_val_frames}, test={total_test_frames}). "
            "Ensure raw_data/train contains labeled images and split/subset settings leave at least one training sample."
        )

    create_dataset_yaml(training_path, custom_classes, use_coco_classes, class_mapping_config)

    test_folder_frame_count = 0
    if test_data_exists:
        _, test_folder_frame_count, _ = process_single_images(
            input_path=test_image_input_path,
            train_output_path=test_path,
            test_output_path=test_path,
            val_split=1,
            test_split=0,
            augment_multiplier=1,
            custom_classes=custom_classes,
            use_coco_classes=use_coco_classes,
            folder_subsets={},
            class_mapping_config=class_mapping_config,
        )

    create_dataset_yaml(test_path, custom_classes, use_coco_classes, class_mapping_config)
    return total_train_frames, total_val_frames, total_test_frames + test_folder_frame_count
