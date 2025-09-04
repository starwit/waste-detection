from pathlib import Path
from typing import List, NamedTuple


class ImageLabelPair(NamedTuple):
    """Pair of image and label paths with scene identifier."""
    image: Path
    label: Path
    scene: str


class ProcessedFolder(NamedTuple):
    """Result of processing a folder into image/label pairs."""
    pairs: List[ImageLabelPair]
    temp_folders: List[Path]
    empty_label_count: int
    # For CVAT processing we also track skipped images; defaults to 0 otherwise
    skip_count: int = 0


class SplitData(NamedTuple):
    """Split result for train/val images and labels."""
    train_images: List[Path]
    train_labels: List[Path]
    val_images: List[Path]
    val_labels: List[Path]

