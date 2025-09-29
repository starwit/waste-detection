import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import yaml
from PIL import Image

SMALL_MAX_AREA = 32 ** 2
MEDIUM_MAX_AREA = 96 ** 2
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
SIZE_BUCKETS: Sequence[str] = ("small", "medium", "large")
DEFAULT_EXPORT_DIR = Path(__file__).resolve().parent / "analysis_output"


class AnalysisError(Exception):
    """Raised when the dataset structure or annotations are invalid."""


@dataclass
class ClassStats:
    class_id: int
    class_name: str
    total_instances: int = 0
    images_with_class: int = 0
    size_distribution: Dict[str, int] = field(
        default_factory=lambda: {bucket: 0 for bucket in SIZE_BUCKETS}
    )

    def add_instance(self, bbox_area: float) -> None:
        self.total_instances += 1
        self.size_distribution[size_bucket(bbox_area)] += 1


@dataclass
class SplitStats:
    split_name: str
    total_images: int = 0
    empty_images: int = 0
    total_objects: int = 0
    class_stats: Dict[int, ClassStats] = field(default_factory=dict)


def size_bucket(area: float) -> str:
    if area < SMALL_MAX_AREA:
        return "small"
    if area < MEDIUM_MAX_AREA:
        return "medium"
    return "large"


class ClassDistributionAnalyzer:
    def __init__(self, dataset_path: Path):
        self.dataset_path = Path(dataset_path)
        self.class_names: Dict[int, str] = {}
        self.splits: Dict[str, SplitStats] = {}

    def analyze_dataset(self) -> None:
        config_path = self._find_dataset_config()
        if config_path is None:
            raise AnalysisError(f"Could not locate dataset.yaml under {self.dataset_path}")

        self._load_class_names(config_path)

        split_definitions = self._discover_splits()
        if not split_definitions:
            raise AnalysisError("No valid dataset splits found")

        for split_name, images_dir, labels_dir in split_definitions:
            self.splits[split_name] = self._analyze_split(split_name, images_dir, labels_dir)

    def print_summary(self) -> None:
        report = self.build_report()
        summary = report["summary"]
        total_images = summary["total_images"]
        total_objects = summary["total_objects"]
        total_empty = summary["total_empty_images"]
        empty_pct = (total_empty / total_images * 100) if total_images else 0

        print("\n" + "=" * 60)
        print("DATASET CLASS DISTRIBUTION ANALYSIS")
        print("=" * 60)

        print("\nOVERALL TOTALS")
        print("-" * 20)
        print(f"Images: {total_images:,} total, {total_empty:,} empty ({empty_pct:.1f}%)")
        print(f"Objects: {total_objects:,} total")
        print(f"Classes: {len(self.class_names)} ({', '.join(self.class_names.values())})")

        for class_row in summary["classes"]:
            print(
                f"  {class_row['class_name']}: {class_row['total_instances']:,} "
                f"instances ({class_row['percentage']:.1f}%)"
            )

        print("\nPER-SPLIT BREAKDOWN")
        print("-" * 20)
        print(f"{'Split':<8} {'Images':<8} {'Objects':<8} {'Empty':<6} {'Empty%':<7}")
        print("-" * 40)

        for split_name in ("train", "val", "test"):
            split_data = report["splits"].get(split_name)
            if not split_data:
                continue

            split_empty_pct = (
                split_data["empty_images"] / split_data["total_images"] * 100
                if split_data["total_images"]
                else 0
            )
            print(
                f"{split_name:<8} {split_data['total_images']:<8,} "
                f"{split_data['total_objects']:<8,} {split_data['empty_images']:<6,} "
                f"{split_empty_pct:<7.1f}"
            )

        for split_name in ("train", "val", "test"):
            split_data = report["splits"].get(split_name)
            if not split_data:
                continue

            print(f"\n{split_name.upper()}:")
            for class_row in split_data["classes"]:
                inst_pct = class_row["instance_percentage"]
                img_pct = class_row["image_percentage"]
                print(
                    f"  {class_row['class_name']}: {class_row['total_instances']:,} "
                    f"instances ({inst_pct:.1f}%) in {class_row['images_with_class']:,} "
                    f"images ({img_pct:.1f}%)"
                )
                sizes = class_row["size_distribution"]
                print(
                    f"    Sizes: {sizes['small']:,} small, {sizes['medium']:,} medium, "
                    f"{sizes['large']:,} large"
                )

        print("\n" + "=" * 60)
        print("Notes:")
        print("  Object sizes: Small <32²px, Medium 32²-96²px, Large ≥96²px")
        print("  Use --export-json for machine-readable output")
        print("=" * 60)

    def export_to_json(self, output_path: Path) -> None:
        json_path = Path(output_path) / "class_distribution.json"
        json_path.write_text(json.dumps(self.build_report(), indent=2), encoding="utf-8")

    def build_report(self) -> Dict[str, object]:
        total_images = sum(split.total_images for split in self.splits.values())
        total_objects = sum(split.total_objects for split in self.splits.values())
        total_empty = sum(split.empty_images for split in self.splits.values())

        class_totals = []
        for class_id, class_name in self.class_names.items():
            total_instances = sum(
                split.class_stats[class_id].total_instances
                for split in self.splits.values()
                if class_id in split.class_stats
            )
            percentage = (total_instances / total_objects * 100) if total_objects else 0
            class_totals.append(
                {
                    "class_id": class_id,
                    "class_name": class_name,
                    "total_instances": total_instances,
                    "percentage": percentage,
                }
            )

        splits_payload: Dict[str, Dict[str, object]] = {}
        for split_name, split_stats in self.splits.items():
            classes_payload = []
            for class_id, stats in split_stats.class_stats.items():
                instance_pct = (
                    stats.total_instances / split_stats.total_objects * 100
                    if split_stats.total_objects
                    else 0
                )
                image_pct = (
                    stats.images_with_class / split_stats.total_images * 100
                    if split_stats.total_images
                    else 0
                )
                classes_payload.append(
                    {
                        "class_id": class_id,
                        "class_name": stats.class_name,
                        "total_instances": stats.total_instances,
                        "images_with_class": stats.images_with_class,
                        "instance_percentage": instance_pct,
                        "image_percentage": image_pct,
                        "size_distribution": stats.size_distribution,
                    }
                )

            splits_payload[split_name] = {
                "total_images": split_stats.total_images,
                "total_objects": split_stats.total_objects,
                "empty_images": split_stats.empty_images,
                "classes": classes_payload,
            }

        return {
            "dataset_path": str(self.dataset_path),
            "class_names": self.class_names,
            "summary": {
                "total_images": total_images,
                "total_objects": total_objects,
                "total_empty_images": total_empty,
                "classes": class_totals,
            },
            "splits": splits_payload,
        }

    def _load_class_names(self, config_path: Path) -> None:
        try:
            with open(config_path, "r", encoding="utf-8") as file:
                config = yaml.safe_load(file)
        except OSError as exc:
            raise AnalysisError(f"Could not read dataset config: {exc}") from exc

        names_data = config.get("names", {})
        if isinstance(names_data, list):
            self.class_names = {index: name for index, name in enumerate(names_data)}
        elif isinstance(names_data, dict):
            try:
                self.class_names = {int(key): value for key, value in names_data.items()}
            except ValueError as exc:
                raise AnalysisError("Class IDs in dataset.yaml must be integers") from exc
        else:
            raise AnalysisError(f"Invalid names format in dataset.yaml: {type(names_data)}")

        if not self.class_names:
            raise AnalysisError("No class names defined in dataset.yaml")

    def _analyze_split(self, split_name: str, images_dir: Path, labels_dir: Path) -> SplitStats:
        split_stats = SplitStats(split_name=split_name)
        split_stats.class_stats = {
            class_id: ClassStats(class_id=class_id, class_name=class_name)
            for class_id, class_name in self.class_names.items()
        }

        image_files = sorted(
            file for file in images_dir.iterdir()
            if file.is_file() and file.suffix.lower() in IMAGE_EXTENSIONS
        )

        for image_file in image_files:
            split_stats.total_images += 1
            label_file = labels_dir / f"{image_file.stem}.txt"
            width, height = self._image_dimensions(image_file)
            annotations = self._parse_yolo_annotation(label_file, width, height)

            if not annotations:
                split_stats.empty_images += 1
                continue

            classes_in_image = set()

            for class_id, bbox_area in annotations:
                stats = split_stats.class_stats.get(class_id)
                if stats is None:
                    continue

                stats.add_instance(bbox_area)
                classes_in_image.add(class_id)
                split_stats.total_objects += 1

            for class_id in classes_in_image:
                split_stats.class_stats[class_id].images_with_class += 1

        return split_stats

    def _discover_splits(self) -> List[Tuple[str, Path, Path]]:
        splits: Dict[str, Tuple[Path, Path]] = {}

        def register(name: str, images: Path, labels: Path) -> None:
            if images.exists():
                splits[name] = (images, labels)

        register("train", self.dataset_path / "train" / "train" / "images", self.dataset_path / "train" / "train" / "labels")
        register("val", self.dataset_path / "train" / "val" / "images", self.dataset_path / "train" / "val" / "labels")
        register("test", self.dataset_path / "test" / "val" / "images", self.dataset_path / "test" / "val" / "labels")

        for name in ("train", "val", "test"):
            register(name, self.dataset_path / name / "images", self.dataset_path / name / "labels")

        return [(name, *paths) for name, paths in splits.items()]

    def _find_dataset_config(self) -> Optional[Path]:
        candidates = (
            self.dataset_path / "dataset.yaml",
            self.dataset_path / "train" / "dataset.yaml",
            self.dataset_path / "test" / "dataset.yaml",
        )
        return next((candidate for candidate in candidates if candidate.exists()), None)

    def _parse_yolo_annotation(
        self, label_path: Path, image_width: int, image_height: int
    ) -> List[Tuple[int, float]]:
        if not label_path.exists():
            return []

        annotations: List[Tuple[int, float]] = []

        try:
            with open(label_path, "r", encoding="utf-8") as file:
                for line in file:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue

                    try:
                        class_id = int(parts[0])
                        norm_width = float(parts[3])
                        norm_height = float(parts[4])
                    except ValueError:
                        continue

                    actual_width = norm_width * image_width
                    actual_height = norm_height * image_height
                    annotations.append((class_id, actual_width * actual_height))
        except OSError:
            return annotations

        return annotations

    def _image_dimensions(self, image_path: Path) -> Tuple[int, int]:
        try:
            with Image.open(image_path) as image:
                return image.size
        except OSError:
            return 1280, 720


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze class distribution in YOLO format dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --dataset datasets/waste-detection
  %(prog)s --dataset datasets/waste-detection --output results/analysis --export-json
  %(prog)s --dataset datasets/waste-detection --quiet
        """,
    )

    parser.add_argument(
        "--dataset",
        "-d",
        type=str,
        required=True,
        help="Path to dataset directory containing dataset.yaml",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=str(DEFAULT_EXPORT_DIR),
        help="Output directory for exports (default: tool-local analysis_output)",
    )
    parser.add_argument(
        "--export-json",
        action="store_true",
        help="Export results to JSON file",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress console output (only show errors)",
    )

    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"Error: dataset path does not exist: {dataset_path}", file=sys.stderr)
        sys.exit(1)

    analyzer = ClassDistributionAnalyzer(dataset_path)

    try:
        analyzer.analyze_dataset()
    except AnalysisError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    if not args.quiet:
        analyzer.print_summary()

    if args.export_json:
        output_path = Path(args.output)
        output_path.mkdir(parents=True, exist_ok=True)
        analyzer.export_to_json(output_path)


if __name__ == "__main__":
    main()
