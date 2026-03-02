from __future__ import annotations

from pathlib import Path
import shutil

import yaml


def organize_training_outputs(
    train_output_dir: Path,
    training_path: Path,
    test_path: Path,
    metadata: dict,
) -> None:
    reorganize_output(train_output_dir, training_path, test_path, metadata)


def reorganize_output(
    train_output_dir: Path,
    training_path: Path,
    test_path: Path,
    metadata: dict,
) -> None:
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
    for plot_file in sorted(train_output_dir.glob("*.png")) + sorted(train_output_dir.glob("*.jpg")):
        shutil.move(str(plot_file), str(plots_dir / plot_file.name))

    print(f"Experiment data organized in {train_output_dir}")
    print(f"Plots saved to {plots_dir}")


__all__ = ["organize_training_outputs", "reorganize_output"]
