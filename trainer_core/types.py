from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class TrainResult:
    model: Any
    train_output_dir: Path
    experiment_name: str
    image_size: int
    train_epochs: int
    training_path: Path
    test_path: Path
    baseline_weights_path: str | None
    fallback_checkpoint: str
    finetune_weights_path: str | None
    reload_metadata: dict[str, Any]
    params: dict[str, Any]


@dataclass
class PersistedTrainResult:
    train_output_dir: Path
    experiment_name: str
    image_size: int
    train_epochs: int
    training_path: Path
    test_path: Path
    baseline_weights_path: str | None
    fallback_checkpoint: str
    finetune_weights_path: str | None
    best_weights_path: Path
    reload_metadata: dict[str, Any]
