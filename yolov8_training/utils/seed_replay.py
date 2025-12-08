from __future__ import annotations

import argparse
from pathlib import Path
import time
import yaml
from ultralytics import YOLO

from yolov8_training.utils.replay import build_or_update_replay_set


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed or update the replay set from a model and prepared dataset.")
    parser.add_argument("--weights", type=str, default=None, help="Path to model weights (.pt). Defaults to params train.pretrained_model_path or evaluation.baseline_weights_path.")
    parser.add_argument("--dataset-name", type=str, default=None, help="Dataset name. Defaults to params.data.dataset_name.")
    parser.add_argument("--max-new", type=int, default=None, help="Max new replay items to add.")
    parser.add_argument("--max-total", type=int, default=None, help="Cap total replay items.")
    parser.add_argument("--dest", type=str, default=None, help="Destination replay folder (images/labels). Defaults to params.prepare.auto_replay.dest.")
    args = parser.parse_args()

    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f) or {}

    data = params.get("data", {})
    prepare_cfg = params.get("prepare", {})
    auto_replay = prepare_cfg.get("auto_replay", {})
    eval_cfg = params.get("evaluation", {})
    train_cfg = params.get("train", {})

    dataset_name = args.dataset_name or data.get("dataset_name")
    if not dataset_name:
        raise SystemExit("dataset_name not found; pass --dataset-name or set data.dataset_name in params.yaml")

    weights = args.weights or train_cfg.get("pretrained_model_path") or eval_cfg.get("baseline_weights_path")
    if not weights:
        raise SystemExit("No weights provided; pass --weights or set train.pretrained_model_path / evaluation.baseline_weights_path")

    training_path = Path("datasets") / dataset_name / "train"
    if not (training_path / "val" / "images").exists():
        raise SystemExit(f"Dataset not prepared. Expected {training_path / 'val' / 'images'} to exist.")

    cfg = dict(auto_replay)
    if args.max_new is not None:
        cfg["max_new"] = args.max_new
    if args.max_total is not None:
        cfg["max_total"] = args.max_total
    if args.dest is not None:
        cfg["dest"] = args.dest

    model = YOLO(str(weights))
    run_id = f"seed-{int(time.time())}"
    build_or_update_replay_set(model, training_path, Path("runs") / run_id, cfg)


if __name__ == "__main__":
    main()

