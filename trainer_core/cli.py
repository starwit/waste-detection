from __future__ import annotations

import argparse
import os
import random
from typing import Sequence

import numpy as np
import torch

from trainer_core.backends.shared import normalize_backend_name, resolve_training_config
from trainer_core.config.loader import load_config
from trainer_core.pipeline.evaluate_stage import run_evaluate_stage
from trainer_core.pipeline.prepare_stage import run_prepare_stage
from trainer_core.pipeline.train_stage import run_train_stage


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["prepare", "train", "evaluate", "all"], required=True)
    parser.add_argument("--config", default="params.yaml", help="Path to config YAML.")
    parser.add_argument("--model", default=None, help="Model key from models.*")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("-d", "--dataset-name", default=None, help="Dataset name override.")
    parser.add_argument("-vs", "--val-split", type=float, default=None)
    parser.add_argument("-ts", "--test-split", type=float, default=None)
    parser.add_argument("--recreate-dataset", action="store_true")
    parser.add_argument("--augment-multiplier", type=int, default=None)
    parser.add_argument(
        "--folder-subset",
        action="append",
        nargs=2,
        metavar=("FOLDER", "RATIO"),
        help="Override folder subsets: --folder-subset scene_a 0.5",
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        help="Config override in key=value format (supports dot paths).",
    )
    return parser


def _set_deterministic_seed(seed: int, backend_hint: str) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    normalized_backend = normalize_backend_name(backend_hint)

    torch.use_deterministic_algorithms(True, warn_only=(normalized_backend == "rfdetr"))
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_all_stages(args, config=None) -> None:
    cfg = config or load_config(getattr(args, "config", "params.yaml"), args=args)
    run_prepare_stage(args, config=cfg)
    train_result = run_train_stage(args, config=cfg)
    run_evaluate_stage(args, train_result=train_result, config=cfg)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    backend_hint = "yolo"
    if args.stage in {"train", "all", "evaluate"}:
        cfg = load_config(args.config, args=args)
        resolved = resolve_training_config(args, cfg)
        backend_hint = resolved["backend"]

    _set_deterministic_seed(args.seed, backend_hint)

    if args.stage == "prepare":
        run_prepare_stage(args)
    elif args.stage == "train":
        run_train_stage(args)
    elif args.stage == "evaluate":
        run_evaluate_stage(args)
    else:
        run_all_stages(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
