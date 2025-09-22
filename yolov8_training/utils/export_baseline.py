import argparse
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml


def _load_params() -> dict:
    try:
        with open("params.yaml", "r") as f:
            return yaml.safe_load(f) or {}
    except Exception as exc:
        print(f"Warning: could not read params.yaml ({exc})")
        return {}


def _resolve_path(path_str: str | None) -> Path | None:
    if not path_str:
        return None
    path = Path(path_str).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    return path


def _default_baseline_path(params: dict) -> Path:
    baseline_rel = (
        params.get("evaluation", {}).get("baseline_weights_path")
        or "models/current_best/best.pt"
    )
    baseline_path = _resolve_path(baseline_rel)
    if baseline_path is None:
        raise ValueError("Could not resolve baseline path from params.yaml")
    return baseline_path


def _determine_sources(args) -> tuple[Path, Path | None, Path | None]:
    run_dir = _resolve_path(args.run_dir)
    weights_path = _resolve_path(args.weights)
    metadata_path = _resolve_path(args.metadata)

    if run_dir is not None:
        if weights_path is None:
            weights_path = run_dir / "weights" / "best.pt"
        if metadata_path is None:
            metadata_path = run_dir / "metadata.yaml"

    if weights_path is None:
        raise ValueError("A weights file must be provided via --weights or --run-dir")

    return weights_path, metadata_path, run_dir


def export_baseline(args):
    params = _load_params()

    baseline_weights_target = _resolve_path(args.baseline_path) or _default_baseline_path(params)
    if baseline_weights_target.suffix != ".pt":
        raise ValueError("Baseline path must point to a .pt file")

    weights_source, metadata_source, run_dir = _determine_sources(args)

    if not weights_source.exists():
        raise FileNotFoundError(f"Weights file not found: {weights_source}")

    baseline_dir = baseline_weights_target.parent
    baseline_dir.mkdir(parents=True, exist_ok=True)

    print(f"Copying weights -> {baseline_weights_target}")
    shutil.copy2(weights_source, baseline_weights_target)

    # Prepare metadata (optional but recommended)
    if args.skip_metadata:
        print("Metadata copy skipped (per --skip-metadata)")
        return

    metadata_to_write: dict = {}
    if metadata_source and metadata_source.exists():
        try:
            with open(metadata_source, "r") as f:
                metadata_to_write = yaml.safe_load(f) or {}
        except Exception as exc:
            print(f"Warning: Could not read metadata from {metadata_source}: {exc}")
            metadata_to_write = {}
    else:
        if metadata_source:
            print(f"Warning: metadata source not found at {metadata_source}; creating minimal file")

    # Enrich metadata with export info
    metadata_to_write.setdefault("source_run", str(run_dir) if run_dir else str(weights_source.parent))
    metadata_to_write["source_weights"] = str(weights_source)
    metadata_to_write["exported_at"] = datetime.now(timezone.utc).isoformat()
    if args.tag:
        metadata_to_write["tag"] = args.tag

    baseline_metadata_path = baseline_dir / "metadata.yaml"
    with open(baseline_metadata_path, "w") as f:
        yaml.safe_dump(metadata_to_write, f, sort_keys=False)

    print(f"Metadata written -> {baseline_metadata_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export a trained run's best weights and metadata to the shared baseline location."
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        help="Path to a training run directory containing weights/best.pt and metadata.yaml",
    )
    parser.add_argument(
        "--weights",
        type=str,
        help="Explicit path to the weights file to export (defaults to run-dir/weights/best.pt)",
    )
    parser.add_argument(
        "--metadata",
        type=str,
        help="Explicit path to metadata yaml (defaults to run-dir/metadata.yaml)",
    )
    parser.add_argument(
        "--baseline-path",
        type=str,
        help="Destination path for the baseline weights (defaults to params.yaml evaluation.baseline_weights_path)",
    )
    parser.add_argument(
        "--skip-metadata",
        action="store_true",
        help="Do not copy or generate metadata.yaml alongside the baseline weights",
    )
    parser.add_argument(
        "--tag",
        type=str,
        help="Optional label to store inside metadata.yaml (e.g., release name or git SHA)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        export_baseline(args)
    except Exception as exc:
        print(f"Error: {exc}")
        return 1
    print("Baseline export complete. Remember to run `dvc add` and `dvc push` if needed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

