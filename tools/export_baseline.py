import argparse
import shutil
import sys
from pathlib import Path

import yaml


def _load_params() -> dict:
    with open("params.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _resolve_path(path_str: str | None) -> Path | None:
    if not path_str:
        return None
    path = Path(path_str).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    return path


def _default_baseline_path(params: dict) -> Path:
    baseline_rel = params.get("evaluation", {}).get("baseline_weights_path")
    baseline_path = _resolve_path(baseline_rel)
    if baseline_path is None:
        raise ValueError(
            "Could not resolve evaluation.baseline_weights_path from params.yaml. "
            "Pass --baseline-path explicitly or configure params.yaml."
        )
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
    if metadata_path is None:
        raise ValueError(
            "A metadata file must be provided via --metadata or implied via --run-dir."
        )

    return weights_path, metadata_path, run_dir


def _load_required_metadata(metadata_source: Path) -> dict:
    if not metadata_source.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_source}")
    with open(metadata_source, "r", encoding="utf-8") as f:
        metadata = yaml.safe_load(f) or {}
    if not isinstance(metadata, dict):
        raise ValueError(f"Metadata file must contain a YAML mapping: {metadata_source}")
    model_backend = str(metadata.get("model_backend", "")).strip()
    if not model_backend:
        raise ValueError(
            f"Metadata file must define model_backend for exported baselines: {metadata_source}"
        )
    return metadata


def _copy_rtmdet_baseline_config(
    *,
    metadata_to_write: dict,
    metadata_source: Path | None,
    run_dir: Path | None,
    baseline_dir: Path,
) -> None:
    model_backend = str(metadata_to_write.get("model_backend", "")).strip().lower()
    if model_backend not in {"rtmdet", "mmdet"}:
        return

    raw_config_path = str(metadata_to_write.get("model_config_path") or "").strip()
    config_source: Path | None = None
    if raw_config_path:
        candidate = Path(raw_config_path).expanduser()
        candidates: list[Path] = []
        if candidate.is_absolute():
            candidates.append(candidate)
        else:
            if metadata_source is not None:
                candidates.append(metadata_source.parent / candidate)
            candidates.append(Path.cwd() / candidate)
        for resolved_candidate in candidates:
            if resolved_candidate.exists():
                config_source = resolved_candidate
                break

    if config_source is None:
        run_config = run_dir / "model_config.py" if run_dir is not None else None
        if run_config is not None and run_config.exists():
            config_source = run_config
        else:
            raise FileNotFoundError(
                "RTMDet baseline export requires a model_config.py artifact. "
                "Pass --metadata pointing at a run metadata file that includes model_config_path, "
                "or export directly from a run directory containing model_config.py."
            )

    baseline_config_path = baseline_dir / "model_config.py"
    shutil.copy2(config_source, baseline_config_path)
    metadata_to_write["model_config_path"] = baseline_config_path.name


def export_baseline(args):
    params = _load_params()

    baseline_weights_target = _resolve_path(args.baseline_path) or _default_baseline_path(params)

    weights_source, metadata_source, run_dir = _determine_sources(args)

    if not weights_source.exists():
        raise FileNotFoundError(f"Weights file not found: {weights_source}")

    baseline_dir = baseline_weights_target.parent
    baseline_dir.mkdir(parents=True, exist_ok=True)

    print(f"Copying weights -> {baseline_weights_target}")
    shutil.copy2(weights_source, baseline_weights_target)

    metadata_to_write = _load_required_metadata(metadata_source)

    # Enrich metadata with export info
    # Convert absolute paths to relative paths starting with "runs/"
    workspace_root = Path.cwd()
    
    if run_dir:
        try:
            run_dir_relative = run_dir.relative_to(workspace_root)
            metadata_to_write.setdefault("source_run", str(run_dir_relative))
        except ValueError:
            # Fallback to absolute path if relative conversion fails
            metadata_to_write.setdefault("source_run", str(run_dir))
    else:
        metadata_to_write.setdefault("source_run", str(weights_source.parent))
    
    try:
        source_weights = str(weights_source.relative_to(workspace_root))
    except ValueError:
        # Fallback to absolute path if relative conversion fails
        source_weights = str(weights_source)
    metadata_to_write["source_weights"] = source_weights
    
    if args.tag:
        metadata_to_write["tag"] = args.tag

    baseline_metadata_path = baseline_dir / "metadata.yaml"

    _copy_rtmdet_baseline_config(
        metadata_to_write=metadata_to_write,
        metadata_source=metadata_source,
        run_dir=run_dir,
        baseline_dir=baseline_dir,
    )

    with open(baseline_metadata_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(metadata_to_write, f, sort_keys=False)

    print(f"Metadata written -> {baseline_metadata_path}")
    return baseline_weights_target, baseline_metadata_path


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
        "--tag",
        type=str,
        help="Optional label to store inside metadata.yaml (e.g., release name or git SHA)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        baseline_weights_target, baseline_metadata_path = export_baseline(args)
    except Exception as exc:
        print(f"Error: {exc}")
        return 1

    baseline_dir = baseline_weights_target.parent
    print("Baseline export complete.")
    print("Next steps:")
    print(f"  dvc add {baseline_weights_target}")
    print("  dvc push")
    print(f"  git add {baseline_weights_target.with_name(baseline_weights_target.name + '.dvc')}")
    print(f"  git add {baseline_metadata_path}")
    model_config_path = baseline_dir / "model_config.py"
    if model_config_path.exists():
        print(f"  git add {model_config_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
