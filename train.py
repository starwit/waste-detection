from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from trainer_core.cli import main as core_main


PROJECT_ROOT = Path(__file__).resolve().parent


def _inject_project_defaults(argv: Sequence[str]) -> list[str]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--stage")
    parser.add_argument("--workspace-root")
    parser.add_argument("--config")
    known, _ = parser.parse_known_args(list(argv))

    forwarded = list(argv)
    if known.stage is None:
        forwarded.extend(["--stage", "train"])
    if known.workspace_root is None:
        forwarded.extend(["--workspace-root", str(PROJECT_ROOT)])
    if known.config is None:
        forwarded.extend(["--config", str(PROJECT_ROOT / "params.yaml")])
    return forwarded


def main(argv: Sequence[str] | None = None) -> int:
    raw_args = list(argv) if argv is not None else list(sys.argv[1:])
    return core_main(_inject_project_defaults(raw_args))


if __name__ == "__main__":
    raise SystemExit(main())
