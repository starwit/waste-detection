from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


RTMDET_CONFIGS = {
    "tiny": "rtmdet_tiny_8xb32-300e_coco",
    "s": "rtmdet_s_8xb32-300e_coco",
    "m": "rtmdet_m_8xb32-300e_coco",
    "l": "rtmdet_l_8xb32-300e_coco",
    "x": "rtmdet_x_8xb32-300e_coco",
}


def _resolve_config_name(variant: str) -> str:
    key = str(variant).strip().lower().replace("-", "_")
    return RTMDET_CONFIGS.get(key, str(variant).strip())


def _download_one(config_name: str, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    cmd = [
        "mim",
        "download",
        "mmdet",
        "--config",
        config_name,
        "--dest",
        str(dest),
    ]
    print(f"Downloading {config_name} -> {dest}")
    subprocess.run(cmd, check=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Download RTMDet configs/checkpoints via OpenMIM.")
    ap.add_argument(
        "--variant",
        action="append",
        dest="variants",
        help="Variant key (tiny/s/m/l/x) or full config name. Can be repeated.",
    )
    ap.add_argument(
        "--dest",
        type=Path,
        default=Path("models/pretrained/mmdet"),
        help="Destination directory for downloaded files.",
    )
    args = ap.parse_args(argv)

    variants = args.variants or ["tiny", "s", "m", "l", "x"]
    try:
        for variant in variants:
            _download_one(_resolve_config_name(str(variant)), args.dest)
    except FileNotFoundError:
        print("Error: 'mim' command not found. Install OpenMIM first: pip install openmim")
        return 1
    except subprocess.CalledProcessError as exc:
        print(f"Error: OpenMIM download failed with exit code {exc.returncode}")
        return int(exc.returncode or 1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
