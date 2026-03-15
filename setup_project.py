"""
────────────
Interactive bootstrap helper for future derived training project work.

This script is not part of the normal workflow for the live waste-detection
project. It is kept only for intentionally bootstrapping a separate derived repo.

DOES:
  • ask for project & dataset names
  • optionally set custom classes
  • patch .dvc/config   (remote URL)
  • patch params.yaml   (dataset_name, experiment_name, classes)
  • create raw_data/train and raw_data/test directories
"""

import configparser
import pathlib
import textwrap

from ruamel.yaml import YAML

ROOT = pathlib.Path(__file__).resolve().parents[0]
CONFIG_FILE = ROOT / ".dvc" / "config"
PARAMS_FILE = ROOT / "params.yaml"

REMOTE_SEC = '\'remote "dvc-hetzner"\''
BASE_URL = "ssh://u420375-sub1.your-storagebox.de"


def prompt(question: str, *, default: str | None = None) -> str:
    suffix = f" [{default}]" if default else ""
    while True:
        ans = input(f"{question}{suffix}: ").strip()
        if ans:
            return ans
        if default is not None:
            return default
        print("Value required.")


def patch_dvc_config(project_slug: str) -> None:
    cfg = configparser.ConfigParser()
    cfg.read(CONFIG_FILE)

    if REMOTE_SEC not in cfg:
        raise ValueError(f"Section {REMOTE_SEC!r} not found in {CONFIG_FILE}")

    cfg[REMOTE_SEC]["url"] = f"{BASE_URL}/{project_slug}"

    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        cfg.write(f)


def patch_params_yaml(dataset: str, experiment: str) -> None:
    yaml = YAML()
    yaml.preserve_quotes = True

    with PARAMS_FILE.open("r", encoding="utf-8") as f:
        params = yaml.load(f)

    # Ensure nested structure exists
    data_cfg = params.setdefault("data", {})

    # Basic fields
    data_cfg["dataset_name"] = dataset
    data_cfg["experiment_name"] = experiment

    # Ask about custom classes
    use_custom = input("Do you want to use custom classes? [y/N]: ").strip().lower() == "y"

    if use_custom:
        class_input = input("Enter custom classes (comma-separated): ").strip()
        classes = [c.strip() for c in class_input.split(",") if c.strip()]
        if classes:
            data_cfg["custom_classes"] = classes
            data_cfg["use_coco_classes"] = False
            # Keep class_mapping consistent with the selected class list.
            data_cfg["class_mapping"] = {cls: [cls] for cls in classes}
        else:
            print("No valid classes entered. Using COCO classes.")
            data_cfg["custom_classes"] = []
            data_cfg["use_coco_classes"] = True
            data_cfg["class_mapping"] = {}
    else:
        # Explicit COCO-class selection.
        data_cfg["custom_classes"] = []
        data_cfg["use_coco_classes"] = True
        data_cfg["class_mapping"] = {}

    with PARAMS_FILE.open("w", encoding="utf-8") as f:
        yaml.dump(params, f)


def make_raw_data_dirs() -> None:
    for sub in ("train", "test"):
        path = ROOT / "raw_data" / sub
        path.mkdir(parents=True, exist_ok=True)


def main() -> int:
    print("╭──────────────────────────────────────────────────────────────╮")
    print("│  Derived project bootstrap helper                            │")
    print("╰──────────────────────────────────────────────────────────────╯\n")
    print("Use this only when intentionally bootstrapping a separate derived project repo.\n")

    project = prompt("Project name (e.g. waste-detection)")
    dataset = prompt("Dataset name (Enter = same as project)", default=project)

    try:
        patch_dvc_config(project)
        patch_params_yaml(dataset, project)
    except Exception as exc:
        print(f"Setup failed: {exc}")
        return 1

    print("\n.dvc/config remote URL set to:")
    print(f"   {BASE_URL}/{project}")
    print(f"params.yaml updated (dataset_name = {dataset})\n")

    make_raw_data_dirs()
    print("Created directories for raw data input")

    print(textwrap.dedent(f"""
        ── NEXT STEPS ── (all manual – nothing was executed for you) ─────────
        1. Review the changes:
             git diff

        2. Commit them when happy:
             git add .dvc/config params.yaml
             git commit -m "Init project '{project}'"

        3. Put your data into  raw_data/train/  (and optionally /test/)
           then:

             dvc add raw_data          
             dvc exp run               # runs preflight + bootstrap + prepare + train + evaluate
             dvc exp show

           If you run the pipeline manually without DVC, bootstrap first:

             python -m train --stage bootstrap
             python -m train --stage all

        4. Push whenever you decide:
             dvc push     # uploads data to Hetzner /{project}
             git push
        ──────────────────────────────────────────────────────────────────────
    """))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
