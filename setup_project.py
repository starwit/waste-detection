"""
────────────
Interactive first-time setup for a repo created from the YOLO-DVC template.

DOES:
  • ask for project & dataset names
  • optionally set custom classes
  • patch .dvc/config   (remote URL)
  • patch params.yaml   (dataset_name, experiment_name, classes)
"""

import configparser
import pathlib
import sys
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
        sys.exit(f"Section {REMOTE_SEC!r} not found in {CONFIG_FILE}")

    cfg[REMOTE_SEC]["url"] = f"{BASE_URL}/{project_slug}"

    with open(CONFIG_FILE, "w") as f:
        cfg.write(f)


def patch_params_yaml(dataset: str, experiment: str) -> None:
    yaml = YAML()
    yaml.preserve_quotes = True

    with PARAMS_FILE.open("r") as f:
        params = yaml.load(f)

    # Basic fields
    params["data"]["dataset_name"] = dataset
    params["data"]["experiment_name"] = experiment

    # Ask about custom classes
    use_custom = input("Do you want to use custom classes? [y/N]: ").strip().lower() == "y"

    if use_custom:
        class_input = input("Enter custom classes (comma-separated): ").strip()
        classes = [c.strip() for c in class_input.split(",") if c.strip()]
        if classes:
            params["data"]["custom_classes"] = classes
            params["data"]["use_coco_classes"] = False
        else:
            print("No valid classes entered. Falling back to COCO classes.")
            params["data"].pop("custom_classes", None)
            params["data"]["use_coco_classes"] = True
    else:
        # COCO fallback: remove custom_classes entirely
        params["data"].pop("custom_classes", None)
        params["data"]["use_coco_classes"] = True

    with PARAMS_FILE.open("w") as f:
        yaml.dump(params, f)



def make_raw_data_dirs() -> None:
    for sub in ("train", "test"):
        path = ROOT / "raw_data" / sub
        path.mkdir(parents=True, exist_ok=True)


def main() -> None:
    print("╭──────────────────────────────────────────────────────────────╮")
    print("│  YOLO-DVC template - initial configuration                   │")
    print("╰──────────────────────────────────────────────────────────────╯\n")

    project = prompt("Project name (e.g. waste-detection)")
    dataset = prompt("Dataset name (Enter = same as project)", default=project)

    patch_dvc_config(project)
    patch_params_yaml(dataset, project)

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
             dvc exp run               # runs prepare + train stages
             dvc exp show

        4. Push whenever you decide:
             dvc push     # uploads data to Hetzner /{project}
             git push
        ──────────────────────────────────────────────────────────────────────
    """))


if __name__ == "__main__":
    main()
