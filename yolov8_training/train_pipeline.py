import argparse
import os
import time
import shutil
import random
from pathlib import Path
import numpy as np
import torch
import yaml
from ultralytics import YOLO
from yolov8_training.utils.data_utils import (
    check_for_test_images,
    create_dataset_yaml,
    process_single_images,
    reorganize_output,
)
from yolov8_training.utils.evaluate import (
    evaluate_and_log_model_results,
    generate_side_by_side_comparisons,
)

from yolov8_training.utils.find_duplicates import DuplicateDetector


def _load_params_yaml():
    """Load and return params from params.yaml as a dict, or {} on failure."""
    try:
        with open("params.yaml", "r") as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        print(f"Warning: Could not load params.yaml: {e}")
        return {}

def load_class_config(params: dict | None = None):
    """Load class configuration from params.yaml"""
    if params is None:
        params = _load_params_yaml()

    custom_classes = params.get("data", {}).get("custom_classes", [])
    use_coco_classes = params.get("data", {}).get("use_coco_classes", True)

    # Filter out None/empty values from custom_classes
    if custom_classes:
        custom_classes = [cls for cls in custom_classes if cls]

    return custom_classes, use_coco_classes

def load_folder_subset_config(params: dict | None = None):
    """Load folder subset configuration from params.yaml"""
    if params is None:
        params = _load_params_yaml()
    return params.get("prepare", {}).get("folder_subsets", {})


def _resolve_save_dir(model, results, default: Path) -> Path:
    """Return Ultralytics' actual save_dir if available, else default.

    Keeps train_model concise while handling version differences
    (trainer.save_dir vs. results.save_dir).
    """
    try:
        trainer = getattr(model, "trainer", None)
        candidate = getattr(trainer, "save_dir", None) if trainer is not None else None
        if candidate:
            return Path(candidate)
    except Exception:
        pass

    try:
        candidate = getattr(results, "save_dir", None)
        if candidate:
            return Path(candidate)
    except Exception:
        pass

    return default


def _load_baseline_from_path(path_candidate: str | None) -> tuple[YOLO | None, str | None]:
    """Return YOLO model and display name loaded from the given path if it exists."""
    if not path_candidate:
        return None, None

    candidate_path = Path(path_candidate).expanduser()
    if not candidate_path.is_absolute():
        candidate_path = Path.cwd() / candidate_path

    if not candidate_path.exists():
        print(f"Warning: Baseline weights not found at {candidate_path}")
        return None, None

    try:
        model_instance = YOLO(str(candidate_path))
    except Exception as load_error:
        print(f"Warning: Could not load baseline model from {candidate_path}: {load_error}")
        return None, None

    display_name = None
    try:
        meta_path = candidate_path.parent / "metadata.yaml"
        if meta_path.exists():
            with open(meta_path, "r") as mf:
                meta = yaml.safe_load(mf) or {}
            display_name = (
                meta.get("experiment_name")
                or meta.get("baseline_display_name")
                or meta.get("run_name")
            )
    except Exception:
        display_name = None

    return model_instance, (display_name or candidate_path.stem)

def train_model(
    dataset_path, model_size, image_size, batch_size, experiment_name, epochs=100,
    finetune_mode=False, pretrained_model_path=None, finetune_lr=None, freeze_backbone=False
):
    """
    Train the YOLO model on the specified dataset.

    Args:
        dataset_path (Path): Path to the dataset directory.
        model_size (str): Size of the YOLO model to train (e.g., 's', 'm', 'l').
        image_size (int): Size of images for training.
        batch_size (int): Batch size for training.
        experiment_name (str): Name for the experiment.
        epochs (int): Number of training epochs.
        finetune_mode (bool): Whether to use fine-tuning mode.
        pretrained_model_path (str): Path to pre-trained model for fine-tuning.
        finetune_lr (float): Learning rate for fine-tuning.
        freeze_backbone (bool): Whether to freeze backbone layers during fine-tuning.

    Returns:
        model (YOLO): The trained YOLO model.
        results: Training results.
        Path: Directory path of the training output.
    """
    # Choose model based on fine-tuning mode
    if finetune_mode and pretrained_model_path and Path(pretrained_model_path).exists():
        print(f"Fine-tuning mode enabled. Loading pre-trained model: {pretrained_model_path}")
        model = YOLO(pretrained_model_path)
        
        # Update experiment name to indicate fine-tuning
        experiment_name = f"{experiment_name}-finetune"
    else:
        if finetune_mode:
            print(f"Warning: Fine-tuning mode enabled but pre-trained model not found at {pretrained_model_path}")
            print("Falling back to training from scratch with YOLO checkpoint")
        # Robust fallback: attempt to load the official YOLO checkpoint.
        # If unavailable (e.g., offline and not cached), raise a clear error.
        checkpoint_name = f"yolov8{model_size}.pt"
        try:
            model = YOLO(checkpoint_name)
        except Exception as e:
            raise RuntimeError(
                "Failed to initialize training without local weights. "
                f"Tried to load official checkpoint '{checkpoint_name}' but it could not be "
                "downloaded or loaded. Ensure network access or provide a valid "
                "train.pretrained_model_path in params.yaml."
            ) from e
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Define project name and ensure unique run name
    project = "runs"
    name = None
    if experiment_name:
        base_name = experiment_name
        name = base_name
        run_number = 1
        while (Path(project) / name).exists():
            name = f"{base_name}_{run_number}"
            run_number += 1

    # Prepare training arguments
    train_args = {
        "data": str(dataset_path / "dataset.yaml"),
        "epochs": epochs,
        "imgsz": image_size,
        "batch": batch_size,
        "device": device,
        "workers": 4,
        "amp": True,
        "project": project,
    }
    if name:
        train_args["name"] = name
    
    # Add fine-tuning specific parameters
    if finetune_mode:
        if finetune_lr is not None:
            train_args["lr0"] = finetune_lr
            print(f"Using fine-tuning learning rate: {finetune_lr}")
        
        if freeze_backbone:
            # Freeze backbone layers (layers 0-9 typically for YOLOv8)
            train_args["freeze"] = list(range(10))
            print("Freezing backbone layers for fine-tuning")

    results = model.train(**train_args)

    default_dir = Path(project) / (name if name else "train")
    output_dir = _resolve_save_dir(model, results, default_dir)

    # Keep return signature compatible with tests: (model, results, output_dir)
    return model, results, output_dir


def process_data(
    image_input_path,
    train_output_path,
    test_output_path,
    val_split,
    test_split,
    augment_multiplier,
    custom_classes=None,
    use_coco_classes=True,
    folder_subsets=None
):
    """
    Process image data for training and validation.

    Args:
        image_input_path (Path): Path to image input data.
        train_output_path (Path): Path to store training data.
        test_output_path (Path): Path to store test data.
        val_split (float): Validation split ratio.
        test_split (float): Test split ratio.
        augment_multiplier (int): Augmentation multiplier.
        custom_classes (list): List of custom class names.
        use_coco_classes (bool): Whether to use COCO classes when custom_classes is empty.
        folder_subsets (dict): Dictionary mapping folder names to subset ratios (0.0-1.0).

    Returns:
        int, int, int: Total frames for training, validation and test.
    """
    if image_input_path.exists():
        return process_single_images(
            image_input_path, train_output_path, test_output_path, val_split, test_split, 
            augment_multiplier, custom_classes, use_coco_classes, folder_subsets
        )
    return 0, 0, 0

def delete_unused_folders():
    """
    Deletes unused/empty folders in the runs/ directory 
    """
    print("Checking for unused folders in 'runs/' directory...")
    runs_dir = Path("runs")
    if not runs_dir.exists():
        return
    for folder in runs_dir.iterdir():
        if folder.is_dir() and not any(folder.iterdir()):
            print(f"Deleting empty folder: {folder}")
            shutil.rmtree(folder)



def run_prepare_stage(args):
    dataset_name = Path(args.dataset_name)
    val_split = float(args.val_split)
    recreate_dataset = args.recreate_dataset
    augment_multiplier=args.augment_multiplier
    
    # Load params once to avoid multiple file reads
    _params = _load_params_yaml()
    # Load class configuration
    custom_classes, use_coco_classes = load_class_config(_params)
    # Load folder subset configuration
    folder_subsets = load_folder_subset_config(_params)
    
    # Override with command-line arguments if provided
    if args.folder_subset:
        print("Overriding folder subset configuration with command-line arguments:")
        for folder_name, ratio_str in args.folder_subset:
            try:
                ratio = float(ratio_str)
                if ratio > 0:
                    folder_subsets[folder_name] = ratio
                    if ratio > 1:
                        print(f"  {folder_name}: {ratio*100:.1f}% (oversampling)")
                    else:
                        print(f"  {folder_name}: {ratio*100:.1f}%")
                else:
                    print(f"  Warning: Invalid ratio {ratio} for {folder_name}. Must be > 0.")
            except ValueError:
                print(f"  Warning: Invalid ratio '{ratio_str}' for {folder_name}. Must be a number.")
    
    if folder_subsets:
        print(f"Final folder subset configuration: {folder_subsets}")
    else:
        print("No folder subset configuration found. Using all images from all folders.")

    if custom_classes and use_coco_classes:
        raise ValueError(
            "Both 'custom_classes' and 'use_coco_classes' are set. "
            "Please choose one strategy."
        )
        
    
    # Define paths
    base_input_path = Path("raw_data")
    train_image_input_path = base_input_path / "train"
    test_image_input_path = base_input_path / "test"

    dataset_path = Path("datasets") / dataset_name
    training_path = dataset_path / "train"
    test_path = dataset_path / "test"

    test_data_exists = check_for_test_images(test_image_input_path)

    if not test_data_exists:
        test_split = args.test_split
        print(f"No dedicated test data found. Using {test_split} for test split.")
    else:
        test_split = 0
        print(f"Dedicated test data found. Using {test_split} for test split.")

    # Create or recreate dataset directory if specified
    if not dataset_path.exists() or recreate_dataset:
        if dataset_path.exists() and recreate_dataset:
            print(f"Recreating dataset '{dataset_name}'...")
            shutil.rmtree(dataset_path)

        dataset_path.mkdir(parents=True, exist_ok=True)
        for path in [training_path, test_path]:
            path.mkdir(parents=True, exist_ok=True)

        # Directories for training/validation frames and labels
        train_img_dir = training_path / "train" / "images"
        train_label_dir = training_path / "train" / "labels"
        val_img_dir = training_path / "val" / "images"
        val_label_dir = training_path / "val" / "labels"
        test_img_dir = test_path / "val" / "images"
        test_label_dir = test_path / "val" / "labels"

        for path in [
            train_img_dir,
            train_label_dir,
            val_img_dir,
            val_label_dir,
            test_img_dir,
            test_label_dir,
        ]:
            path.mkdir(parents=True, exist_ok=True)

        # Process training data
        total_train_frames, total_val_frames, total_test_frames = process_data(
            image_input_path=train_image_input_path,
            train_output_path=training_path,
            test_output_path=test_path,
            val_split=val_split,
            test_split=test_split,
            augment_multiplier=augment_multiplier,
            custom_classes=custom_classes,
            use_coco_classes=use_coco_classes,
            folder_subsets=folder_subsets
        )

        create_dataset_yaml(training_path, custom_classes, use_coco_classes)

        test_folder_frame_count = 0
        if test_data_exists:
            # Process test data
            _, test_folder_frame_count, _ = process_data(
                image_input_path=test_image_input_path,
                train_output_path=test_path,
                test_output_path=test_path,
                val_split=1,
                test_split=0,
                augment_multiplier=1,
                custom_classes=custom_classes,
                use_coco_classes=use_coco_classes,
                folder_subsets={}  # Don't apply subsets to test data
            )

        create_dataset_yaml(test_path, custom_classes, use_coco_classes)

        print(f"Total training frames: {total_train_frames}")
        print(f"Total validation frames: {total_val_frames}")
        print(f"Total test frames: {total_test_frames + test_folder_frame_count}")
    else:
        print(f"Dataset '{dataset_name}' already exists. Skipping dataset creation.")

    print("Testing for duplicates between train and test folders...")
    # Initialize duplicate detector to compare the train and test folders
    detector = DuplicateDetector(phash_threshold=2, ssim_threshold=0.95)

    # Compare folders
    matches = detector.compare_folders(training_path, test_path)

    # Print results
    detector.print_folder_comparison_results(matches)
    return

def run_train_eval_stage(args):
    experiment_name = os.getenv("DVC_EXP_NAME")
    train_epochs = int(args.epochs)
    model_size = args.model_size
    image_size = int(args.image_size)
    batch_size = int(args.batch_size)
    val_split = float(args.val_split)

    # Load fine-tuning parameters from params.yaml
    try:
        params = _load_params_yaml()
        train_params = params.get("train", {})
        evaluation_params = params.get("evaluation", {})
        finetune_mode = train_params.get("finetune_mode", False)
        pretrained_model_path = train_params.get("pretrained_model_path", None)
        finetune_lr = train_params.get("finetune_lr", None)
        finetune_epochs = train_params.get("finetune_epochs", None)
        freeze_backbone = train_params.get("freeze_backbone", False)
        baseline_weights_path = evaluation_params.get("baseline_weights_path")
        # Use fine-tuning epochs if in fine-tuning mode and specified
        if finetune_mode and finetune_epochs is not None:
            train_epochs = finetune_epochs
            print(f"Fine-tuning mode: Using {finetune_epochs} epochs")
    except Exception as e:
        print(f"Warning: Could not load fine-tuning params from params.yaml: {e}")
        finetune_mode = False
        pretrained_model_path = None
        finetune_lr = None
        freeze_backbone = False
        baseline_weights_path = None

    # Define paths
    dataset_name = Path(args.dataset_name)
    dataset_path = Path("datasets") / dataset_name
    training_path = dataset_path / "train"
    test_path = dataset_path / "test"

    # Train the model
    model, results, train_output_dir = train_model(
        training_path,
        model_size,
        image_size,
        batch_size,
        experiment_name,
        epochs=train_epochs,
        finetune_mode=finetune_mode,
        pretrained_model_path=pretrained_model_path,
        finetune_lr=finetune_lr,
        freeze_backbone=freeze_backbone,
    )

    # Determine the final experiment display name (append suffix in finetune mode)
    final_experiment_name = (
        f"{experiment_name}-finetune" if experiment_name and finetune_mode and pretrained_model_path and Path(pretrained_model_path).exists() else experiment_name
    )

    # Determine baseline model for comparison
    baseline_model = None
    baseline_display_name = None

    # 1) Highest priority: explicitly configured baseline weights
    baseline_model, baseline_display_name = _load_baseline_from_path(baseline_weights_path)

    # 2) Next, fall back to the pre-trained weights used for fine-tuning (if available)
    if baseline_model is None and finetune_mode and pretrained_model_path and Path(pretrained_model_path).exists():
        baseline_model, baseline_display_name = _load_baseline_from_path(pretrained_model_path)

    # 3) Final fallback: COCO checkpoint matching the requested model size
    if baseline_model is None:
        baseline_checkpoint = f"yolov8{model_size}.pt"
        try:
            baseline_model = YOLO(baseline_checkpoint)
        except Exception as e:
            raise RuntimeError(
                "Failed to load a baseline model. No local baseline weights were found "
                "and loading the official YOLO checkpoint also failed. "
                f"Attempted checkpoint: '{baseline_checkpoint}'. Ensure network access or set "
                "evaluation.baseline_weights_path to a valid local file."
            ) from e
        baseline_display_name = f"yolov8{model_size}-coco"

    # Evaluate and log results for the baseline model
    evaluate_and_log_model_results(
        model=baseline_model,
        model_name=baseline_display_name or "baseline",
        test_path=test_path,
        image_size=image_size,
        output_dir=train_output_dir,
        val_split=val_split,
        train_epochs=0,
        is_original=True,
        baseline_model=None,  # Don't pass baseline for the baseline evaluation
    )

    # Evaluate and log results for the trained/fine-tuned model
    # Note: final_experiment_name includes the "-finetune" suffix if in fine-tuning mode
    retrained_metadata = evaluate_and_log_model_results(
        model=model,
        model_name=final_experiment_name,
        test_path=test_path,
        image_size=image_size,
        output_dir=train_output_dir,
        val_split=val_split,
        train_epochs=train_epochs,
        baseline_model=baseline_model,  # Pass the correct baseline model
        baseline_display_name=baseline_display_name,
    )

    # Organize output files
    reorganize_output(train_output_dir, training_path, test_path, retrained_metadata)

    # Generate side-by-side comparisons
    generate_side_by_side_comparisons(
        original_model=baseline_model,
        retrained_model=model,
        test_img_dir=test_path / "val" / "images",
        output_dir=train_output_dir,
    )

    delete_unused_folders()

def main(args):
    """
    Main function to set up dataset, process data, and train the YOLO model.

    Args:
        args: Parsed command line arguments.
    """
    run_prepare_stage(args)
    run_train_eval_stage(args)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=["prepare", "train"],
        help="Specify which pipeline stage to run."
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "-d", "--dataset-name", required=True, help="Name of the dataset"
    )
    """
    parser.add_argument(
        "-name", "--experiment-name", required=True, help="Name of the experiment"
    )
    """
    parser.add_argument(
        "-ms",
        "--model-size",
        choices=["n", "s", "m", "l", "x"],
        default="m",
        help="YOLO model size",
    )
    parser.add_argument(
        "-ims", "--image-size", type=int, default=640, help="Image size"
    )
    parser.add_argument("-e", "--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument(
        "-b", "--batch-size", type=int, default=8, help="Training batch size"
    )
    parser.add_argument(
        "-vs", "--val-split", type=float, default=0.1, help="Validation split ratio"
    )
    parser.add_argument(
        "-ts", "--test-split", type=float, default=0.1, help="Test split ratio"
    )
    parser.add_argument(
        "--recreate-dataset", action="store_true", help="Recreate dataset if it exists"
    )
    parser.add_argument(
        "--augment-multiplier", type=int, default=1, 
        help="By which factor the training data will be multiplied with augmented data. Default is 1, meaning there is no added augmentation by default."
    )
    parser.add_argument(
        "--folder-subset", action="append", nargs=2, metavar=('FOLDER', 'RATIO'),
        help="Use subset/oversample images from specific folders. Format: --folder-subset uavvaste 0.5 (50%%) --folder-subset small_dataset 2.0 (200%% = oversample)"
    )
    args = parser.parse_args()

    # Add seeds for reproducibility
    SEED = args.seed

    os.environ["PYTHONHASHSEED"] = str(SEED)

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False  

    # --- This logic calls the correct function based on the --stage argument ---
    if args.stage == "prepare":
        run_prepare_stage(args)
    elif args.stage == "train":
        run_train_eval_stage(args)
    else:  
        main(args)
