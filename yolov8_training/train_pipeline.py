import argparse
import os
import time
import shutil
import random
from pathlib import Path
import numpy as np
import torch
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

def train_model(
    dataset_path, model_size, image_size, batch_size, experiment_name, epochs=100
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

    Returns:
        model (YOLO): The trained YOLO model.
        results: Training results.
        Path: Directory path of the training output.
    """
    model = YOLO(f"yolov8{model_size}.pt")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Define project name and ensure unique run name
    project = "runs"
    base_name = experiment_name
    name = base_name
    run_number = 1
    while (Path(project) / name).exists():
        name = f"{base_name}_{run_number}"
        run_number += 1

    results = model.train(
        data=str(dataset_path / "dataset.yaml"),
        epochs=epochs,
        imgsz=image_size,
        batch=batch_size,
        device=device,
        workers=4,
        amp=True,
        project=project,
        name=name,
    )

    return model, results, Path(project) / name


def process_data(
    image_input_path,
    train_output_path,
    test_output_path,
    val_split,
    test_split,
    augment_multiplier
):
    """
    Process image data for training and validation.

    Args:
        image_input_path (Path): Path to image input data.
        train_output_path (Path): Path to store training data.
        test_output_path (Path): Path to store test data.
        val_split (float): Validation split ratio.
        test_split (float): Test split ratio.

    Returns:
        int, int, int: Total frames for training, validation and test.
    """
    if image_input_path.exists():
        return process_single_images(
            image_input_path, train_output_path, test_output_path, val_split, test_split, augment_multiplier
        )
    return 0, 0, 0

def delete_unused_folders():
    """
    Deletes unused/empty folders in the runs/ directory 
    """
    print("Checking for unused folders in 'runs/' directory...")
    runs_dir = Path("runs")
    for folder in runs_dir.iterdir():
        if folder.is_dir() and not any(folder.iterdir()):
            print(f"Deleting empty folder: {folder}")
            shutil.rmtree(folder)



def run_prepare_stage(args):
    dataset_name = Path(args.dataset_name)
    val_split = float(args.val_split)
    recreate_dataset = args.recreate_dataset
    augment_multiplier=args.augment_multiplier
    
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
            augment_multiplier=augment_multiplier
        )

        create_dataset_yaml(training_path)

        test_folder_frame_count = 0
        if test_data_exists:
            # Process test data
            _, test_folder_frame_count, _ = process_data(
                image_input_path=test_image_input_path,
                train_output_path=test_path,
                test_output_path=test_path,
                val_split=1,
                test_split=0,
                augment_multiplier=1
            )

        create_dataset_yaml(test_path)

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
    )

    # Evaluate and log results for the original model
    evaluate_and_log_model_results(
        model=YOLO(f"yolov8{model_size}.pt"),
        model_name=experiment_name,
        test_path=test_path,
        image_size=image_size,
        output_dir=train_output_dir,
        val_split=val_split,
        train_epochs=0,
        is_original=True,
    )

    # Evaluate and log results for the retrained model
    retrained_metadata = evaluate_and_log_model_results(
        model=model,
        model_name=experiment_name,
        test_path=test_path,
        image_size=image_size,
        output_dir=train_output_dir,
        val_split=val_split,
        train_epochs=train_epochs,
    )

    # Organize output files
    reorganize_output(train_output_dir, training_path, test_path, retrained_metadata)

    # Generate side-by-side comparisons
    generate_side_by_side_comparisons(
        original_model=YOLO(f"yolov8{model_size}.pt"),
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
