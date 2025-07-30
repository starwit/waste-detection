import matplotlib.pyplot as plt
from pathlib import Path
import os
import numpy as np
import cv2
from tqdm import tqdm


def create_annotated_video(
    base_path, output_path, video_filename, video_fps=30, display_first_frame=False
):
    base_path = Path(base_path)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    frames_path = base_path / "frames"
    labels_path = base_path / "labels"

    # frames_path = base_path / "splits" / "fold_0" / "frames"
    # labels_path = base_path / "labels"

    """
    # List all files in the directory
    files = os.listdir(labels_path)

    # Sort files to ensure correct order
    files.sort()

    # Delete files from frame_000000.txt to frame_000018.txt
    for i in range(19):
        file_to_delete = f"frame_{i:06d}.txt"
        if file_to_delete in files:
            os.remove(os.path.join(labels_path, file_to_delete))
            files.remove(file_to_delete)  # Remove from list as well

    # Rename the remaining files
    for i, filename in enumerate(files):
        new_filename = f"frame_{i:06d}.txt"
        os.rename(os.path.join(labels_path, filename), os.path.join(labels_path, new_filename))

    print("Files have been deleted and renamed successfully.")
    """

    image_files = sorted([f for f in os.listdir(frames_path) if f.endswith(".jpg")])

    # Get video dimensions from the first image
    first_image = cv2.imread(str(frames_path / image_files[0]))
    height, width = first_image.shape[:2]

    # Create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(
        str(output_path / video_filename), fourcc, video_fps, (width, height)
    )

    i = 0
    for image_filename in tqdm(image_files, desc="Processing images"):
        # for image_filename in image_files:
        i = i + 1
        # print(i)
        # frame_number = image_filename.split('_')[1].split('.')[0]
        image_path = frames_path / image_filename
        # label_path = labels_path / f"frame_{frame_number}.txt"
        label_path = labels_path / Path(image_filename).with_suffix(".txt")

        # Read image and labels
        image = cv2.imread(str(image_path))
        # image = cv2.resize(image, (1920, 1080))
        # print(image.shape)

        with open(label_path, "r") as f:
            labels = f.readlines()

        # Annotate image
        for line in labels:
            parts = line.strip().split()
            if len(parts) == 5:
                label, x_center, y_center, w, h = parts
                x_center, y_center, w, h = map(float, (x_center, y_center, w, h))

                # Convert YOLO format to pixel coordinates
                x1 = int((x_center - w / 2) * width)
                y1 = int((y_center - h / 2) * height)
                x2 = int((x_center + w / 2) * width)
                y2 = int((y_center + h / 2) * height)

                # Draw bounding box
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Add label
                cv2.putText(
                    image,
                    label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 0),
                    2,
                )

        # Write the frame to video
        out.write(image)

        # Display the first frame if the flag is set
        if display_first_frame:
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            plt.title("First Annotated Frame")
            plt.show()
            break

    # Release the VideoWriter
    out.release()
    print(f"Video saved as {output_path / video_filename}")


# Usage
base_path = "datasets/train/num_test/train"
output_path = base_path
video_filename = "annotated_train_video.mp4"
video_fps = 25

create_annotated_video(
    base_path, output_path, video_filename, video_fps, display_first_frame=False
)
