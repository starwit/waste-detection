import argparse
import json
from pathlib import Path
from textwrap import dedent
from typing import Dict, List, Tuple, NamedTuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
import zipfile
import itertools

class FrameStats(NamedTuple):
    detection_count: int
    confidences: List[float]


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Annotate video frames, filter and store as YOLO dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("-i", "--input-dir", required=True, type=Path, help="Input directory containing videos")
    parser.add_argument("-p", "--pattern", default="*.mp4", help="Globbing pattern for video files")
    parser.add_argument("-o", "--output-dir", default=Path("./output"), type=Path, help="Output directory for filtered frames")
    parser.add_argument("-c", "--min-count", type=int, default=1, help="Minimum object count threshold")
    parser.add_argument("-w", "--weights", required=True, type=Path, help="Path to YOLO model weights")
    parser.add_argument("-n", "--min-interval", type=int, default=100, help="Minimum frame interval between picks")
    parser.add_argument("-f", "--min-confidence", type=float, default=0.0, help="Minimum detection confidence for objects to be considered")
    return parser.parse_args()


def get_video_files(input_dir: Path, pattern: str) -> List[Path]:
    """Get list of video files matching the pattern."""
    return sorted(input_dir.glob(pattern))


def extract_frames_from_video(video_path: Path):
    """Extract frames from a video file as a generator."""
    cap = cv2.VideoCapture(str(video_path))
    
    # Try to get total frame count for progress bar
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        total_frames = None  # Unknown frame count
    
    frame_idx = 0
    pbar = tqdm(total=total_frames, desc=f"Processing {video_path.name}", unit="frames")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        yield frame_idx, frame
        frame_idx += 1
        pbar.update(1)
    
    pbar.close()
    cap.release()


def run_inference(model: YOLO, frame: np.ndarray, min_confidence: float) -> List[Dict]:
    """Run YOLO inference on a single frame."""
    results = model(frame, verbose=False)
    detections = []
    
    for result in results:
        boxes = result.boxes
        if boxes is not None:
            for box in boxes:
                detection = {
                    'class_id': int(box.cls.item()),
                    'confidence': float(box.conf.item()),
                    'bbox': box.xywhn.cpu().numpy().tolist()[0]  # normalized coordinates
                }
                if detection['confidence'] >= min_confidence:
                    detections.append(detection)
    
    return detections


def filter_frames_by_interval(frame_candidates: List[Tuple[int, int]], min_interval: int) -> List[Tuple[int, int]]:
    """Filter frames to maintain minimum interval between selected frames."""
    if not frame_candidates:
        return []
    
    # Sort by frame index
    sorted_candidates = sorted(frame_candidates, key=lambda x: x[0])
    selected = [sorted_candidates[0]]
    
    for frame_idx, count in sorted_candidates[1:]:
        if frame_idx - selected[-1][0] >= min_interval:
            selected.append((frame_idx, count))
    
    return selected


def save_frame_and_label(frame: np.ndarray, detections: List[Dict], output_dir: Path, filename: str):
    """Save frame as JPG and detections in YOLO format."""
    # Save frame
    frame_path = output_dir / 'images' / 'train' / f"{filename}.jpg"
    cv2.imwrite(str(frame_path), frame)

    # Use ultralytics annotation tool to also save annotated frames into separate directory
    annotated_frame_path = output_dir / 'annotated_images' / f"{filename}.jpg"
    annotated_frame = frame.copy()
    annotator = Annotator(annotated_frame, line_width=2, example=None, font_size=24)
    for det in detections:
        # bbox is [x_center, y_center, width, height] normalized
        h, w = frame.shape[:2]
        x_c, y_c, bw, bh = det['bbox']
        # Convert normalized xywh to pixel xyxy
        x1 = int((x_c - bw / 2) * w)
        y1 = int((y_c - bh / 2) * h)
        x2 = int((x_c + bw / 2) * w)
        y2 = int((y_c + bh / 2) * h)
        label = f"{det['class_id']}, {det['confidence']:.2f}"
        annotator.box_label([x1, y1, x2, y2], label, color=(0,0,255))
    annotated_frame = annotator.result()
    cv2.imwrite(str(annotated_frame_path), annotated_frame)
    
    # Save label
    label_path = output_dir / 'labels' / 'train' / f"{filename}.txt"
    with open(label_path, 'w') as f:
        for det in detections:
            bbox = det['bbox']
            f.write(f"{det['class_id']} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\n")

def create_2d_histogram(values: List[Tuple[float, float]], output_file: Path, title: str, x_label: str, y_label: str):
    """Create and save a 2D histogram from a list of (x, y) values."""
    if not values:
        print("No values provided for 2D histogram.")
        return
    x_vals, y_vals = [v[0] for v in values], [v[1] for v in values]
    plt.figure(figsize=(10, 8))
    plt.hist2d(x_vals, y_vals, bins=(max(x_vals) - min(x_vals) + 1, 10), range=[[min(x_vals), max(x_vals) + 1], [0, 1]], cmap='Blues')
    plt.colorbar(label='Frequency')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

def create_histogram(values: List[float], output_file: Path, title: str, x_label: str):
    """Create and save histogram of object count distribution."""
    plt.figure(figsize=(10, 6))
    plt.hist(values, bins='auto', 
             edgecolor='black', alpha=0.7)
    plt.xlabel(x_label)
    plt.ylabel('Frequency')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

def process_video(video_path: Path, model: YOLO, args) -> Tuple[List[int], List[FrameStats]]:
    """Process a single video file and return object counts of selected frames, all frames, and all confidence values."""
    print(f"Processing video: {video_path}")
    
    video_name = video_path.stem
    selected_frame_stats: List[FrameStats] = []
    all_frame_stats: List[FrameStats] = []
    last_selected_frame = -args.min_interval  # Initialize to allow first frame selection
    
    for frame_idx, frame in extract_frames_from_video(video_path):
        detections = run_inference(model, frame, args.min_confidence)
        object_count = len(detections)
        stats = FrameStats(object_count, [det['confidence'] for det in detections])
        all_frame_stats.append(stats)
        
        # Check if frame meets criteria and interval requirement
        if (object_count >= args.min_count and 
            frame_idx - last_selected_frame >= args.min_interval):
            
            filename = f"{video_name}_frame_{frame_idx:06d}"
            save_frame_and_label(frame, detections, args.output_dir, filename)
            selected_frame_stats.append(stats)
            last_selected_frame = frame_idx
    
    print(f"Selected {len(selected_frame_stats)} frames")
    return selected_frame_stats, all_frame_stats

def main():
    """Main function."""
    args = parse_arguments()
    
    # Create output directories
    (args.output_dir / 'images' / 'train').mkdir(parents=True, exist_ok=True)
    (args.output_dir / 'labels' / 'train').mkdir(parents=True, exist_ok=True)
    (args.output_dir / 'annotated_images').mkdir(parents=True, exist_ok=True)
    
    # Load YOLO model
    print(f"Loading YOLO model from {args.weights}")
    model = YOLO(str(args.weights))
    print(f'Model has classes {model.names}')
    
    # Get video files
    video_files = get_video_files(args.input_dir, args.pattern)
    print(f"Found {len(video_files)} video files")
    
    if not video_files:
        print("No video files found matching the pattern")
        return
    
    # Process all videos
    all_frame_stats: List[FrameStats] = []
    selected_frame_stats: List[FrameStats] = []

    for video_file in video_files:
        selected_stats, all_stats = process_video(video_file, model, args)
        selected_frame_stats.extend(selected_stats)
        all_frame_stats.extend(all_stats)

    # Create data.yaml in output directory to make it a valid YOLO dataset
    data_yaml_content = dedent("""\
        path: .
        train: images/train

        names:
    """) + '\n'.join([f'    {idx}: {name}' for idx, name in model.names.items()])
    data_yaml_path = args.output_dir / 'data.yaml'
    with open(data_yaml_path, 'w') as f:
        f.write(data_yaml_content)

    zip_path = args.output_dir / 'dataset.zip'
    with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_STORED) as zipf:
        # Add data.yaml
        zipf.write(data_yaml_path, arcname='data.yaml')
        # Add images directory
        images_dir = args.output_dir / 'images'
        for file in images_dir.rglob('*'):
            if file.is_file():
                zipf.write(file, arcname=file.relative_to(args.output_dir))
        # Add labels directory
        labels_dir = args.output_dir / 'labels'
        for file in labels_dir.rglob('*'):
            if file.is_file():
                zipf.write(file, arcname=file.relative_to(args.output_dir))
    print(f"Created dataset archive at {zip_path}")
    
    # Create histograms
    if selected_frame_stats:
        detection_counts = [s.detection_count for s in selected_frame_stats]
        create_histogram(detection_counts, args.output_dir / 'object_count_histogram_selected.png', title='Distribution of Object Counts in Selected Frames', x_label='Object Count')
        print(f"Saved {len(detection_counts)} frames total")
        print(f"Selected frames object count statistics: min={min(detection_counts)}, max={max(detection_counts)}, avg={np.mean(detection_counts):.2f}")
    else:
        print("No frames met the criteria")
    
    if all_frame_stats:
        detection_counts = [s.detection_count for s in all_frame_stats]
        confidences = list(itertools.chain.from_iterable([s.confidences for s in all_frame_stats]))
        create_histogram(detection_counts, args.output_dir / 'object_count_histogram_all.png', title='Distribution of Object Counts in All Frames', x_label='Object Count')
        create_histogram(confidences, args.output_dir / 'confidence_histogram_all.png', title='Distribution of Detection Confidence', x_label='Confidence')
        counts_confidences = [(s.detection_count, conf) for s in all_frame_stats for conf in s.confidences]
        create_2d_histogram(counts_confidences, args.output_dir / 'count_confidences_histogram_all.png', title='Count vs Confidence', x_label='Frame with count', y_label='Confidences')
        print(f"All frames object count statistics: min={min(detection_counts)}, max={max(detection_counts)}, avg={np.mean(detection_counts):.2f}")
        print(f"All frames confidence statistics: min={min(confidences)}, max={max(confidences)}, avg={np.mean(confidences):.2f}")

    
    # Save processing summary
    summary = {
        'total_videos': len(video_files),
        'total_frames_processed': len(all_frame_stats),
        'total_frames_saved': len(selected_frame_stats),
        'min_count_threshold': args.min_count,
        'min_interval': args.min_interval,
    }
    
    summary_path = args.output_dir / 'processing_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

if __name__ == "__main__":
    main()
