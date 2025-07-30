import os
from collections import defaultdict
from ultralytics import YOLO
from PIL import Image
import torch
from tqdm import tqdm

def count_objects(image_path):
    # Run YOLOv8 inference on the image
    results = model(image_path, verbose = False)
    
    # Filter for cars, trucks, and buses
    vehicle_count = sum(1 for detection in results[0].boxes.data if int(detection[5]) in [2, 5, 7])  # 2=car, 5=bus, 7=truck in COCO dataset
    
    return vehicle_count

def get_scene_name(filename):
    # Split the filename at the first underscore
    parts = filename.split('_', 1)
    if len(parts) > 0:
        return parts[0]
    else:
        # If there's no underscore, return the whole filename (without extension)
        return os.path.splitext(filename)[0]

def analyze_folder(folder_path):
    scene_stats = defaultdict(lambda: {'total': 0, '0': 0, '1': 0, '2': 0, '3': 0, 'more': 0})
    overall_stats = {'total': 0, '0': 0, '1': 0, '2': 0, '3': 0, 'more': 0}

    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]

    for filename in tqdm(image_files, desc="Processing images", unit="image"):      
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            overall_stats['total'] += 1
            image_path = os.path.join(folder_path, filename)
            
            scene_name = get_scene_name(filename)
            scene_stats[scene_name]['total'] += 1
            
            object_count = count_objects(image_path)
            
            if object_count > 3:
                count_key = 'more'
            else:
                count_key = str(object_count)
            
            scene_stats[scene_name][count_key] += 1
            overall_stats[count_key] += 1

    return scene_stats, overall_stats

def print_stats(stats, name):
    print(f"\nStatistics for {name}:")
    print(f"Total images analyzed: {stats['total']}")
    print(f"Images without any objects: {stats['0']}")
    print(f"Images with 1 object: {stats['1']}")
    print(f"Images with 2 objects: {stats['2']}")
    print(f"Images with 3 objects: {stats['3']}")
    print(f"Images with more than 3 objects: {stats['more']}")

if __name__ == "__main__":
    model = YOLO('yolov8x.pt') 
    
    folder_path = "yolov8_training/raw_data/images/10min_snapshots"
    scene_stats, overall_stats = analyze_folder(folder_path)

    # Print overall statistics
    print_stats(overall_stats, "Overall")

    # Print statistics for each scene
    for scene_name, stats in scene_stats.items():
        print_stats(stats, scene_name)