#!/usr/bin/env python3

import argparse
import re
from pathlib import Path
from typing import List, Tuple

from src.classifier import (find_cleaning_segments_classifier,
                            run_cleaning_classifier)
from src.gps import (FacilityArea, calculate_speeds, extract_gps_segment,
                     find_cleaning_segments_gps, parse_gps_log)
from src.types import CleaningSegment, FacilityArea
from src.video import extract_video_segment, parse_video_timestamp


def get_file_pairs(input_dir: Path) -> List[Tuple[Path, Path]]:
    """Find matching video and GPS file pairs."""
    pairs = []
    video_pattern = re.compile(r'(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})_video\.(mkv|mp4|avi)$')
    
    for video_file in input_dir.glob('*_video.*'):
        match = video_pattern.match(video_file.name)
        if match:
            prefix = match.group(1)
            gps_file = input_dir / f"{prefix}_gps.log"
            if gps_file.exists():
                pairs.append((video_file, gps_file))
                
    return pairs


def parse_facility_area(area_str: str) -> FacilityArea:
    """Parse facility area from command line string format."""
    coords = [float(x.strip()) for x in area_str.split(',')]
    if len(coords) != 4:
        raise ValueError("Facility area must have 4 coordinates: lat1,lon1,lat2,lon2")
    return FacilityArea((coords[0], coords[1]), (coords[2], coords[3]))


def intersect_segments(segments_a: List[CleaningSegment], segments_b: List[CleaningSegment]) -> List[CleaningSegment]:
    filtered_segments: List[CleaningSegment] = []

    for a in segments_a:
        for b in segments_b:
            start = max(a.start_offset, b.start_offset)
            end = min(a.end_offset, b.end_offset)
            if (end-start).total_seconds() > 0:
                filtered_segments.append(CleaningSegment(
                    start_offset=start,
                    end_offset=end,
                ))
            
    return filtered_segments


def main():
    parser = argparse.ArgumentParser(description='Filter cleaning runs from street cleaner videos')
    parser.add_argument('input_dir', type=Path, help='Directory containing video and GPS files')
    parser.add_argument('output_dir', type=Path, help='Output directory for filtered video segments')
    parser.add_argument('--min-duration', type=int, default=120,
                        help='Minimum duration (seconds) for cleaning segments (default: 120)')
    parser.add_argument('--classifier-weights', type=Path, 
                        help='Weights for cleaning run classifier (needs to output classes [notcleaning, cleaning])')
    parser.add_argument('--classifier-stride-sec', type=int, default=1,
                        help='Stride for classifier in seconds (default: 1)')
    parser.add_argument('--classifier-threshold', type=float, default=0.9,
                        help='Threshold for classifier to consider a segment as cleaning (default: 0.9)')
    parser.add_argument('--classifier-debounce-interval', type=int, default=3,
                        help='How many consecutive "cleaning" results are required (default: 3)')
    parser.add_argument('--gps-max-speed', type=float, default=20.0, 
                        help='Maximum speed (km/h) for cleaning activity (default: 20.0)')
    parser.add_argument('--gps-min-distance', type=float, default=20.0,
                        help='Minimum maximum extent (meters) of segment (default: 20.0)')
    parser.add_argument('--gps-speed-window-size', type=int, default=5,
                        help='Window size for speed calculation (default: 5)')
    parser.add_argument('--facility-area', type=str,
                        help='Facility area to exclude: lat1,lon1,lat2,lon2')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.input_dir.exists():
        print(f"Error: Input directory {args.input_dir} does not exist")
        return 1
        
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    facility_area = None
    if args.facility_area:
        try:
            facility_area = parse_facility_area(args.facility_area)
        except ValueError as e:
            print(f"Error: {e}")
            return 1
    
    # Find file pairs
    file_pairs = get_file_pairs(args.input_dir)
    print(f"Found {len(file_pairs)} video/GPS file pairs")
    
    total_segments = 0
    
    for video_file, gps_file in file_pairs:
        print(f"Processing {video_file.name}...")
        
        # Parse GPS data
        print(f"  Parse corresponding GPS log {gps_file.name}")
        gps_points = parse_gps_log(gps_file)
        if len(gps_points) < 2:
            print(f"  Warning: Insufficient GPS data in {gps_file.name}")
            continue
            
        # Calculate speeds
        gps_points = calculate_speeds(gps_points, args.gps_speed_window_size)

        # Find cleaning segments using GPS data
        segments: List[CleaningSegment] = find_cleaning_segments_gps(gps_points, args.gps_max_speed, 
                                        args.min_duration, args.gps_min_distance, facility_area)
        print(f"  GPS filter found {len(segments)} cleaning segments")
        
        # Find cleaning segments using a visual classifier (i.e. is equipment active)
        if args.classifier_weights:
            print(f"  Running cleaning classifier")
            classifier_output = run_cleaning_classifier(video_file, args.classifier_weights, args.classifier_stride_sec, args.classifier_debounce_interval, args.classifier_threshold)
            classifier_segments = find_cleaning_segments_classifier(classifier_output, args.min_duration)
            print(f"  Visual classification found {len(classifier_segments)} cleaning segments. Merging GPS segments.")
            segments = intersect_segments(classifier_segments, segments)

        print(f"  Found {len(segments)} cleaning segments")
        
        # Extract video and GPS segments
        for i, segment in enumerate(segments):
            # Video segment
            video_output_name = f"{video_file.stem}_segment_{i:03d}{video_file.suffix}"
            video_output_file = args.output_dir / video_output_name
            
            # GPS segment
            gps_output_name = f"{gps_file.stem}_segment_{i:03d}.log"
            gps_output_file = args.output_dir / gps_output_name
            
            video_success = extract_video_segment(video_file, video_output_file, segment)
            
            gps_success = extract_gps_segment(gps_file, gps_output_file, segment)
            
            if video_success and gps_success:
                duration = (segment.end_offset - segment.start_offset).total_seconds()
                print(f"    Extracted segment {i+1}: {duration:.1f}s -> {video_output_name}, {gps_output_name}")
                total_segments += 1
            else:
                print(f"    Failed to extract segment {i+1} (video: {video_success}, gps: {gps_success})")
    
    print(f"\nProcessing complete. Extracted {total_segments} cleaning segments.")
    return 0


if __name__ == '__main__':
    exit(main())
