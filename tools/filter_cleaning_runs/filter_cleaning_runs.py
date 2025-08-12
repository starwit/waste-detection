#!/usr/bin/env python3

import argparse
import os
import re
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, NamedTuple, Optional, Tuple

import cv2
from geopy.distance import geodesic
from ultralytics import YOLO


class GPSPoint(NamedTuple):
    timestamp: datetime
    lat: float
    lon: float

class CleaningStatus(NamedTuple):
    timestamp: datetime
    is_cleaning: bool

class CleaningSegment(NamedTuple):
    start_time: datetime
    end_time: datetime
    start_index: int
    end_index: int


class FacilityArea(NamedTuple):
    top_left: Tuple[float, float]  # (lat, lon)
    bottom_right: Tuple[float, float]  # (lat, lon)


def dm_to_deg(dm, direction):
    if not dm or dm == '':
        return None
    if '.' not in dm:
        return None
    d, m = dm.split('.', 1)
    deg = int(d[:-2])
    min = float(d[-2:] + '.' + m)
    val = deg + min / 60
    if direction in ('S', 'W'):
        val = -val
    return val


# Fast NMEA GGA parser (only supports GGA); pynmeagps is extremely slow
def parse_nmea_gga(nmea_str):
    """Parse GGA message. Return None if not GGA"""
    if not nmea_str.startswith('$GPGGA'):
        return None
    parts = nmea_str.split(',')
    if len(parts) < 7:
        return None
    # Latitude
    lat_raw = parts[2]
    lat_dir = parts[3]
    lon_raw = parts[4]
    lon_dir = parts[5]
    quality = int(parts[6]) if parts[6].isdigit() else 0

    lat = dm_to_deg(lat_raw, lat_dir)
    lon = dm_to_deg(lon_raw, lon_dir)
    return {
        'quality': quality,
        'lat': lat,
        'lon': lon
    }


def parse_gps_log(gps_file: Path) -> List[GPSPoint]:
    """Parse GPS log file and extract GGA messages with coordinates and timestamps."""
    gps_points = []
    
    with open(gps_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
                
            # Split timestamp and NMEA message
            parts = line.split(';', 1)
            if len(parts) != 2:
                continue
                
            timestamp_str, nmea_msg = parts
            
            try:
                # Parse timestamp
                timestamp = datetime.fromisoformat(timestamp_str)
                
                # Parse NMEA message
                msg = parse_nmea_gga(nmea_msg)
                
                # Only process GGA messages with valid fix
                if msg is not None and msg['quality'] > 0:
                    gps_points.append(GPSPoint(
                        timestamp=timestamp,
                        lat=float(msg['lat']),
                        lon=float(msg['lon'])
                    ))
            except Exception:
                continue  # Skip invalid lines
                
    return gps_points


def calculate_speeds(gps_points: List[GPSPoint], window_size: int) -> List[float]:
    """Calculate speed using sliding window average over GPS positions."""
    if len(gps_points) < 2:
        return []
        
    speeds = []
    
    for i in range(len(gps_points)):
        # Define window bounds
        start_idx = max(0, i - window_size // 2)
        end_idx = min(len(gps_points), i + window_size // 2 + 1)
        
        if end_idx - start_idx < 2:
            speeds.append(0.0)
            continue
            
        # Calculate distances and time differences within window
        total_distance = 0.0
        total_time = 0.0
        
        for j in range(start_idx, end_idx - 1):
            p1, p2 = gps_points[j], gps_points[j + 1]
            distance = geodesic((p1.lat, p1.lon), (p2.lat, p2.lon)).meters
            time_diff = (p2.timestamp - p1.timestamp).total_seconds()
            
            if time_diff > 0:
                total_distance += distance
                total_time += time_diff
        
        # Calculate average speed in km/h
        if total_time > 0:
            speed_ms = total_distance / total_time
            speed_kmh = speed_ms * 3.6
            speeds.append(speed_kmh)
        else:
            speeds.append(0.0)
            
    return speeds


def is_in_facility_area(point: GPSPoint, facility_area: Optional[FacilityArea]) -> bool:
    """Check if GPS point is within the defined facility area."""
    if facility_area is None:
        return False
        
    lat_min = min(facility_area.top_left[0], facility_area.bottom_right[0])
    lat_max = max(facility_area.top_left[0], facility_area.bottom_right[0])
    lon_min = min(facility_area.top_left[1], facility_area.bottom_right[1])
    lon_max = max(facility_area.top_left[1], facility_area.bottom_right[1])
    
    return (lat_min <= point.lat <= lat_max and 
            lon_min <= point.lon <= lon_max)


def calculate_segment_max_distance(gps_points: List[GPSPoint], start_index: int, end_index: int) -> float:
    """Calculate the maximum distance using min/max extents in both dimensions."""
    if start_index >= end_index or end_index >= len(gps_points):
        return 0.0
        
    segment_points = gps_points[start_index:end_index + 1]
    
    # Find min/max coordinates
    lats = [point.lat for point in segment_points]
    lons = [point.lon for point in segment_points]
    
    lat_min, lat_max = min(lats), max(lats)
    lon_min, lon_max = min(lons), max(lons)
    
    # Calculate distance between opposite corners of bounding box
    return geodesic((lat_min, lon_min), (lat_max, lon_max)).meters


def run_cleaning_classifier(video_file: Path, video_start: datetime, weights: Path, stride_sec: int, debounce_interval: int, threshold: float) -> List[CleaningStatus]:
    """Run the cleaning classifier on the video file and return debounced cleaning status."""
    visual_cleaning_status: List[CleaningStatus] = []

    # Determine framerate of video
    try:
        cap = cv2.VideoCapture()
        cap.open(str(video_file))
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = 1 / video_fps
    except Exception as e:
        raise IOError('Could not open video', e)
    finally:
        cap.release()

    stable_interval_len = 0

    # Initialize YOLO model
    model = YOLO(model=weights)

    cleaning_class_id = {v: k for k, v in model.names.items()}.get('cleaning')
    if cleaning_class_id is None:
        raise ValueError('Model does not have class named `cleaning`')
    
    prev_cleaning = False

    print(f"Video stride: {int(stride_sec *video_fps)}")

    # Iterate over video frames and run cleaning classification model on it
    for idx, result in enumerate(model.predict(source=str(video_file), vid_stride=int(stride_sec * video_fps), stream=True, verbose=False)):
        p = result.probs
        frame_timestamp = video_start + timedelta(seconds=idx * stride_sec)

        # Assume (and add) non cleaning or cleanung status based on conditions
        print(f"{idx:05d} {p.top1} {float(p.top1conf):.2f}")
        cleaning_detected = p.top1 == cleaning_class_id and p.top1conf >= threshold
        if cleaning_detected:
            if prev_cleaning == False:
                stable_interval_len = 0
            elif prev_cleaning == True and stable_interval_len >= debounce_interval:
                print(f"{timedelta(seconds=idx * stride_sec)}: CLEANING")
                visual_cleaning_status.append(CleaningStatus(
                    is_cleaning=True,
                    timestamp=frame_timestamp,
                ))
            prev_cleaning = True
            stable_interval_len += 1
        else:
            if prev_cleaning == True:
                stable_interval_len = 0
            elif prev_cleaning == False and stable_interval_len >= debounce_interval:
                print(f"{timedelta(seconds=idx * stride_sec)}: NOT CLEANING")
                visual_cleaning_status.append(CleaningStatus(
                    is_cleaning=False,
                    timestamp=frame_timestamp,
                ))
            prev_cleaning = False
            stable_interval_len += 1

    return visual_cleaning_status


def parse_video_timestamp(filename: str) -> Optional[datetime]:
    """Parse timestamp from video filename pattern YYYY-MM-DD_HH-mm-ss into datetime."""
    video_pattern = re.compile(r'(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})_video\.(mkv|mp4|avi)$')
    match = video_pattern.match(filename)
    
    if not match:
        return None
    
    timestamp_str = match.group(1)
    try:        
        return datetime.strptime(timestamp_str, '%Y-%m-%d_%H-%M-%S').replace(tzinfo=None)
    except ValueError:
        return None


def find_cleaning_segments_gps(gps_points: List[GPSPoint], speeds: List[float], 
                          max_speed: float, min_duration: int, min_distance: float,
                          facility_area: Optional[FacilityArea]) -> List[CleaningSegment]:
    """Identify cleaning run segments based on speed, duration and distance criteria."""
    if len(gps_points) != len(speeds):
        return []
        
    segments = []
    current_start = None
    
    for i, (point, speed) in enumerate(zip(gps_points, speeds)):
        # Skip points in facility area
        if is_in_facility_area(point, facility_area):
            if current_start is not None:
                # End current segment if it meets all criteria
                duration = (gps_points[i-1].timestamp - gps_points[current_start].timestamp).total_seconds()
                if duration >= min_duration:
                    max_distance = calculate_segment_max_distance(gps_points, current_start, i-1)
                    if max_distance >= min_distance:
                        segments.append(CleaningSegment(
                            start_time=gps_points[current_start].timestamp,
                            end_time=gps_points[i-1].timestamp,
                            start_index=current_start,
                            end_index=i-1
                        ))
                current_start = None
            continue
            
        # Check if speed indicates cleaning activity
        if speed <= max_speed:
            if current_start is None:
                current_start = i
        else:
            if current_start is not None:
                # Check if segment meets all criteria
                duration = (gps_points[i-1].timestamp - gps_points[current_start].timestamp).total_seconds()
                if duration >= min_duration:
                    max_distance = calculate_segment_max_distance(gps_points, current_start, i-1)
                    if max_distance >= min_distance:
                        segments.append(CleaningSegment(
                            start_time=gps_points[current_start].timestamp,
                            end_time=gps_points[i-1].timestamp,
                            start_index=current_start,
                            end_index=i-1
                        ))
                current_start = None
    
    # Handle case where segment extends to end of data
    if current_start is not None:
        duration = (gps_points[-1].timestamp - gps_points[current_start].timestamp).total_seconds()
        if duration >= min_duration:
            max_distance = calculate_segment_max_distance(gps_points, current_start, len(gps_points)-1)
            if max_distance >= min_distance:
                segments.append(CleaningSegment(
                    start_time=gps_points[current_start].timestamp,
                    end_time=gps_points[-1].timestamp,
                    start_index=current_start,
                    end_index=len(gps_points)-1
                ))
    
    return segments


def extract_video_segment(video_file: Path, output_file: Path, 
                         start_time: datetime, end_time: datetime,
                         video_start_time: datetime) -> bool:
    """Extract video segment using ffmpeg."""
    # Calculate relative timestamps
    start_offset = (start_time - video_start_time).total_seconds()
    duration = (end_time - start_time).total_seconds()
    
    if start_offset < 0 or duration <= 0:
        return False
        
    cmd = [
        'ffmpeg', '-i', str(video_file),
        '-ss', str(start_offset),
        '-t', str(duration),
        '-c', 'copy',
        '-avoid_negative_ts', 'make_zero',
        str(output_file)
    ]
    
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError:
        return False


def extract_gps_segment(gps_file: Path, output_file: Path, 
                       start_index: int, end_index: int) -> bool:
    """Extract GPS segment and save to new file."""
    try:
        with open(gps_file, 'r') as infile:
            lines = infile.readlines()
        
        # Extract only the lines corresponding to the segment indices
        extracted_lines = []
        line_count = 0
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Check if this line contains a valid GPS point
            parts = line.split(';', 1)
            if len(parts) == 2:
                try:
                    timestamp_str, nmea_msg = parts
                    timestamp = datetime.fromisoformat(timestamp_str)
                    msg = parse_nmea_gga(nmea_msg)
                    
                    if msg is not None and msg['quality'] > 0:
                        if start_index <= line_count <= end_index:
                            extracted_lines.append(line + '\n')
                        line_count += 1
                except Exception:
                    continue
        
        # Write extracted lines to output file
        with open(output_file, 'w') as outfile:
            outfile.writelines(extracted_lines)
            
        return len(extracted_lines) > 0
        
    except Exception:
        return False


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


def find_cleaning_segments_classifier(cleaning_status: List[CleaningStatus]) -> List[CleaningSegment]:
    pass


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
        speeds = calculate_speeds(gps_points, args.gps_speed_window_size)

        video_start = parse_video_timestamp(video_file.name)
        if video_start is None:
            print(f"  Could not parse timestamp from filename")
            continue

        if args.classifier_weights:
            print(f"  Running cleaning classifier")
            visual_cleaning_status = run_cleaning_classifier(video_file, video_start, args.classifier_weights, args.classifier_stride_sec, args.classifier_debounce_interval, args.classifier_threshold)

        # Find cleaning segments
        segments = find_cleaning_segments_gps(gps_points, speeds, args.gps_max_speed, 
                                        args.min_duration, args.gps_min_distance, facility_area)
        
        print(f"  Found {len(segments)} cleaning segments")
        
        # Extract video and GPS segments
        video_start_time = gps_points[0].timestamp
        for i, segment in enumerate(segments):
            # Video segment
            video_output_name = f"{video_file.stem}_segment_{i:03d}{video_file.suffix}"
            video_output_file = args.output_dir / video_output_name
            
            # GPS segment
            gps_output_name = f"{gps_file.stem}_segment_{i:03d}.log"
            gps_output_file = args.output_dir / gps_output_name
            
            video_success = extract_video_segment(video_file, video_output_file, 
                                                 segment.start_time, segment.end_time,
                                                 video_start_time)
            
            gps_success = extract_gps_segment(gps_file, gps_output_file,
                                            segment.start_index, segment.end_index)
            
            if video_success and gps_success:
                duration = (segment.end_time - segment.start_time).total_seconds()
                print(f"    Extracted segment {i+1}: {duration:.1f}s -> {video_output_name}, {gps_output_name}")
                total_segments += 1
            else:
                print(f"    Failed to extract segment {i+1} (video: {video_success}, gps: {gps_success})")
    
    print(f"\nProcessing complete. Extracted {total_segments} cleaning segments.")
    return 0


if __name__ == '__main__':
    exit(main())
