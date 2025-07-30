#!/usr/bin/env python3

import argparse
import os
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Tuple, Optional, NamedTuple
import subprocess

import pynmeagps
from geopy.distance import geodesic


class GPSPoint(NamedTuple):
    timestamp: datetime
    lat: float
    lon: float


class CleaningSegment(NamedTuple):
    start_time: datetime
    end_time: datetime
    start_index: int
    end_index: int


class FacilityArea(NamedTuple):
    top_left: Tuple[float, float]  # (lat, lon)
    bottom_right: Tuple[float, float]  # (lat, lon)


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
                msg = pynmeagps.NMEAReader.parse(nmea_msg)
                
                # Only process GGA messages with valid fix
                if msg.msgID == 'GGA' and msg.quality > 0:
                    gps_points.append(GPSPoint(
                        timestamp=timestamp,
                        lat=float(msg.lat),
                        lon=float(msg.lon)
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


def find_cleaning_segments(gps_points: List[GPSPoint], speeds: List[float], 
                          max_speed: float, min_duration: int,
                          facility_area: Optional[FacilityArea]) -> List[CleaningSegment]:
    """Identify cleaning run segments based on speed and duration criteria."""
    if len(gps_points) != len(speeds):
        return []
        
    segments = []
    current_start = None
    
    for i, (point, speed) in enumerate(zip(gps_points, speeds)):
        # Skip points in facility area
        if is_in_facility_area(point, facility_area):
            if current_start is not None:
                # End current segment if it was long enough
                duration = (gps_points[i-1].timestamp - gps_points[current_start].timestamp).total_seconds()
                if duration >= min_duration:
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
                # Check if segment is long enough
                duration = (gps_points[i-1].timestamp - gps_points[current_start].timestamp).total_seconds()
                if duration >= min_duration:
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


def main():
    parser = argparse.ArgumentParser(description='Filter cleaning runs from street cleaner videos')
    parser.add_argument('input_dir', type=Path, help='Directory containing video and GPS files')
    parser.add_argument('output_dir', type=Path, help='Output directory for filtered video segments')
    parser.add_argument('--max-speed', type=float, default=15.0, 
                       help='Maximum speed (km/h) for cleaning activity (default: 15.0)')
    parser.add_argument('--min-duration', type=int, default=120,
                       help='Minimum duration (seconds) for cleaning segments (default: 120)')
    parser.add_argument('--window-size', type=int, default=5,
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
        gps_points = parse_gps_log(gps_file)
        if len(gps_points) < 2:
            print(f"  Warning: Insufficient GPS data in {gps_file.name}")
            continue
            
        # Calculate speeds
        speeds = calculate_speeds(gps_points, args.window_size)
        
        # Find cleaning segments
        segments = find_cleaning_segments(gps_points, speeds, args.max_speed, 
                                        args.min_duration, facility_area)
        
        print(f"  Found {len(segments)} cleaning segments")
        
        # Extract video segments
        video_start_time = gps_points[0].timestamp
        for i, segment in enumerate(segments):
            output_name = f"{video_file.stem}_segment_{i:03d}{video_file.suffix}"
            output_file = args.output_dir / output_name
            
            success = extract_video_segment(video_file, output_file, 
                                          segment.start_time, segment.end_time,
                                          video_start_time)
            
            if success:
                duration = (segment.end_time - segment.start_time).total_seconds()
                print(f"    Extracted segment {i+1}: {duration:.1f}s -> {output_name}")
                total_segments += 1
            else:
                print(f"    Failed to extract segment {i+1}")
    
    print(f"\nProcessing complete. Extracted {total_segments} cleaning segments.")
    return 0


if __name__ == '__main__':
    exit(main())
