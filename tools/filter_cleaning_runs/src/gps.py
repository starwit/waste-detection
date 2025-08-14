from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

from geopy.distance import geodesic
from src.types import CleaningSegment, FacilityArea, GPSPoint


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
    start_timestamp = None
    
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

                if start_timestamp is None:
                    start_timestamp = timestamp
                
                # Parse NMEA message
                msg = parse_nmea_gga(nmea_msg)
                
                # Only process GGA messages with valid fix
                if msg is not None and msg['quality'] > 0:
                    gps_points.append(GPSPoint(
                        offset=timestamp - start_timestamp,
                        timestamp=timestamp,
                        lat=float(msg['lat']),
                        lon=float(msg['lon']),
                        speed_kmh=None
                    ))
            except Exception as e:
                continue  # Skip invalid lines
                
    return gps_points


def calculate_speeds(gps_points: List[GPSPoint], window_size: int) -> List[GPSPoint]:
    """Calculate speed using sliding window average over GPS positions."""
    if len(gps_points) < 2:
        return []
        
    speeds = []
    
    for i, point in enumerate(gps_points):
        # Define window bounds
        start_idx = max(0, i - window_size // 2)
        end_idx = min(len(gps_points), i + window_size // 2 + 1)
        
        if end_idx - start_idx < 2:
            speeds.append(point._replace(speed_kmh=0.0))
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
            speeds.append(point._replace(speed_kmh=speed_kmh))
        else:
            speeds.append(point._replace(speed_kmh=0.0))
            
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


def find_cleaning_segments_gps(gps_points: List[GPSPoint], 
                          max_speed: float, min_duration: int, min_distance: float,
                          facility_area: Optional[FacilityArea]) -> List[CleaningSegment]:
    """Identify cleaning run segments based on speed, duration and distance criteria."""
    segments = []
    current_start = None
    
    for i, point in enumerate(gps_points):
        # Skip points in facility area
        if is_in_facility_area(point, facility_area):
            if current_start is not None:
                # End current segment if it meets all criteria
                duration = (gps_points[i-1].timestamp - gps_points[current_start].timestamp).total_seconds()
                if duration >= min_duration:
                    max_distance = calculate_segment_max_distance(gps_points, current_start, i-1)
                    if max_distance >= min_distance:
                        segments.append(CleaningSegment(
                            start_offset=gps_points[current_start].offset,
                            end_offset=gps_points[i-1].offset,
                        ))
                current_start = None
            continue
            
        # Check if speed indicates cleaning activity
        if point.speed_kmh <= max_speed:
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
                            start_offset=gps_points[current_start].offset,
                            end_offset=gps_points[i-1].offset,
                        ))
                current_start = None
    
    # Handle case where segment extends to end of data
    if current_start is not None:
        duration = (gps_points[-1].timestamp - gps_points[current_start].timestamp).total_seconds()
        if duration >= min_duration:
            max_distance = calculate_segment_max_distance(gps_points, current_start, len(gps_points)-1)
            if max_distance >= min_distance:
                segments.append(CleaningSegment(
                    start_offset=gps_points[current_start].offset,
                    end_offset=gps_points[-1].offset,
                ))
    
    return segments


def extract_gps_segment(gps_file: Path, output_file: Path, segment: CleaningSegment) -> bool:
    """Extract GPS segment and save to new file."""
    try:
        with open(gps_file, 'r') as infile:
            lines = infile.readlines()
        
        # Extract only the lines corresponding to the segment indices
        extracted_lines = []

        start_timestamp = None
        
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

                    if start_timestamp is None:
                        start_timestamp = timestamp

                    current_offset = timestamp - start_timestamp
                    
                    if msg is not None and msg['quality'] > 0:
                        if segment.start_offset <= current_offset <= segment.end_offset:
                            extracted_lines.append(line + '\n')
                except Exception:
                    continue
        
        # Write extracted lines to output file
        with open(output_file, 'w') as outfile:
            outfile.writelines(extracted_lines)
            
        return len(extracted_lines) > 0
        
    except Exception:
        return False