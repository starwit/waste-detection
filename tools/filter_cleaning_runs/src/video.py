import re
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

from .types import CleaningSegment


def extract_video_segment(video_file: Path, output_file: Path, segment: CleaningSegment) -> bool:
    """Extract video segment using ffmpeg."""
    # Calculate relative timestamps
    start_offset = segment.start_offset.total_seconds()
    duration = (segment.end_offset - segment.start_offset).total_seconds()
    
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


def parse_video_timestamp(filename: str) -> Optional[datetime]:
    """Parse timestamp from video filename pattern YYYY-MM-DD_HH-mm-ss into datetime."""
    video_pattern = re.compile(r'(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})_video\.(mkv|mp4|avi)$')
    match = video_pattern.match(filename)
    
    if not match:
        return None
    
    timestamp_str = match.group(1)
    try:        
        return datetime.strptime(timestamp_str, '%Y-%m-%d_%H-%M-%S').replace(tzinfo=ZoneInfo("Europe/Berlin"))
    except ValueError:
        return None