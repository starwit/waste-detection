from datetime import datetime, timedelta
from typing import NamedTuple, Tuple


class GPSPoint(NamedTuple):
    timestamp: datetime
    offset: timedelta
    speed_kmh: float
    lat: float
    lon: float


class FacilityArea(NamedTuple):
    top_left: Tuple[float, float]  # (lat, lon)
    bottom_right: Tuple[float, float]  # (lat, lon)


class ClassifierStatus(NamedTuple):
    offset: timedelta
    is_cleaning: bool


class CleaningSegment(NamedTuple):
    start_offset: timedelta
    end_offset: timedelta