from datetime import datetime
from typing import NamedTuple, Tuple


class GPSPoint(NamedTuple):
    timestamp: datetime
    speed_kmh: float
    lat: float
    lon: float


class FacilityArea(NamedTuple):
    top_left: Tuple[float, float]  # (lat, lon)
    bottom_right: Tuple[float, float]  # (lat, lon)


class ClassifierStatus(NamedTuple):
    timestamp: datetime
    is_cleaning: bool


class CleaningSegment(NamedTuple):
    start_time: datetime
    end_time: datetime