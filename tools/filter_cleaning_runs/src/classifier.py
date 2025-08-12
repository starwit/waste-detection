from datetime import datetime, timedelta
from pathlib import Path
from typing import List
from tqdm import tqdm

import cv2
from src.types import ClassifierStatus, CleaningSegment
from ultralytics import YOLO


def run_cleaning_classifier(video_file: Path, video_start: datetime, weights: Path, stride_sec: int, debounce_interval: int, threshold: float) -> List[ClassifierStatus]:
    """Run the cleaning classifier on the video file and return debounced cleaning status."""
    visual_cleaning_status: List[ClassifierStatus] = []

    # Determine framerate of video
    try:
        cap = cv2.VideoCapture()
        cap.open(str(video_file))
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        video_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    except Exception as e:
        raise IOError('Could not open video', e)
    finally:
        cap.release()

    total_frames_to_process = (video_frame_count / video_fps) / stride_sec

    # Initialize YOLO model
    model = YOLO(model=weights)

    cleaning_class_id = {v: k for k, v in model.names.items()}.get('cleaning')
    if cleaning_class_id is None:
        raise ValueError('Model does not have class named `cleaning`')
    
    stable_interval_len = 0
    prev_cleaning = False

    # Iterate over video frames and run cleaning classification model on it
    for idx, result in enumerate(tqdm(model.predict(source=str(video_file), vid_stride=int(stride_sec * video_fps), stream=True, verbose=False), total=total_frames_to_process)):
        p = result.probs
        frame_timestamp = video_start + timedelta(seconds=idx * stride_sec)

        # Assume (and add) non cleaning or cleanung status based on conditions
        cleaning_detected = p.top1 == cleaning_class_id and p.top1conf >= threshold
        if cleaning_detected:
            if prev_cleaning == False:
                stable_interval_len = 0
            elif prev_cleaning == True and stable_interval_len >= debounce_interval:
                visual_cleaning_status.append(ClassifierStatus(
                    is_cleaning=True,
                    timestamp=frame_timestamp,
                ))
            prev_cleaning = True
            stable_interval_len += 1
        else:
            if prev_cleaning == True:
                stable_interval_len = 0
            elif prev_cleaning == False and stable_interval_len >= debounce_interval:
                visual_cleaning_status.append(ClassifierStatus(
                    is_cleaning=False,
                    timestamp=frame_timestamp,
                ))
            prev_cleaning = False
            stable_interval_len += 1
    
    return visual_cleaning_status


def find_cleaning_segments_classifier(cleaning_status: List[ClassifierStatus], min_length_s: int) -> List[CleaningSegment]:
    segments: List[CleaningSegment] = []

    segment_start = cleaning_status[0] if cleaning_status[0].is_cleaning else None
    
    # segment_start != None means we're in a segment
    for status in cleaning_status[1:]:
        if segment_start is not None and not status.is_cleaning:
            # we're at the end of a segment -> save new segment if it satisfies the length constraint
            if (status.timestamp - segment_start.timestamp).total_seconds() >= min_length_s:
                segments.append(CleaningSegment(
                    start_time=segment_start.timestamp,
                    end_time=status.timestamp
                ))
            segment_start = None
        elif segment_start is None and status.is_cleaning:
            # we're at the start of a new segment -> remember start point
            segment_start = status
        else:
            # in all other cases skip, because we're not at a transition point
            continue

    # Case when last segment extends to end of classifier output
    if segment_start is not None and (status.timestamp - segment_start.timestamp).total_seconds() >= min_length_s:
        segments.append(CleaningSegment(
            start_time=segment_start.timestamp,
            end_time=cleaning_status[-1].timestamp
        ))
    
    return segments