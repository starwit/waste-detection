import os
# Suppress output of ffmpeg decoding errors
os.environ["OPENCV_FFMPEG_LOGLEVEL"] = "0"

from datetime import datetime, timedelta
from pathlib import Path
from typing import List

import cv2
from src.types import ClassifierStatus, CleaningSegment
from tqdm import tqdm
from ultralytics import YOLO


def run_cleaning_classifier(video_file: Path, weights: Path, stride_sec: int, debounce_interval: int, threshold: float) -> List[ClassifierStatus]:
    """Run the cleaning classifier on the video file and return debounced cleaning status."""
    visual_cleaning_status: List[ClassifierStatus] = []

    # Determine framerate of video
    try:
        cap = cv2.VideoCapture()
        cap.open(str(video_file))
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        video_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        total_frames_to_process = int((video_frame_count / video_fps) / stride_sec)
        stride = stride_sec * video_fps

        if not cap.isOpened():
            raise IOError('Could not open video')

        # Initialize YOLO model
        model = YOLO(model=weights)

        cleaning_class_id = {v: k for k, v in model.names.items()}.get('cleaning')
        if cleaning_class_id is None:
            raise ValueError('Model does not have class named `cleaning`')
        
        stable_interval_len = 0
        prev_cleaning = False

        pbar = tqdm(total=total_frames_to_process)
        frame_idx = 0
        frames_processed_count = 0
        while True:
            ret, frame = cap.read()

            if not ret:
                break

            frame_idx += 1

            # check if this frame should be processed
            if frame_idx % stride != 0:
                continue

            frames_processed_count += 1
            pbar.update(1)

            results = model.predict(frame, verbose=False)
            p = results[0].probs

            frame_offset = timedelta(seconds=cap.get(cv2.CAP_PROP_POS_MSEC) / 1000)

            # Assume (and add) non cleaning or cleanung status based on conditions
            cleaning_detected = p.top1 == cleaning_class_id and p.top1conf >= threshold
            if cleaning_detected:
                if prev_cleaning == False:
                    stable_interval_len = 0
                elif prev_cleaning == True and stable_interval_len >= debounce_interval:
                    visual_cleaning_status.append(ClassifierStatus(
                        is_cleaning=True,
                        offset=frame_offset,
                    ))
                prev_cleaning = True
                stable_interval_len += 1
            else:
                if prev_cleaning == True:
                    stable_interval_len = 0
                elif prev_cleaning == False and stable_interval_len >= debounce_interval:
                    visual_cleaning_status.append(ClassifierStatus(
                        is_cleaning=False,
                        offset=frame_offset,
                    ))
                prev_cleaning = False
                stable_interval_len += 1
            
    except Exception as e:
        raise Exception('Error running cleaning classifier', e)
    finally:
        cap.release()        
    
    return visual_cleaning_status


def find_cleaning_segments_classifier(cleaning_status: List[ClassifierStatus], min_length_s: int) -> List[CleaningSegment]:
    segments: List[CleaningSegment] = []

    segment_start = cleaning_status[0] if cleaning_status[0].is_cleaning else None
    
    # segment_start != None means we're in a segment
    for status in cleaning_status[1:]:
        if segment_start is not None and not status.is_cleaning:
            # we're at the end of a segment -> save new segment if it satisfies the length constraint
            if (status.offset - segment_start.offset).total_seconds() >= min_length_s:
                segments.append(CleaningSegment(
                    start_offset=segment_start.offset,
                    end_offset=status.offset,
                ))
            segment_start = None
        elif segment_start is None and status.is_cleaning:
            # we're at the start of a new segment -> remember start point
            segment_start = status
        else:
            # in all other cases skip, because we're not at a transition point
            continue

    # Case when last segment extends to end of classifier output
    if segment_start is not None and (status.offset - segment_start.offset).total_seconds() >= min_length_s:
        segments.append(CleaningSegment(
            start_offset=segment_start.offset,
            end_offset=cleaning_status[-1].offset
        ))
    
    return segments