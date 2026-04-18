"""
scenes.py
---------
Scene detection module for the Viral Clipper pipeline.

Uses PySceneDetect to identify natural cut boundaries in videos,
extract representative thumbnails, and align emotion peaks to clean
scene boundaries.
"""

import os
import json
import pathlib
import logging
from typing import Optional

import cv2
import numpy as np
from scenedetect import detect, ContentDetector, FrameTimecode

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MIN_SCENE_DURATION = 3.0  # seconds
MAX_SCENE_DURATION = 120.0  # seconds
DEFAULT_THRESHOLD = 27.0
SNAP_TOLERANCE_SEC = 3.0


# ---------------------------------------------------------------------------
# 1. detect_scenes
# ---------------------------------------------------------------------------

def detect_scenes(video_path: str, threshold: float = 27.0) -> list[dict]:
    """
    Detect scene boundaries using PySceneDetect's ContentDetector.

    Parameters
    ----------
    video_path : str
        Path to the video file.
    threshold : float
        Content detection sensitivity threshold (default 27.0).
        Lower values = more sensitive, more scenes detected.

    Returns
    -------
    list[dict]
        List of detected scenes, each with:
        - start_time (float): scene start in seconds
        - end_time (float): scene end in seconds
        - duration (float): end_time - start_time
        - start_frame (int): frame number at start
        - end_frame (int): frame number at end
    """
    try:
        # Detect scenes using ContentDetector with specified threshold
        scenes = detect(video_path, ContentDetector(threshold=threshold))

        if not scenes:
            logger.info(f"No scenes detected in {video_path}")
            return []

        # Get video properties for frame calculations
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            logger.warning(f"Invalid FPS: {fps}, defaulting to 30")
            fps = 30.0
        cap.release()

        # Convert scenedetect TimeCodes to readable format
        result = []
        for i, (start_tc, end_tc) in enumerate(scenes):
            # Convert FrameTimecode to seconds
            start_time = start_tc.get_seconds()
            end_time = end_tc.get_seconds()
            duration = end_time - start_time

            result.append({
                "scene_number": i + 1,
                "start_time": round(start_time, 3),
                "end_time": round(end_time, 3),
                "duration": round(duration, 3),
                "start_frame": int(start_tc.get_frames()),
                "end_frame": int(end_tc.get_frames()),
            })

        logger.info(f"Detected {len(result)} scenes in {video_path}")
        return result

    except Exception as e:
        logger.error(f"Error detecting scenes: {e}")
        return []


# ---------------------------------------------------------------------------
# 2. filter_scenes
# ---------------------------------------------------------------------------

def filter_scenes(
    scenes: list[dict],
    min_dur: float = MIN_SCENE_DURATION,
    max_dur: float = MAX_SCENE_DURATION,
) -> list[dict]:
    """
    Filter scenes by duration to remove too-short or too-long clips.

    Parameters
    ----------
    scenes : list[dict]
        Raw scene list from detect_scenes().
    min_dur : float
        Minimum scene duration in seconds (default 3.0).
    max_dur : float
        Maximum scene duration in seconds (default 120.0).

    Returns
    -------
    list[dict]
        Filtered scene list, maintaining original scene_number.
    """
    filtered = []
    for scene in scenes:
        duration = scene["duration"]
        if min_dur <= duration <= max_dur:
            filtered.append(scene)
        else:
            reason = "too short" if duration < min_dur else "too long"
            logger.debug(
                f"Filtering scene {scene['scene_number']}: "
                f"duration {duration:.2f}s ({reason})"
            )

    logger.info(
        f"Filtered scenes: {len(scenes)} total → {len(filtered)} usable"
    )
    return filtered


# ---------------------------------------------------------------------------
# 3. extract_thumbnail
# ---------------------------------------------------------------------------

def extract_thumbnail(
    video_path: str,
    timestamp: float,
    save_path: str,
) -> str:
    """
    Extract a single frame at the specified timestamp and save as JPEG.

    Parameters
    ----------
    video_path : str
        Path to the video file.
    timestamp : float
        Timestamp in seconds to extract frame from.
    save_path : str
        Full path (including filename) to save the thumbnail as JPEG.

    Returns
    -------
    str
        The save_path if successful, empty string on error.
    """
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Failed to open video: {video_path}")
            return ""

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30.0

        # Seek to timestamp
        frame_number = int(timestamp * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

        ret, frame = cap.read()
        cap.release()

        if not ret or frame is None:
            logger.error(f"Failed to read frame at {timestamp}s")
            return ""

        # Ensure save directory exists
        save_dir = os.path.dirname(save_path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)

        # Save thumbnail
        success = cv2.imwrite(save_path, frame)
        if success:
            logger.debug(f"Extracted thumbnail: {save_path}")
            return save_path
        else:
            logger.error(f"Failed to save thumbnail: {save_path}")
            return ""

    except Exception as e:
        logger.error(f"Error extracting thumbnail: {e}")
        return ""


# ---------------------------------------------------------------------------
# 4. snap_to_scene_boundary
# ---------------------------------------------------------------------------

def snap_to_scene_boundary(
    peak_windows: list[dict],
    scenes: list[dict],
    tolerance_sec: float = SNAP_TOLERANCE_SEC,
) -> list[dict]:
    """
    Align peak emotion windows to nearest scene boundaries.

    For each peak window (start, end), find the nearest scene boundary
    within tolerance_sec. If found, snap to that boundary; otherwise,
    keep the original timestamp.

    Parameters
    ----------
    peak_windows : list[dict]
        Peak emotion windows with 'start' and 'end' keys (in seconds).
    scenes : list[dict]
        Filtered scene list from filter_scenes().
    tolerance_sec : float
        Maximum distance to snap to a scene boundary (default 3.0).

    Returns
    -------
    list[dict]
        List of aligned clips, each with:
        - original_start (float)
        - original_end (float)
        - aligned_start (float)
        - aligned_end (float)
        - snapped (bool): True if either start or end was snapped
    """
    if not peak_windows or not scenes:
        return []

    # Extract all scene boundaries (starts and ends)
    boundaries = []
    for scene in scenes:
        boundaries.append(scene["start_time"])
        boundaries.append(scene["end_time"])
    boundaries = sorted(set(boundaries))

    aligned_clips = []
    for window in peak_windows:
        orig_start = window["start"]
        orig_end = window["end"]

        # Find nearest boundary for start
        aligned_start = orig_start
        start_snapped = False
        for boundary in boundaries:
            if abs(boundary - orig_start) <= tolerance_sec:
                aligned_start = boundary
                start_snapped = True
                break

        # Find nearest boundary for end
        aligned_end = orig_end
        end_snapped = False
        for boundary in boundaries:
            if abs(boundary - orig_end) <= tolerance_sec:
                aligned_end = boundary
                end_snapped = True
                break

        snapped = start_snapped or end_snapped

        aligned_clips.append({
            "original_start": round(orig_start, 3),
            "original_end": round(orig_end, 3),
            "aligned_start": round(aligned_start, 3),
            "aligned_end": round(aligned_end, 3),
            "snapped": snapped,
        })

    logger.info(f"Snapped {sum(c['snapped'] for c in aligned_clips)}/{len(aligned_clips)} clips")
    return aligned_clips


# ---------------------------------------------------------------------------
# 5. run_scene_detection (main orchestrator)
# ---------------------------------------------------------------------------

def run_scene_detection(
    video_path: str,
    job_id: str,
    peak_windows: list[dict] = None,
    threshold: float = DEFAULT_THRESHOLD,
) -> dict:
    """
    Full scene detection pipeline: detect → filter → extract thumbnails →
    align to peaks → persist → return results.

    Parameters
    ----------
    video_path : str
        Path to the video file.
    job_id : str
        Job identifier for file naming.
    peak_windows : list[dict], optional
        Peak emotion windows to align (from emotions.py).
        If None, aligned_clips will be empty.
    threshold : float
        ContentDetector threshold (default 27.0).

    Returns
    -------
    dict
        Result dictionary with keys:
        - job_id (str)
        - scenes (list[dict]): filtered scenes with thumbnails
        - aligned_clips (list[dict]): emotion windows snapped to boundaries
        - total_scenes (int): count before filtering
        - usable_scenes (int): count after filtering
    """
    if peak_windows is None:
        peak_windows = []

    # Step 1: Detect raw scenes
    raw_scenes = detect_scenes(video_path, threshold=threshold)
    total_scenes = len(raw_scenes)

    # Step 2: Filter by duration
    filtered_scenes = filter_scenes(raw_scenes)
    usable_scenes = len(filtered_scenes)

    # Step 3: Extract thumbnails for each filtered scene
    temp_dir = "temp/thumbnails"
    os.makedirs(temp_dir, exist_ok=True)

    for scene in filtered_scenes:
        # Midpoint of scene for representative frame
        midpoint = (scene["start_time"] + scene["end_time"]) / 2.0
        scene_num = scene["scene_number"]
        thumbnail_filename = f"scene_{job_id}_{scene_num}.jpg"
        thumbnail_path = os.path.join(temp_dir, thumbnail_filename)

        # Extract and save thumbnail
        saved_path = extract_thumbnail(video_path, midpoint, thumbnail_path)
        scene["thumbnail_path"] = saved_path if saved_path else ""

    # Step 4: Snap peak windows to scene boundaries
    aligned_clips = snap_to_scene_boundary(peak_windows, filtered_scenes)

    # Step 5: Prepare result dictionary
    result = {
        "job_id": job_id,
        "scenes": filtered_scenes,
        "aligned_clips": aligned_clips,
        "total_scenes": total_scenes,
        "usable_scenes": usable_scenes,
    }

    # Step 6: Save to JSON
    output_path = os.path.join("temp", f"scenes_{job_id}.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    try:
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)
        logger.info(f"Saved scene results to {output_path}")
    except Exception as e:
        logger.error(f"Failed to save results to {output_path}: {e}")

    return result
