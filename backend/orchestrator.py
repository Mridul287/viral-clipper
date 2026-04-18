"""
orchestrator.py
---------------
Main orchestrator that combines emotions, scenes, and export modules
into a single end-to-end video processing pipeline.

Coordinates the three processing steps, handles errors gracefully,
and provides unified progress reporting.
"""

import json
import logging
import time
import os
from typing import Callable, Optional

# Import sub-modules (these will be mocked in tests)
import emotions as em
import scenes as sc
import export as exp

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_OPTIONS = {
    "emotion_interval": 1,
    "scene_threshold": 27.0,
    "use_scene_snap": True,
    "caption_style": "srt",
    "top_n_clips": 5,
}


# ---------------------------------------------------------------------------
# 1. match_emotion_to_clip
# ---------------------------------------------------------------------------

def match_emotion_to_clip(
    clip_start: float,
    clip_end: float,
    frame_emotions: list[dict],
) -> str:
    """
    Find the dominant emotion within a given time range.

    Parameters
    ----------
    clip_start : float
        Clip start time in seconds.
    clip_end : float
        Clip end time in seconds.
    frame_emotions : list[dict]
        List of frame emotions from emotions module with keys:
        timestamp, emotion, confidence, intensity.

    Returns
    -------
    str
        Dominant emotion name, or empty string if no data.
    """
    # Filter emotions within clip range
    emotions_in_range = [
        fe for fe in frame_emotions
        if clip_start <= fe.get("timestamp", 0) <= clip_end
    ]

    if not emotions_in_range:
        return ""

    # Count occurrences of each emotion (weighted by confidence or intensity)
    emotion_scores = {}
    for fe in emotions_in_range:
        emotion = fe.get("emotion", "")
        intensity = fe.get("intensity", 0.0)

        if emotion:
            emotion_scores[emotion] = emotion_scores.get(emotion, 0.0) + intensity

    if not emotion_scores:
        return ""

    # Return emotion with highest total score
    return max(emotion_scores, key=emotion_scores.get)


# ---------------------------------------------------------------------------
# 2. apply_scene_snapping
# ---------------------------------------------------------------------------

def apply_scene_snapping(
    top_clips: list[dict],
    aligned_clips: list[dict],
) -> list[dict]:
    """
    Update clip boundaries to align with scene cuts.

    For each top_clip, find the nearest aligned_clip (scene boundary) and
    update start/end if snapped. Track which clips were snapped.

    Parameters
    ----------
    top_clips : list[dict]
        Original clips with start, end keys.
    aligned_clips : list[dict]
        Scene-aligned clips from scene detection with aligned_start, aligned_end.

    Returns
    -------
    list[dict]
        Updated clips with start, end, scene_snapped keys.
    """
    if not aligned_clips:
        # No scene data, return clips unchanged
        snapped_clips = []
        for clip in top_clips:
            clip_copy = dict(clip)
            clip_copy["scene_snapped"] = False
            snapped_clips.append(clip_copy)
        return snapped_clips

    snapped_clips = []
    for clip in top_clips:
        clip_copy = dict(clip)
        orig_start = clip.get("start", 0.0)
        orig_end = clip.get("end", 0.0)

        # Find best matching aligned_clip (one with most overlap)
        best_aligned = None
        best_overlap = 0

        for aligned in aligned_clips:
            aligned_start = aligned.get("aligned_start", 0.0)
            aligned_end = aligned.get("aligned_end", 0.0)

            # Calculate overlap
            overlap_start = max(orig_start, aligned_start)
            overlap_end = min(orig_end, aligned_end)
            overlap = max(0, overlap_end - overlap_start)

            if overlap > best_overlap:
                best_overlap = overlap
                best_aligned = aligned

        # Apply snapping if we found a match
        if best_aligned and best_aligned.get("snapped", False):
            clip_copy["start"] = best_aligned["aligned_start"]
            clip_copy["end"] = best_aligned["aligned_end"]
            clip_copy["scene_snapped"] = True
            logger.debug(f"Snapped clip {best_aligned}")
        else:
            clip_copy["scene_snapped"] = False

        snapped_clips.append(clip_copy)

    return snapped_clips


# ---------------------------------------------------------------------------
# 3. run_video_pipeline (main orchestrator)
# ---------------------------------------------------------------------------

def run_video_pipeline(
    job_id: str,
    source_video_path: str,
    top_clips: list[dict],
    words: list[dict] = None,
    options: dict = None,
    on_progress: Optional[Callable] = None,
) -> dict:
    """
    Execute the full video processing pipeline in order.

    Coordinates emotion detection, scene detection, optional scene snapping,
    and clip export. Handles errors gracefully and reports progress.

    Parameters
    ----------
    job_id : str
        Job identifier.
    source_video_path : str
        Path to source video.
    top_clips : list[dict]
        List of clips with start, end, suggested_title, clip_type, final_score.
    words : list[dict], optional
        Transcribed words with word, start, end.
    options : dict, optional
        Configuration overrides (see DEFAULT_OPTIONS).
    on_progress : Callable, optional
        Callback function(step: int, total: int, message: str) for progress updates.

    Returns
    -------
    dict
        Combined result with status, steps_completed, and enriched clips.
    """
    start_time = time.time()

    # Merge options with defaults
    if options is None:
        options = {}
    config = {**DEFAULT_OPTIONS, **options}

    if words is None:
        words = []

    # Default progress callback
    if on_progress is None:
        def on_progress(step: int, total: int, message: str):
            logger.info(f"[{step}/{total}] {message}")

    # Initialize result structure
    result = {
        "job_id": job_id,
        "status": "success",
        "steps_completed": [],
        "emotion_summary": {},
        "scene_summary": {},
        "clips": [],
        "total_clips": 0,
        "processing_time_sec": 0.0,
    }

    total_steps = 4
    current_step = 1

    try:
        # ===== STEP 1: Emotion Detection =====
        on_progress(current_step, total_steps, "Detecting emotions...")
        try:
            emotion_result = em.detect_emotions(
                source_video_path,
                job_id,
            )
            frame_emotions = emotion_result.get("frame_emotions", [])
            peak_windows = emotion_result.get("peak_windows", [])

            result["emotion_summary"] = {
                "total_frames": len(frame_emotions),
                "peak_windows": len(peak_windows),
            }
            result["steps_completed"].append("emotion")
            logger.info(f"Emotion detection: {len(frame_emotions)} frames, "
                       f"{len(peak_windows)} peaks")

        except Exception as e:
            logger.warning(f"Emotion detection failed: {e}, continuing...")
            frame_emotions = []
            peak_windows = []

        current_step += 1

        # ===== STEP 2: Scene Detection =====
        on_progress(current_step, total_steps, "Detecting scenes...")
        try:
            scene_result = sc.run_scene_detection(
                source_video_path,
                job_id,
                peak_windows=peak_windows,
                threshold=config["scene_threshold"],
            )
            aligned_clips = scene_result.get("aligned_clips", [])
            total_scenes = scene_result.get("total_scenes", 0)
            usable_scenes = scene_result.get("usable_scenes", 0)

            result["scene_summary"] = {
                "total_scenes": total_scenes,
                "usable_scenes": usable_scenes,
            }
            result["steps_completed"].append("scene")
            logger.info(f"Scene detection: {total_scenes} total, "
                       f"{usable_scenes} usable")

        except Exception as e:
            logger.warning(f"Scene detection failed: {e}, continuing...")
            aligned_clips = []
            result["scene_summary"] = {"total_scenes": 0, "usable_scenes": 0}

        current_step += 1

        # ===== STEP 3: Scene Snapping (optional) =====
        on_progress(current_step, total_steps, "Snapping to scene boundaries...")
        clips_to_export = list(top_clips)  # Make a copy

        if config["use_scene_snap"] and aligned_clips:
            try:
                clips_to_export = apply_scene_snapping(top_clips, aligned_clips)
                result["steps_completed"].append("scene_snap")
                logger.info(f"Applied scene snapping to {len(clips_to_export)} clips")
            except Exception as e:
                logger.warning(f"Scene snapping failed: {e}, using original boundaries")
                for clip in clips_to_export:
                    clip["scene_snapped"] = False
        else:
            for clip in clips_to_export:
                clip["scene_snapped"] = False
            if config["use_scene_snap"]:
                result["steps_completed"].append("scene_snap")

        current_step += 1

        # ===== STEP 4: Clip Export =====
        on_progress(current_step, total_steps, "Exporting clips...")

        # Apply top_n_clips limit
        clips_limited = clips_to_export[:config["top_n_clips"]]

        try:
            export_result = exp.export_all_clips(
                source_video_path,
                clips_limited,
                words,
                job_id,
            )
            exported_clips = export_result.get("clips", [])

            # Enrich exported clips with emotion and scene data
            enriched_clips = []
            for rank, clip in enumerate(exported_clips, start=1):
                clip_start = clip.get("start", 0.0)
                clip_end = clip.get("end", 0.0)

                # Find dominant emotion
                dominant_emotion = match_emotion_to_clip(
                    clip_start,
                    clip_end,
                    frame_emotions,
                )

                # Find matching original clip (by rank)
                orig_clip = None
                if rank - 1 < len(clips_limited):
                    orig_clip = clips_limited[rank - 1]
                    scene_snapped = orig_clip.get("scene_snapped", False)
                else:
                    scene_snapped = False

                # Get metadata from original clip
                if orig_clip:
                    clip.update({
                        "clip_type": orig_clip.get("clip_type", ""),
                        "suggested_title": orig_clip.get("suggested_title", ""),
                        "final_score": orig_clip.get("final_score", 0.0),
                    })

                clip.update({
                    "dominant_emotion": dominant_emotion,
                    "scene_snapped": scene_snapped,
                })

                enriched_clips.append(clip)

            result["clips"] = enriched_clips
            result["total_clips"] = len(enriched_clips)
            result["steps_completed"].append("export")

            if len(enriched_clips) == 0 and len(clips_limited) > 0:
                result["status"] = "failed"
            elif len(enriched_clips) < len(clips_limited):
                result["status"] = "partial"

            logger.info(f"Exported {len(enriched_clips)} clips")

        except Exception as e:
            logger.error(f"Clip export failed: {e}")
            result["status"] = "failed"

    except Exception as e:
        logger.error(f"Pipeline error: {e}")
        result["status"] = "failed"

    # Calculate total time
    result["processing_time_sec"] = round(time.time() - start_time, 2)

    # Save result JSON
    json_path = os.path.join("temp", f"final_{job_id}.json")
    os.makedirs(os.path.dirname(json_path), exist_ok=True)

    try:
        with open(json_path, "w") as f:
            json.dump(result, f, indent=2)
        logger.info(f"Pipeline result saved: {json_path}")
    except Exception as e:
        logger.error(f"Failed to save result JSON: {e}")

    return result
