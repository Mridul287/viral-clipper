"""
emotions.py
-----------
Emotion detection module for the Viral Clipper pipeline.

Uses DeepFace to analyse sampled video frames, compute per-frame emotion
intensity scores, locate peak emotional windows, and persist results to disk.
"""

import os
import json
import pathlib
import logging
from typing import Optional

import cv2
import numpy as np
from deepface import DeepFace

# ---------------------------------------------------------------------------
# JSON Encoder for numpy types
# ---------------------------------------------------------------------------

class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""
    def default(self, obj):  # pylint: disable=method-hidden
        if isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        return super().default(obj)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EMOTION_WEIGHTS: dict[str, float] = {
    "surprise": 1.0,
    "angry":    0.9,
    "happy":    0.8,
    "fear":     0.8,
    "sad":      0.6,
    "disgust":  0.5,
    "neutral":  0.1,
}

# Output directory – mirrors the existing project convention
TEMP_DIR = pathlib.Path(os.environ.get("CLIPPER_TEMP_DIR", "temp"))


# ---------------------------------------------------------------------------
# 1. sample_frames
# ---------------------------------------------------------------------------

def sample_frames(
    video_path: str,
    interval_sec: int = 1,
) -> list[tuple[float, np.ndarray]]:
    """
    Sample frames from a video file at a fixed time interval.

    Parameters
    ----------
    video_path : str
        Absolute or relative path to the video file.
    interval_sec : int
        Interval between sampled frames in seconds (default 1).

    Returns
    -------
    list[tuple[float, np.ndarray]]
        Each element is ``(timestamp_seconds, bgr_frame_array)``.
        The list is ordered chronologically.

    Raises
    ------
    FileNotFoundError
        If the video file cannot be opened by OpenCV.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video file: {video_path}")

    fps: float = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        # Fallback – assume 30 fps so we don't divide by zero
        fps = 30.0

    frame_interval: int = max(1, int(round(fps * interval_sec)))
    frames: list[tuple[float, np.ndarray]] = []
    frame_index: int = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Only keep frames that fall on the sampling grid
        if frame_index % frame_interval == 0:
            timestamp: float = frame_index / fps
            frames.append((timestamp, frame))

        frame_index += 1

    cap.release()
    logger.info("Sampled %d frames from '%s'", len(frames), video_path)
    return frames


# ---------------------------------------------------------------------------
# 2. analyze_frame
# ---------------------------------------------------------------------------

def analyze_frame(frame: np.ndarray) -> Optional[dict]:
    """
    Run DeepFace emotion analysis on a single BGR frame.

    Parameters
    ----------
    frame : np.ndarray
        A BGR image array as returned by ``cv2.VideoCapture.read()``.

    Returns
    -------
    dict or None
        On success: ``{"emotion": str, "confidence": float}`` where
        *confidence* is always in [0.0, 1.0].
        Returns ``None`` when no face is detected (handled silently).
    """
    try:
        # DeepFace expects BGR or RGB; it handles both.
        results = DeepFace.analyze(
            img_path=frame,
            actions=["emotion"],
            enforce_detection=True,   # raises ValueError when no face found
            silent=True,
        )

        # DeepFace may return a list (multiple faces) or a single dict.
        # We take the first detected face only.
        result = results[0] if isinstance(results, list) else results

        # emotion_confidence is a dict {emotion_label: percentage (0-100)}
        # Convert numpy float32 to Python float for JSON serialization
        emotion_dict = result.get("emotion", {})
        emotion_scores: dict[str, float] = {k: float(v) for k, v in emotion_dict.items()}
        dominant_emotion: str = result.get("dominant_emotion", "neutral")

        # Convert from percentage to fraction and clamp to [0, 1]
        raw_confidence: float = emotion_scores.get(dominant_emotion, 0.0)
        confidence: float = float(np.clip(raw_confidence / 100.0, 0.0, 1.0))

        # Normalise emotion label to lowercase so it matches EMOTION_WEIGHTS
        dominant_emotion = dominant_emotion.lower().strip()

        logger.debug("Face detected: emotion=%s, confidence=%.3f", dominant_emotion, confidence)
        return {"emotion": dominant_emotion, "confidence": confidence, "scores": emotion_scores}

    except (ValueError, Exception) as exc:  # noqa: BLE001
        # ValueError is raised by DeepFace when no face is detected.
        # We log at DEBUG to avoid polluting logs for black/landscape frames.
        logger.debug("No face detected in frame: %s", exc)
        return None


# ---------------------------------------------------------------------------
# 3. calculate_intensity
# ---------------------------------------------------------------------------

def calculate_intensity(emotion: str, confidence: float) -> float:
    """
    Compute the emotion intensity score for a single frame.

    Formula::

        intensity = confidence × emotion_weight

    Parameters
    ----------
    emotion : str
        Dominant emotion label (e.g. ``"happy"``).
    confidence : float
        Confidence value in [0.0, 1.0].

    Returns
    -------
    float
        Intensity score in [0.0, 1.0].
    """
    weight: float = EMOTION_WEIGHTS.get(emotion.lower(), 0.0)
    intensity: float = float(np.clip(confidence * weight, 0.0, 1.0))
    return round(intensity, 6)


# ---------------------------------------------------------------------------
# 4. find_peak_windows
# ---------------------------------------------------------------------------

def find_peak_windows(
    frame_emotions: list[dict],
    threshold: float = 0.6,
    merge_gap_sec: float = 2.0,
) -> list[dict]:
    """
    Identify and merge consecutive high-intensity emotion windows.

    A *peak window* is a contiguous run of frames whose intensity exceeds
    *threshold*.  Adjacent windows separated by less than *merge_gap_sec*
    are merged into a single window.

    Parameters
    ----------
    frame_emotions : list[dict]
        Each item must have keys: ``timestamp``, ``emotion``,
        ``confidence``, ``intensity``.
    threshold : float
        Minimum intensity to include a frame in a peak window (default 0.6).
    merge_gap_sec : float
        Maximum gap in seconds between two windows before they are merged
        (default 2.0).  A gap *strictly less than* this value triggers a
        merge.

    Returns
    -------
    list[dict]
        Sorted list of windows, each with keys:
        ``start``, ``end``, ``dominant_emotion``, ``avg_intensity``.
    """
    # ------------------------------------------------------------------
    # Step 1: collect raw (contiguous) windows from high-intensity frames
    # ------------------------------------------------------------------
    raw_windows: list[dict] = []
    current_window: Optional[dict] = None

    for entry in frame_emotions:
        ts: float = entry["timestamp"]
        intensity: float = entry["intensity"]
        emotion: str = entry["emotion"]

        if intensity >= threshold:
            if current_window is None or (ts - current_window["end"]) > 1.1:
                # Start a new window
                if current_window is not None:
                    raw_windows.append(current_window)
                current_window = {
                    "start": ts,
                    "end": ts,
                    "_emotions": [emotion],
                    "_intensities": [intensity],
                }
            else:
                # Extend the current window
                current_window["end"] = ts
                current_window["_emotions"].append(emotion)
                current_window["_intensities"].append(intensity)
        else:
            if current_window is not None:
                raw_windows.append(current_window)
                current_window = None

    # Don't forget the last open window
    if current_window is not None:
        raw_windows.append(current_window)

    if not raw_windows:
        return []

    # ------------------------------------------------------------------
    # Step 2: merge windows whose gap is strictly less than merge_gap_sec
    # ------------------------------------------------------------------
    merged: list[dict] = [raw_windows[0]]

    for window in raw_windows[1:]:
        prev = merged[-1]
        gap: float = window["start"] - prev["end"]

        if gap < merge_gap_sec:
            # Merge: extend the previous window
            prev["end"] = window["end"]
            prev["_emotions"].extend(window["_emotions"])
            prev["_intensities"].extend(window["_intensities"])
        else:
            merged.append(window)

    # ------------------------------------------------------------------
    # Step 3: compute summary fields for each final window
    # ------------------------------------------------------------------
    result: list[dict] = []
    for w in merged:
        dominant_emotion = _most_frequent(w["_emotions"])
        avg_intensity = float(np.mean(w["_intensities"]))
        result.append(
            {
                "start": round(w["start"], 3),
                "end": round(w["end"], 3),
                "dominant_emotion": dominant_emotion,
                "avg_intensity": round(avg_intensity, 6),
            }
        )

    # Guarantee sorted order (input frames should already be sorted, but
    # defensive sorting is cheap)
    result.sort(key=lambda x: x["start"])
    return result


# ---------------------------------------------------------------------------
# 5. detect_emotions  (main orchestrator)
# ---------------------------------------------------------------------------

def detect_emotions(video_path: str, job_id: str) -> dict:
    """
    Full pipeline: sample → analyse → score → find peaks → persist → return.

    Parameters
    ----------
    video_path : str
        Path to the input video file.
    job_id : str
        Unique identifier for this processing job.

    Returns
    -------
    dict
        A results dictionary with keys:
        ``job_id``, ``frame_emotions``, ``peak_windows``, ``emotion_summary``.
        Also persisted to ``<TEMP_DIR>/emotions_{job_id}.json``.
    """
    logger.info("[%s] Starting emotion detection on '%s'", job_id, video_path)
    logger.info("[%s] TEMP_DIR = '%s'", job_id, TEMP_DIR)

    try:
        # ── 1. Sample frames ────────────────────────────────────────────────────
        sampled: list[tuple[float, np.ndarray]] = sample_frames(video_path, interval_sec=1)
        logger.info("[%s] Sampled %d frames from video", job_id, len(sampled))

        # ── 2. Analyse each frame ───────────────────────────────────────────────
        frame_emotions: list[dict] = []

        for timestamp, frame in sampled:
            analysis = analyze_frame(frame)
            if analysis is None:
                # No face detected – skip silently per spec
                continue

            emotion: str = analysis["emotion"]
            confidence: float = analysis["confidence"]
            intensity: float = calculate_intensity(emotion, confidence)

            frame_emotions.append(
                {
                    "timestamp": round(timestamp, 3),
                    "emotion": emotion,
                    "confidence": round(confidence, 6),
                    "intensity": intensity,
                    "all_scores": analysis["scores"]
                }
            )

        logger.info("[%s] Detected faces in %d frames", job_id, len(frame_emotions))

        # ── 3. Find peak windows ────────────────────────────────────────────────
        peak_windows: list[dict] = find_peak_windows(frame_emotions)
        logger.info("[%s] Found %d peak windows", job_id, len(peak_windows))

        # ── 4. Compute summary ──────────────────────────────────────────────────
        emotion_summary: dict = _build_summary(frame_emotions, peak_windows)

        # ── 5. Assemble result ──────────────────────────────────────────────────
        result: dict = {
            "job_id": job_id,
            "frame_emotions": frame_emotions,
            "peak_windows": peak_windows,
            "emotion_summary": emotion_summary,
        }

        # ── 6. Persist to disk ──────────────────────────────────────────────────
        _save_result(result, job_id)

        logger.info("[%s] Emotion detection complete.", job_id)
        return result

    except Exception as e:
        logger.error("[%s] Emotion detection error: %s", job_id, str(e), exc_info=True)
        raise


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _most_frequent(items: list[str]) -> str:
    """Return the most frequently occurring string in *items*."""
    if not items:
        return "neutral"
    return max(set(items), key=items.count)


def _build_summary(
    frame_emotions: list[dict],
    peak_windows: list[dict],
) -> dict:
    """Compute the ``emotion_summary`` block."""
    if not frame_emotions:
        return {
            "most_frequent": "none",
            "highest_intensity_moment": 0.0,
            "peak_count": 0,
        }

    all_emotions = [fe["emotion"] for fe in frame_emotions]
    most_frequent: str = _most_frequent(all_emotions)

    highest_intensity_moment: float = max(
        fe["intensity"] for fe in frame_emotions
    )

    return {
        "most_frequent": most_frequent,
        "highest_intensity_moment": round(highest_intensity_moment, 6),
        "peak_count": len(peak_windows),
    }


def _save_result(result: dict, job_id: str) -> None:
    """Persist *result* as JSON to the temp directory."""
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    output_path = TEMP_DIR / f"emotions_{job_id}.json"
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, cls=NumpyEncoder)
    logger.info("Results saved to '%s'", output_path)
