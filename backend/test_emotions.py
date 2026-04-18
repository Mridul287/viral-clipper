"""
test_emotions.py
----------------
Pytest suite for backend/emotions.py

Covers all 10 test cases specified in the task:

  TC-01  Valid MP4 with face → result dict has all 4 required keys
  TC-02  Every frame_emotions item has timestamp, emotion, confidence, intensity
  TC-03  confidence values are always in [0.0, 1.0]
  TC-04  intensity follows the formula: confidence × emotion_weight
  TC-05  peak_windows are sorted by start time with no overlaps
  TC-06  Video with no faces → empty frame_emotions & peak_windows, no crash
  TC-07  Merge: gap < 2 s → single merged window
  TC-08  No-merge: gap > 2 s → two separate windows remain
  TC-09  emotions_{job_id}.json saved in /temp with valid JSON structure
  TC-10  sample_frames at interval=1 on a 10-second video → ≈10 frames

DeepFace.analyze is mocked via unittest.mock so no GPU is required.
"""

import json
import pathlib
import shutil
import tempfile
import pytest
import numpy as np
import cv2
from unittest.mock import patch, MagicMock

# Module under test
import emotions as em


# ---------------------------------------------------------------------------
# Helper: synthetic video generator
# ---------------------------------------------------------------------------

def create_test_video(
    duration_sec: int,
    fps: int,
    output_path: str,
    blank: bool = False,
) -> str:
    """
    Generate a synthetic MP4 file using cv2.VideoWriter.

    Parameters
    ----------
    duration_sec : int
        Length of the video in seconds.
    fps : int
        Frames per second.
    output_path : str
        Where to write the file (must end in .mp4 or .avi).
    blank : bool
        If True, frames are pure black (simulates no-face scenario).
        If False, frames contain a coloured gradient pattern.

    Returns
    -------
    str
        Absolute path to the written video file.
    """
    width, height = 320, 240
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    total_frames = duration_sec * fps
    for i in range(total_frames):
        if blank:
            frame = np.zeros((height, width, 3), dtype=np.uint8)
        else:
            # Simple colour-cycle pattern – definitely not blank
            colour = int((i / total_frames) * 255)
            frame = np.full((height, width, 3), colour, dtype=np.uint8)
        writer.write(frame)

    writer.release()
    return output_path


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def tmp_dir():
    """Temporary directory that lives for the entire test module."""
    d = tempfile.mkdtemp(prefix="test_emotions_")
    yield pathlib.Path(d)
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture(scope="module")
def face_video(tmp_dir):
    """10-second synthetic video with coloured frames (non-blank)."""
    path = str(tmp_dir / "face_video.mp4")
    return create_test_video(duration_sec=10, fps=30, output_path=path, blank=False)


@pytest.fixture(scope="module")
def blank_video(tmp_dir):
    """10-second synthetic video with all-black frames (no face)."""
    path = str(tmp_dir / "blank_video.mp4")
    return create_test_video(duration_sec=10, fps=30, output_path=path, blank=True)


# ---------------------------------------------------------------------------
# Shared mock factory
# ---------------------------------------------------------------------------

def _make_deepface_result(emotion: str = "happy", score: float = 90.0) -> list[dict]:
    """
    Build a minimal DeepFace.analyze return value.

    score : float
        Percentage value (0–100) for the dominant emotion.
    """
    emotion_scores = {e: 0.0 for e in em.EMOTION_WEIGHTS}
    emotion_scores[emotion] = score
    return [
        {
            "dominant_emotion": emotion,
            "emotion": emotion_scores,
        }
    ]


# ---------------------------------------------------------------------------
# TC-01  Valid MP4 with face → result dict has all 4 required keys
# ---------------------------------------------------------------------------

class TestTC01_ValidVideoKeys:
    def test_result_has_four_required_keys(self, face_video, tmp_dir):
        """detect_emotions must return a dict with job_id, frame_emotions,
        peak_windows, and emotion_summary."""
        mock_result = _make_deepface_result("happy", 85.0)

        # Redirect TEMP_DIR so we don't pollute the real backend/temp folder
        with patch.object(em, "TEMP_DIR", tmp_dir), \
             patch("emotions.DeepFace.analyze", return_value=mock_result):
            result = em.detect_emotions(face_video, job_id="tc01")

        assert isinstance(result, dict), "Result must be a dict"
        for key in ("job_id", "frame_emotions", "peak_windows", "emotion_summary"):
            assert key in result, f"Missing key: {key}"


# ---------------------------------------------------------------------------
# TC-02  Every frame_emotions item has the 4 required keys
# ---------------------------------------------------------------------------

class TestTC02_FrameEmotionKeys:
    def test_each_frame_has_required_keys(self, face_video, tmp_dir):
        """Every entry in frame_emotions must have timestamp, emotion,
        confidence, and intensity."""
        mock_result = _make_deepface_result("surprise", 75.0)

        with patch.object(em, "TEMP_DIR", tmp_dir), \
             patch("emotions.DeepFace.analyze", return_value=mock_result):
            result = em.detect_emotions(face_video, job_id="tc02")

        assert result["frame_emotions"], "frame_emotions must not be empty for a face video"
        for item in result["frame_emotions"]:
            for key in ("timestamp", "emotion", "confidence", "intensity"):
                assert key in item, f"frame_emotions item missing key: {key}"


# ---------------------------------------------------------------------------
# TC-03  confidence values are always in [0.0, 1.0]
# ---------------------------------------------------------------------------

class TestTC03_ConfidenceRange:
    def test_confidence_in_unit_interval(self, face_video, tmp_dir):
        """confidence must be in [0.0, 1.0] for every detected frame."""
        mock_result = _make_deepface_result("angry", 120.0)  # intentionally > 100

        with patch.object(em, "TEMP_DIR", tmp_dir), \
             patch("emotions.DeepFace.analyze", return_value=mock_result):
            result = em.detect_emotions(face_video, job_id="tc03")

        for item in result["frame_emotions"]:
            assert 0.0 <= item["confidence"] <= 1.0, (
                f"confidence out of range: {item['confidence']}"
            )


# ---------------------------------------------------------------------------
# TC-04  intensity follows the formula
# ---------------------------------------------------------------------------

class TestTC04_IntensityFormula:
    def test_intensity_formula_unit(self):
        """calculate_intensity must return confidence × weight, clamped to [0,1]."""
        # happy weight = 0.8
        emotion, confidence = "happy", 0.75
        expected = round(0.75 * 0.8, 6)
        result = em.calculate_intensity(emotion, confidence)
        assert result == pytest.approx(expected, abs=1e-6)

    def test_intensity_formula_via_pipeline(self, face_video, tmp_dir):
        """Verify the formula end-to-end for a known DeepFace output."""
        # DeepFace returns 80% → confidence = 0.80; emotion = "surprise" (weight 1.0)
        mock_result = _make_deepface_result("surprise", 80.0)
        expected_intensity = round(0.80 * 1.0, 6)

        with patch.object(em, "TEMP_DIR", tmp_dir), \
             patch("emotions.DeepFace.analyze", return_value=mock_result):
            result = em.detect_emotions(face_video, job_id="tc04")

        for item in result["frame_emotions"]:
            assert item["intensity"] == pytest.approx(expected_intensity, abs=1e-5), (
                f"intensity mismatch: got {item['intensity']}, expected {expected_intensity}"
            )


# ---------------------------------------------------------------------------
# TC-05  peak_windows are sorted by start time, no overlaps
# ---------------------------------------------------------------------------

class TestTC05_PeakWindowOrder:
    def test_windows_sorted_no_overlap(self, face_video, tmp_dir):
        """peak_windows must be sorted by start time and must not overlap."""
        # surprise at 90% → intensity = 0.9 × 1.0 = 0.9 > threshold 0.6
        mock_result = _make_deepface_result("surprise", 90.0)

        with patch.object(em, "TEMP_DIR", tmp_dir), \
             patch("emotions.DeepFace.analyze", return_value=mock_result):
            result = em.detect_emotions(face_video, job_id="tc05")

        windows = result["peak_windows"]
        for i in range(1, len(windows)):
            assert windows[i]["start"] >= windows[i - 1]["end"], (
                f"Overlapping or unsorted windows: {windows[i-1]} and {windows[i]}"
            )


# ---------------------------------------------------------------------------
# TC-06  Blank / no-face video → empty lists, no crash
# ---------------------------------------------------------------------------

class TestTC06_NoFaceVideo:
    def test_blank_video_returns_empty_and_no_crash(self, blank_video, tmp_dir):
        """A video with no detectable faces must return empty lists gracefully."""
        with patch.object(em, "TEMP_DIR", tmp_dir), \
             patch("emotions.DeepFace.analyze", side_effect=ValueError("No face detected")):
            result = em.detect_emotions(blank_video, job_id="tc06")

        assert result["frame_emotions"] == [], "frame_emotions must be empty"
        assert result["peak_windows"] == [], "peak_windows must be empty"
        assert result["job_id"] == "tc06"


# ---------------------------------------------------------------------------
# TC-07  Merge logic: gap < 2 s → single merged window
# ---------------------------------------------------------------------------

class TestTC07_MergeClose:
    def test_windows_merged_when_gap_less_than_2s(self):
        """Windows at t=5-6 and t=7.5-8.5 (gap 1.5 s) must merge into one."""
        frame_emotions = [
            # Window A
            {"timestamp": 5.0, "emotion": "angry", "confidence": 0.9, "intensity": 0.81},
            {"timestamp": 6.0, "emotion": "angry", "confidence": 0.9, "intensity": 0.81},
            # Gap of 1.5 s (< 2 s)
            {"timestamp": 7.5, "emotion": "angry", "confidence": 0.9, "intensity": 0.81},
            {"timestamp": 8.5, "emotion": "angry", "confidence": 0.9, "intensity": 0.81},
        ]
        windows = em.find_peak_windows(frame_emotions, threshold=0.6, merge_gap_sec=2.0)

        assert len(windows) == 1, (
            f"Expected 1 merged window, got {len(windows)}: {windows}"
        )
        assert windows[0]["start"] == pytest.approx(5.0)
        assert windows[0]["end"] == pytest.approx(8.5)


# ---------------------------------------------------------------------------
# TC-08  No-merge: gap > 2 s → two separate windows
# ---------------------------------------------------------------------------

class TestTC08_NoMergeFar:
    def test_windows_stay_separate_when_gap_exceeds_2s(self):
        """Windows at t=5-6 and t=9-10 (gap 3 s) must remain two windows."""
        frame_emotions = [
            {"timestamp": 5.0, "emotion": "fear", "confidence": 0.9, "intensity": 0.72},
            {"timestamp": 6.0, "emotion": "fear", "confidence": 0.9, "intensity": 0.72},
            # Gap of 3 s (> 2 s)
            {"timestamp": 9.0, "emotion": "fear", "confidence": 0.9, "intensity": 0.72},
            {"timestamp": 10.0, "emotion": "fear", "confidence": 0.9, "intensity": 0.72},
        ]
        windows = em.find_peak_windows(frame_emotions, threshold=0.6, merge_gap_sec=2.0)

        assert len(windows) == 2, (
            f"Expected 2 separate windows, got {len(windows)}: {windows}"
        )
        assert windows[0]["start"] == pytest.approx(5.0)
        assert windows[0]["end"] == pytest.approx(6.0)
        assert windows[1]["start"] == pytest.approx(9.0)
        assert windows[1]["end"] == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# TC-09  JSON file saved in /temp with valid structure
# ---------------------------------------------------------------------------

class TestTC09_JsonPersisted:
    def test_json_file_saved_with_correct_structure(self, face_video, tmp_dir):
        """emotions_{job_id}.json must exist in TEMP_DIR with all 4 keys."""
        mock_result = _make_deepface_result("happy", 70.0)

        with patch.object(em, "TEMP_DIR", tmp_dir), \
             patch("emotions.DeepFace.analyze", return_value=mock_result):
            em.detect_emotions(face_video, job_id="tc09")

        json_path = tmp_dir / "emotions_tc09.json"
        assert json_path.exists(), f"JSON file not found at {json_path}"

        with open(json_path, encoding="utf-8") as fh:
            data = json.load(fh)

        for key in ("job_id", "frame_emotions", "peak_windows", "emotion_summary"):
            assert key in data, f"JSON missing key: {key}"

        assert data["job_id"] == "tc09"
        assert isinstance(data["frame_emotions"], list)
        assert isinstance(data["peak_windows"], list)
        assert isinstance(data["emotion_summary"], dict)


# ---------------------------------------------------------------------------
# TC-10  sample_frames at interval=1 on a 10-second video → ≈10 frames
# ---------------------------------------------------------------------------

class TestTC10_SampleFrameCount:
    def test_sample_count_matches_duration(self, face_video):
        """A 10-second video at interval=1 should yield roughly 10 frames (±1)."""
        frames = em.sample_frames(face_video, interval_sec=1)

        assert len(frames) >= 9, f"Expected ≥9 frames, got {len(frames)}"
        assert len(frames) <= 11, f"Expected ≤11 frames, got {len(frames)}"

    def test_sample_returns_correct_types(self, face_video):
        """Each element must be a (float, ndarray) tuple."""
        frames = em.sample_frames(face_video, interval_sec=1)
        for ts, arr in frames:
            assert isinstance(ts, float), f"timestamp must be float, got {type(ts)}"
            assert isinstance(arr, np.ndarray), "frame must be np.ndarray"

    def test_sample_frames_nonexistent_video(self):
        """sample_frames should raise FileNotFoundError for a missing file."""
        with pytest.raises(FileNotFoundError):
            em.sample_frames("/nonexistent/path/video.mp4")


# ---------------------------------------------------------------------------
# Additional edge-case unit tests for calculate_intensity
# ---------------------------------------------------------------------------

class TestCalculateIntensity:
    @pytest.mark.parametrize("emotion,confidence,expected", [
        ("surprise", 1.0,  1.0),   # max possible
        ("neutral",  1.0,  0.1),   # low weight
        ("disgust",  0.5,  0.25),  # 0.5 × 0.5
        ("unknown",  0.9,  0.0),   # unknown emotion → weight 0
        ("happy",    0.0,  0.0),   # zero confidence
    ])
    def test_intensity_parametrised(self, emotion, confidence, expected):
        result = em.calculate_intensity(emotion, confidence)
        assert result == pytest.approx(expected, abs=1e-6)


# ---------------------------------------------------------------------------
# Additional edge-case unit tests for find_peak_windows
# ---------------------------------------------------------------------------

class TestFindPeakWindowsEdgeCases:
    def test_empty_input_returns_empty(self):
        assert em.find_peak_windows([]) == []

    def test_no_frames_above_threshold(self):
        frames = [
            {"timestamp": 1.0, "emotion": "neutral", "confidence": 0.5, "intensity": 0.05},
            {"timestamp": 2.0, "emotion": "neutral", "confidence": 0.5, "intensity": 0.05},
        ]
        assert em.find_peak_windows(frames, threshold=0.6) == []

    def test_single_frame_above_threshold(self):
        frames = [
            {"timestamp": 3.0, "emotion": "happy", "confidence": 0.9, "intensity": 0.72},
        ]
        windows = em.find_peak_windows(frames, threshold=0.6)
        assert len(windows) == 1
        assert windows[0]["start"] == pytest.approx(3.0)
        assert windows[0]["end"] == pytest.approx(3.0)

    def test_exact_gap_at_boundary_not_merged(self):
        """A gap of exactly 2.0 s is NOT strictly less than 2.0, so no merge."""
        frames = [
            {"timestamp": 1.0, "emotion": "angry", "confidence": 0.9, "intensity": 0.81},
            # gap = 3.0 - 1.0 = 2.0 s  (not < 2.0)
            {"timestamp": 3.0, "emotion": "angry", "confidence": 0.9, "intensity": 0.81},
        ]
        windows = em.find_peak_windows(frames, threshold=0.6, merge_gap_sec=2.0)
        assert len(windows) == 2
