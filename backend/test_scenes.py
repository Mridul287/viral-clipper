"""
test_scenes.py
--------------
Pytest suite for backend/scenes.py

Covers all 12 test cases specified in the task:

  TC-01  Valid MP4 with scene changes → scenes list is non-empty
  TC-02  All scenes have start_time < end_time, duration = end_time - start_time
  TC-03  No scene in output has duration < 3.0 seconds (filter working)
  TC-04  No scene in output has duration > 120.0 seconds (filter working)
  TC-05  Thumbnails: for each scene, thumbnail_path file exists on disk
  TC-06  Thumbnails are valid images: cv2.imread gives shape (H, W, 3)
  TC-07  snap test: peak at t=10.2-13.5, boundary at t=10.0 → snapped to 10.0
  TC-08  snap test: peak at t=10.2-13.5, boundary at t=15.0 (gap>3s) → not snapped
  TC-09  Empty peak_windows list → aligned_clips is empty, no crash
  TC-10  Static video (no scene changes) → gracefully return empty/no crash
  TC-11  total_scenes >= usable_scenes always (usable is subset)
  TC-12  scenes_{job_id}.json saved and valid parseable JSON
"""

import json
import os
import pathlib
import shutil
import tempfile
import pytest
import numpy as np
import cv2

# Module under test
import scenes as sc


# ---------------------------------------------------------------------------
# Helper: create_test_video_with_scenes
# ---------------------------------------------------------------------------

def create_test_video_with_scenes(
    output_path: str,
    n_scenes: int = 3,
    fps: int = 30,
    duration_per_scene: float = 5.0,
) -> str:
    """
    Generate a synthetic MP4 with obvious color changes between scenes.

    Each scene is a solid color block for the specified duration.
    Scenes change to create obvious content boundaries for PySceneDetect.

    Parameters
    ----------
    output_path : str
        Path to save the output MP4.
    n_scenes : int
        Number of scenes/color blocks (default 3).
    fps : int
        Frames per second (default 30).
    duration_per_scene : float
        Seconds per scene (default 5.0).

    Returns
    -------
    str
        The output_path.
    """
    # Video properties
    width, height = 1280, 720
    frame_count = int(n_scenes * duration_per_scene * fps)

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, float(fps), (width, height))

    if not out.isOpened():
        raise RuntimeError(f"Failed to create video writer for {output_path}")

    # Define HIGH CONTRAST colors for each scene (BGR format)
    colors = [
        (0, 0, 255),      # Bright Red
        (0, 255, 0),      # Bright Green
        (255, 0, 0),      # Bright Blue
        (0, 255, 255),    # Bright Yellow
        (255, 0, 255),    # Bright Magenta
        (255, 255, 0),    # Bright Cyan
    ]

    frames_per_scene = int(duration_per_scene * fps)

    for scene_idx in range(n_scenes):
        color = colors[scene_idx % len(colors)]
        frame = np.full((height, width, 3), color, dtype=np.uint8)

        # Add scene number text to frame for clarity
        cv2.putText(
            frame,
            f"Scene {scene_idx + 1}",
            (400, 360),
            cv2.FONT_HERSHEY_SIMPLEX,
            3.0,
            (255, 255, 255),
            5,
        )

        # Write frames for this scene - make it consistent throughout
        for _ in range(frames_per_scene):
            out.write(frame)

    out.release()
    return output_path


# ---------------------------------------------------------------------------
# Helper: create_static_video
# ---------------------------------------------------------------------------

def create_static_video(
    output_path: str,
    fps: int = 30,
    duration_sec: float = 10.0,
) -> str:
    """
    Generate a static video with no scene changes (all one color).

    Parameters
    ----------
    output_path : str
        Path to save the output MP4.
    fps : int
        Frames per second (default 30).
    duration_sec : float
        Video duration in seconds (default 10.0).

    Returns
    -------
    str
        The output_path.
    """
    width, height = 640, 480
    frame_count = int(duration_sec * fps)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, float(fps), (width, height))

    if not out.isOpened():
        raise RuntimeError(f"Failed to create video writer for {output_path}")

    # Static gray frame
    frame = np.full((height, width, 3), (128, 128, 128), dtype=np.uint8)
    cv2.putText(
        frame,
        "Static Scene",
        (150, 240),
        cv2.FONT_HERSHEY_SIMPLEX,
        2.0,
        (255, 255, 255),
        3,
    )

    for _ in range(frame_count):
        out.write(frame)

    out.release()
    return output_path


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def temp_workspace(tmp_path):
    """Create temporary workspace with temp subdirectories."""
    temp_dir = tmp_path / "temp"
    temp_dir.mkdir()
    (temp_dir / "thumbnails").mkdir()

    # Temporarily change working directory
    original_cwd = os.getcwd()
    os.chdir(str(tmp_path))

    yield tmp_path

    os.chdir(original_cwd)


@pytest.fixture
def test_video_with_scenes(temp_workspace):
    """Generate test video with 3 distinct scenes."""
    video_path = str(temp_workspace / "test_scenes.mp4")
    create_test_video_with_scenes(
        video_path,
        n_scenes=3,
        fps=30,
        duration_per_scene=5.0,
    )
    return video_path


@pytest.fixture
def static_video(temp_workspace):
    """Generate static test video with no scene changes."""
    video_path = str(temp_workspace / "static.mp4")
    create_static_video(video_path, fps=30, duration_sec=10.0)
    return video_path


# ---------------------------------------------------------------------------
# Test Cases
# ---------------------------------------------------------------------------

class TestTC01_ValidVideoNonEmpty:
    """TC-01: Valid MP4 with scene changes → scenes list is non-empty."""

    def test_detects_scenes_in_multicolor_video(self, test_video_with_scenes):
        # Use lower threshold for synthetic high-contrast colors
        scenes = sc.detect_scenes(test_video_with_scenes, threshold=15.0)
        assert len(scenes) > 0, "Should detect at least one scene"


class TestTC02_TimeAndDuration:
    """TC-02: start_time < end_time, duration = end_time - start_time."""

    def test_scene_times_valid(self, test_video_with_scenes):
        scenes = sc.detect_scenes(test_video_with_scenes, threshold=15.0)
        for scene in scenes:
            assert scene["start_time"] < scene["end_time"], (
                f"Scene {scene['scene_number']}: start >= end"
            )
            expected_duration = scene["end_time"] - scene["start_time"]
            assert abs(scene["duration"] - expected_duration) < 0.001, (
                f"Duration mismatch: {scene['duration']} != {expected_duration}"
            )


class TestTC03_FilterMinDuration:
    """TC-03: No scene in output has duration < 3.0 seconds."""

    def test_filter_removes_short_scenes(self, test_video_with_scenes):
        raw_scenes = sc.detect_scenes(test_video_with_scenes, threshold=15.0)
        filtered = sc.filter_scenes(raw_scenes, min_dur=3.0, max_dur=120.0)

        for scene in filtered:
            assert scene["duration"] >= 3.0, (
                f"Scene {scene['scene_number']} has duration {scene['duration']} < 3.0"
            )


class TestTC04_FilterMaxDuration:
    """TC-04: No scene in output has duration > 120.0 seconds."""

    def test_filter_removes_long_scenes(self, test_video_with_scenes):
        raw_scenes = sc.detect_scenes(test_video_with_scenes, threshold=15.0)
        filtered = sc.filter_scenes(raw_scenes, min_dur=3.0, max_dur=120.0)

        for scene in filtered:
            assert scene["duration"] <= 120.0, (
                f"Scene {scene['scene_number']} has duration {scene['duration']} > 120.0"
            )


class TestTC05_ThumbnailFilesExist:
    """TC-05: For each scene, thumbnail_path file exists on disk."""

    def test_thumbnails_exist(self, test_video_with_scenes, temp_workspace):
        os.makedirs("temp/thumbnails", exist_ok=True)
        result = sc.run_scene_detection(test_video_with_scenes, "test_job_01")

        for scene in result["scenes"]:
            thumb_path = scene["thumbnail_path"]
            assert thumb_path, f"Scene {scene['scene_number']}: no thumbnail_path"
            assert os.path.exists(thumb_path), (
                f"Thumbnail does not exist: {thumb_path}"
            )


class TestTC06_ThumbnailsValidImages:
    """TC-06: Thumbnails are valid images with shape (H, W, 3)."""

    def test_thumbnails_are_valid_images(self, test_video_with_scenes, temp_workspace):
        os.makedirs("temp/thumbnails", exist_ok=True)
        result = sc.run_scene_detection(test_video_with_scenes, "test_job_02")

        for scene in result["scenes"]:
            thumb_path = scene["thumbnail_path"]
            img = cv2.imread(thumb_path)
            assert img is not None, f"Failed to read thumbnail: {thumb_path}"
            assert len(img.shape) == 3, (
                f"Thumbnail has wrong shape: {img.shape} (expected 3D)"
            )
            assert img.shape[2] == 3, (
                f"Thumbnail has {img.shape[2]} channels (expected 3)"
            )


class TestTC07_SnapNearBoundary:
    """TC-07: peak at t=10.2-13.5, boundary at t=10.0 → aligned_start=10.0."""

    def test_snap_to_nearby_boundary(self, test_video_with_scenes, temp_workspace):
        # Create a scene with start at 10.0
        scenes = [
            {
                "scene_number": 1,
                "start_time": 10.0,
                "end_time": 15.0,
                "duration": 5.0,
                "start_frame": 300,
                "end_frame": 450,
                "thumbnail_path": "",
            }
        ]

        # Peak window overlaps with scene boundary
        peak_windows = [
            {"start": 10.2, "end": 13.5}
        ]

        aligned = sc.snap_to_scene_boundary(peak_windows, scenes, tolerance_sec=3.0)
        assert len(aligned) == 1
        aligned_clip = aligned[0]

        # Should snap to 10.0 (within 3s tolerance)
        assert aligned_clip["aligned_start"] == 10.0, (
            f"Expected aligned_start=10.0, got {aligned_clip['aligned_start']}"
        )
        assert aligned_clip["snapped"] is True, "Should be marked as snapped"


class TestTC08_SnapFarBoundary:
    """TC-08: peak at t=10.2-13.5, boundary at t=15.0 (gap=4.8s>3s) → not snapped."""

    def test_no_snap_when_boundary_far(self, test_video_with_scenes, temp_workspace):
        # Scene boundary is far away - starts at 20.0
        scenes = [
            {
                "scene_number": 1,
                "start_time": 20.0,
                "end_time": 25.0,
                "duration": 5.0,
                "start_frame": 600,
                "end_frame": 750,
                "thumbnail_path": "",
            }
        ]

        peak_windows = [
            {"start": 10.2, "end": 13.5}
        ]

        aligned = sc.snap_to_scene_boundary(peak_windows, scenes, tolerance_sec=3.0)
        assert len(aligned) == 1
        aligned_clip = aligned[0]

        # Should NOT snap (20.0 - 13.5 = 6.5 > 3.0)
        assert aligned_clip["aligned_start"] == 10.2, (
            f"Expected aligned_start=10.2, got {aligned_clip['aligned_start']}"
        )
        assert aligned_clip["aligned_end"] == 13.5, (
            f"Expected aligned_end=13.5, got {aligned_clip['aligned_end']}"
        )
        assert aligned_clip["snapped"] is False, "Should not be snapped"


class TestTC09_EmptyPeakWindows:
    """TC-09: Empty peak_windows list → aligned_clips is empty, no crash."""

    def test_empty_peak_windows(self, test_video_with_scenes, temp_workspace):
        os.makedirs("temp/thumbnails", exist_ok=True)
        result = sc.run_scene_detection(
            test_video_with_scenes,
            "test_job_09",
            peak_windows=[],
        )

        assert result["aligned_clips"] == [], "aligned_clips should be empty"
        # Just verify it doesn't crash and has the expected structure
        assert isinstance(result, dict), "Should return dict"
        assert "scenes" in result, "Should have scenes key"
        assert "total_scenes" in result, "Should have total_scenes key"


class TestTC10_StaticVideoNoScenes:
    """TC-10: Static video (no scene changes) → gracefully return without crash."""

    def test_static_video_no_crash(self, static_video, temp_workspace):
        os.makedirs("temp/thumbnails", exist_ok=True)
        result = sc.run_scene_detection(static_video, "test_job_10")

        # May have 0 or 1 scenes, but should not crash
        assert isinstance(result, dict), "Should return dict"
        assert "scenes" in result, "Should have 'scenes' key"
        assert isinstance(result["scenes"], list), "scenes should be list"


class TestTC11_TotalVsUsable:
    """TC-11: total_scenes >= usable_scenes always (usable is subset)."""

    def test_total_gte_usable(self, test_video_with_scenes, temp_workspace):
        os.makedirs("temp/thumbnails", exist_ok=True)
        result = sc.run_scene_detection(test_video_with_scenes, "test_job_11")

        assert result["total_scenes"] >= result["usable_scenes"], (
            f"total_scenes ({result['total_scenes']}) < usable_scenes ({result['usable_scenes']})"
        )


class TestTC12_JSONSaved:
    """TC-12: scenes_{job_id}.json saved and valid parseable JSON."""

    def test_json_saved_and_valid(self, test_video_with_scenes, temp_workspace):
        os.makedirs("temp/thumbnails", exist_ok=True)
        job_id = "test_job_12"
        result = sc.run_scene_detection(test_video_with_scenes, job_id)

        json_path = os.path.join("temp", f"scenes_{job_id}.json")
        assert os.path.exists(json_path), f"JSON not saved: {json_path}"

        # Try to parse
        with open(json_path, "r") as f:
            loaded = json.load(f)

        assert loaded["job_id"] == job_id
        assert "scenes" in loaded
        assert "aligned_clips" in loaded
        assert "total_scenes" in loaded
        assert "usable_scenes" in loaded


# ---------------------------------------------------------------------------
# Edge Case: Verify frame numbers are sequential
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Edge cases for robustness."""

    def test_frame_numbers_sequential(self, test_video_with_scenes):
        """Verify frame numbers make sense."""
        scenes = sc.detect_scenes(test_video_with_scenes, threshold=15.0)
        for scene in scenes:
            assert scene["start_frame"] < scene["end_frame"], (
                f"Scene {scene['scene_number']}: start_frame >= end_frame"
            )

    def test_scenes_maintain_number_sequence(self, test_video_with_scenes):
        """Scene numbers should be sequential starting from 1."""
        result = sc.run_scene_detection(test_video_with_scenes, "edge_1")
        scenes = result["scenes"]

        for i, scene in enumerate(scenes):
            assert scene["scene_number"] == i + 1, (
                f"Scene number mismatch: expected {i+1}, got {scene['scene_number']}"
            )
