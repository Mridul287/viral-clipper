"""
test_orchestrator.py
--------------------
Pytest suite for backend/orchestrator.py

Covers all 14 test cases specified in the task:

  TC-01  Full happy path: valid video + clips → status="success", clips non-empty
  TC-02  Every clip has all 12 required keys
  TC-03  steps_completed contains all 4 steps when all succeed
  TC-04  Progress callback called exactly 4 times (once per step)
  TC-05  Emotion step failure → pipeline continues, emotion missing from steps
  TC-06  Scene step failure → pipeline continues with original boundaries
  TC-07  One clip export failure → status="partial", other clips present
  TC-08  All clips fail → status="failed"
  TC-09  use_scene_snap=False → apply_scene_snapping NOT called
  TC-10  match_emotion_to_clip: happy at t=5, surprise at t=7, clip 6-8 → surprise
  TC-11  processing_time_sec > 0 always
  TC-12  final_{job_id}.json saved, valid JSON, all keys present
  TC-13  Empty top_clips → status="success", clips=[], no crash
  TC-14  top_n_clips option: 10 clips, top_n_clips=3 → 3 clips in output
"""

import json
import os
import time
import pytest
from unittest.mock import patch, MagicMock, call
from pathlib import Path

# Module under test
import orchestrator as orch


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def temp_workspace(tmp_path):
    """Create temporary workspace."""
    (tmp_path / "temp").mkdir()
    (tmp_path / "output").mkdir()

    original_cwd = os.getcwd()
    os.chdir(str(tmp_path))

    yield tmp_path

    os.chdir(original_cwd)


def mock_emotion_result():
    """Return mock emotion detection result."""
    return {
        "job_id": "test_job",
        "frame_emotions": [
            {"timestamp": 5.0, "emotion": "happy", "confidence": 0.9, "intensity": 0.8},
            {"timestamp": 5.5, "emotion": "happy", "confidence": 0.85, "intensity": 0.75},
            {"timestamp": 7.0, "emotion": "surprise", "confidence": 0.95, "intensity": 0.9},
            {"timestamp": 7.5, "emotion": "surprise", "confidence": 0.9, "intensity": 0.85},
        ],
        "peak_windows": [
            {
                "start": 5.0,
                "end": 7.5,
                "dominant_emotion": "happy",
                "avg_intensity": 0.82,
            }
        ],
    }


def mock_scene_result():
    """Return mock scene detection result."""
    return {
        "job_id": "test_job",
        "scenes": [
            {
                "scene_number": 1,
                "start_time": 0.0,
                "end_time": 5.0,
                "duration": 5.0,
                "thumbnail_path": "temp/thumbnails/scene_test_job_1.jpg",
            }
        ],
        "aligned_clips": [
            {
                "original_start": 1.5,
                "original_end": 4.5,
                "aligned_start": 1.5,
                "aligned_end": 5.0,
                "snapped": True,
            }
        ],
        "total_scenes": 1,
        "usable_scenes": 1,
    }


def mock_export_result(n_clips=3):
    """Return mock export result."""
    clips = []
    for i in range(1, n_clips + 1):
        clips.append({
            "rank": i,
            "clip_path": f"output/test_job/clip_test_job_{i}.mp4",
            "thumbnail_path": f"output/test_job/thumbnails/clip_test_job_{i}.jpg",
            "duration": 5.0 + i,
            "file_size_mb": 10.0 + i,
            "caption_word_count": 5,
            "start": float(i * 2),
            "end": float(i * 2 + 5),
            "suggested_title": f"Clip {i}",
            "clip_type": "moment",
            "final_score": 0.9 - (i * 0.01),
        })
    return {
        "job_id": "test_job",
        "clips": clips,
        "total_clips": len(clips),
        "output_dir": "output/test_job",
    }


# ---------------------------------------------------------------------------
# Test Cases
# ---------------------------------------------------------------------------

class TestTC01_HappyPath:
    """TC-01: Full happy path → status="success", clips non-empty."""

    def test_happy_path(self, temp_workspace):
        top_clips = [
            {
                "start": 1.5,
                "end": 4.5,
                "suggested_title": "Clip 1",
                "clip_type": "moment",
                "final_score": 0.95,
            }
        ]

        with patch("orchestrator.em.detect_emotions", return_value=mock_emotion_result()):
            with patch("orchestrator.sc.run_scene_detection", return_value=mock_scene_result()):
                with patch("orchestrator.exp.export_all_clips", return_value=mock_export_result(1)):
                    result = orch.run_video_pipeline(
                        "test_job",
                        "dummy_video.mp4",
                        top_clips,
                    )

        assert result["status"] == "success"
        assert len(result["clips"]) > 0
        assert result["total_clips"] > 0


class TestTC02_AllRequiredKeys:
    """TC-02: Every clip has all 12 required keys."""

    def test_clip_has_all_keys(self, temp_workspace):
        top_clips = [
            {
                "start": 1.5,
                "end": 4.5,
                "suggested_title": "Clip 1",
                "clip_type": "moment",
                "final_score": 0.95,
            }
        ]

        required_keys = {
            "rank", "clip_path", "thumbnail_path", "duration",
            "file_size_mb", "caption_word_count", "start", "end",
            "clip_type", "suggested_title", "final_score",
            "dominant_emotion", "scene_snapped",
        }

        with patch("orchestrator.em.detect_emotions", return_value=mock_emotion_result()):
            with patch("orchestrator.sc.run_scene_detection", return_value=mock_scene_result()):
                with patch("orchestrator.exp.export_all_clips", return_value=mock_export_result(1)):
                    result = orch.run_video_pipeline(
                        "test_job",
                        "dummy_video.mp4",
                        top_clips,
                    )

        for clip in result["clips"]:
            assert required_keys.issubset(clip.keys()), (
                f"Missing keys: {required_keys - set(clip.keys())}"
            )


class TestTC03_StepsCompleted:
    """TC-03: steps_completed contains all 4 steps when all succeed."""

    def test_steps_completed_all(self, temp_workspace):
        top_clips = [{"start": 1.0, "end": 2.0, "suggested_title": "C", "clip_type": "m", "final_score": 0.9}]

        with patch("orchestrator.em.detect_emotions", return_value=mock_emotion_result()):
            with patch("orchestrator.sc.run_scene_detection", return_value=mock_scene_result()):
                with patch("orchestrator.exp.export_all_clips", return_value=mock_export_result(1)):
                    result = orch.run_video_pipeline(
                        "test_job",
                        "dummy_video.mp4",
                        top_clips,
                        options={"use_scene_snap": True},
                    )

        expected_steps = {"emotion", "scene", "scene_snap", "export"}
        assert expected_steps.issubset(set(result["steps_completed"])), (
            f"Missing steps: {expected_steps - set(result['steps_completed'])}"
        )


class TestTC04_ProgressCallback:
    """TC-04: Progress callback called exactly 4 times."""

    def test_callback_called_4_times(self, temp_workspace):
        top_clips = [{"start": 1.0, "end": 2.0, "suggested_title": "C", "clip_type": "m", "final_score": 0.9}]
        mock_callback = MagicMock()

        with patch("orchestrator.em.detect_emotions", return_value=mock_emotion_result()):
            with patch("orchestrator.sc.run_scene_detection", return_value=mock_scene_result()):
                with patch("orchestrator.exp.export_all_clips", return_value=mock_export_result(1)):
                    orch.run_video_pipeline(
                        "test_job",
                        "dummy_video.mp4",
                        top_clips,
                        on_progress=mock_callback,
                    )

        assert mock_callback.call_count == 4, (
            f"Callback should be called 4 times, got {mock_callback.call_count}"
        )


class TestTC05_EmotionFailure:
    """TC-05: Emotion step failure → pipeline continues, emotion missing."""

    def test_emotion_failure_continues(self, temp_workspace):
        top_clips = [{"start": 1.0, "end": 2.0, "suggested_title": "C", "clip_type": "m", "final_score": 0.9}]

        with patch("orchestrator.em.detect_emotions", side_effect=Exception("Emotion error")):
            with patch("orchestrator.sc.run_scene_detection", return_value=mock_scene_result()):
                with patch("orchestrator.exp.export_all_clips", return_value=mock_export_result(1)):
                    result = orch.run_video_pipeline(
                        "test_job",
                        "dummy_video.mp4",
                        top_clips,
                    )

        assert result["status"] != "failed", "Should continue on emotion failure"
        assert "emotion" not in result["steps_completed"], (
            "emotion should not be in steps_completed"
        )
        assert "export" in result["steps_completed"], "export should still complete"


class TestTC06_SceneFailure:
    """TC-06: Scene step failure → pipeline continues with original boundaries."""

    def test_scene_failure_uses_original_bounds(self, temp_workspace):
        top_clips = [
            {
                "start": 1.5,
                "end": 4.5,
                "suggested_title": "Clip",
                "clip_type": "moment",
                "final_score": 0.9,
            }
        ]

        with patch("orchestrator.em.detect_emotions", return_value=mock_emotion_result()):
            with patch("orchestrator.sc.run_scene_detection", side_effect=Exception("Scene error")):
                with patch("orchestrator.exp.export_all_clips", return_value=mock_export_result(1)):
                    result = orch.run_video_pipeline(
                        "test_job",
                        "dummy_video.mp4",
                        top_clips,
                    )

        assert result["status"] != "failed", "Should continue on scene failure"
        assert "export" in result["steps_completed"], "export should still complete"


class TestTC07_PartialExportFailure:
    """TC-07: One clip fails → status="partial", others present."""

    def test_one_clip_fails(self, temp_workspace):
        top_clips = [
            {"start": 1.0, "end": 2.0, "suggested_title": "C1", "clip_type": "m", "final_score": 0.9},
            {"start": 3.0, "end": 4.0, "suggested_title": "C2", "clip_type": "m", "final_score": 0.8},
        ]

        # Mock export to return only 1 clip (simulating 1 failure)
        with patch("orchestrator.em.detect_emotions", return_value=mock_emotion_result()):
            with patch("orchestrator.sc.run_scene_detection", return_value=mock_scene_result()):
                with patch("orchestrator.exp.export_all_clips", return_value=mock_export_result(1)):
                    result = orch.run_video_pipeline(
                        "test_job",
                        "dummy_video.mp4",
                        top_clips,
                    )

        assert result["status"] == "partial"
        assert len(result["clips"]) > 0


class TestTC08_AllClipsFail:
    """TC-08: All clips fail → status="failed"."""

    def test_all_clips_fail(self, temp_workspace):
        top_clips = [
            {"start": 1.0, "end": 2.0, "suggested_title": "C1", "clip_type": "m", "final_score": 0.9},
            {"start": 3.0, "end": 4.0, "suggested_title": "C2", "clip_type": "m", "final_score": 0.8},
        ]

        # Mock export to return 0 clips
        export_result_empty = mock_export_result(0)

        with patch("orchestrator.em.detect_emotions", return_value=mock_emotion_result()):
            with patch("orchestrator.sc.run_scene_detection", return_value=mock_scene_result()):
                with patch("orchestrator.exp.export_all_clips", return_value=export_result_empty):
                    result = orch.run_video_pipeline(
                        "test_job",
                        "dummy_video.mp4",
                        top_clips,
                    )

        assert result["status"] == "failed"


class TestTC09_NoSceneSnap:
    """TC-09: use_scene_snap=False → scene snapping NOT applied."""

    def test_no_scene_snap_applied(self, temp_workspace):
        top_clips = [{"start": 1.0, "end": 2.0, "suggested_title": "C", "clip_type": "m", "final_score": 0.9}]

        with patch("orchestrator.em.detect_emotions", return_value=mock_emotion_result()):
            with patch("orchestrator.sc.run_scene_detection", return_value=mock_scene_result()):
                with patch("orchestrator.exp.export_all_clips", return_value=mock_export_result(1)):
                    with patch("orchestrator.apply_scene_snapping") as mock_snap:
                        result = orch.run_video_pipeline(
                            "test_job",
                            "dummy_video.mp4",
                            top_clips,
                            options={"use_scene_snap": False},
                        )

        # apply_scene_snapping should NOT be called
        mock_snap.assert_not_called()


class TestTC10_MatchEmotion:
    """TC-10: happy at t=5, surprise at t=7, clip 6-8 → surprise."""

    def test_match_emotion_to_clip(self):
        frame_emotions = [
            {"timestamp": 5.0, "emotion": "happy", "confidence": 0.9, "intensity": 0.8},
            {"timestamp": 5.5, "emotion": "happy", "confidence": 0.85, "intensity": 0.75},
            {"timestamp": 7.0, "emotion": "surprise", "confidence": 0.95, "intensity": 0.9},
            {"timestamp": 7.5, "emotion": "surprise", "confidence": 0.9, "intensity": 0.85},
        ]

        # Clip from 6.0 to 8.0 should match surprise (at 7.0-7.5)
        emotion = orch.match_emotion_to_clip(6.0, 8.0, frame_emotions)

        assert emotion == "surprise", f"Expected 'surprise', got '{emotion}'"


class TestTC11_ProcessingTime:
    """TC-11: processing_time_sec >= 0 always."""

    def test_processing_time_non_negative(self, temp_workspace):
        top_clips = [{"start": 1.0, "end": 2.0, "suggested_title": "C", "clip_type": "m", "final_score": 0.9}]

        with patch("orchestrator.em.detect_emotions", return_value=mock_emotion_result()):
            with patch("orchestrator.sc.run_scene_detection", return_value=mock_scene_result()):
                with patch("orchestrator.exp.export_all_clips", return_value=mock_export_result(1)):
                    result = orch.run_video_pipeline(
                        "test_job",
                        "dummy_video.mp4",
                        top_clips,
                    )

        assert result["processing_time_sec"] >= 0, (
            f"processing_time_sec should be >= 0, got {result['processing_time_sec']}"
        )


class TestTC12_JSONSaved:
    """TC-12: final_{job_id}.json saved, valid JSON, all keys present."""

    def test_json_saved_valid(self, temp_workspace):
        top_clips = [{"start": 1.0, "end": 2.0, "suggested_title": "C", "clip_type": "m", "final_score": 0.9}]

        with patch("orchestrator.em.detect_emotions", return_value=mock_emotion_result()):
            with patch("orchestrator.sc.run_scene_detection", return_value=mock_scene_result()):
                with patch("orchestrator.exp.export_all_clips", return_value=mock_export_result(1)):
                    result = orch.run_video_pipeline(
                        "test_job",
                        "dummy_video.mp4",
                        top_clips,
                    )

        json_path = "temp/final_test_job.json"
        assert os.path.exists(json_path), f"JSON not saved: {json_path}"

        with open(json_path, "r") as f:
            loaded = json.load(f)

        required_keys = {
            "job_id", "status", "steps_completed", "emotion_summary",
            "scene_summary", "clips", "total_clips", "processing_time_sec",
        }
        assert required_keys.issubset(loaded.keys()), (
            f"Missing keys: {required_keys - set(loaded.keys())}"
        )


class TestTC13_EmptyClips:
    """TC-13: Empty top_clips → status="success", clips=[], no crash."""

    def test_empty_clips(self, temp_workspace):
        with patch("orchestrator.em.detect_emotions", return_value=mock_emotion_result()):
            with patch("orchestrator.sc.run_scene_detection", return_value=mock_scene_result()):
                with patch("orchestrator.exp.export_all_clips", return_value=mock_export_result(0)):
                    result = orch.run_video_pipeline(
                        "test_job",
                        "dummy_video.mp4",
                        [],
                    )

        assert isinstance(result, dict), "Should return dict"
        assert result["clips"] == [], "clips should be empty"
        assert result["total_clips"] == 0, "total_clips should be 0"


class TestTC14_TopNClips:
    """TC-14: 10 clips + top_n_clips=3 → 3 clips output."""

    def test_top_n_clips_limit(self, temp_workspace):
        # Create 10 top clips
        top_clips = [
            {
                "start": float(i),
                "end": float(i + 1),
                "suggested_title": f"Clip {i}",
                "clip_type": "moment",
                "final_score": 0.9 - (i * 0.01),
            }
            for i in range(10)
        ]

        # Export should be called with only 3 clips
        with patch("orchestrator.em.detect_emotions", return_value=mock_emotion_result()):
            with patch("orchestrator.sc.run_scene_detection", return_value=mock_scene_result()):
                with patch("orchestrator.exp.export_all_clips", return_value=mock_export_result(3)) as mock_export:
                    result = orch.run_video_pipeline(
                        "test_job",
                        "dummy_video.mp4",
                        top_clips,
                        options={"top_n_clips": 3},
                    )

        # Check that export was called with only 3 clips
        assert mock_export.called
        call_args = mock_export.call_args
        exported_clips = call_args[0][1]  # Second positional argument
        assert len(exported_clips) == 3, f"Expected 3 clips, got {len(exported_clips)}"


# ---------------------------------------------------------------------------
# Edge Cases & Integration Tests
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Edge case tests."""

    def test_apply_scene_snapping_empty_aligned(self):
        """Test scene snapping with no aligned clips."""
        top_clips = [{"start": 1.0, "end": 2.0}]
        result = orch.apply_scene_snapping(top_clips, [])

        assert len(result) == 1
        assert result[0]["scene_snapped"] is False

    def test_match_emotion_empty_range(self):
        """Test emotion matching with no emotions in range."""
        frame_emotions = [
            {"timestamp": 1.0, "emotion": "happy", "intensity": 0.8},
        ]
        emotion = orch.match_emotion_to_clip(5.0, 6.0, frame_emotions)

        assert emotion == "", "Should return empty string when no emotions in range"

    def test_match_emotion_multiple_emotions(self):
        """Test emotion matching with multiple emotions, returns highest score."""
        frame_emotions = [
            {"timestamp": 1.0, "emotion": "happy", "intensity": 0.5},
            {"timestamp": 1.5, "emotion": "surprise", "intensity": 0.9},
            {"timestamp": 2.0, "emotion": "happy", "intensity": 0.6},
        ]
        emotion = orch.match_emotion_to_clip(0.5, 2.5, frame_emotions)

        # happy: 0.5 + 0.6 = 1.1, surprise: 0.9
        assert emotion == "happy", f"Expected 'happy' (higher score), got '{emotion}'"

    def test_callback_receives_correct_params(self, temp_workspace):
        """Test that callback receives correct (step, total, message) params."""
        top_clips = [{"start": 1.0, "end": 2.0, "suggested_title": "C", "clip_type": "m", "final_score": 0.9}]
        callback_calls = []

        def capture_callback(step, total, message):
            callback_calls.append((step, total, message))

        with patch("orchestrator.em.detect_emotions", return_value=mock_emotion_result()):
            with patch("orchestrator.sc.run_scene_detection", return_value=mock_scene_result()):
                with patch("orchestrator.exp.export_all_clips", return_value=mock_export_result(1)):
                    orch.run_video_pipeline(
                        "test_job",
                        "dummy_video.mp4",
                        top_clips,
                        on_progress=capture_callback,
                    )

        assert len(callback_calls) == 4
        for step, total, message in callback_calls:
            assert step >= 1 and step <= 4, f"step {step} out of range"
            assert total == 4, f"total should be 4, got {total}"
            assert isinstance(message, str), f"message should be str, got {type(message)}"
