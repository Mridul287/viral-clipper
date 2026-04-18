import pytest
import os
import cv2
import numpy as np
import pathlib
import shutil
from unittest.mock import patch

from reframe import reframe_clip, reframe_all_clips, calculate_crop_box, FaceTracker

TEST_DIR = pathlib.Path("temp_test_reframe")

def create_test_video(path: str, width: int, height: int, duration_sec: float, fps: float = 30.0, has_face: bool = True):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(path, fourcc, fps, (width, height))
    total_frames = int(duration_sec * fps)
    
    for i in range(total_frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        if has_face:
            # Simulate a face moving horizontally
            fx = int((width / 4) + (width / 2) * (i / max(1, total_frames)))
            fy = height // 3
            fw, fh = 100, 100
            cv2.rectangle(frame, (fx, fy), (fx+fw, fy+fh), (255, 255, 255), -1)
        out.write(frame)
    out.release()
    return path

@pytest.fixture(autouse=True)
def setup_teardown():
    TEST_DIR.mkdir(parents=True, exist_ok=True)
    yield
    if TEST_DIR.exists():
        shutil.rmtree(TEST_DIR, ignore_errors=True)

# Math tests
def test_calculate_crop_box_center():
    # Test 5: center_x=960, source 1920x1080 -> width = 608. 960 - 304 = 656.
    left, top, right, bottom = calculate_crop_box(960, 1920, 1080)
    assert left == 656
    assert right == 1264
    assert right - left == 608

def test_calculate_crop_box_left_edge():
    # Test 6: center_x=100 (near left edge), source 1920x1080 -> left=0 (clamped)
    left, top, right, bottom = calculate_crop_box(100, 1920, 1080)
    assert left == 0
    assert right == 608

def test_calculate_crop_box_right_edge():
    # Test 7: center_x=1900 (near right edge), source 1920x1080 -> right=1920 (clamped)
    left, top, right, bottom = calculate_crop_box(1900, 1920, 1080)
    assert right == 1920
    assert left == 1920 - 608

def test_smoothing():
    # Test 8: Smoothing test: push [100,100,100,900,900,900] into deque with window=6 -> smooth_center_x should be 500
    tracker = FaceTracker(smooth_window=6)
    pts = [100, 100, 100, 900, 900, 900]
    result = 0
    for p in pts:
        result = tracker.get_smooth_center_x(p)
    assert result == 500

@patch('cv2.CascadeClassifier.detectMultiScale')
def test_reframe_horizontal_video(mock_detect):
    # Test 1, 2, 3, 4, 11, 12, 14
    mock_detect.return_value = [[500, 400, 80, 80]] # Mock uniform stationary face
    
    input_vid = str(TEST_DIR / "horiz.mp4")
    output_vid = str(TEST_DIR / "horiz_vert.mp4")
    
    create_test_video(input_vid, 1920, 1080, 1.0, 30.0, True)
    
    res = reframe_clip(input_vid, "job123", output_vid)
    
    assert os.path.exists(output_vid)
    assert res["target_resolution"] == [1080, 1920]
    assert res["total_frames"] == 30
    assert res["frames_with_face"] + res["frames_without_face"] == 30
    assert 0.0 <= res["face_detection_rate"] <= 1.0
    assert res["processing_time_sec"] > 0
    
    # Check output specs
    cap = cv2.VideoCapture(output_vid)
    assert cap.get(cv2.CAP_PROP_FRAME_WIDTH) == 1080
    assert cap.get(cv2.CAP_PROP_FRAME_HEIGHT) == 1920
    # Since OpenCV VideoWriter sometimes drops a frame at end, we allow ±1 or match exactly.
    assert abs(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 30) <= 2
    assert int(cap.get(cv2.CAP_PROP_FPS)) == 30
    cap.release()

@patch('cv2.CascadeClassifier.detectMultiScale')
def test_reframe_no_face(mock_detect):
    # Test 9: No face video -> fallback center, no crash, detect rate 0.0
    mock_detect.return_value = ()
    
    input_vid = str(TEST_DIR / "noface.mp4")
    output_vid = str(TEST_DIR / "noface_vert.mp4")
    
    create_test_video(input_vid, 1920, 1080, 0.5, 30.0, False)
    
    res = reframe_clip(input_vid, "job123", output_vid)
    
    assert os.path.exists(output_vid)
    assert res["face_detection_rate"] == 0.0
    assert res["frames_with_face"] == 0

def test_already_vertical():
    # Test 10: Already vertical video (1080x1920 input) -> skip reframe, copy
    input_vid = str(TEST_DIR / "vert.mp4")
    output_vid = str(TEST_DIR / "vert_out.mp4")
    
    create_test_video(input_vid, 1080, 1920, 0.5, 30.0, False)
    
    res = reframe_clip(input_vid, "job123", output_vid)
    
    assert os.path.exists(output_vid)
    assert res["source_resolution"] == [1080, 1920]
    assert res["target_resolution"] == [1080, 1920]
    
@patch('cv2.CascadeClassifier.detectMultiScale')
def test_reframe_all_clips(mock_detect):
    # Test 13: reframe_all_clips with 3 dicts
    mock_detect.return_value = ()
    
    clips = []
    for i in range(3):
        vid = str(TEST_DIR / f"clip_{i}.mp4")
        create_test_video(vid, 1920, 1080, 0.2, 30.0, False)
        clips.append({"path": vid})
        
    results = reframe_all_clips(clips, "job123", str(TEST_DIR))
    
    assert len(results) == 3
    for res in results:
        assert os.path.exists(res["output_path"])
        assert res["target_resolution"] == [1080, 1920]
