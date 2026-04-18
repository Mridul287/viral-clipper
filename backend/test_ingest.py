import os
import uuid
import pytest
import pathlib
import librosa
from ingest import ingest_from_file, ingest_from_url, extract_audio, TEMP_DIR

# Sample paths for testing
VALID_VIDEO_PATH = "test_data/sample.mp4"
INVALID_EXT_PATH = "test_data/sample.txt"
# Using a very stable, short video (Me at the zoo - first youtube video)
VALID_YT_URL = "https://www.youtube.com/watch?v=jNQXAC9IVRw" 
INVALID_YT_URL = "https://www.youtube.com/watch?v=invalid_id_XYZ"

@pytest.fixture(scope="session", autouse=True)
def setup_test_data():
    """
    Creates dummy test files needed for testing using moviepy.
    """
    test_dir = pathlib.Path("test_data")
    test_dir.mkdir(exist_ok=True)
    
    # Create an empty txt file
    with open(INVALID_EXT_PATH, "w") as f:
        f.write("test")
        
    # Generate a dummy 1s MP4 file using moviepy
    if not os.path.exists(VALID_VIDEO_PATH):
        try:
            import numpy as np
            import moviepy as mp
            # Create a 1s black clip with a 1s silent audio track
            clip = mp.ColorClip(size=(64, 64), color=(0,0,0), duration=1)
            # Create a silent audio clip
            audio = mp.AudioClip(lambda t: [0], duration=1, fps=16000)
            clip = clip.with_audio(audio)
            clip.write_videofile(VALID_VIDEO_PATH, fps=1, logger=None)
            clip.close()
        except Exception as e:
            print(f"Warning: Could not generate dummy video: {e}")

def test_ingest_valid_file():
    """1. Pass a valid local MP4 file -> should return dict with audio_path ending in .wav"""
    job_id = str(uuid.uuid4())
    result = ingest_from_file(VALID_VIDEO_PATH, job_id)
    assert "job_id" in result
    assert result["job_id"] == job_id
    assert result["audio_path"].endswith(".wav")
    assert result["source"] == "file"
    assert os.path.exists(result["audio_path"])

def test_ingest_valid_url():
    """2. Pass a valid YouTube URL -> should download and return audio_path"""
    job_id = str(uuid.uuid4())
    result = ingest_from_url(VALID_YT_URL, job_id)
    assert "job_id" in result
    assert result["job_id"] == job_id
    assert result["audio_path"].endswith(".wav")
    assert result["source"] == "url"
    assert os.path.exists(result["audio_path"])

def test_ingest_invalid_url():
    """3. Pass an invalid URL -> should raise ValueError with message"""
    job_id = str(uuid.uuid4())
    with pytest.raises(ValueError, match="Invalid or unreachable URL"):
        ingest_from_url(INVALID_YT_URL, job_id)

def test_ingest_unsupported_file():
    """4. Pass an unsupported file type like .txt -> should raise ValueError"""
    job_id = str(uuid.uuid4())
    with pytest.raises(ValueError, match="Unsupported file format"):
        ingest_from_file(INVALID_EXT_PATH, job_id)

def test_output_wav_format():
    """5. Check output .wav file with librosa: sample_rate == 16000, channels == 1"""
    job_id = str(uuid.uuid4())
    result = ingest_from_file(VALID_VIDEO_PATH, job_id)
    audio_path = result["audio_path"]
    
    # By default librosa loads as mono. We use sr=None to get original SR
    y, sr = librosa.load(audio_path, sr=None, mono=False)
    
    assert sr == 16000
    # If mono=False, y will be 1D array if mono, 2D if stereo
    assert len(y.shape) == 1  # Ensures mono 

def test_unique_job_ids():
    """6. Confirm job_id is preserved when passed"""
    job_id1 = str(uuid.uuid4())
    job_id2 = str(uuid.uuid4())
    res1 = ingest_from_file(VALID_VIDEO_PATH, job_id1)
    res2 = ingest_from_file(VALID_VIDEO_PATH, job_id2)
    assert res1["job_id"] == job_id1
    assert res2["job_id"] == job_id2
    assert res1["job_id"] != res2["job_id"]

def test_temp_folder_creation():
    """7. Confirm /temp folder is created automatically"""
    import shutil
    import time
    
    # Clean up first to test creation
    if TEMP_DIR.exists():
        # Give OS a moment to release file handles if any
        try:
            shutil.rmtree(TEMP_DIR)
        except PermissionError:
            # If rmtree fails, we can just check if we can write to it instead
            # or ignore if it's a persistent handle issue on Windows
            pass
        
    # We don't assert it doesn't exist here because rmtree might have partially failed
    # but ingest_from_file should work regardless.
    job_id = str(uuid.uuid4())
    ingest_from_file(VALID_VIDEO_PATH, job_id)
    assert TEMP_DIR.exists()
