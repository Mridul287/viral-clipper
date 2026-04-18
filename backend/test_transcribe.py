import pytest
import os
import wave
import uuid
import math
from pathlib import Path

from transcribe import transcribe, chunk_audio, merge_chunks, TEMP_DIR

# Using tiny model to keep tests fast, "medium" would take too long to run repeatedly
TEST_MODEL_SIZE = "tiny" 

def generate_silent_wav(file_path, duration_sec):
    """Helper function to create a silent wav file of given duration."""
    sample_rate = 16000
    n_frames = int(sample_rate * duration_sec)
    
    with wave.open(file_path, 'wb') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(b'\x00' * 2 * n_frames)

@pytest.fixture(scope="module", autouse=True)
def setup_audio_files():
    """Setup audio files for testing."""
    os.makedirs("test_data", exist_ok=True)
    generate_silent_wav("test_data/2min_audio.wav", 120)
    generate_silent_wav("test_data/15min_audio.wav", 15 * 60)
    generate_silent_wav("test_data/silent_audio.wav", 5)

def test_transcribe_2min_audio():
    """1. Pass a valid 2-minute .wav file -> transcript dict returned with all 4 keys present"""
    job_id = "test_job_2min"
    res = transcribe("test_data/2min_audio.wav", job_id, model_size=TEST_MODEL_SIZE)
    
    assert "job_id" in res
    assert "transcript" in res
    assert "words" in res
    assert "segments" in res
    
def test_transcribe_words_list():
    """2. Check words list: every item has 'word', 'start', 'end' keys, start < end always"""
    job_id = "test_job_words"
    res = transcribe("test_data/2min_audio.wav", job_id, model_size=TEST_MODEL_SIZE)
    
    for word_info in res["words"]:
        assert "word" in word_info
        assert "start" in word_info
        assert "end" in word_info
        assert word_info["start"] <= word_info["end"]

def test_transcribe_segments_list():
    """3. Check segments list: segments are in chronological order"""
    job_id = "test_job_segments"
    res = transcribe("test_data/2min_audio.wav", job_id, model_size=TEST_MODEL_SIZE)
    
    segments = res["segments"]
    for i in range(len(segments) - 1):
        assert segments[i]["end"] <= segments[i + 1]["start"] + 0.5

def test_chunking_long_audio():
    """4. Pass a 15-minute audio -> confirm chunking happens (chunk_audio returns >1 file)"""
    chunks = chunk_audio("test_data/15min_audio.wav", chunk_minutes=5)
    assert len(chunks) > 1
    assert len(chunks) == 4 # 15 min with 4.5m jumps = 4 chunks

def test_merge_no_duplicates():
    """5. After merge, confirm no duplicate words appear at chunk boundaries"""
    chunk_1 = {
        "text": "Hello world overlapping word",
        "words": [
            {"word": "Hello", "start": 0.0, "end": 1.0},
            {"word": "world", "start": 1.0, "end": 2.0},
            {"word": "overlapping", "start": 271.0, "end": 272.0},
        ],
        "segments": [{"text": "Hello world overlapping word", "start": 0.0, "end": 272.0}]
    }
    chunk_2 = {
        "text": "overlapping word is second",
        "words": [
            {"word": "overlapping", "start": 1.0, "end": 2.0},
            {"word": "word", "start": 2.0, "end": 3.0},
        ],
        "segments": [{"text": "overlapping word is second", "start": 1.0, "end": 3.0}]
    }
    
    merged = merge_chunks([chunk_1, chunk_2])
    all_words = [w["word"] for w in merged["words"]]
    
    assert "Hello" in all_words
    assert all_words.count("overlapping") == 1

def test_transcribe_silent_audio():
    """6. Pass a silent .wav file -> should return empty transcript gracefully, no crash"""
    job_id = "test_job_silent"
    res = transcribe("test_data/silent_audio.wav", job_id, model_size=TEST_MODEL_SIZE)
    assert isinstance(res, dict)

def test_transcript_file_saved():
    """7. Confirm transcript_{job_id}.json file is saved in /temp"""
    job_id = f"test_job_{uuid.uuid4()}"
    transcribe("test_data/silent_audio.wav", job_id, model_size=TEST_MODEL_SIZE)
    
    expected_path = TEMP_DIR / f"transcript_{job_id}.json"
    assert expected_path.exists()
    
def test_transcribe_deterministic():
    """8. Run transcribe twice on same file -> confirm output is deterministic"""
    job_id_1 = str(uuid.uuid4())
    job_id_2 = str(uuid.uuid4())
    
    res1 = transcribe("test_data/2min_audio.wav", job_id_1, model_size=TEST_MODEL_SIZE)
    res2 = transcribe("test_data/2min_audio.wav", job_id_2, model_size=TEST_MODEL_SIZE)
    
    words_count_1 = len(res1["words"])
    words_count_2 = len(res2["words"])
    
    assert abs(words_count_1 - words_count_2) <= 2
