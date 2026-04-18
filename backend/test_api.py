"""
test_api.py — FastAPI endpoint tests using TestClient (httpx).
The pipeline function is fully mocked so no AI processing occurs.
"""

import io
import time
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

import job_store
from main import app

client = TestClient(app, raise_server_exceptions=False)

# ---------------------------------------------------------------------------
# Shared mock result (what a completed pipeline job would produce)
# ---------------------------------------------------------------------------

MOCK_RESULT = {
    "job_id": "mock-job",
    "top_clips": [
        {
            "rank": 1,
            "start": 0.0,
            "end": 10.0,
            "text": "This one secret changed everything.",
            "scores": {"funny": 7, "surprising": 9, "quotable": 9, "emotional": 8, "virality": 10},
            "final_score": 9.3,
            "clip_type": "hook",
            "suggested_title": "The Secret That Changed Everything",
        }
    ],
}


def _fake_pipeline_success(job_id: str, source: dict) -> None:
    """Simulates a successful pipeline run updating job_store."""
    job_store.update_job(job_id, status="transcribing", progress_percent=25)
    job_store.update_job(job_id, status="scoring", progress_percent=60)
    job_store.update_job(job_id, status="done", progress_percent=100, result=MOCK_RESULT)


def _fake_pipeline_failure(job_id: str, source: dict) -> None:
    """Simulates a mid-pipeline crash."""
    job_store.update_job(job_id, status="transcribing", progress_percent=25)
    job_store.update_job(job_id, status="failed", error="Whisper ran out of memory")


def _dummy_wav() -> tuple:
    """Returns (filename, file-like bytes) for a fake .wav upload."""
    return ("test_video.wav", io.BytesIO(b"RIFF" + b"\x00" * 36), "audio/wav")


# ---------------------------------------------------------------------------
# Test 1 — File upload returns 200 with job_id + status=queued
# ---------------------------------------------------------------------------

def test_upload_file_returns_queued():
    """1. POST /upload with a valid file → 200 with job_id and status='queued'."""
    with patch("main.run_pipeline", side_effect=_fake_pipeline_success):
        resp = client.post(
            "/upload",
            files={"file": _dummy_wav()},
        )
    assert resp.status_code == 200
    data = resp.json()
    assert "job_id" in data
    assert data["status"] == "queued"


# ---------------------------------------------------------------------------
# Test 2 — URL upload returns 200 with job_id
# ---------------------------------------------------------------------------

def test_upload_url_returns_job_id():
    """2. POST /upload with a YouTube URL → 200 with job_id."""
    with patch("main.run_pipeline", side_effect=_fake_pipeline_success):
        resp = client.post(
            "/upload",
            data={"url": "https://www.youtube.com/watch?v=jNQXAC9IVRw"},
        )
    assert resp.status_code == 200
    assert "job_id" in resp.json()


# ---------------------------------------------------------------------------
# Test 3 — GET /status with valid job_id has all required keys
# ---------------------------------------------------------------------------

def test_status_valid_job_has_required_keys():
    """3. GET /status/{valid_job_id} → dict with all required status keys."""
    with patch("main.run_pipeline"):
        resp = client.post("/upload", files={"file": _dummy_wav()})
    job_id = resp.json()["job_id"]

    status_resp = client.get(f"/status/{job_id}")
    assert status_resp.status_code == 200
    data = status_resp.json()
    assert "job_id" in data
    assert "status" in data
    assert "progress_percent" in data
    assert "error" in data


# ---------------------------------------------------------------------------
# Test 4 — GET /status with fake job_id → 404
# ---------------------------------------------------------------------------

def test_status_unknown_job_returns_404():
    """4. GET /status/{fake_job_id} → 404 JSON error."""
    resp = client.get("/status/this-id-does-not-exist")
    assert resp.status_code == 404
    assert "error" in resp.json()


# ---------------------------------------------------------------------------
# Test 5 — GET /results before job completes → 202
# ---------------------------------------------------------------------------

def test_results_before_complete_returns_202():
    """5. GET /results/{job_id} before done → 202 with 'not complete' message."""
    # Create job but do NOT complete it
    with patch("main.run_pipeline"):  # pipeline is a no-op so job stays queued
        resp = client.post("/upload", files={"file": _dummy_wav()})
    job_id = resp.json()["job_id"]

    res_resp = client.get(f"/results/{job_id}")
    assert res_resp.status_code == 202
    assert "not complete" in res_resp.json()["message"].lower()


# ---------------------------------------------------------------------------
# Test 6 — GET /results after job completes → top_clips array
# ---------------------------------------------------------------------------

def test_results_after_complete_returns_clips():
    """6. GET /results/{job_id} after done → top_clips array present."""
    with patch("main.run_pipeline", side_effect=_fake_pipeline_success):
        resp = client.post("/upload", files={"file": _dummy_wav()})
    job_id = resp.json()["job_id"]

    res_resp = client.get(f"/results/{job_id}")
    assert res_resp.status_code == 200
    assert "top_clips" in res_resp.json()


# ---------------------------------------------------------------------------
# Test 7 — Upload unsupported file type → 400
# ---------------------------------------------------------------------------

def test_upload_unsupported_file_returns_400():
    """7. POST /upload with a .exe file → 400 unsupported format."""
    bad_file = ("malware.exe", io.BytesIO(b"MZ\x90\x00"), "application/octet-stream")
    with patch("main.run_pipeline"):
        resp = client.post("/upload", files={"file": bad_file})
    assert resp.status_code == 400
    assert "error" in resp.json()


# ---------------------------------------------------------------------------
# Test 8 — No file and no URL → 422
# ---------------------------------------------------------------------------

def test_upload_no_body_returns_422():
    """8. POST /upload with no file and no URL → 422 validation error."""
    resp = client.post("/upload")
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Test 9 — GET /health → 200 with boolean fields
# ---------------------------------------------------------------------------

def test_health_returns_booleans():
    """9. GET /health → 200 with ollama_connected and whisper_loaded as booleans."""
    with patch("main.http_requests.get", side_effect=Exception("offline")):
        resp = client.get("/health")

    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert isinstance(data["ollama_connected"], bool)
    assert isinstance(data["whisper_loaded"], bool)


# ---------------------------------------------------------------------------
# Test 10 — Pipeline failure → status=failed with error message
# ---------------------------------------------------------------------------

def test_pipeline_failure_sets_failed_status():
    """10. Simulate pipeline failure → GET /status returns status='failed'."""
    with patch("main.run_pipeline", side_effect=_fake_pipeline_failure):
        resp = client.post("/upload", files={"file": _dummy_wav()})
    job_id = resp.json()["job_id"]

    status_resp = client.get(f"/status/{job_id}")
    data = status_resp.json()
    assert data["status"] == "failed"
    assert data["error"] is not None


# ---------------------------------------------------------------------------
# Test 11 — Same file uploaded twice → two different job_ids
# ---------------------------------------------------------------------------

def test_same_file_twice_gets_different_job_ids():
    """11. Upload same file twice → two different job_ids returned."""
    with patch("main.run_pipeline"):
        r1 = client.post("/upload", files={"file": _dummy_wav()})
        r2 = client.post("/upload", files={"file": _dummy_wav()})

    assert r1.json()["job_id"] != r2.json()["job_id"]


# ---------------------------------------------------------------------------
# Test 12 — Progress increases monotonically: 0 → 25 → 60 → 100
# ---------------------------------------------------------------------------

def test_progress_increases_monotonically():
    """12. Progress percent must be non-decreasing throughout the pipeline."""
    progress_snapshots = []

    def capturing_pipeline(job_id: str, source: dict) -> None:
        progress_snapshots.append(job_store.get_job(job_id)["progress_percent"])
        job_store.update_job(job_id, status="transcribing", progress_percent=25)
        progress_snapshots.append(job_store.get_job(job_id)["progress_percent"])
        job_store.update_job(job_id, status="scoring", progress_percent=60)
        progress_snapshots.append(job_store.get_job(job_id)["progress_percent"])
        job_store.update_job(job_id, status="done", progress_percent=100, result=MOCK_RESULT)
        progress_snapshots.append(job_store.get_job(job_id)["progress_percent"])

    with patch("main.run_pipeline", side_effect=capturing_pipeline):
        resp = client.post("/upload", files={"file": _dummy_wav()})

    assert progress_snapshots == [0, 25, 60, 100], f"Got: {progress_snapshots}"
    for i in range(len(progress_snapshots) - 1):
        assert progress_snapshots[i] <= progress_snapshots[i + 1], (
            f"Progress went backwards: {progress_snapshots}"
        )


# ---------------------------------------------------------------------------
# Test 13 — File too large (500 MB limit) → 413
# ---------------------------------------------------------------------------

def test_upload_file_too_large_returns_413():
    """13. POST /upload with a file > 500 MB → 413 error."""
    # We don't need to actually create a 500MB file. 
    # We can mock the file content to be a large stream or just provide many bytes.
    # However, since the code reads in 1MB chunks, we can provide a small file 
    # but patch the MAX_FILE_SIZE_BYTES if we wanted, or just give it slightly more than 500MB.
    # To keep it fast, let's patch the limit to 10 bytes for this test.
    with patch("main.MAX_FILE_SIZE_BYTES", 10):
        large_file = ("too_large.mp4", io.BytesIO(b"A" * 20), "video/mp4")
        resp = client.post("/upload", files={"file": large_file})
        
    assert resp.status_code == 413
    assert "exceeds" in resp.json()["error"].lower()


# ---------------------------------------------------------------------------
# Test 14 — GET /results with fake job_id → 404
# ---------------------------------------------------------------------------

def test_results_unknown_job_returns_404():
    """14. GET /results/{fake_job_id} → 404 JSON error."""
    resp = client.get("/results/non-existent-job-uuid")
    assert resp.status_code == 404
    assert "error" in resp.json()
