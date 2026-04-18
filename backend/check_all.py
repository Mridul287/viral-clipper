import requests
import time
import os

BASE_URL = "http://localhost:8000"

def log(msg):
    print(f"[TEST] {msg}")

def run_checks():
    log("Starting comprehensive API checks...")

    # 1. GET /health
    try:
        resp = requests.get(f"{BASE_URL}/health")
        log(f"Health Check: {resp.status_code} - {resp.json()}")
    except Exception as e:
        log(f"Health Check failed (is the server running?): {e}")
        return

    # 2. POST /upload (URL)
    url_payload = {"url": "https://www.youtube.com/watch?v=jNQXAC9IVRw"}
    resp = requests.post(f"{BASE_URL}/upload", data=url_payload)
    log(f"Upload URL: {resp.status_code} - {resp.json()}")
    job_id = resp.json().get("job_id")

    # 3. GET /status/{job_id}
    if job_id:
        log(f"Polling status for job {job_id}...")
        for _ in range(5):
            s_resp = requests.get(f"{BASE_URL}/status/{job_id}")
            log(f"Status: {s_resp.json()}")
            if s_resp.json().get("status") in ["done", "failed"]:
                break
            time.sleep(1)

    # 4. GET /results/{job_id}
    if job_id:
        r_resp = requests.get(f"{BASE_URL}/results/{job_id}")
        log(f"Results: {r_resp.status_code} - {r_resp.json()}")

    # 5. POST /upload (File) - Small dummy file
    with open("test_upload.mp4", "wb") as f:
        f.write(b"fake mp4 content" * 100)
    
    with open("test_upload.mp4", "rb") as f:
        files = {"file": ("test_upload.mp4", f, "video/mp4")}
        f_resp = requests.post(f"{BASE_URL}/upload", files=files)
        log(f"Upload File: {f_resp.status_code} - {f_resp.json()}")
    
    os.remove("test_upload.mp4")

    # 6. Error Case: Unsupported Format
    with open("test.txt", "w") as f: f.write("hello")
    with open("test.txt", "rb") as f:
        files = {"file": ("test.txt", f, "text/plain")}
        e_resp = requests.post(f"{BASE_URL}/upload", files=files)
        log(f"Upload Unsupported: {e_resp.status_code} - {e_resp.json()}")
    os.remove("test.txt")

    # 7. Error Case: Unknown Job ID
    u_resp = requests.get(f"{BASE_URL}/status/invalid-job-id")
    log(f"Unknown Job Status: {u_resp.status_code} - {u_resp.json()}")

    log("Checks complete.")

if __name__ == "__main__":
    run_checks()
