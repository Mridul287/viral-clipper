"""
main.py — FastAPI REST API for the AI Video Clip Tool.
Exposes 4 endpoints: /upload, /status/{job_id}, /results/{job_id}, /health
"""

import uuid
import pathlib
import shutil
from typing import Optional

import requests as http_requests
from fastapi import FastAPI, BackgroundTasks, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import job_store
from pipeline import run_pipeline

app = FastAPI(title="Viral Clipper API", version="1.0.0")

# Allow requests from the Vite dev server and any local origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static output files (thumbnails, clips, etc)
OUTPUT_DIR = pathlib.Path("output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
try:
    app.mount("/output", StaticFiles(directory=str(OUTPUT_DIR)), name="output")
except Exception as e:
    print(f"Warning: Could not mount output directory: {e}")

# Serve uploaded video files from temp directory
TEMP_DIR = pathlib.Path("temp")
TEMP_DIR.mkdir(parents=True, exist_ok=True)
try:
    app.mount("/temp", StaticFiles(directory=str(TEMP_DIR)), name="temp")
except Exception as e:
    print(f"Warning: Could not mount temp directory: {e}")

MAX_FILE_SIZE_BYTES = 500 * 1024 * 1024  # 500 MB
SUPPORTED_EXTENSIONS = {".mp4", ".mov", ".avi", ".mp3", ".wav", ".webm"}


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class UrlUploadRequest(BaseModel):
    url: str


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _error(msg: str, code: int) -> JSONResponse:
    return JSONResponse(status_code=code, content={"error": msg, "code": code})


# ---------------------------------------------------------------------------
# POST /upload
# ---------------------------------------------------------------------------

@app.post("/upload")
async def upload(
    background_tasks: BackgroundTasks,
    file: Optional[UploadFile] = File(default=None),
    url: Optional[str] = Form(default=None),
):
    """
    Accepts either a multipart video file or a JSON/form URL.
    Immediately returns a job_id and queues the pipeline as a background task.
    """
    job_id = str(uuid.uuid4())

    # ---- URL upload ----
    if url and not file:
        job_store.create_job(job_id)
        background_tasks.add_task(run_pipeline, job_id, {"type": "url", "url": url})
        return {
            "job_id": job_id,
            "status": "queued",
            "message": f"URL job queued. Poll /status/{job_id} for updates.",
        }

    # ---- File upload ----
    if file:
        suffix = pathlib.Path(file.filename).suffix.lower()

        if suffix not in SUPPORTED_EXTENSIONS:
            return _error(
                f"Unsupported file format '{suffix}'. Allowed: {sorted(SUPPORTED_EXTENSIONS)}",
                400,
            )

        # Stream to disk to check size without fully loading into memory
        dest = TEMP_DIR / f"{job_id}_upload{suffix}"
        size = 0
        try:
            with open(dest, "wb") as f_out:
                while chunk := await file.read(1024 * 1024):  # 1 MB chunks
                    size += len(chunk)
                    if size > MAX_FILE_SIZE_BYTES:
                        raise ValueError("File too large")
                    f_out.write(chunk)
        except ValueError:
            dest.unlink(missing_ok=True)
            return _error("File exceeds 500 MB limit.", 413)
        except Exception as e:
            dest.unlink(missing_ok=True)
            return _error(f"Upload failed: {str(e)}", 500)

        job_store.create_job(job_id)
        background_tasks.add_task(
            run_pipeline, job_id, {"type": "file", "path": str(dest)}
        )
        return {
            "job_id": job_id,
            "status": "queued",
            "message": f"File job queued. Poll /status/{job_id} for updates.",
        }

    # ---- Neither provided ----
    return _error("Provide either a 'file' (multipart) or a 'url' (form field).", 422)


# ---------------------------------------------------------------------------
# POST /reframe
# ---------------------------------------------------------------------------

@app.post("/reframe")
async def reframe(
    background_tasks: BackgroundTasks,
    file: Optional[UploadFile] = File(default=None),
    url: Optional[str] = Form(default=None)
):
    """
    Accepts a video file or URL and reframes it to vertical using FaceTracker.
    Returns a job_id and streams the result back.
    """
    import reframe as reframer
    import yt_dlp
    
    job_id = str(uuid.uuid4())
    
    # ---- URL logic ----
    if url and not file:
        job_store.create_job(job_id)
        job_store.update_job(job_id, status="reframing", progress_percent=10)
        
        def do_reframe_url():
            try:
                ydl_opts = {
                    'format': 'best',
                    'outtmpl': str(TEMP_DIR / f"{job_id}_raw.%(ext)s"),
                    'noplaylist': True,
                    'quiet': True,
                    'no_warnings': True,
                }
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(url, download=True)
                    ext = info.get('ext', 'mp4')
                    dest = str(TEMP_DIR / f"{job_id}_raw.{ext}")
                
                out_path = str(TEMP_DIR / f"{job_id}_vertical.{ext}")
                res = reframer.reframe_clip(dest, job_id, out_path)
                
                out_rel = pathlib.Path(out_path).name
                job_store.update_job(
                    job_id,
                    status="done",
                    progress_percent=100,
                    result={"url": f"/api/temp/{out_rel}", "meta": res}
                )
            except Exception as e:
                job_store.update_job(job_id, status="failed", error=str(e))
                
        background_tasks.add_task(do_reframe_url)
        return {
            "job_id": job_id,
            "status": "queued",
            "message": f"Reframe URL job queued. Poll /status/{job_id} for updates.",
        }

    # ---- File Logic ----
    if file:
        suffix = pathlib.Path(file.filename).suffix.lower()

        if suffix not in SUPPORTED_EXTENSIONS:
            return _error(
                f"Unsupported file format '{suffix}'. Allowed: {sorted(SUPPORTED_EXTENSIONS)}",
                400,
            )

        # Stream to disk 
        dest = TEMP_DIR / f"{job_id}_raw{suffix}"
        try:
            with open(dest, "wb") as f_out:
                while chunk := await file.read(1024 * 1024):
                    f_out.write(chunk)
        except Exception as e:
            dest.unlink(missing_ok=True)
            return _error(f"Upload failed: {str(e)}", 500)

        job_store.create_job(job_id)
        job_store.update_job(job_id, status="reframing", progress_percent=15)
        
        def do_reframe_file():
            try:
                out_path = str(TEMP_DIR / f"{job_id}_vertical{suffix}")
                res = reframer.reframe_clip(str(dest), job_id, out_path)
                
                # Form final URL for download
                out_rel = pathlib.Path(out_path).name
                job_store.update_job(
                    job_id,
                    status="done",
                    progress_percent=100,
                    result={"url": f"/api/temp/{out_rel}", "meta": res}
                )
            except Exception as e:
                job_store.update_job(job_id, status="failed", error=str(e))
                
        background_tasks.add_task(do_reframe_file)
        
        return {
            "job_id": job_id,
            "status": "queued",
            "message": f"Reframe job queued. Poll /status/{job_id} for updates.",
        }
        
    return _error("Provide either a 'file' (multipart) or a 'url' (form field).", 422)


# ---------------------------------------------------------------------------
# GET /status/{job_id}
# ---------------------------------------------------------------------------

@app.get("/status/{job_id}")
def get_status(job_id: str):
    """Returns current job status and progress percentage."""
    job = job_store.get_job(job_id)
    if job is None:
        return _error(f"Job '{job_id}' not found.", 404)

    response = {
        "job_id": job_id,
        "status": job["status"],
        "progress_percent": job["progress_percent"],
        "error": job["error"],
    }
    
    # Include video_path if available, converted to API URL
    if job.get("video_path"):
        video_path = str(job["video_path"])
        # Normalize path separators
        video_path = video_path.replace("\\", "/")
        # Remove leading ./ if present
        if video_path.startswith("./"):
            video_path = video_path[2:]
        # Ensure it's rooted at /temp or /api/temp
        if not video_path.startswith("/"):
            if video_path.startswith("temp/"):
                video_path = "/" + video_path
            elif "temp/" in video_path:
                # Extract from full path
                idx = video_path.index("temp/")
                video_path = "/" + video_path[idx:]
        response["video_path"] = video_path
    
    # Include original source URL if available (for URL uploads)
    if job.get("source_url"):
        response["source_url"] = job["source_url"]
    
    return response


# ---------------------------------------------------------------------------
# GET /results/{job_id}
# ---------------------------------------------------------------------------

@app.get("/results/{job_id}")
def get_results(job_id: str):
    """Returns the final scored clips. 202 if not yet done."""
    job = job_store.get_job(job_id)
    if job is None:
        return _error(f"Job '{job_id}' not found.", 404)

    if job["status"] != "done":
        return JSONResponse(
            status_code=202,
            content={"message": "Job not complete yet", "status": job["status"]},
        )

    return job["result"]


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------

@app.get("/health")
def health_check():
    """Returns API health + connectivity checks for Ollama and Whisper."""

    # Check Ollama reachability
    ollama_connected = False
    try:
        r = http_requests.get("http://localhost:11434", timeout=2)
        ollama_connected = r.status_code == 200
    except Exception:
        pass

    # Check Whisper importability (model not loaded into memory here)
    whisper_loaded = False
    try:
        import whisper  # noqa: F401
        whisper_loaded = True
    except ImportError:
        pass

    return {
        "status": "ok",
        "ollama_connected": ollama_connected,
        "whisper_loaded": whisper_loaded,
    }



