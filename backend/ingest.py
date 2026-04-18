import os
import uuid
import pathlib
import moviepy as mp
import yt_dlp
from typing import Dict, Any

SUPPORTED_EXTENSIONS = {'.mp4', '.mov', '.avi', '.mp3', '.wav'}
TEMP_DIR = pathlib.Path("temp")

def get_duration(file_path: str) -> float:
    """Helper to get duration using moviepy."""
    try:
        with mp.VideoFileClip(file_path) as clip:
            return float(clip.duration)
    except Exception:
        # Try as audio clip if video fails
        try:
            with mp.AudioFileClip(file_path) as clip:
                return float(clip.duration)
        except Exception:
            return 0.0

def extract_audio(source_path: str, job_id: str) -> str:
    """
    Extracts audio from a given video/audio file and saves it
    as a 16kHz mono WAV file (required for Whisper).
    """
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    out_path = str(TEMP_DIR / f"{job_id}.wav")
    
    try:
        # AudioFileClip works for both video and audio files
        # and is more reliable for audio-only streams like those from yt-dlp
        with mp.AudioFileClip(source_path) as clip:
            # Write audio with specific params for Whisper
            # Use a temporary name for audio write to avoid conflicts if needed,
            # but job_id.wav should be unique already.
            clip.write_audiofile(
                out_path, 
                fps=16000, 
                nbytes=2, 
                codec='pcm_s16le', 
                ffmpeg_params=["-ac", "1"], 
                logger=None
            )
        return out_path
    except Exception as e:
        raise RuntimeError(f"Audio extraction error: {str(e)}") from e

def ingest_from_file(file_path: str, job_id: str) -> Dict[str, Any]:
    """
    Ingests video/audio from a local file.
    """
    path = pathlib.Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError("Unsupported file format")

    duration = get_duration(file_path)
    
    audio_path = extract_audio(file_path, job_id)
    
    return {
        "job_id": job_id,
        "audio_path": audio_path,
        "video_path": file_path,  # Store the source video path
        "duration_seconds": duration,
        "source": "file"
    }

def ingest_from_url(url: str, job_id: str) -> Dict[str, Any]:
    """
    Downloads media from a URL using yt-dlp and extracts the audio.
    """
    print(f"[ingest] Starting URL ingestion for: {url}")
    print(f"[ingest] Job ID: {job_id}")
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    
    # Try to download best video format first (for thumbnails), then fallback to audio
    ydl_opts_video = {
        'format': 'best[ext=mp4]/best[ext=webm]/best',
        'outtmpl': str(TEMP_DIR / f"{job_id}_video.%(ext)s"),
        'noplaylist': True,
        'quiet': False,
        'no_warnings': True,
    }
    
    actual_download_path = None
    duration = 0.0
    
    try:
        print(f"[ingest] Attempting video download with yt-dlp...")
        with yt_dlp.YoutubeDL(ydl_opts_video) as ydl:
            info_dict = ydl.extract_info(url, download=True)
            duration = info_dict.get('duration', 0.0)
            ext = info_dict.get('ext', 'mp4')
            actual_download_path = str(TEMP_DIR / f"{job_id}_video.{ext}")
            print(f"[ingest] Video download succeeded: {actual_download_path}, duration: {duration}s, ext: {ext}")
            
            # Verify the file actually exists
            if os.path.exists(actual_download_path):
                file_size = os.path.getsize(actual_download_path)
                print(f"[ingest] Downloaded file verified: {file_size} bytes")
            else:
                print(f"[ingest] WARNING: Downloaded file path doesn't exist yet: {actual_download_path}")
                # Try to find the actual file
                import glob
                search_pattern = str(TEMP_DIR / f"{job_id}_video.*")
                found_files = glob.glob(search_pattern)
                print(f"[ingest] Files matching pattern {search_pattern}: {found_files}")
                if found_files:
                    actual_download_path = found_files[0]
                    print(f"[ingest] Using found file: {actual_download_path}")
            
    except Exception as e:
        print(f"[ingest] Video download failed: {str(e)[:200]}. Will try audio-only fallback.")
        # Fallback: try audio-only download for thumbnail-less processing
        ydl_opts_audio = {
            'format': 'bestaudio/best',
            'outtmpl': str(TEMP_DIR / f"{job_id}_audio.%(ext)s"),
            'noplaylist': True,
            'quiet': False,
            'no_warnings': True,
        }
        try:
            print(f"[ingest] Attempting audio-only download...")
            with yt_dlp.YoutubeDL(ydl_opts_audio) as ydl:
                info_dict = ydl.extract_info(url, download=True)
                duration = info_dict.get('duration', 0.0)
                ext = info_dict.get('ext', 'webm')
                actual_download_path = str(TEMP_DIR / f"{job_id}_audio.{ext}")
                print(f"[ingest] Audio download succeeded: {actual_download_path}, duration: {duration}s")
                
                if os.path.exists(actual_download_path):
                    file_size = os.path.getsize(actual_download_path)
                    print(f"[ingest] Downloaded audio file verified: {file_size} bytes")
                else:
                    print(f"[ingest] WARNING: Audio file path doesn't exist: {actual_download_path}")
                    import glob
                    search_pattern = str(TEMP_DIR / f"{job_id}_audio.*")
                    found_files = glob.glob(search_pattern)
                    print(f"[ingest] Files matching pattern {search_pattern}: {found_files}")
                    if found_files:
                        actual_download_path = found_files[0]
                        print(f"[ingest] Using found file: {actual_download_path}")
        except yt_dlp.utils.DownloadError as e2:
            print(f"[ingest] Audio download also failed: {str(e2)[:200]}")
            raise ValueError(f"Failed to download from URL: {e2}") from e2

    if not actual_download_path:
        print(f"[ingest] ERROR: No download path determined!")
        raise RuntimeError(f"Downloaded file not found")
    
    if not os.path.exists(actual_download_path):
        print(f"[ingest] ERROR: File does not exist at path: {actual_download_path}")
        raise RuntimeError(f"Downloaded file not found: {actual_download_path}")

    print(f"[ingest] Starting audio extraction from: {actual_download_path}")
    audio_path = extract_audio(actual_download_path, job_id)
    print(f"[ingest] Audio extraction complete: {audio_path}")
    
    # Keep the video file for thumbnail generation, don't delete it
    result = {
        "job_id": job_id,
        "audio_path": audio_path,
        "video_path": actual_download_path if os.path.exists(actual_download_path) else None,
        "duration_seconds": float(duration),
        "source": "url"
    }
    print(f"[ingest] Ingest complete: {result}")
    return result
