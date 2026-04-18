import os
import json
import pathlib
import subprocess
import yt_dlp
import imageio_ffmpeg
from typing import Dict, Any

# Use the bundled ffmpeg from imageio_ffmpeg — works even without system ffmpeg.
_FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()

SUPPORTED_EXTENSIONS = {'.mp4', '.mov', '.avi', '.mp3', '.wav'}
TEMP_DIR = pathlib.Path("temp")

def get_duration(file_path: str) -> float:
    """
    Probe video/audio duration using ffmpeg directly.
    Avoids MoviePy's broken AudioFileClip (KeyError: 'audio_bitrate' in v2.1.2).
    """
    try:
        cmd = [
            _FFMPEG.replace("ffmpeg", "ffprobe") if "ffprobe" in _FFMPEG else _FFMPEG,
            "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            str(file_path),
        ]
        # imageio_ffmpeg bundles ffmpeg only; use ffmpeg to probe instead
        probe_cmd = [
            _FFMPEG, "-i", str(file_path),
            "-f", "null", "-",
        ]
        # Simpler: parse from ffmpeg stderr '-i' output
        result = subprocess.run(
            [_FFMPEG, "-i", str(file_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        stderr = result.stderr.decode("utf-8", errors="replace")
        import re
        m = re.search(r"Duration:\s*(\d+):(\d+):([\d.]+)", stderr)
        if m:
            h, mn, s = int(m.group(1)), int(m.group(2)), float(m.group(3))
            return h * 3600 + mn * 60 + s
    except Exception:
        pass
    return 0.0


def extract_audio(source_path: str, job_id: str) -> str:
    """
    Extracts audio from a given video/audio file and saves it
    as a 16kHz mono WAV file (required for Whisper).

    Uses ffmpeg subprocess directly to avoid MoviePy 2.1.2 bug
    (KeyError: 'audio_bitrate' when audio bitrate is missing from metadata).
    If the source has no audio track, generates a short silent WAV so that
    Whisper still receives a valid input without crashing the pipeline.
    """
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    out_path = str(TEMP_DIR / f"{job_id}.wav")

    cmd = [
        _FFMPEG,
        "-y",               # overwrite if exists
        "-i", source_path,
        "-vn",              # no video
        "-acodec", "pcm_s16le",
        "-ar", "16000",     # 16 kHz sample rate for Whisper
        "-ac", "1",         # mono
        out_path,
    ]

    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=300,
        )
        if result.returncode != 0:
            err = result.stderr.decode("utf-8", errors="replace")
            # Check if failure is due to no audio stream in the source file
            if "does not contain any stream" in err or "Invalid argument" in err:
                print(f"[ingest] Source has no audio track — generating 1 s silent WAV for {job_id}")
                silent_cmd = [
                    _FFMPEG, "-y",
                    "-f", "lavfi",
                    "-i", "anullsrc=channel_layout=mono:sample_rate=16000",
                    "-t", "1",
                    "-acodec", "pcm_s16le",
                    out_path,
                ]
                subprocess.run(
                    silent_cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=30,
                    check=True,
                )
            else:
                raise RuntimeError(f"ffmpeg failed (code {result.returncode}): {err[-800:]}")
        if not os.path.exists(out_path) or os.path.getsize(out_path) == 0:
            raise RuntimeError(f"ffmpeg produced no output file at {out_path}")
        return out_path
    except subprocess.TimeoutExpired:
        raise RuntimeError("Audio extraction timed out after 5 minutes")
    except RuntimeError:
        raise
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
