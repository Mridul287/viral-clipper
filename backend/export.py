"""
export.py
---------
Clip export module for the Viral Clipper pipeline.

Trims video clips to specified timestamps, burns in captions from
transcribed words, extracts thumbnails, and generates output metadata.
"""

import os
import json
import subprocess
import logging
import tempfile
from pathlib import Path
from typing import Optional
import cv2
import imageio_ffmpeg

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_PADDING_SEC = 0.5
DEFAULT_CRF = 23
DEFAULT_PRESET = "fast"
DEFAULT_AUDIO_BITRATE = "128k"


# ---------------------------------------------------------------------------
# 1. trim_clip
# ---------------------------------------------------------------------------

def trim_clip(
    source: str,
    start: float,
    end: float,
    output_path: str,
    padding_sec: float = DEFAULT_PADDING_SEC,
) -> bool:
    """
    Trim video clip from source using FFmpeg with re-encoding.

    Parameters
    ----------
    source : str
        Path to source video file.
    start : float
        Clip start time in seconds.
    end : float
        Clip end time in seconds.
    output_path : str
        Path to save trimmed clip (must include filename and .mp4 extension).
    padding_sec : float
        Padding to add before start and after end (default 0.5).

    Returns
    -------
    bool
        True if successful, False otherwise.
    """
    try:
        # Get video duration to clamp padding
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            logger.error(f"Failed to open source video: {source}")
            return False

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        duration_sec = frame_count / fps if fps > 0 else 0
        cap.release()

        if duration_sec <= 0:
            logger.error(f"Invalid video duration: {duration_sec}")
            return False

        # Clamp padding to video bounds
        trim_start = max(0.0, start - padding_sec)
        trim_end = min(duration_sec, end + padding_sec)

        if trim_start >= trim_end:
            logger.error(f"Invalid trim range: {trim_start} >= {trim_end}")
            return False

        # Ensure output directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

        # FFmpeg command for trimming and re-encoding
        cmd = [
            ffmpeg_exe,
            "-i", source,
            "-ss", str(trim_start),
            "-to", str(trim_end),
            "-c:v", "libx264",
            "-crf", str(DEFAULT_CRF),
            "-preset", DEFAULT_PRESET,
            "-c:a", "aac",
            "-b:a", DEFAULT_AUDIO_BITRATE,
            "-y",  # Overwrite output
            output_path,
        ]

        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )

        if result.returncode != 0:
            logger.error(
                f"FFmpeg trim failed: {result.stderr.decode()}"
            )
            return False

        logger.info(f"Trimmed clip saved: {output_path}")
        return True

    except Exception as e:
        logger.error(f"Error trimming clip: {e}")
        return False


# ---------------------------------------------------------------------------
# 2. generate_srt
# ---------------------------------------------------------------------------

def generate_srt(
    words: list[dict],
    clip_start: float,
    clip_end: float,
) -> str:
    """
    Generate SRT subtitle content from word timestamps.

    Timestamps are adjusted to be relative to clip_start (0:00:00 at clip start).

    Parameters
    ----------
    words : list[dict]
        List of word dicts with keys: word, start, end (all in seconds).
    clip_start : float
        Clip start time in seconds (used to offset word timestamps).
    clip_end : float
        Clip end time in seconds (filter words outside this range).

    Returns
    -------
    str
        SRT file content (multiple subtitle blocks).
    """
    def seconds_to_srt_time(sec: float) -> str:
        """Convert seconds to SRT time format: HH:MM:SS,mmm"""
        hours = int(sec // 3600)
        remainder = sec % 3600
        minutes = int(remainder // 60)
        seconds = remainder % 60

        return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}".replace(".", ",")

    # Filter words to only those within clip range
    filtered_words = [
        w for w in words
        if clip_start <= w.get("start", 0) <= clip_end
    ]

    if not filtered_words:
        return ""

    srt_blocks = []
    for i, word_dict in enumerate(filtered_words, start=1):
        word = word_dict.get("word", "")
        start = word_dict.get("start", 0)
        end = word_dict.get("end", 0)

        # Adjust to be relative to clip start
        rel_start = start - clip_start
        rel_end = end - clip_start

        # Clamp to non-negative
        rel_start = max(0.0, rel_start)
        rel_end = max(0.0, rel_end)

        srt_time_start = seconds_to_srt_time(rel_start)
        srt_time_end = seconds_to_srt_time(rel_end)

        block = f"{i}\n{srt_time_start} --> {srt_time_end}\n{word}\n"
        srt_blocks.append(block)

    return "\n".join(srt_blocks)


# ---------------------------------------------------------------------------
# 3. burn_captions
# ---------------------------------------------------------------------------

def burn_captions(
    clip_path: str,
    srt_content: str,
    output_path: str,
) -> bool:
    """
    Burn SRT captions into video file using FFmpeg.

    Parameters
    ----------
    clip_path : str
        Path to video clip with audio and content.
    srt_content : str
        SRT subtitle file content as string.
    output_path : str
        Path to save video with burned captions.

    Returns
    -------
    bool
        True if successful, False otherwise.
    """
    try:
        if not srt_content or not srt_content.strip():
            # No subtitles to burn, just copy the file
            logger.debug(f"No captions to burn, copying file from {clip_path}")
            import shutil
            shutil.copy2(clip_path, output_path)
            return True

        # Write SRT to temporary file
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".srt",
            delete=False,
            encoding="utf-8",
        ) as f:
            f.write(srt_content)
            srt_path = f.name

        try:
            # Escape SRT path for FFmpeg (Windows)
            srt_path_escaped = srt_path.replace("\\", "/").replace(":", "\\:")

            ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

            # FFmpeg command to burn subtitles
            cmd = [
                ffmpeg_exe,
                "-i", clip_path,
                "-vf", (
                    f"subtitles={srt_path_escaped}:"
                    "force_style='FontSize=24,"
                    "PrimaryColour=&HFFFFFF,"
                    "OutlineColour=&H000000,"
                    "Outline=2'"
                ),
                "-c:a", "copy",
                "-y",
                output_path,
            ]

            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

            if result.returncode != 0:
                logger.warning(
                    f"FFmpeg caption burn may have issues, "
                    f"but continuing: {result.stderr.decode()[:200]}"
                )
                # Still consider it a success if output file was created
                if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                    logger.info(f"Captions burned: {output_path}")
                    return True
                return False

            logger.info(f"Captions burned: {output_path}")
            return True

        finally:
            # Clean up temp SRT file
            if os.path.exists(srt_path):
                os.remove(srt_path)

    except Exception as e:
        logger.error(f"Error burning captions: {e}")
        return False


# ---------------------------------------------------------------------------
# 4. extract_clip_thumbnail
# ---------------------------------------------------------------------------

def extract_clip_thumbnail(
    clip_path: str,
    save_path: str,
    timestamp_sec: float = 1.0,
) -> str:
    """
    Extract a frame from the clip and save as JPEG thumbnail.

    Parameters
    ----------
    clip_path : str
        Path to video clip.
    save_path : str
        Path to save thumbnail JPEG.
    timestamp_sec : float
        Timestamp in seconds to extract frame from (default 1.0).

    Returns
    -------
    str
        The save_path if successful, empty string on error.
    """
    try:
        cap = cv2.VideoCapture(clip_path)
        if not cap.isOpened():
            logger.error(f"Failed to open clip: {clip_path}")
            return ""

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30.0

        # Get frame at timestamp
        frame_number = int(timestamp_sec * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

        ret, frame = cap.read()
        cap.release()

        if not ret or frame is None:
            logger.error(f"Failed to read frame at {timestamp_sec}s from {clip_path}")
            return ""

        # Ensure directory exists
        save_dir = os.path.dirname(save_path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)

        # Save as JPEG
        success = cv2.imwrite(save_path, frame)
        if success:
            logger.debug(f"Extracted thumbnail: {save_path}")
            return save_path
        else:
            logger.error(f"Failed to save thumbnail: {save_path}")
            return ""

    except Exception as e:
        logger.error(f"Error extracting thumbnail: {e}")
        return ""


# ---------------------------------------------------------------------------
# 5. export_all_clips (main orchestrator)
# ---------------------------------------------------------------------------

def export_all_clips(
    source_video: str,
    top_clips: list[dict],
    words: list[dict] = None,
    job_id: str = "default",
) -> dict:
    """
    Export all clips from source video with captions burned in.

    Parameters
    ----------
    source_video : str
        Path to source video file.
    top_clips : list[dict]
        List of clip dicts, each with: start, end, suggested_title, clip_type, final_score.
    words : list[dict], optional
        List of transcribed words with: word, start, end (all in seconds).
        If None, no captions will be burned.
    job_id : str
        Job identifier for file naming.

    Returns
    -------
    dict
        Result dictionary with keys:
        - job_id (str)
        - clips (list[dict]): exported clip metadata
        - total_clips (int): number of clips exported
        - output_dir (str): directory containing output files
    """
    if words is None:
        words = []

    # Create output directories
    export_dir = os.path.join("output", job_id)
    thumbnail_dir = os.path.join(export_dir, "thumbnails")
    os.makedirs(export_dir, exist_ok=True)
    os.makedirs(thumbnail_dir, exist_ok=True)

    exported_clips = []

    for rank, clip_spec in enumerate(top_clips, start=1):
        clip_start = clip_spec.get("start", 0.0)
        clip_end = clip_spec.get("end", 0.0)
        suggested_title = clip_spec.get("suggested_title", f"clip_{rank}")

        # Step 1: Trim clip
        trimmed_path = os.path.join(
            export_dir,
            f"clip_{job_id}_{rank}.mp4",
        )

        success = trim_clip(source_video, clip_start, clip_end, trimmed_path)
        if not success:
            logger.error(f"Failed to trim clip {rank}, skipping")
            continue

        # Step 2: Generate SRT captions
        srt_content = generate_srt(words, clip_start, clip_end)
        caption_word_count = len(
            [w for w in words if clip_start <= w.get("start", 0) <= clip_end]
        )

        # Step 3: Burn captions
        captioned_path = trimmed_path  # Same as trimmed (or separate if needed)
        if srt_content:
            temp_captioned = trimmed_path.replace(".mp4", "_captioned.mp4")
            success = burn_captions(trimmed_path, srt_content, temp_captioned)
            if success:
                # Replace original with captioned version
                import shutil
                shutil.move(temp_captioned, captioned_path)
            else:
                logger.warning(f"Failed to burn captions, using untrimmed clip")

        # Step 4: Get clip duration and file size
        cap = cv2.VideoCapture(captioned_path)
        if cap.isOpened():
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            duration = frame_count / fps if fps > 0 else 0
            cap.release()
        else:
            duration = 0

        file_size_mb = os.path.getsize(captioned_path) / (1024 * 1024)

        # Step 5: Extract thumbnail
        thumbnail_path = os.path.join(
            thumbnail_dir,
            f"clip_{job_id}_{rank}.jpg",
        )
        extract_clip_thumbnail(captioned_path, thumbnail_path)

        # Step 6: Build output metadata
        clip_output = {
            "rank": rank,
            "clip_path": captioned_path,
            "thumbnail_path": thumbnail_path,
            "duration": round(duration, 2),
            "file_size_mb": round(file_size_mb, 2),
            "caption_word_count": caption_word_count,
            "start": round(clip_start, 3),
            "end": round(clip_end, 3),
            "suggested_title": suggested_title,
        }

        exported_clips.append(clip_output)
        logger.info(f"Exported clip {rank}: {captioned_path}")

    # Step 7: Prepare result dictionary
    result = {
        "job_id": job_id,
        "clips": exported_clips,
        "total_clips": len(exported_clips),
        "output_dir": os.path.abspath(export_dir),
    }

    # Step 8: Save JSON metadata
    json_path = os.path.join("temp", f"export_{job_id}.json")
    os.makedirs(os.path.dirname(json_path), exist_ok=True)

    try:
        with open(json_path, "w") as f:
            json.dump(result, f, indent=2)
        logger.info(f"Saved export metadata: {json_path}")
    except Exception as e:
        logger.error(f"Failed to save export metadata: {e}")

    return result
