"""
thumbnail_generator.py — Generate video thumbnails for scored clips
"""

import cv2
import os
import pathlib
from typing import Optional

def extract_frame_as_thumbnail(
    video_path: str,
    output_path: str,
    timestamp_sec: float = 1.0,
    scale_width: int = 600,
) -> Optional[str]:
    """
    Extract a single frame from video at timestamp and save as JPEG.
    
    Returns the output path if successful, None otherwise.
    """
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Failed to open video: {video_path}")
            return None

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_number = int(timestamp_sec * fps)
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        cap.release()

        if not ret or frame is None:
            print(f"Failed to read frame at {timestamp_sec}s")
            return None

        # Resize frame
        h, w = frame.shape[:2]
        scale = scale_width / w
        new_h = int(h * scale)
        frame = cv2.resize(frame, (scale_width, new_h))

        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Save as JPEG
        if cv2.imwrite(output_path, frame):
            return output_path
        return None

    except Exception as e:
        print(f"Error extracting thumbnail: {e}")
        return None


def generate_clip_thumbnails(
    video_path: str,
    top_clips: list,
    job_id: str,
    output_base: str = "output",
    base_url: str = "http://localhost:8000",
) -> list:
    """
    Generate thumbnails for all clips and return enhanced clip data.
    """
    thumbnail_dir = pathlib.Path(output_base) / job_id / "thumbnails"
    thumbnail_dir.mkdir(parents=True, exist_ok=True)

    enhanced_clips = []
    
    for clip in top_clips:
        # Extract thumbnail at 25% through the clip
        clip_duration = clip["end"] - clip["start"]
        thumbnail_timestamp = clip["start"] + (clip_duration * 0.25)
        
        thumb_filename = f"clip_{clip['rank']:02d}.jpg"
        thumb_path = thumbnail_dir / thumb_filename
        
        # Extract the frame
        extracted = extract_frame_as_thumbnail(
            video_path,
            str(thumb_path),
            timestamp_sec=thumbnail_timestamp,
        )
        
        # Add thumbnail URL to clip data (absolute URL so frontend can access it)
        clip_data = clip.copy()
        if extracted:
            # Return absolute URL that frontend can access from port 5173
            clip_data["thumbnail_url"] = f"{base_url}/output/{job_id}/thumbnails/{thumb_filename}"
        
        enhanced_clips.append(clip_data)
    
    return enhanced_clips
