import cv2
import numpy as np
import time
import logging
import pathlib
import shutil
import subprocess
import imageio_ffmpeg
from collections import deque
from typing import Tuple, List, Dict, Optional, Any

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")


class FaceTracker:
    """
    Tracks face position and provides smoothed center points to avoid jitter.
    """
    def __init__(self, smooth_window: int = 15):
        self.smooth_window = smooth_window
        self.history = deque(maxlen=smooth_window)
        # Load Haar Cascade
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        if self.face_cascade.empty():
            logger.warning(f"Failed to load Haar cascade from {cascade_path}")

    def detect_face(self, frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """
        Detects the largest face in the frame using OpenCV Haar Cascade.
        Resizes frame to 480p height for faster detection.
        Returns: (x, y, w, h) in original frame coordinates, or None.
        """
        original_h, original_w = frame.shape[:2]
        
        # Resize for performance: Scale down to 480p
        target_h = 480
        if original_h > target_h:
            scale = target_h / float(original_h)
            target_w = int(original_w * scale)
            small_frame = cv2.resize(frame, (target_w, target_h))
            scale_back = 1.0 / scale
        else:
            small_frame = frame
            scale_back = 1.0

        gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        if len(faces) == 0:
            return None

        # Pick largest face
        largest_face = max(faces, key=lambda f: f[2] * f[3])
        x, y, w, h = largest_face

        # Scale coordinates back
        return (
            int(x * scale_back),
            int(y * scale_back),
            int(w * scale_back),
            int(h * scale_back)
        )

    def get_smooth_center_x(self, raw_center_x: int) -> int:
        """
        Adds raw_center_x to the deque history and returns the moving average.
        """
        self.history.append(raw_center_x)
        return int(sum(self.history) / len(self.history))


def calculate_crop_box(
    center_x: int,
    source_width: int,
    source_height: int,
    target_width: int = 1080,
    target_height: int = 1920
) -> Tuple[int, int, int, int]:
    """
    Calculates crop boundaries given a target aspect ratio and desired center x.
    Returns: (left, top, right, bottom)
    """
    # Target aspect ratio
    target_aspect = target_width / target_height
    
    # Calculate crop dimensions
    crop_height = source_height
    crop_width = int(round(source_height * target_aspect))
    
    # Clamp width if needed
    if crop_width > source_width:
        crop_width = source_width
        crop_height = int(round(source_width / target_aspect))

    # Calculate left boundary, clamped to bounds
    left = center_x - (crop_width // 2)
    left = max(0, min(left, source_width - crop_width))
    right = left + crop_width
    
    # Height boundaries
    top = max(0, (source_height - crop_height) // 2)
    bottom = top + crop_height

    return (left, top, right, bottom)


def reframe_clip(
    clip_path: str,
    job_id: str,
    output_path: str,
    options: dict = None
) -> dict:
    """
    Reframes a given horizontal video clip to vertical 9:16 keeping face centered.
    """
    t0 = time.time()
    
    if options is None:
        options = {}
        
    target_width = options.get("target_width", 1080)
    target_height = options.get("target_height", 1920)
    smooth_window = options.get("smooth_window", 15)
    fallback_center = options.get("fallback_center", True)
    
    if not pathlib.Path(clip_path).exists():
        raise FileNotFoundError(f"Source video not found: {clip_path}")

    pathlib.Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(clip_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video file: {clip_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0: fps = 30.0
    source_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    source_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames_estimate = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if source_height >= source_width:
        logger.warning(f"Video {clip_path} is already vertical. Skipping reframe.")
        cap.release()
        shutil.copy2(clip_path, output_path)
        return {
            "job_id": job_id,
            "input_path": clip_path,
            "output_path": output_path,
            "source_resolution": [source_width, source_height],
            "target_resolution": [source_width, source_height],
            "total_frames": total_frames_estimate,
            "frames_with_face": 0,
            "frames_without_face": total_frames_estimate,
            "face_detection_rate": 0.0,
            "processing_time_sec": round(time.time() - t0, 3)
        }

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    temp_cv2_out = str(pathlib.Path(output_path).with_suffix(".temp.mp4"))
    out = cv2.VideoWriter(temp_cv2_out, fourcc, fps, (target_width, target_height))
    
    if not out.isOpened():
        cap.release()
        raise RuntimeError(f"Failed to open VideoWriter for {temp_cv2_out}")

    tracker = FaceTracker(smooth_window=smooth_window)
    
    frames_with_face = 0
    frames_without_face = 0
    actual_frames = 0
    last_known_center_x = source_width // 2

    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        actual_frames += 1
        
        face_rect = tracker.detect_face(frame)
        if face_rect is not None:
            x, y, w, h = face_rect
            raw_center_x = x + (w // 2)
            last_known_center_x = raw_center_x
            frames_with_face += 1
        else:
            raw_center_x = last_known_center_x if fallback_center else (source_width // 2)
            frames_without_face += 1
            
        smooth_center_x = tracker.get_smooth_center_x(raw_center_x)
        
        left, top, right, bottom = calculate_crop_box(
            smooth_center_x, 
            source_width, 
            source_height, 
            target_width, 
            target_height
        )
        
        cropped_frame = frame[top:bottom, left:right]
        resized_frame = cv2.resize(cropped_frame, (target_width, target_height))
        
        out.write(resized_frame)

    cap.release()
    out.release()
    
    # Remux temp video with original audio and convert to web-playable H264
    _FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
    mux_cmd = [
        _FFMPEG, "-y",
        "-v", "warning",
        "-i", temp_cv2_out,
        "-i", clip_path,
        "-c:v", "libx264",
        "-preset", "fast",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-map", "0:v:0",
        "-map", "1:a:0?",
        "-shortest",
        output_path
    ]
    
    try:
        subprocess.run(mux_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg muxing failed: {e.stderr.decode('utf-8', errors='replace')}")
        # If ffmpeg fails, fallback to passing the temp openCV output as primary
        shutil.copy2(temp_cv2_out, output_path)
    finally:
        # Clean up temp file
        pathlib.Path(temp_cv2_out).unlink(missing_ok=True)
    
    total_processed = frames_with_face + frames_without_face
    detection_rate = frames_with_face / total_processed if total_processed > 0 else 0.0

    return {
        "job_id": job_id,
        "input_path": clip_path,
        "output_path": output_path,
        "source_resolution": [source_width, source_height],
        "target_resolution": [target_width, target_height],
        "total_frames": actual_frames,
        "frames_with_face": frames_with_face,
        "frames_without_face": frames_without_face,
        "face_detection_rate": round(detection_rate, 4),
        "processing_time_sec": round(time.time() - t0, 3)
    }

def reframe_all_clips(
    clips: List[Dict[str, Any]],
    job_id: str,
    output_dir: str,
    options: dict = None
) -> List[Dict[str, Any]]:
    """
    Batch reframe all clips.
    """
    results = []
    logger.info(f"[{job_id}] Starting batch reframe for {len(clips)} clips")
    
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    for idx, clip in enumerate(clips):
        clip_path = clip.get("output_path", clip.get("path"))
        if not clip_path:
            logger.error(f"Clip idx {idx} missing path")
            continue
            
        p = pathlib.Path(clip_path)
        out_path = str(pathlib.Path(output_dir) / f"{p.stem}_vertical{p.suffix}")
        
        logger.info(f"Reframing {clip_path} -> {out_path}")
        try:
            res = reframe_clip(clip_path, job_id, out_path, options)
            res["original_clip_meta"] = clip
            results.append(res)
        except Exception as e:
            logger.error(f"Failed to reframe {clip_path}: {e}")
            
    return results
