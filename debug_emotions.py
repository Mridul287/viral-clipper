import sys
from emotions import detect_emotions
from pathlib import Path

# Get the latest video file in temp directory
videos = list(Path("backend/temp").glob("*_video.*"))
if videos:
    video_path = str(videos[0])
    print(f"Testing with: {video_path}")
    try:
        detect_emotions(video_path, "debug_emotions")
        print("Success")
    except Exception as e:
        print(f"FAILED: {e}")
else:
    print("No video found for testing.")
