"""
pipeline.py — Orchestrates the full AI processing pipeline.
Calls ingest → transcribe → score in order and updates job_store at each step.
"""

import traceback
from typing import Any, Dict

import job_store
from ingest import ingest_from_file, ingest_from_url
from transcribe import transcribe
from scoring import score_all_segments
from thumbnail_generator import generate_clip_thumbnails
from emotions import detect_emotions


def run_pipeline(job_id: str, source: Dict[str, Any]) -> None:
    """
    Executes the full pipeline for a given job.
    `source` must be either {"type": "file", "path": str}
    or {"type": "url", "url": str}.

    Updates job_store at every stage so callers can poll /status.
    """
    try:
        # ----------------------------------------------------------------
        # Step 1 — Ingest (download + extract audio → .wav)
        # ----------------------------------------------------------------
        job_store.update_job(job_id, status="transcribing", progress_percent=25)

        if source["type"] == "file":
            ingest_result = ingest_from_file(source["path"], job_id)
        elif source["type"] == "url":
            ingest_result = ingest_from_url(source["url"], job_id)
        else:
            raise ValueError(f"Unknown source type: {source['type']}")

        audio_path = ingest_result["audio_path"]
        video_path = ingest_result.get("video_path")
        
        # Store video path and source info in job for frontend access
        update_data = {"video_path": video_path}
        if source["type"] == "url":
            update_data["source_url"] = source["url"]
        
        print(f"[pipeline] Storing video metadata: {update_data}")
        job_store.update_job(job_id, **update_data)

        # ----------------------------------------------------------------
        # Step 2 — Transcribe (Whisper → structured transcript)
        # ----------------------------------------------------------------
        job_store.update_job(job_id, status="scoring", progress_percent=60)

        transcript_result = transcribe(audio_path, job_id)
        segments = transcript_result.get("segments", [])
        print(f"[pipeline] Transcription complete. Got {len(segments)} segments")
        
        if not segments:
            print(f"[pipeline] Warning: No segments found in transcription")

        # ----------------------------------------------------------------
        # Step 3 — Score (Ollama → ranked viral clips)
        # ----------------------------------------------------------------
        print(f"[pipeline] Starting scoring for {len(segments)} segments...")
        job_store.update_job(job_id, status="scoring", progress_percent=65)
        
        score_result = score_all_segments(segments, job_id)
        print(f"[pipeline] Scoring complete. Got {len(score_result.get('top_clips', []))} top clips")

        # ----------------------------------------------------------------
        # Step 4 — Generate Thumbnails (extract frames for each clip)
        # ----------------------------------------------------------------
        # Get source video path from ingest result
        source_video = ingest_result.get("video_path") or source.get("path")
        
        if source_video and score_result.get("top_clips"):
            try:
                enhanced_clips = generate_clip_thumbnails(
                    source_video,
                    score_result["top_clips"],
                    job_id,
                )
                score_result["top_clips"] = enhanced_clips
                print(f"[pipeline] Generated {len(enhanced_clips)} thumbnails for job {job_id}")
            except Exception as e:
                print(f"[pipeline] Thumbnail generation failed (non-fatal): {e}")

        # ----------------------------------------------------------------
        # Step 5 — Detect Emotions
        # ----------------------------------------------------------------
        if source_video:
            try:
                print(f"[pipeline] Starting emotion detection for {source_video}...")
                job_store.update_job(job_id, status="scoring", progress_percent=85)
                emotions_result = detect_emotions(source_video, job_id)
                score_result["emotions"] = emotions_result
                print(f"[pipeline] Emotion detection complete. Frames detected: {len(emotions_result.get('frame_emotions', []))}")
            except Exception as e:
                print(f"[pipeline] Emotion detection failed (non-fatal): {e}")
                import traceback
                print(traceback.format_exc())

        # ----------------------------------------------------------------
        # Done
        # ----------------------------------------------------------------
        # Include segments in the final result so frontend can access captions
        if segments:
            score_result["segments"] = segments
        
        job_store.update_job(
            job_id,
            status="done",
            progress_percent=100,
            result=score_result,
        )

    except Exception as exc:
        job_store.update_job(
            job_id,
            status="failed",
            error=str(exc),
        )
        # Log full traceback server-side
        print(f"[pipeline] Job {job_id} failed:\n{traceback.format_exc()}")
