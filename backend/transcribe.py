import json
import torch
import whisper
import pathlib
import uuid
import soundfile as sf
import os
from typing import Dict, Any, List

TEMP_DIR = pathlib.Path("temp")

def chunk_audio(audio_path: str, chunk_minutes: int = 5) -> List[str]:
    """
    Chunks a long audio file into overlapping segments to prevent memory issues.
    If the audio is <= 10 minutes, returns the original file path in a list.
    """
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load audio natively using soundfile to avoid ffmpeg CLI dependency
    audio, sr_orig = sf.read(audio_path, dtype='float32')
    
    # Ensure mono
    if len(audio.shape) > 1:
        import numpy as np
        audio = np.mean(audio, axis=1)
        
    # We expect ingest.py to already provide 16kHz audio, but ensure sr is handled correctly
    sr = 16000
    if sr_orig != 16000:
        try:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr_orig, target_sr=16000)
        except ImportError:
            print("Warning: Audio is not 16kHz and librosa is unavailable. Skipping resample.")
            
    duration_minutes = len(audio) / sr / 60
    
    if duration_minutes <= 10:
        return [audio_path]
        
    chunk_size = int(chunk_minutes * 60 * sr)
    overlap = int(0.5 * 60 * sr) # 30s overlap
    
    chunks = []
    start = 0
    idx = 0
    job_id = str(uuid.uuid4())
    
    while start < len(audio):
        end = min(start + chunk_size, len(audio))
        chunk_arr = audio[start:end]
        
        chunk_path = str(TEMP_DIR / f"{job_id}_chunk_{idx}.wav")
        sf.write(chunk_path, chunk_arr, sr)
        chunks.append(chunk_path)
        
        if end == len(audio):
            break
        start += chunk_size - overlap
        idx += 1
        
    return chunks

def merge_chunks(chunk_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Merges transcription results from overlapping chunks.
    Drops duplicate words and segments in overlapping boundaries.
    """
    merged_words = []
    merged_segments = []
    
    for i, res in enumerate(chunk_results):
        offset = i * (4.5 * 60) # chunk length 5min minus 30s overlap = 4.5m advanced
        
        is_last = (i == len(chunk_results) - 1)
        
        for w in res.get("words", []):
            # If not the last chunk, drop words that happen in the overlap (last 30s of a 5m chunk -> 270s)
            if is_last or w["end"] <= 270.0:
                merged_words.append({
                    "word": w["word"],
                    "start": round(w["start"] + offset, 3),
                    "end": round(w["end"] + offset, 3)
                })
        
        for s in res.get("segments", []):
            if is_last or s["end"] <= 270.0:
                merged_segments.append({
                    "text": s["text"],
                    "start": round(s["start"] + offset, 3),
                    "end": round(s["end"] + offset, 3),
                    "speaker": s.get("speaker", "Speaker 1")
                })
                
    merged_transcript = " ".join([seg["text"].strip() for seg in merged_segments])
    
    return {
        "transcript": merged_transcript,
        "words": merged_words,
        "segments": merged_segments
    }

def format_whisper_result(result: Dict[str, Any]) -> Dict[str, Any]:
    """Formats a Whisper result dict to match our expected schema."""
    words = []
    segments = []
    
    for seg in result.get("segments", []):
        text = seg.get("text", "").strip()
        if not text:
            continue
            
        segments.append({
            "text": text,
            "start": round(seg["start"], 3),
            "end": round(seg["end"], 3),
            "speaker": "Speaker 1"
        })
        
        for w in seg.get("words", []):
            words.append({
                "word": w.get("word", "").strip(),
                "start": round(w["start"], 3),
                "end": round(w["end"], 3)
            })
            
    return {
        "transcript": result.get("text", "").strip(),
        "words": words,
        "segments": segments
    }

def transcribe(audio_path: str, job_id: str, model_size: str = "medium") -> Dict[str, Any]:
    """
    Transcribes an audio file using Whisper. Handles long audio via chunking
    and ensures word-level timestamps and speaker attribution exist.
    """
    print(f"[transcribe] Starting transcription for job {job_id}, audio file: {audio_path}")
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if not os.path.exists(audio_path):
        print(f"[transcribe] ERROR: Audio file not found at {audio_path}")
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    
    print(f"[transcribe] Audio file exists, file size: {os.path.getsize(audio_path)} bytes")
    print(f"[transcribe] Loading Whisper model: {model_size} on {device}")
    model = whisper.load_model(model_size, device=device)
    
    chunks = chunk_audio(audio_path, chunk_minutes=5)
    print(f"[transcribe] Audio split into {len(chunks)} chunk(s)")
    
    chunk_results = []
    for i, chunk_path in enumerate(chunks):
        print(f"[transcribe] Processing chunk {i+1}/{len(chunks)}: {chunk_path}")
        # Load audio natively again to bypass whisper's implicit ffmpeg CLI
        audio_chunk, sr_chunk = sf.read(chunk_path, dtype='float32')
        if len(audio_chunk.shape) > 1:
            import numpy as np
            audio_chunk = np.mean(audio_chunk, axis=1)
            
        # Require word timestamps
        res = model.transcribe(audio_chunk, word_timestamps=True)
        formatted_res = format_whisper_result(res)
        chunk_results.append(formatted_res)
        print(f"[transcribe] Chunk {i+1} complete: {len(formatted_res.get('segments', []))} segments, {len(formatted_res.get('words', []))} words")
        
        # Cleanup temp chunks
        if chunk_path != audio_path:
            try:
                os.remove(chunk_path)
            except OSError:
                pass
                
    if len(chunk_results) == 1:
        final_result = chunk_results[0]
    else:
        print(f"[transcribe] Merging {len(chunk_results)} chunk results")
        final_result = merge_chunks(chunk_results)
        
    final_result["job_id"] = job_id
    
    print(f"[transcribe] Final result: {len(final_result.get('segments', []))} segments, {len(final_result.get('words', []))} words from transcript")
    
    save_path = TEMP_DIR / f"transcript_{job_id}.json"
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(final_result, f, ensure_ascii=False, indent=2)
    print(f"[transcribe] Transcript saved to {save_path}")
    print(f"[transcribe] Transcription complete for job {job_id}")
        
    return final_result
