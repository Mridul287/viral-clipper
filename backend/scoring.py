"""
scoring.py — Viral Segment Scoring Engine
Uses a locally running Ollama LLM (llama3) to score transcript segments
for viral potential across multiple dimensions.
"""

import json
import re
import pathlib
import requests
from typing import Dict, Any, List

TEMP_DIR = pathlib.Path("temp")
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3"

# Weighted formula coefficients
WEIGHTS = {
    "virality": 0.4,
    "quotable": 0.3,
    "emotional": 0.2,
    "surprising": 0.1,
}

DEFAULT_SCORES = {
    "funny": 5,
    "surprising": 5,
    "quotable": 5,
    "emotional": 5,
    "virality": 5,
    "clip_type": "key_fact",
    "suggested_title": "Untitled Clip"
}

SCORE_KEYS = {"funny", "surprising", "quotable", "emotional", "virality"}
REQUIRED_KEYS = SCORE_KEYS | {"clip_type", "suggested_title"}
VALID_CLIP_TYPES = {"funny", "motivational", "intense", "key_fact", "hook"}


def _build_prompt(segment_text: str) -> str:
    """Constructs the exact prompt to send to Ollama for a given segment."""
    return f"""You are a viral content editor. Analyze this transcript segment and respond ONLY
in valid JSON with no extra text, no markdown, no explanation.

Segment: "{segment_text}"

Return this exact JSON structure:
{{
  "funny": <1-10>,
  "surprising": <1-10>,
  "quotable": <1-10>,
  "emotional": <1-10>,
  "virality": <1-10>,
  "clip_type": "<funny|motivational|intense|key_fact|hook>",
  "suggested_title": "<max 8 words>"
}}"""


def parse_llm_response(raw_text: str) -> dict:
    """
    Safely parses JSON from LLM response text.
    Handles incomplete JSON, markdown code fences, and malformed output.
    Falls back to DEFAULT_SCORES on any parse failure.
    """
    if not raw_text or not raw_text.strip():
        print(f"[scoring] parse_llm_response: Empty response, using defaults")
        return dict(DEFAULT_SCORES)

    # Attempt 1: direct parse
    try:
        data = json.loads(raw_text.strip())
        if isinstance(data, dict):
            print(f"[scoring] Successfully parsed JSON directly")
            return _validate_and_fill_scores(data)
    except json.JSONDecodeError as e:
        print(f"[scoring] Direct JSON parse failed: {str(e)[:100]}")

    # Attempt 2: Try to fix incomplete JSON by adding missing closing braces BEFORE regex
    open_braces = raw_text.count('{')
    close_braces = raw_text.count('}')
    if open_braces > close_braces:
        print(f"[scoring] Incomplete JSON detected: open_braces={open_braces}, close_braces={close_braces}")
        fixed_text = raw_text + '}' * (open_braces - close_braces)
        print(f"[scoring] Added {open_braces - close_braces} closing brace(s), retrying parse")
        try:
            data = json.loads(fixed_text)
            if isinstance(data, dict):
                print(f"[scoring] Successfully parsed fixed incomplete JSON")
                return _validate_and_fill_scores(data)
        except json.JSONDecodeError as e1:
            print(f"[scoring] Fixed JSON parse failed: {str(e1)[:100]}")

    # Attempt 3: extract JSON block from markdown fences or surrounding text WITH closing brace
    match = re.search(r'\{[\s\S]*\}', raw_text)
    if match:
        json_str = match.group()
        print(f"[scoring] Extracted JSON block (length: {len(json_str)})")
        
        try:
            data = json.loads(json_str)
            if isinstance(data, dict):
                print(f"[scoring] Successfully parsed extracted JSON")
                return _validate_and_fill_scores(data)
        except json.JSONDecodeError as e2:
            print(f"[scoring] Extracted JSON parse failed: {str(e2)[:100]}")

    # Final fallback
    print(f"[scoring] All parsing attempts failed, using DEFAULT_SCORES")
    return dict(DEFAULT_SCORES)


def _validate_and_fill_scores(data: dict) -> dict:
    """
    Validates parsed JSON and fills in any missing required keys with defaults.
    This allows partial responses from the LLM to still work.
    """
    result = dict(DEFAULT_SCORES)
    
    # Update with any keys that were successfully parsed
    for key in REQUIRED_KEYS | {"funny", "surprising", "quotable", "emotional", "virality"}:
        if key in data:
            value = data[key]
            # Validate score ranges for numeric fields
            if key in SCORE_KEYS:
                try:
                    value = max(1, min(10, int(value)))  # Clamp to [1, 10]
                except (TypeError, ValueError):
                    value = DEFAULT_SCORES[key]
            result[key] = value
    
    print(f"[scoring] Final scores after validation: {result}")
    return result


def score_segment(segment: dict) -> dict:
    """
    Calls the Ollama API to score a single transcript segment.
    Returns the parsed score dict merged with the original segment info.
    If the segment text is empty, returns an empty dict to signal skipping.
    """
    text = segment.get("text", "").strip()

    # Skip empty segments
    if not text:
        return {}

    prompt = _build_prompt(text)

    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,  # Get full response at once
            },
            timeout=90,  # Allow up to 90 seconds for LLM response (increased from 60)
        )
        response.raise_for_status()
        raw = response.json().get("response", "")
        print(f"[scoring] Ollama raw response (length: {len(raw)}, first 500 chars): {raw[:500]}")
    except requests.Timeout:
        print(f"[scoring] Ollama request TIMEOUT after 90s. Using default scores.")
        raw = ""
    except requests.RequestException as e:
        # Network / Ollama not running — fall back to defaults
        print(f"[scoring] Ollama request failed: {e}. Using default scores.")
        raw = ""

    scores = parse_llm_response(raw)

    return {
        "start": segment.get("start", 0.0),
        "end": segment.get("end", 0.0),
        "text": text,
        "scores": {k: scores.get(k, 5) for k in SCORE_KEYS},
        "clip_type": scores.get("clip_type", "key_fact"),
        "suggested_title": scores.get("suggested_title", "Untitled Clip"),
    }


def _compute_final_score(scores: dict) -> float:
    """
    Computes final viral score using weighted formula:
    final_score = (virality*0.4) + (quotable*0.3) + (emotional*0.2) + (surprising*0.1)
    Result is clamped to [0.0, 10.0].
    """
    raw = sum(scores.get(k, 0) * w for k, w in WEIGHTS.items())
    return round(min(max(raw, 0.0), 10.0), 4)


def score_all_segments(
    segments: List[dict],
    job_id: str,
    top_n: int = 5
) -> dict:
    """
    Scores all segments, filters out low-scorers (< 5.0), sorts by final_score,
    and returns the top N results. Saves output to /temp/scores_{job_id}.json
    """
    TEMP_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[scoring] Starting to score {len(segments)} segments...")
    scored = []
    for idx, seg in enumerate(segments):
        print(f"[scoring] Processing segment {idx+1}/{len(segments)}: {seg.get('text', '')[:60]}...")
        result = score_segment(seg)

        # Skip empty / skipped segments
        if not result:
            print(f"[scoring] Segment {idx+1} skipped (empty)")
            continue

        final_score = _compute_final_score(result["scores"])

        # Filter below threshold (1.0 keeps even Ollama-fallback default clips)
        if final_score < 1.0:
            print(f"[scoring] Segment {idx+1} filtered (score: {final_score})")
            continue

        result["final_score"] = final_score
        scored.append(result)
        print(f"[scoring] Segment {idx+1} scored: {final_score} - {result.get('suggested_title', 'N/A')}")

    print(f"[scoring] Scored {len(scored)} valid segments")
    
    # Sort descending by final_score
    scored.sort(key=lambda x: x["final_score"], reverse=True)

    # Take top N
    top_clips = []
    for rank, clip in enumerate(scored[:top_n], start=1):
        top_clips.append({
            "rank": rank,
            "start": clip["start"],
            "end": clip["end"],
            "text": clip["text"],
            "scores": clip["scores"],
            "final_score": clip["final_score"],
            "clip_type": clip["clip_type"],
            "suggested_title": clip["suggested_title"],
        })

    output = {
        "job_id": job_id,
        "top_clips": top_clips,
    }

    # Persist to disk
    save_path = TEMP_DIR / f"scores_{job_id}.json"
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    return output
