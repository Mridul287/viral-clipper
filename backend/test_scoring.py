"""
test_scoring.py — Tests for the viral segment scoring engine.
All Ollama HTTP calls are mocked with unittest.mock so no Ollama
installation is required to run the suite.
"""

import json
import pytest
import pathlib
from unittest.mock import patch, MagicMock

from scoring import (
    score_segment,
    score_all_segments,
    parse_llm_response,
    _compute_final_score,
    DEFAULT_SCORES,
    TEMP_DIR,
)

# ---------------------------------------------------------------------------
# Helpers / Fixtures
# ---------------------------------------------------------------------------

GOOD_SCORES = {
    "funny": 7,
    "surprising": 8,
    "quotable": 9,
    "emotional": 6,
    "virality": 8,
    "clip_type": "hook",
    "suggested_title": "The AI Secret Nobody Talks About",
}

LOW_SCORES = {
    "funny": 2,
    "surprising": 2,
    "quotable": 2,
    "emotional": 2,
    "virality": 2,
    "clip_type": "key_fact",
    "suggested_title": "A Very Boring Moment",
}


def _mock_ollama_response(scores_dict: dict):
    """Returns a mock requests.Response whose .json() yields the Ollama format."""
    mock_resp = MagicMock()
    mock_resp.raise_for_status.return_value = None
    mock_resp.json.return_value = {"response": json.dumps(scores_dict)}
    return mock_resp


def _make_segment(text: str, start: float = 0.0, end: float = 5.0) -> dict:
    return {"text": text, "start": start, "end": end}


# ---------------------------------------------------------------------------
# Test 1 — Basic call returns all required keys
# ---------------------------------------------------------------------------

def test_top_clips_have_required_keys():
    """1. Pass 3 mock segments → all top_clips items have required keys."""
    segments = [
        _make_segment("This one secret will change your life.", 0, 10),
        _make_segment("Science proves multitasking is a myth.", 10, 20),
        _make_segment("You have been lied to your entire life.", 20, 30),
    ]

    with patch("scoring.requests.post", return_value=_mock_ollama_response(GOOD_SCORES)):
        result = score_all_segments(segments, job_id="test1")

    required = {"rank", "start", "end", "text", "scores", "final_score",
                "clip_type", "suggested_title"}

    assert "top_clips" in result
    for clip in result["top_clips"]:
        assert required.issubset(clip.keys()), f"Missing keys in: {clip.keys()}"


# ---------------------------------------------------------------------------
# Test 2 — final_score always within [0.0, 10.0]
# ---------------------------------------------------------------------------

def test_final_score_within_bounds():
    """2. All final_score values must be in [0.0, 10.0]."""
    segments = [_make_segment(f"Segment number {i}", i * 5, i * 5 + 5) for i in range(5)]

    with patch("scoring.requests.post", return_value=_mock_ollama_response(GOOD_SCORES)):
        result = score_all_segments(segments, job_id="test2")

    for clip in result["top_clips"]:
        assert 0.0 <= clip["final_score"] <= 10.0


# ---------------------------------------------------------------------------
# Test 3 — Results are sorted descending by final_score
# ---------------------------------------------------------------------------

def test_clips_sorted_descending():
    """3. top_clips[0].final_score >= top_clips[1].final_score always."""
    # Mix high and low scorers
    scores_list = [GOOD_SCORES, LOW_SCORES, GOOD_SCORES, GOOD_SCORES]
    responses = iter([_mock_ollama_response(s) for s in scores_list])

    segments = [_make_segment(f"Segment {i}", i * 5, i * 5 + 5) for i in range(4)]

    with patch("scoring.requests.post", side_effect=lambda *a, **kw: next(responses)):
        result = score_all_segments(segments, job_id="test3", top_n=10)

    clips = result["top_clips"]
    for i in range(len(clips) - 1):
        assert clips[i]["final_score"] >= clips[i + 1]["final_score"]


# ---------------------------------------------------------------------------
# Test 4 — Segments with final_score < 5.0 are filtered out
# ---------------------------------------------------------------------------

def test_low_score_segments_filtered():
    """4. Segments with final_score < 5.0 must NOT appear in top_clips."""
    segments = [
        _make_segment("This is mediocre content.", 0, 5),
        _make_segment("This is absolutely flat.", 5, 10),
    ]

    with patch("scoring.requests.post", return_value=_mock_ollama_response(LOW_SCORES)):
        result = score_all_segments(segments, job_id="test4")

    # low_scores yields final_score of:
    # (2*0.4) + (2*0.3) + (2*0.2) + (2*0.1) = 2.0 — well below 5.0
    assert result["top_clips"] == []


# ---------------------------------------------------------------------------
# Test 5 — Empty text segment is skipped gracefully
# ---------------------------------------------------------------------------

def test_empty_text_segment_skipped():
    """5. A segment with text='' should be skipped, no crash."""
    segments = [
        _make_segment("", 0, 5),         # should be skipped
        _make_segment("Valid content about AI growth.", 5, 10),
    ]

    with patch("scoring.requests.post", return_value=_mock_ollama_response(GOOD_SCORES)):
        result = score_all_segments(segments, job_id="test5")

    # Only the valid segment should appear
    for clip in result["top_clips"]:
        assert clip["text"] != ""


# ---------------------------------------------------------------------------
# Test 6 — Malformed JSON falls back to default scores
# ---------------------------------------------------------------------------

def test_malformed_json_uses_defaults():
    """6. parse_llm_response returns DEFAULT_SCORES on malformed input."""
    malformed_cases = [
        "This is not JSON at all",
        "{funny: 7, virality: 9}",      # unquoted keys
        '{"funny": 7',                  # truncated
        "",                             # empty string
        None,                           # NoneType handled
    ]

    for bad_input in malformed_cases:
        result = parse_llm_response(bad_input)
        assert result["funny"] == DEFAULT_SCORES["funny"]
        assert result["virality"] == DEFAULT_SCORES["virality"]


# ---------------------------------------------------------------------------
# Test 7 — top_n=3 returns at most 3 items from 10 segments
# ---------------------------------------------------------------------------

def test_top_n_limit():
    """7. Pass 10 segments, top_n=3 → only 3 clips returned."""
    segments = [_make_segment(f"Powerful insight number {i}", i * 5, i * 5 + 5)
                for i in range(10)]

    with patch("scoring.requests.post", return_value=_mock_ollama_response(GOOD_SCORES)):
        result = score_all_segments(segments, job_id="test7", top_n=3)

    assert len(result["top_clips"]) == 3


# ---------------------------------------------------------------------------
# Test 8 — scores_{job_id}.json saved in /temp with correct structure
# ---------------------------------------------------------------------------

def test_scores_json_saved():
    """8. Confirm scores_{job_id}.json is saved in /temp after run."""
    job_id = "test8_save"
    segments = [_make_segment("AI will replace every knowledge worker.", 0, 8)]

    with patch("scoring.requests.post", return_value=_mock_ollama_response(GOOD_SCORES)):
        score_all_segments(segments, job_id=job_id)

    save_path = TEMP_DIR / f"scores_{job_id}.json"
    assert save_path.exists()

    with open(save_path, encoding="utf-8") as f:
        data = json.load(f)

    assert data["job_id"] == job_id
    assert "top_clips" in data


# ---------------------------------------------------------------------------
# Test 9 — Manual formula verification
# ---------------------------------------------------------------------------

def test_final_score_formula():
    """9. Verify final_score = (virality*0.4) + (quotable*0.3) + (emotional*0.2) + (surprising*0.1)."""
    scores = {"funny": 7, "surprising": 6, "quotable": 9, "emotional": 8, "virality": 10}

    expected = (10 * 0.4) + (9 * 0.3) + (8 * 0.2) + (6 * 0.1)
    expected = round(min(max(expected, 0.0), 10.0), 4)

    computed = _compute_final_score(scores)
    assert computed == expected, f"Expected {expected}, got {computed}"
