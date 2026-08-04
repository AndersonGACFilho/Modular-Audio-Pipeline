"""Faithful transcript formatting for documentation and archival output."""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List

from .utils import format_timestamp


def normalize_whitespace(text: str) -> str:
    """Clean layout only; never rewrite, remove, or infer spoken content."""
    return re.sub(r"\s+", " ", text).strip()


def archival_segments(segments: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Copy segments while applying only whitespace normalization to their text."""
    normalized = []
    for segment in segments:
        copy = dict(segment)
        copy["text"] = normalize_whitespace(str(copy.get("text", "")))
        normalized.append(copy)
    return normalized


def documentation_text(segments: Iterable[Dict[str, Any]]) -> str:
    """Render a readable, speaker-labelled transcript without semantic edits."""
    lines = []
    for segment in segments:
        start = format_timestamp(float(segment.get("original_start", segment.get("start", 0))))
        speaker = segment.get("speaker", "Unknown")
        suggestion = segment.get("speaker_suggestion")
        if suggestion:
            speaker = (
                f"{speaker} (possivelmente {suggestion['name']}, "
                f"{suggestion['confidence']:.0%})"
            )
        text = segment.get("text", "")
        if text:
            lines.append(f"[{start}] {speaker}: {text}")
    return "\n".join(lines)
