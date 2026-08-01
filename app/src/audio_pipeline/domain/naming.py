"""Deterministic, filesystem-safe names for contextual pipeline outputs."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict


def _slug(value: str) -> str:
    """Convert context text to a short, portable filename component."""
    normalized = value.strip().lower()
    normalized = re.sub(r"[^a-z0-9]+", "-", normalized)
    return normalized.strip("-")[:48]


def contextual_output_stem(source_file: str, formatting: Dict[str, Any] | None) -> str:
    """Build an output stem from a detected profile and optional context.

    Falls back to the original media stem whenever routing metadata is absent.
    """
    source_stem = Path(source_file).stem
    if not formatting or not formatting.get("profile"):
        return source_stem

    components = []
    timestamp_match = re.search(
        r"(\d{4}-\d{2}-\d{2})[ _](\d{2}[-:]\d{2}[-:]\d{2})", source_stem
    )
    if timestamp_match:
        components.extend((timestamp_match.group(1), timestamp_match.group(2).replace(":", "-")))
    else:
        date_match = re.search(r"\d{4}-\d{2}-\d{2}", source_stem)
        if date_match:
            components.append(date_match.group(0))
        else:
            components.append(_slug(source_stem))

    components.append(_slug(str(formatting["profile"])))
    for key in ("organization", "project"):
        value = formatting.get(key)
        if value:
            component = _slug(str(value))
            if component and component not in components:
                components.append(component)

    return "_".join(component for component in components if component) or source_stem
