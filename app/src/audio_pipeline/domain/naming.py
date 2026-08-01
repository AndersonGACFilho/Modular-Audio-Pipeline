"""Deterministic, filesystem-safe names for contextual pipeline outputs."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Optional


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


def rename_source_media(source_file: str, output_stem: str) -> str:
    """Rename source media to the contextual stem without overwriting another file."""
    source = Path(source_file)
    target = source.with_name(f"{output_stem}{source.suffix.lower()}")
    if target == source:
        return str(source)
    if target.exists():
        raise FileExistsError(f"Contextual media name already exists: {target}")
    source.rename(target)
    return str(target)


def rename_derived_artifact(
    artifact_file: str,
    source_stem: str,
    output_stem: str,
) -> Optional[str]:
    """Rename a generated audio artifact while preserving its processing suffix.

    For example, ``recording_16000Hz_denoised_norm.wav`` becomes
    ``contextual-name_16000Hz_denoised_norm.wav``. Missing files are ignored so
    callers can safely pass temporary artifacts that were intentionally removed.
    """
    artifact = Path(artifact_file)
    if not artifact.exists() or artifact.stem == source_stem:
        return None

    suffix = artifact.stem.removeprefix(source_stem)
    if not suffix:
        return None

    target = artifact.with_name(f"{output_stem}{suffix}{artifact.suffix.lower()}")
    if target == artifact:
        return str(artifact)
    if target.exists():
        raise FileExistsError(f"Contextual artifact name already exists: {target}")
    artifact.rename(target)
    return str(target)
