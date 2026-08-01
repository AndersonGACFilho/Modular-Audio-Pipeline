"""Filesystem operations for output artifact names."""

from pathlib import Path
from typing import Optional


class LocalArtifactRenamer:
    """Local-filesystem implementation of the artifact-renaming port."""

    def rename_source_media(self, source_file: str, output_stem: str) -> str:
        return rename_source_media(source_file, output_stem)

    def rename_derived_artifact(self, artifact_file: str, source_stem: str, output_stem: str) -> Optional[str]:
        return rename_derived_artifact(artifact_file, source_stem, output_stem)


def rename_source_media(source_file: str, output_stem: str) -> str:
    source = Path(source_file)
    target = source.with_name(f"{output_stem}{source.suffix.lower()}")
    if target == source:
        return str(source)
    if target.exists():
        raise FileExistsError(f"Contextual media name already exists: {target}")
    source.rename(target)
    return str(target)


def rename_derived_artifact(artifact_file: str, source_stem: str, output_stem: str) -> Optional[str]:
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
