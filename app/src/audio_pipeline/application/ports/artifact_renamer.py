"""Port for physical artifact-renaming operations."""

from typing import Protocol


class ArtifactRenamer(Protocol):
    def rename_source_media(self, source_file: str, output_stem: str) -> str: ...
    def rename_derived_artifact(self, artifact_file: str, source_stem: str, output_stem: str) -> str | None: ...
