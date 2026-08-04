"""GPU workload profiles applied by the application's composition root."""

from __future__ import annotations

from dataclasses import dataclass

from .config import PipelineConfig


@dataclass(frozen=True)
class GPUWorkloadProfile:
    """Batch sizes that determine GPU pressure during inference."""

    transcription_batch_size: int
    diarization_segmentation_batch_size: int
    diarization_embedding_batch_size: int


GPU_WORKLOAD_PROFILES = {
    "fast": GPUWorkloadProfile(4, 32, 32),
    "balanced": GPUWorkloadProfile(2, 16, 16),
    "background": GPUWorkloadProfile(1, 4, 4),
}


def apply_gpu_workload_profile(config: PipelineConfig, name: str) -> GPUWorkloadProfile:
    """Apply a named profile to a config before infrastructure is constructed."""
    try:
        profile = GPU_WORKLOAD_PROFILES[name.casefold()]
    except KeyError as error:
        available = ", ".join(GPU_WORKLOAD_PROFILES)
        raise ValueError(f"Unknown GPU workload profile '{name}'. Choose one of: {available}.") from error

    config.transcription.batch_size = profile.transcription_batch_size
    config.diarization.segmentation_batch_size = profile.diarization_segmentation_batch_size
    config.diarization.embedding_batch_size = profile.diarization_embedding_batch_size
    return profile
