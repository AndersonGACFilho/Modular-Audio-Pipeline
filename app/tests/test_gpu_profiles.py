import pytest

from audio_pipeline.config import PipelineConfig
from audio_pipeline.config.gpu_profiles import apply_gpu_workload_profile


def test_background_gpu_profile_reduces_all_inference_batches():
    config = PipelineConfig()

    profile = apply_gpu_workload_profile(config, "background")

    assert profile.transcription_batch_size == 1
    assert config.transcription.batch_size == 1
    assert config.diarization.segmentation_batch_size == 4
    assert config.diarization.embedding_batch_size == 4


def test_unknown_gpu_profile_is_actionable():
    with pytest.raises(ValueError, match="fast, balanced, background"):
        apply_gpu_workload_profile(PipelineConfig(), "quiet")
