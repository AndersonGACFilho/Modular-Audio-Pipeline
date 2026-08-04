import logging

from audio_pipeline.infrastructure.speech.diarizer import _DiarizationProgressHook


def test_diarization_progress_hook_logs_pyannote_step(caplog):
    hook = _DiarizationProgressHook()

    with caplog.at_level(logging.INFO):
        hook("segmentation", None, total=16, completed=10)

    assert hook.status == "segmentation (10/16)"
    assert "Diarization progress: segmentation (10/16)" in caplog.messages
