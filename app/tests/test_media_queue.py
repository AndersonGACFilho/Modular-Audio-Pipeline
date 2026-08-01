from audio_pipeline.infrastructure.media.handler import MediaHandler


def test_media_handler_queues_all_supported_files_in_deterministic_order(tmp_path, monkeypatch):
    for name in ("z.mp4", "a.mp3", "b.mp4", "ignored.txt"):
        (tmp_path / name).write_bytes(b"media")
    handler = MediaHandler(str(tmp_path), str(tmp_path / "temp"))
    monkeypatch.setattr(handler, "_has_audio_stream", lambda _: True)

    queued = handler.list_media_files()

    assert [path.rsplit("\\", 1)[-1] for path in queued] == ["a.mp3", "b.mp4", "z.mp4"]
