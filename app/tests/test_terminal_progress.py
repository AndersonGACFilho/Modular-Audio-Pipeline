from rich.console import Console

from shared.observability.terminal_progress import TerminalProgress


def test_terminal_progress_is_enabled_in_pycharm_console(monkeypatch):
    monkeypatch.setenv("PYCHARM_HOSTED", "1")

    progress = TerminalProgress(Console(force_terminal=True))

    assert progress._enabled is True


def test_terminal_progress_is_enabled_in_a_real_terminal(monkeypatch):
    monkeypatch.delenv("PYCHARM_HOSTED", raising=False)

    progress = TerminalProgress(Console(force_terminal=True))

    assert progress._enabled is True


def test_footer_shows_current_file_and_remaining_count():
    progress = TerminalProgress(Console(force_terminal=True), total_files=3)
    progress.set_file("recording.mp4", index=2)

    assert progress._description() == "Arquivo 2/3 (restam 1): recording.mp4 - Starting (0/16)"
