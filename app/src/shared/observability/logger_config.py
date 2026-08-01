import logging
from datetime import date
from pathlib import Path

from rich.logging import RichHandler
from rich.console import Console

CONSOLE_FORMAT = (
    "%(name)s.%(funcName)s:%(lineno)d"
    " | %(message)s"
)

FILE_FORMAT = (
    "%(asctime)s"
    " | %(levelname)-8s"
    " | %(name)s.%(funcName)s:%(lineno)d"
    " | %(message)s"
)

def configure_logging(
    entrypoint: str,
    level: int = logging.INFO,
    log_directory: str | Path = "logs",
    console: Console | None = None,
) -> Console:
    """
    Configures logging for the entire application.

    Args:
        entrypoint: Name of the application entry point. It becomes the log
            directory name.
        level: Minimum logging level.
        log_directory: Root directory for application logs.
    """
    log_path = Path(log_directory) / entrypoint
    log_path.mkdir(parents=True, exist_ok=True)

    log_date = date.today().isoformat()
    application_log_path = log_path / f"application_{log_date}.log"
    error_log_path = log_path / f"error_{log_date}.log"

    console = console or Console()
    console_handler = RichHandler(
        console=console,
        rich_tracebacks=True,
        show_time=True,
        show_level=True,
        show_path=False,
        markup=False,
    )

    console_handler.setLevel(level)

    console_handler.setFormatter(
        logging.Formatter(CONSOLE_FORMAT)
    )

    file_handler = logging.FileHandler(
        filename=application_log_path,
        encoding="utf-8",
    )

    file_handler.setLevel(level)

    file_handler.setFormatter(
        logging.Formatter(
            fmt=FILE_FORMAT,
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )

    error_handler = logging.FileHandler(
        filename=error_log_path,
        encoding="utf-8",
        delay=True,
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(
        logging.Formatter(
            fmt=FILE_FORMAT,
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )

    logging.basicConfig(
        level=level,
        handlers=[
            console_handler,
            file_handler,
            error_handler,
        ],
        force=True,
    )

    logging.captureWarnings(True)
    return console
