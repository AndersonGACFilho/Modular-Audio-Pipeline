import logging
from typing import Any

class ClassLogger:
    """
    Descriptor that provides a logger named after its module and class.
    """

    def __get__(
            self,
            instance: Any,
            owner: type,
    ) -> logging.Logger:
        logger_name = f"{owner.__module__}.{owner.__qualname__}"

        return logging.getLogger(logger_name)

class LoggerMixin:
    """
    Add an automatically named logger to classes that inherit from this mixin.
    """
    logger = ClassLogger()
