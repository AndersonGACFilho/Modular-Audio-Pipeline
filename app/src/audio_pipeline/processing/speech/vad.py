"""Deprecated compatibility imports for voice activity detection."""

from ...infrastructure.speech.vad import NoOpVADFilter, SileroVADFilter, VADFilter

__all__ = ["NoOpVADFilter", "SileroVADFilter", "VADFilter"]
