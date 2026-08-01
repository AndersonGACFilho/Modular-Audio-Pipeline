"""Technical processing capabilities required by the application workflow."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, Tuple, runtime_checkable


@dataclass
class DiarizationSegment:
    speaker: str
    start: float
    end: float
    track: str = ""


@dataclass
class TimestampMapping:
    processed_start: float
    processed_end: float
    original_start: float
    original_end: float


@runtime_checkable
class MediaHandlerProtocol(Protocol):
    def find_media_file(self) -> Tuple[str, bool]: ...
    def find_specific_file(self, input_path: str) -> Tuple[str, bool]: ...
    def get_media_info(self, file_path: str) -> Dict[str, Any]: ...
    def convert_to_wav(self, input_path: str) -> str: ...


@runtime_checkable
class PreprocessorProtocol(Protocol):
    def reduce_stationary_noise(self, input_wav: str, noise_sample_path: Optional[str] = None) -> str: ...
    def normalize_audio(self, input_wav: str) -> str: ...
    def normalize_loudness(self, input_wav: str, target_lufs: float = -16.0) -> str: ...
    def remove_silence(self, input_wav: str, preserve_timestamps: bool = False) -> Tuple[str, List[TimestampMapping]]: ...


@runtime_checkable
class VocalSeparatorProtocol(Protocol):
    def extract_vocals(self, input_wav: str) -> str: ...


@runtime_checkable
class VADProtocol(Protocol):
    def filter_voice(self, input_wav: str, output_dir: str) -> Tuple[str, List[TimestampMapping]]: ...


@runtime_checkable
class TranscriberProtocol(Protocol):
    def transcribe(self, input_wav: str) -> Dict[str, Any]: ...


@runtime_checkable
class DiarizerProtocol(Protocol):
    def diarize(self, audio_path: str, min_speakers: int = 2, max_speakers: int = 5) -> List[DiarizationSegment]: ...


@runtime_checkable
class RedundancyRemoverProtocol(Protocol):
    def remove(self, segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]: ...


class SegmentMergerProtocol(Protocol):
    def merge(self, segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]: ...
