"""
audio_pipeline.segment_merger

Utilities for merging adjacent transcription/diarization segments.

Provides SegmentMerger which merges adjacent segments by speaker and
concatenates text when input segments are dictionaries. Docstrings follow
pydoc conventions to support Sphinx/pydoc extraction.
"""

from typing import List, Optional, Union, Any, Dict
from .protocols import DiarizationSegment


class SegmentMerger:
    """
    Merges adjacent segments belonging to the same speaker.
    Supports both DiarizationSegment objects and transcription dictionaries.
    """

    def __init__(self, max_gap_s: float = 0.5):
        self.max_gap_s = max_gap_s

    def _get(self, seg: Union[DiarizationSegment, dict], name: str, default: Any = None):
        """Safely retrieves a value from a dictionary or an object attribute."""
        if isinstance(seg, dict):
            return seg.get(name, default)
        return getattr(seg, name, default)

    def merge(
            self,
            segments: List[Union[DiarizationSegment, dict]],
            max_gap_s: Optional[float] = None
    ) -> List[Union[DiarizationSegment, Dict[str, Any]]]:
        """
        Returns a new list of merged segments.
        If input contains dictionaries (transcriptions), it preserves/merges text and returns dictionaries.
        """
        if not segments:
            return []

        gap = self.max_gap_s if max_gap_s is None else max_gap_s

        def get_start(s):
            return float(self._get(s, "start", 0.0))

        segs = sorted(segments, key=get_start)

        merged = []

        first_seg = segs[0]
        cur_speaker = self._get(first_seg, "speaker")
        cur_start = float(self._get(first_seg, "start", 0.0))
        cur_end = float(self._get(first_seg, "end", 0.0))
        cur_track = str(self._get(first_seg, "track", "0"))
        cur_text = self._get(first_seg, "text")

        for seg in segs[1:]:
            s_speaker = self._get(seg, "speaker")
            s_start = float(self._get(seg, "start", 0.0))
            s_end = float(self._get(seg, "end", 0.0))
            s_track = str(self._get(seg, "track", "0"))
            s_text = self._get(seg, "text")

            gap_between = s_start - cur_end

            if s_speaker == cur_speaker and gap_between <= gap:
                cur_end = max(cur_end, s_end)

                if cur_text is not None and s_text is not None:
                    if cur_text.strip():
                        cur_text = f"{cur_text.strip()} {s_text.strip()}"
                    else:
                        cur_text = s_text
                elif cur_text is None and s_text is not None:
                    cur_text = s_text

            else:
                merged.append(self._make_output(
                    first_seg, cur_speaker, cur_start, cur_end, cur_track, cur_text
                ))

                cur_speaker = s_speaker
                cur_start = s_start
                cur_end = s_end
                cur_track = s_track
                cur_text = s_text
                first_seg = seg

        merged.append(self._make_output(
            first_seg, cur_speaker, cur_start, cur_end, cur_track, cur_text
        ))

        return merged

    def _make_output(self, template_obj, speaker, start, end, track, text):
        """Creates the output segment maintaining the input type (dict or object)."""
        if isinstance(template_obj, dict):
            return {
                "speaker": speaker,
                "start": start,
                "end": end,
                "track": track,
                "text": text if text is not None else ""
            }
        else:
            return DiarizationSegment(
                speaker=speaker,
                start=start,
                end=end,
                track=track
            )