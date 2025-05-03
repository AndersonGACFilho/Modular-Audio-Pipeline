from difflib import SequenceMatcher
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)

class RedundancyRemover:
    """
    Removes near-duplicate transcription segments based on text similarity.
    """
    def __init__(self, similarity_threshold: float = 0.9):
        """
        Args:
            similarity_threshold (float): ratio above which two texts are considered duplicates.
        """
        self.sim_thr = similarity_threshold

    def is_similar(self, a: str, b: str) -> bool:
        """
        Compare two strings for similarity.
        """
        return SequenceMatcher(None, a, b).ratio() >= self.sim_thr

    def remove(self, segments: List[Dict]) -> List[Dict]:
        """
        Iterate through segments and drop any whose text is too similar
        to the previous kept segment.
        """
        if not segments:
            return []
        filtered = [segments[0]]
        for seg in segments[1:]:
            if self.is_similar(filtered[-1]["text"], seg["text"]):
                logger.info("Dropped redundant segment: %r", seg["text"])
                continue
            filtered.append(seg)
        return filtered
