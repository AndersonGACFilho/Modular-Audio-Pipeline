"""
Redundancy Remover for the Audio Pipeline.

Removes near-duplicate transcription segments based on text similarity.
"""

from difflib import SequenceMatcher
from typing import List, Dict, Optional, Callable
import logging
import re

from .protocols import RedundancyRemoverProtocol
from .config import PipelineConfig, RedundancyConfig

logger = logging.getLogger(__name__)


class RedundancyRemover(RedundancyRemoverProtocol):
    """
    Removes near-duplicate transcription segments based on text similarity.
    
    Features:
    - Configurable similarity threshold
    - Text normalization before comparison
    - Support for custom similarity functions
    """
    
    def __init__(
        self,
        similarity_threshold: float = 0.85,
        normalize_text: bool = True,
        custom_similarity_fn: Optional[Callable[[str, str], float]] = None
    ):
        """
        Initialize RedundancyRemover.
        
        Args:
            similarity_threshold: Ratio above which texts are considered duplicates (0-1)
            normalize_text: Whether to normalize text before comparison
            custom_similarity_fn: Optional custom similarity function
        """
        if not 0 <= similarity_threshold <= 1:
            raise ValueError(f"similarity_threshold must be 0-1, got: {similarity_threshold}")
        
        self.threshold = similarity_threshold
        self.normalize = normalize_text
        self.custom_similarity_fn = custom_similarity_fn
    
    @classmethod
    def from_config(cls, config: PipelineConfig) -> "RedundancyRemover":
        """Create remover from pipeline configuration."""
        return cls(similarity_threshold=config.redundancy.similarity_threshold)
    
    def _normalize_text(self, text: str) -> str:
        """
        Normalize text for comparison.
        
        Removes punctuation, extra whitespace, and converts to lowercase.
        """
        # Remove punctuation
        text = re.sub(r'[^\w\s]', '', text)
        # Normalize whitespace
        text = ' '.join(text.split())
        # Lowercase
        text = text.lower()
        return text
    
    def get_similarity(self, a: str, b: str) -> float:
        """
        Calculate similarity ratio between two strings.
        
        Args:
            a: First string
            b: Second string
            
        Returns:
            Similarity ratio (0-1)
        """
        if self.custom_similarity_fn:
            return self.custom_similarity_fn(a, b)
        
        if self.normalize:
            a = self._normalize_text(a)
            b = self._normalize_text(b)
        
        return SequenceMatcher(None, a, b).ratio()
    
    def is_similar(self, a: str, b: str) -> bool:
        """
        Check if two texts are similar enough to be considered duplicates.
        
        Args:
            a: First text
            b: Second text
            
        Returns:
            True if similarity >= threshold
        """
        return self.get_similarity(a, b) >= self.threshold
    
    def remove(self, segments: List[Dict]) -> List[Dict]:
        """
        Remove redundant segments from a list.
        
        Iterates through segments and drops any whose text is too similar
        to the previous kept segment.
        
        Args:
            segments: List of segment dicts with 'text' key
            
        Returns:
            Filtered list of segments
        """
        if not segments:
            return []
        
        filtered = [segments[0]]
        dropped_count = 0
        
        for seg in segments[1:]:
            current_text = seg.get("text", "").strip()
            last_text = filtered[-1].get("text", "").strip()
            
            if not current_text:
                logger.debug("Dropping empty segment")
                dropped_count += 1
                continue
            
            similarity = self.get_similarity(last_text, current_text)
            
            if similarity >= self.threshold:
                logger.debug(
                    f"Dropped redundant segment (similarity={similarity:.2f}): "
                    f"{current_text[:50]}..."
                )
                dropped_count += 1
                continue
            
            filtered.append(seg)
        
        if dropped_count > 0:
            logger.info(f"Removed {dropped_count} redundant segments")
        
        return filtered
    
    def remove_with_merging(
        self,
        segments: List[Dict],
        merge_gap_s: float = 0.5
    ) -> List[Dict]:
        """
        Remove redundant segments with optional merging of consecutive same-speaker segments.
        
        Args:
            segments: List of segment dicts
            merge_gap_s: Maximum gap between segments to merge
            
        Returns:
            Filtered and merged list of segments
        """
        if not segments:
            return []
        
        # First remove redundancies
        filtered = self.remove(segments)
        
        if len(filtered) <= 1:
            return filtered
        
        # Then merge consecutive same-speaker segments
        merged = [filtered[0].copy()]
        
        for seg in filtered[1:]:
            last = merged[-1]
            
            # Check if same speaker and close in time
            same_speaker = seg.get("speaker") == last.get("speaker")
            gap = seg.get("start", 0) - last.get("end", 0)
            
            if same_speaker and gap <= merge_gap_s:
                # Merge: extend end time and concatenate text
                last["end"] = seg.get("end", last["end"])
                last["text"] = last.get("text", "") + " " + seg.get("text", "")
                logger.debug(f"Merged consecutive segment from {seg.get('speaker')}")
            else:
                merged.append(seg.copy())
        
        return merged
    
    def find_duplicates(
        self,
        segments: List[Dict]
    ) -> List[tuple]:
        """
        Find all pairs of potentially duplicate segments.
        
        Useful for analysis without modifying the list.
        
        Args:
            segments: List of segment dicts
            
        Returns:
            List of (index_i, index_j, similarity) tuples
        """
        duplicates = []
        
        for i, seg_i in enumerate(segments):
            text_i = seg_i.get("text", "").strip()
            
            for j, seg_j in enumerate(segments[i + 1:], start=i + 1):
                text_j = seg_j.get("text", "").strip()
                similarity = self.get_similarity(text_i, text_j)
                
                if similarity >= self.threshold:
                    duplicates.append((i, j, similarity))
        
        return duplicates


class NoOpRedundancyRemover(RedundancyRemoverProtocol):
    """
    No-operation remover that passes through all segments unchanged.
    
    Used when redundancy removal is disabled.
    """
    
    def is_similar(self, a: str, b: str) -> bool:
        return False
    
    def remove(self, segments: List[Dict]) -> List[Dict]:
        return segments
