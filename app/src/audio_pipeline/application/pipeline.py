"""
audio_pipeline.pipeline

Orchestrator for the audio processing and transcription pipeline.

This module coordinates all pipeline steps: media discovery/conversion,
preprocessing (denoise, normalization, silence removal), optional vocal
separation, VAD, transcription, diarization, redundancy removal and final
output serialization.

The public API is the AudioPipeline class which accepts components via
dependency injection for testing and customization.
"""

import os
import json
import time
from pathlib import Path
from typing import List, Dict, Optional, Any, Callable
from dataclasses import dataclass

from ..config import PipelineConfig
from ..documentation import archival_segments, documentation_text
from ..domain.exceptions import AudioPipelineError, MediaNotFoundError
from ..domain.naming import contextual_output_stem
from ..infrastructure.storage.transcription_snapshot import TranscriptionSnapshotWriter
from .ports.artifact_renamer import ArtifactRenamer
from .ports.media_processing import (
    MediaHandlerProtocol,
    PreprocessorProtocol,
    VocalSeparatorProtocol,
    VADProtocol,
    TranscriberProtocol,
    DiarizerProtocol,
    RedundancyRemoverProtocol,
    SegmentMergerProtocol,
    DiarizationSegment,
    TimestampMapping
)
from ..utils import CheckpointManager, ensure_directory, get_audio_duration
from shared.observability import LoggerMixin

# Optional LLM post-processor (imported lazily).
# Declared here to satisfy static analysis
HybridLLMPostProcessor = None


@dataclass
class PipelineResult:
    """Result returned by AudioPipeline.run().

    Args:
        success:
            Whether the pipeline completed successfully.
        input_file:
            Path to the input media file processed.
        output_file:
            Path to the output transcription JSON file.
        segments:
            List of transcription segments with timing and speaker info.
        error:
            Optional error message if the pipeline failed.
        metadata:
            Additional metadata about the processing run.
        llm_analysis:
            Optional LLM analysis results if LLM post-processing was used.
    """
    success: bool
    input_file: str
    output_file: Optional[str]
    segments: List[Dict[str, Any]]
    error: Optional[str] = None
    metadata: Dict[str, Any] = None
    llm_analysis: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class AudioPipeline(LoggerMixin):
    """Coordinates the full audio processing pipeline.

    The pipeline composes modular components and supports dependency
    injection for testing or custom implementations. The main entry point
    is run(input_file: Optional[str]) -> PipelineResult.
    """

    def __init__(
        self,
        config: PipelineConfig,
        media_handler: MediaHandlerProtocol,
        preprocessor: PreprocessorProtocol,
        separator: VocalSeparatorProtocol,
        vad: VADProtocol,
        transcriber: TranscriberProtocol,
        diarizer: DiarizerProtocol,
        redundancy_remover: RedundancyRemoverProtocol,
        segment_merger: SegmentMergerProtocol,
        checkpoint_manager: Optional[CheckpointManager] = None,
        artifact_renamer: Optional[ArtifactRenamer] = None,
        llm_processor_factory: Optional[Callable[[], Any]] = None,
        progress_callback: Optional[Callable[[str], None]] = None,
        file_callback: Optional[Callable[[str], None]] = None,
    ):
        """Create AudioPipeline.

        Parameters
        ----------
        config:
            Validated PipelineConfig instance.
        media_handler, preprocessor, separator, vad, transcriber, diarizer,
        redundancy_remover:
            Concrete adapters implementing the corresponding application ports.
        """
        self.config = config
        self.config.validate()

        # Setup directories
        self.media_dir = ensure_directory(self.config.media_dir)
        self.temp_dir = ensure_directory(self.config.temp_dir)
        self.results_dir = ensure_directory(self.config.results_dir)

        # Setup checkpoint manager
        self.checkpoint_manager = checkpoint_manager
        self.media = media_handler
        self.preprocessor = preprocessor
        self.separator = separator
        self.vad = vad
        self.transcriber = transcriber
        self.diarizer = diarizer
        self.redundancy = redundancy_remover
        self.segment_merger = segment_merger
        self.artifact_renamer = artifact_renamer
        self.llm_processor_factory = llm_processor_factory
        self.progress_callback = progress_callback
        self.file_callback = file_callback

        # LLM Post-Processor — intentionally NOT loaded here.
        # It is initialized lazily inside run(), AFTER transcription and
        # diarization models have been unloaded, to avoid VRAM exhaustion
        # on GPUs with ≤ 8 GB of memory.
        self.llm_processor = None

        # Timestamp mappings
        self._timestamp_mappings: List[TimestampMapping] = []
        self._snapshot_writer: Optional[TranscriptionSnapshotWriter] = None
        self._snapshot_segments: List[Dict[str, Any]] = []

    def _map_timestamp_to_original(
        self,
        processed_time: float,
        mappings: List[TimestampMapping]
    ) -> float:
        """Map timestamp from processed audio back to original.

        Parameters
        ----------
        processed_time:
            Time in seconds in the processed audio timeline.
        mappings:
            List of TimestampMapping objects produced during preprocessing.

        Returns
        -------
        float
            Corresponding time in the original audio timeline if mapping
            exists; otherwise returns processed_time unchanged.
        """
        if not mappings:
            return processed_time

        for mapping in mappings:
            if mapping.processed_start <= processed_time <= mapping.processed_end:
                ratio = (processed_time - mapping.processed_start) / \
                        (mapping.processed_end - mapping.processed_start + 1e-10)
                original_time = mapping.original_start + \
                               ratio * (mapping.original_end - mapping.original_start)
                return original_time

        return processed_time

    def _release_audio_models(self) -> None:
        """Release ASR and diarization models before LLM processing."""
        for component_name, component in (("transcriber", self.transcriber), ("diarizer", self.diarizer)):
            unload = getattr(component, "unload_model", None)
            if unload:
                self.logger.info(f"Unloading {component_name} to free VRAM for LLM...")
                unload()
        try:
            import gc
            import torch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            self.logger.debug("Unable to clear CUDA cache after audio model unload", exc_info=True)

    def _compose_timestamp_mappings(
        self, outer: List[TimestampMapping], inner: List[TimestampMapping]
    ) -> List[TimestampMapping]:
        """Compose processed→intermediate and intermediate→original mappings."""
        composed: List[TimestampMapping] = []
        for inner_mapping in inner:
            inner_duration = inner_mapping.original_end - inner_mapping.original_start
            if inner_duration <= 0:
                continue
            for outer_mapping in outer:
                start = max(inner_mapping.original_start, outer_mapping.processed_start)
                end = min(inner_mapping.original_end, outer_mapping.processed_end)
                if end <= start:
                    continue
                processed_duration = inner_mapping.processed_end - inner_mapping.processed_start
                processed_start = inner_mapping.processed_start + (start - inner_mapping.original_start) / inner_duration * processed_duration
                processed_end = inner_mapping.processed_start + (end - inner_mapping.original_start) / inner_duration * processed_duration
                outer_duration = outer_mapping.processed_end - outer_mapping.processed_start
                original_start = outer_mapping.original_start + (start - outer_mapping.processed_start) / outer_duration * (outer_mapping.original_end - outer_mapping.original_start)
                original_end = outer_mapping.original_start + (end - outer_mapping.processed_start) / outer_duration * (outer_mapping.original_end - outer_mapping.original_start)
                composed.append(TimestampMapping(processed_start, processed_end, original_start, original_end))
        return composed

    def _align_transcription_with_speakers(
        self,
        transcription_segments: List[Dict],
        diarization_segments: List[DiarizationSegment]
    ) -> List[Dict]:
        """Align transcription segments with diarization speaker labels.

        Parameters
        ----------
        transcription_segments:
            List of dicts produced by the transcriber with 'start' and 'end'.
        diarization_segments:
            List of DiarizationSegment instances returned by the diarizer.

        Returns
        -------
        List[Dict]
            List of aligned segments containing speaker labels.
        """
        aligned = []

        for seg in transcription_segments:
            start = seg["start"]
            end = seg["end"]
            text = seg.get("text", "").strip()

            if not text:
                continue

            speaker = "Unknown"
            max_overlap = 0

            for diar_seg in diarization_segments:
                overlap_start = max(start, diar_seg.start)
                overlap_end = min(end, diar_seg.end)
                overlap_duration = max(0, overlap_end - overlap_start)

                if overlap_duration > max_overlap:
                    max_overlap = overlap_duration
                    speaker = diar_seg.speaker

            aligned.append({
                "speaker": speaker,
                "start": start,
                "end": end,
                "text": text
            })

        return aligned

    @staticmethod
    def _apply_speaker_name_suggestions(
        segments: List[Dict[str, Any]], suggestions: Dict[str, Any]
    ) -> None:
        """Attach LLM suggestions while preserving the diarizer's speaker labels."""
        by_speaker = {
            item["speaker"]: item for item in suggestions.get("suggestions", [])
            if item.get("suggested_name")
        }
        for segment in segments:
            suggestion = by_speaker.get(segment.get("speaker"))
            if suggestion:
                segment["speaker_suggestion"] = {
                    "name": suggestion["suggested_name"],
                    "confidence": suggestion["confidence"],
                    "evidence": suggestion["evidence"],
                }

    def _report_progress(self, stage: str) -> None:
        if self.progress_callback:
            self.progress_callback(stage.replace("_", " ").capitalize())

    def _report_file(self, file_path: str) -> None:
        if self.file_callback:
            self.file_callback(file_path)

    def _save_transcription_snapshot(
        self,
        writer: TranscriptionSnapshotWriter,
        stage: str,
        segments: List[Dict[str, Any]],
        metrics: Dict[str, Any],
        **kwargs: Any,
    ) -> None:
        """Persist the readable transcript after a completed processing stage."""
        path = writer.write(stage, segments, metrics, **kwargs)
        self.logger.info("Saved transcription snapshot after %s: %s", stage, path)

    def _measure_stage(
        self, metrics: Dict[str, Any], name: str, operation: Callable[[], Any]
    ) -> Any:
        """Run an operation and record its elapsed wall-clock duration."""
        self._report_progress(name)
        started_at = time.perf_counter()
        try:
            return operation()
        finally:
            metrics["stage_durations_s"][name] = round(
                time.perf_counter() - started_at, 3
            )
            snapshot_writer = getattr(self, "_snapshot_writer", None)
            if snapshot_writer is not None:
                try:
                    self._save_transcription_snapshot(
                        snapshot_writer, name, getattr(self, "_snapshot_segments", []), metrics
                    )
                except OSError as error:
                    self.logger.warning("Could not save transcription snapshot after %s: %s", name, error)

    @staticmethod
    def _finalize_metrics(metrics: Dict[str, Any], started_at: float) -> Dict[str, Any]:
        metrics["total_duration_s"] = round(time.perf_counter() - started_at, 3)
        return metrics

    def run(self, input_file: Optional[str] = None) -> PipelineResult:
        """Execute the full audio processing pipeline.

        Parameters
        ----------
        input_file:
            Optional path to a single media file to process. If None the
            pipeline will process the first discovered file in the configured
            media directory.

        Returns
        -------
        PipelineResult
            Result object containing success flag, output path, segments and
            optional error information.
        """
        started_at = time.perf_counter()
        metrics: Dict[str, Any] = {"stage_durations_s": {}, "media": {}, "segments": {}}
        self._snapshot_writer = None
        self._snapshot_segments = []

        try:
            # Step 1: Find media file
            if input_file:
                media_file, is_video = self._measure_stage(
                    metrics, "media_discovery", lambda: self.media.find_specific_file(input_file)
                )
            else:
                media_file, is_video = self._measure_stage(
                    metrics, "media_discovery", self.media.find_media_file
                )

            base = Path(media_file).stem
            self._report_file(str(media_file))
            self.logger.info(f"Processing: {media_file}")
            self._snapshot_writer = TranscriptionSnapshotWriter(self.results_dir, media_file)
            source_info = self.media.get_media_info(media_file)
            metrics["media"] = {
                "input_size_bytes": os.path.getsize(media_file),
                "source_duration_s": source_info.get("duration"),
                "source_format": Path(media_file).suffix.lower(),
            }
            self._save_transcription_snapshot(
                self._snapshot_writer, "media_discovery", self._snapshot_segments, metrics
            )

            # Step 2: Convert to WAV
            ext = Path(media_file).suffix.lower()
            if is_video or ext != '.wav':
                wav = self._measure_stage(
                    metrics, "media_conversion", lambda: self.media.convert_to_wav(media_file)
                )
            else:
                wav = media_file
                self._report_progress("media conversion")
                metrics["stage_durations_s"]["media_conversion"] = 0.0

            # Step 3: Preprocess
            all_mappings: List[TimestampMapping] = []

            # Noise reduction
            if self.config.noise_reduction.enabled:
                self.logger.info("Reducing noise...")
                denoised = self._measure_stage(
                    metrics, "noise_reduction", lambda: self.preprocessor.reduce_stationary_noise(wav)
                )
            else:
                denoised = wav
                self._report_progress("noise reduction")
                metrics["stage_durations_s"]["noise_reduction"] = 0.0

            # Vocal separation
            if self.config.vocal_separation.enabled or self.config.vocal_separation.auto_detect:
                self.logger.info("Checking if vocal separation needed...")
                vocals = self._measure_stage(
                    metrics, "vocal_separation", lambda: self.separator.extract_vocals(denoised)
                )
            else:
                vocals = denoised
                self._report_progress("vocal separation")
                metrics["stage_durations_s"]["vocal_separation"] = 0.0

            # Normalization
            self.logger.info("Normalizing audio...")
            norm = self._measure_stage(
                metrics, "peak_normalization", lambda: self.preprocessor.normalize_audio(vocals)
            )
            loudnorm = self._measure_stage(
                metrics, "loudness_normalization", lambda: self.preprocessor.normalize_loudness(norm)
            )

            # Silence removal
            if self.config.preserve_timestamps:
                self.logger.info("Removing silence (preserving timestamps)...")
                silence_removed, silence_mappings = self._measure_stage(
                    metrics, "silence_removal", lambda: self.preprocessor.remove_silence(
                        loudnorm, preserve_timestamps=True
                    )
                )
                all_mappings.extend(silence_mappings)
            else:
                silence_removed, _ = self._measure_stage(
                    metrics, "silence_removal", lambda: self.preprocessor.remove_silence(loudnorm)
                )

            # Step 4: VAD
            if self.config.vad.enabled:
                self.logger.info(f"Applying VAD ({self.config.vad.provider})...")
                voiced_wav, vad_mappings = self._measure_stage(
                    metrics, "voice_activity_detection", lambda: self.vad.filter_voice(
                        silence_removed, self.results_dir
                    )
                )
                if self.config.preserve_timestamps:
                    all_mappings = self._compose_timestamp_mappings(all_mappings, vad_mappings)
            else:
                voiced_wav = silence_removed
                self._report_progress("voice activity detection")
                metrics["stage_durations_s"]["voice_activity_detection"] = 0.0

            metrics["media"]["post_silence_duration_s"] = round(get_audio_duration(silence_removed), 3)
            metrics["media"]["post_vad_duration_s"] = round(get_audio_duration(voiced_wav), 3)

            # Step 5: Transcribe
            self.logger.info(f"Transcribing ({self.config.transcription.backend})...")
            transcription_result = self._measure_stage(
                metrics, "transcription", lambda: self.transcriber.transcribe(voiced_wav)
            )
            raw_segments = transcription_result.get("segments", [])
            self._snapshot_segments = raw_segments
            metrics["segments"]["transcribed"] = len(raw_segments)
            self.logger.info(f"✓ Transcribed {len(raw_segments)} segments")
            self._save_transcription_snapshot(
                self._snapshot_writer, "transcription", self._snapshot_segments, metrics
            )

            # Step 6: Diarize
            if self.config.diarization.enabled:
                self.logger.info("Diarizing speakers...")
                diarization_segments = self._measure_stage(
                    metrics, "diarization", lambda: self.diarizer.diarize(
                        voiced_wav,
                        min_speakers=self.config.diarization.min_speakers,
                        max_speakers=self.config.diarization.max_speakers
                    )
                )
            else:
                diarization_segments = []
                self._report_progress("diarization")
                metrics["stage_durations_s"]["diarization"] = 0.0

            # Step 7: Align
            self.logger.info("Aligning transcription with speakers...")
            aligned_segments = self._measure_stage(
                metrics, "speaker_alignment", lambda: self._align_transcription_with_speakers(
                    raw_segments, diarization_segments
                )
            )
            self._snapshot_segments = aligned_segments
            metrics["segments"]["aligned"] = len(aligned_segments)
            self._save_transcription_snapshot(
                self._snapshot_writer, "speaker_alignment", self._snapshot_segments, metrics
            )

            # Step 8: Map timestamps
            self._report_progress("timestamp mapping")
            if self.config.preserve_timestamps and all_mappings:
                self.logger.info("Mapping timestamps to original audio...")
                mapping_started_at = time.perf_counter()
                for seg in aligned_segments:
                    original_start = self._map_timestamp_to_original(seg["start"], all_mappings)
                    original_end = self._map_timestamp_to_original(seg["end"], all_mappings)
                    if original_end < original_start:
                        self.logger.warning(
                            "Invalid timestamp mapping: processed=%s-%s, original=%s-%s",
                            seg["start"], seg["end"], original_start, original_end,
                        )
                        original_end = max(original_start, all_mappings[-1].original_end)
                    seg["original_start"] = original_start
                    seg["original_end"] = original_end
                metrics["stage_durations_s"]["timestamp_mapping"] = round(
                    time.perf_counter() - mapping_started_at, 3
                )
                metrics["timestamp_mappings"] = len(all_mappings)
            else:
                metrics["stage_durations_s"]["timestamp_mapping"] = 0.0
            self._save_transcription_snapshot(
                self._snapshot_writer, "timestamp_mapping", self._snapshot_segments, metrics
            )

            # This archival representation is the canonical transcript. It is
            # never passed through an LLM or a lossy redundancy filter.
            archived_segments = archival_segments(aligned_segments)
            self._snapshot_segments = archived_segments
            metrics["segments"]["archival"] = len(archived_segments)

            # Step 9: Remove redundancies
            self.logger.info("Removing redundant segments...")
            final_segments = self._measure_stage(
                metrics, "redundancy_removal", lambda: self.redundancy.remove(aligned_segments)
            )
            metrics["segments"]["after_redundancy_removal"] = len(final_segments)
            self.logger.info(f"✓ Final: {len(final_segments)} segments")
            self._snapshot_segments = final_segments
            self._save_transcription_snapshot(
                self._snapshot_writer, "redundancy_removal", self._snapshot_segments, metrics
            )

            # Step 10: Merge short segments if needed
            if self.config.segment_merging.enabled:
                self.logger.info("Merging short segments...")
                final_segments = self._measure_stage(
                    metrics, "segment_merging", lambda: self.segment_merger.merge(final_segments)
                )
            else:
                self._report_progress("segment merging")
                metrics["stage_durations_s"]["segment_merging"] = 0.0
            metrics["segments"]["final"] = len(final_segments)
            self._snapshot_segments = final_segments
            self._save_transcription_snapshot(
                self._snapshot_writer, "segment_merging", self._snapshot_segments, metrics
            )

            # Step 11: LLM Post-Processing
            # Unload transcription and diarization models first to free VRAM
            # before loading the LLM (critical on GPUs with ≤ 8 GB VRAM).
            self._report_progress("llm analysis")
            llm_analysis = None
            speaker_suggestions = None
            llm_started_at = time.perf_counter()
            if self.config.llm.enabled:
                if self.llm_processor and hasattr(self.llm_processor, "unload_model"):
                    self.llm_processor.unload_model()
                self.llm_processor = None
                if hasattr(self.transcriber, 'unload_model'):
                    self.logger.info("Unloading transcriber to free VRAM for LLM...")
                    self.transcriber.unload_model()
                if hasattr(self.diarizer, 'unload_model'):
                    self.logger.info("Unloading diarizer to free VRAM for LLM...")
                    self.diarizer.unload_model()
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        import gc; gc.collect()

                    if self.llm_processor_factory is None:
                        raise RuntimeError("No LLM processor factory was configured.")
                    self.llm_processor = self.llm_processor_factory()
                    info = self.llm_processor.get_backend_info()
                    self.logger.info(f"✓ LLM initialized: {info['backend']} ({info['model']})")
                except Exception as e:
                    self.logger.warning(f"LLM post-processing disabled: {e}")
                    self.llm_processor = None

            if self.llm_processor:
                try:
                    if self.config.llm.speaker_labeling_enabled:
                        self.logger.info("Suggesting speaker names from dialogue evidence...")
                        speaker_suggestions = self.llm_processor.suggest_speaker_names(
                            archived_segments
                        )
                        self._apply_speaker_name_suggestions(
                            archived_segments, speaker_suggestions
                        )

                    self.logger.info("Analyzing with LLM...")
                    full_text = " ".join([s["text"] for s in archived_segments])
                    llm_analysis = self.llm_processor.process(
                        full_text, source_path=str(media_file)
                    )

                    if "error" not in llm_analysis:
                        self.logger.info("✓ LLM analysis complete")
                        self.logger.info(f"  Summary: {llm_analysis['summary'][:80]}...")
                        self.logger.info(f"  Topics: {len(llm_analysis['topics'])}")
                        self.logger.info(f"  Actions: {len(llm_analysis['action_items'])}")
                    else:
                        self.logger.warning(f"LLM analysis failed: {llm_analysis['error']}")

                except Exception as e:
                    self.logger.warning(f"LLM processing failed: {e}")
                    llm_analysis = {"error": str(e)}

            metrics["stage_durations_s"]["llm_post_processing"] = round(
                time.perf_counter() - llm_started_at, 3
            )
            metrics["llm"] = {
                "enabled": self.config.llm.enabled,
                "backend": (
                    self.llm_processor.get_backend_info().get("backend")
                    if self.llm_processor else None
                ),
                "status": (
                    "success" if llm_analysis and "error" not in llm_analysis
                    else "failed" if llm_analysis else "skipped"
                ),
            }
            self._snapshot_segments = archived_segments
            self._save_transcription_snapshot(
                self._snapshot_writer,
                "llm_post_processing",
                self._snapshot_segments,
                metrics,
                llm_analysis=llm_analysis,
                speaker_suggestions=speaker_suggestions,
            )
            self._finalize_metrics(metrics, started_at)

            formatting = (
                llm_analysis.get("formatting")
                if llm_analysis and "error" not in llm_analysis else None
            )
            output_stem = contextual_output_stem(str(media_file), formatting)
            original_media_file = media_file
            if formatting and self.artifact_renamer:
                try:
                    media_file = self.artifact_renamer.rename_source_media(media_file, output_stem)
                    metrics["media"]["renamed_from"] = original_media_file
                    metrics["media"]["renamed_to"] = media_file
                except OSError as error:
                    self.logger.warning("Could not rename source media for indexing: %s", error)
                    metrics["media"]["rename_error"] = str(error)

                generated_artifacts = (
                    wav,
                    denoised,
                    vocals,
                    norm,
                    loudnorm,
                    silence_removed,
                    voiced_wav,
                )
                artifact_renames = []
                for artifact in dict.fromkeys(generated_artifacts):
                    try:
                        renamed_artifact = self.artifact_renamer.rename_derived_artifact(artifact, base, output_stem)
                        if renamed_artifact:
                            artifact_renames.append({"from": artifact, "to": renamed_artifact})
                    except OSError as error:
                        self.logger.warning("Could not rename generated artifact %s: %s", artifact, error)
                if artifact_renames:
                    metrics["media"]["generated_artifact_renames"] = artifact_renames

            # Step 12: Save results and metrics
            self._report_progress("saving results")
            output_data = {
                "metadata": {
                    "source_file": str(media_file),
                    "config": {
                        "model": self.config.transcription.model,
                        "language": self.config.transcription.language,
                        "vad_provider": self.config.vad.provider,
                        "transcription_backend": self.config.transcription.backend,
                    },
                    "metrics": metrics,
                },
                "segments": archived_segments,
                "documentation": {
                    "fidelity": "verbatim_with_whitespace_normalization",
                    "text": documentation_text(archived_segments),
                },
            }

            if self.llm_processor and self.config.llm.speaker_labeling_enabled:
                output_data["speaker_suggestions"] = speaker_suggestions

            if llm_analysis and "error" not in llm_analysis:
                output_data["llm_analysis"] = llm_analysis

            out_path = os.path.join(self.results_dir, f"{output_stem}_transcription.json")
            if os.path.exists(out_path):
                with open(out_path, "r", encoding="utf-8") as existing_file:
                    existing_source = json.load(existing_file).get("metadata", {}).get("source_file")
                if existing_source != str(media_file):
                    import hashlib
                    source_hash = hashlib.sha256(str(Path(media_file).resolve()).encode()).hexdigest()[:8]
                    out_path = os.path.join(self.results_dir, f"{output_stem}_{source_hash}_transcription.json")
            serialization_started_at = time.perf_counter()
            json.dumps(output_data, ensure_ascii=False, indent=2)
            metrics["stage_durations_s"]["output_serialization"] = round(
                time.perf_counter() - serialization_started_at, 3
            )
            self._finalize_metrics(metrics, started_at)
            with open(out_path, "w", encoding="utf-8") as file:
                json.dump(output_data, file, ensure_ascii=False, indent=2)

            self._save_transcription_snapshot(
                self._snapshot_writer,
                "saving_results",
                archived_segments,
                metrics,
                llm_analysis=llm_analysis,
                speaker_suggestions=speaker_suggestions,
                completed=True,
                final_output=out_path,
            )

            self.logger.info(f"✓ Saved transcription: {out_path}")

            return PipelineResult(
                success=True,
                input_file=str(media_file),
                output_file=out_path,
                segments=archived_segments,
                llm_analysis=llm_analysis,
                metadata={
                    "model": self.config.transcription.model,
                    "backend": self.config.transcription.backend,
                    "vad": self.config.vad.provider,
                    "llm_enabled": self.config.llm.enabled,
                    "metrics": metrics,
                }
            )

        except MediaNotFoundError as e:
            self.logger.error(f"Media not found: {e}")
            return PipelineResult(
                success=False,
                input_file=str(input_file) if input_file else "",
                output_file=None,
                segments=[],
                error=str(e),
                metadata={"metrics": self._finalize_metrics(metrics, started_at)}
            )

        except AudioPipelineError as e:
            self.logger.error(f"Pipeline error: {e}")
            return PipelineResult(
                success=False,
                input_file=str(input_file) if input_file else "",
                output_file=None,
                segments=[],
                error=str(e),
                metadata={"metrics": self._finalize_metrics(metrics, started_at)}
            )

        except Exception as e:
            self.logger.exception(f"Unexpected error: {e}")
            return PipelineResult(
                success=False,
                input_file=str(input_file) if input_file else "",
                output_file=None,
                segments=[],
                error=f"Unexpected error: {e}",
                metadata={"metrics": self._finalize_metrics(metrics, started_at)}
            )

    def run_transcription_only(self, input_wav: str) -> PipelineResult:
        """Run transcription only on a pre-processed WAV file."""
        try:
            result = self.transcriber.transcribe(input_wav)
            segments = result.get("segments", [])

            return PipelineResult(
                success=True,
                input_file=input_wav,
                output_file=None,
                segments=segments
            )
        except Exception as e:
            return PipelineResult(
                success=False,
                input_file=input_wav,
                output_file=None,
                segments=[],
                error=str(e)
            )

    def cleanup(self, remove_temp: bool = True, clear_checkpoints: bool = False) -> None:
        """Cleanup temporary files and unload models."""
        import shutil

        self.logger.info("Cleaning up...")

        # Unload models
        if hasattr(self.transcriber, 'unload_model'):
            self.transcriber.unload_model()
        if hasattr(self.diarizer, 'unload_model'):
            self.diarizer.unload_model()

        # Clear checkpoint cache only when explicitly requested. Checkpoints
        # live outside the temporary directory so they can support resuming.
        if clear_checkpoints and self.checkpoint_manager:
            self.checkpoint_manager.clear()

        # Remove temp directory
        if remove_temp and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
            self.logger.info(f"✓ Cleaned up temp directory: {self.temp_dir}")
