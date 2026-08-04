"""
Hybrid LLM Post-Processor.

Backend priority (auto-selected):
  1. Ollama  - if running locally at OLLAMA_HOST (default: http://localhost:11434)
  2. OpenAI  - if OPENAI_API_KEY is set
  3. Local HuggingFace model - always available as final fallback

Override via config:
  llm.use_ollama   = true/false   (default: true  - probe automatically)
  llm.ollama_model = "llama3"     (default: first available model on the server)
  llm.use_openai   = true/false
  llm.openai_model = "gpt-4o-mini"
  llm.local_model  = null         (auto-select by VRAM) or HuggingFace model id
"""

from __future__ import annotations

from typing import Callable, List, Optional, Dict, Any, Literal
from pydantic import BaseModel, Field, ValidationError

from ...config.profiles import PROFILE_INSTRUCTIONS, ProfileRouter
import os
import json
import logging
import re
import threading
import time
from contextlib import contextmanager

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class ActionItem(BaseModel):
    description: str
    owner: Optional[str] = Field(None)
    priority: str = Field("Medium", description="High, Medium, Low")


class MeetingAnalysis(BaseModel):
    summary: str = Field(..., description="Executive summary of the transcription")
    topics: List[str] = Field(..., description="Main topics discussed")
    action_items: List[ActionItem] = Field(..., description="Extracted tasks")
    sentiment: str = Field(..., description="Overall tone: Positive, Neutral, or Negative")
    profile_data: Dict[str, Any] = Field(
        default_factory=dict,
        description="Profile-specific structured data extracted from the meeting",
    )


class SpeakerNameSuggestion(BaseModel):
    """A text-based, non-biometric suggestion for one diarized speaker."""

    speaker: str
    suggested_name: Optional[str] = None
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: List[str] = Field(default_factory=list)


class SpeakerNameSuggestions(BaseModel):
    suggestions: List[SpeakerNameSuggestion] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class HybridLLMPostProcessor:
    """
    Smart LLM processor that automatically selects the best available backend:
      1. Ollama  (local server - zero in-process VRAM cost for the pipeline)
      2. OpenAI  (cloud API)
      3. Local HuggingFace model (downloaded on demand)
    """

    # HuggingFace fallback models ordered by quality/size
    RECOMMENDED_HF_MODELS = [
        "mistralai/Mistral-7B-Instruct-v0.2",   # Best quality
        "microsoft/Phi-3-mini-4k-instruct",      # Good balance
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",   # Fastest/smallest
    ]

    # Preferred Ollama models to try when no model is specified (in order)
    PREFERRED_OLLAMA_MODELS = [
        "llama3", "llama3.1", "llama3.2",
        "mistral", "phi3", "gemma2", "qwen2",
    ]
    LLM_HEARTBEAT_SECONDS = 30

    def __init__(
        self,
        # OpenAI options
        model: str = "gpt-4o-mini",
        # Ollama options
        ollama_host: Optional[str] = None,
        ollama_model: Optional[str] = None,
        use_ollama: bool = True,
        use_openai: bool = True,
        ollama_num_ctx: int = 8192,
        ollama_keep_alive: str | int = "5m",
        request_timeout_s: int = 900,
        chunk_size_chars: int = 6_000,
        chunk_max_length: int = 384,
        disable_thinking: bool = True,
        # HuggingFace options
        local_model: Optional[str] = None,
        device: str = "auto",
        max_length: int = 2048,
        local_max_new_tokens: int = 384,
        local_attention_implementation: str = "sdpa",
        # Shared options
        temperature: float = 0.3,
        force_local: bool = False,
        lazy_load: bool = False,
        profile_id: Optional[str] = None,
        profile_prompt: Optional[str] = None,
        output_language: str = "pt-BR",
        progress_callback: Optional[Callable[[str], None]] = None,
    ):
        """
        Initialize hybrid processor.

        Args:
            model:        OpenAI model name.
            ollama_host:  Ollama base URL. Defaults to OLLAMA_HOST env var or
                          http://localhost:11434.
            ollama_model: Ollama model name. Auto-detected if None.
            use_ollama:   Probe for Ollama before trying other backends.
            local_model:  HuggingFace model id. Auto-selected by VRAM if None.
            device:       'cuda', 'cpu', or 'auto'.
            max_length:   Max new tokens for generation.
            temperature:  Sampling temperature.
            force_local:  Skip Ollama and OpenAI; go straight to HuggingFace.
            lazy_load:    Defer HuggingFace model loading until process() is called.
        """
        self.openai_model = model
        self.ollama_host = (
            ollama_host or os.getenv("OLLAMA_HOST", "http://localhost:11434")
        ).rstrip("/")
        self.ollama_model = ollama_model

        self.use_ollama = use_ollama
        self.ollama_num_ctx = ollama_num_ctx
        self.ollama_keep_alive = ollama_keep_alive
        self.request_timeout_s = request_timeout_s
        self.chunk_size_chars = chunk_size_chars
        self.chunk_max_length = chunk_max_length
        self.disable_thinking = disable_thinking

        self.use_openai = use_openai

        self.local_model_name = local_model
        self.device = device
        self.max_length = max_length
        self.local_max_new_tokens = local_max_new_tokens
        self.local_attention_implementation = local_attention_implementation
        self.temperature = temperature
        self.profile_id = profile_id
        self.profile_prompt = profile_prompt
        self.output_language = output_language
        self.progress_callback = progress_callback
        self._lazy_load = lazy_load
        self.profile_router = ProfileRouter()

        self.api_key = os.getenv("OPENAI_API_KEY")
        self.backend: Literal["ollama", "openai", "local"] = "local"

        # --- Backend selection ---
        if not force_local and use_ollama:
            detected = self._detect_ollama()
            if detected:
                self.backend = "ollama"
                self.ollama_model = detected
                logger.info(f"✓ Ollama available → model: {self.ollama_model}")
                return

        if not force_local and use_openai and self.api_key:
            self.backend = "openai"
            logger.info(f"Using OpenAI API: {self.openai_model}")
            self._init_openai()
            return

        # Fallback: HuggingFace
        self.backend = "local"
        logger.info("Using local HuggingFace model (fallback)")
        if not lazy_load:
            self._init_local()
        else:
            import torch
            if self.device == "auto":
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            if not self.local_model_name:
                self.local_model_name = self._select_best_hf_model()
            logger.info(f"Local HF model selected (lazy): {self.local_model_name}")
            self.pipe = None
            self.tokenizer = None
            self.model = None

    # ------------------------------------------------------------------
    # Ollama
    # ------------------------------------------------------------------

    def _detect_ollama(self) -> Optional[str]:
        """
        Probe the Ollama server at self.ollama_host.

        Returns the model name to use, or None if Ollama is unreachable / empty.
        """
        try:
            import urllib.request

            req = urllib.request.Request(
                f"{self.ollama_host}/api/tags", method="GET"
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode())

            available: List[str] = [
                m["name"].split(":")[0] for m in data.get("models", [])
            ]

            if not available:
                logger.info("Ollama is running but has no models pulled yet.")
                return None

            logger.info(f"Ollama available - models: {available}")

            # Honour caller-specified model
            if self.ollama_model:
                base = self.ollama_model.split(":")[0]
                if base in available or self.ollama_model in available:
                    return self.ollama_model
                logger.warning(
                    f"Requested Ollama model '{self.ollama_model}' not found on server. "
                    "Auto-selecting from available models."
                )

            # Prefer known-good models
            for preferred in self.PREFERRED_OLLAMA_MODELS:
                if preferred in available:
                    return preferred

            # Fall back to whatever is installed
            return available[0]

        except Exception as error:
            logger.warning(
                "Ollama unavailable at %s (%s). Falling back to the next configured LLM backend.",
                self.ollama_host,
                error,
            )
            return None

    def _process_ollama(
        self,
        text: str,
        max_length: Optional[int] = None,
        user_prompt: Optional[str] = None,
        response_model: type[BaseModel] = MeetingAnalysis,
    ) -> Dict[str, Any]:
        """Send request to Ollama /api/chat."""
        import urllib.request

        payload = {
            "model": self.ollama_model,
            "stream": False,
            "think": not self.disable_thinking,
            "keep_alive": self.ollama_keep_alive,
            "format": response_model.model_json_schema(),
            "options": {
                "temperature": self.temperature,
                "num_predict": max_length or self.max_length,
                "num_ctx": self.ollama_num_ctx,
            },
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are an expert meeting analyst. "
                        "Always respond with valid JSON and nothing else."
                    ),
                },
                {"role": "user", "content": user_prompt or self._build_prompt(text)},
            ],
        }

        data = json.dumps(payload).encode()
        req = urllib.request.Request(
            f"{self.ollama_host}/api/chat",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        with urllib.request.urlopen(req, timeout=self.request_timeout_s) as resp:
            result = json.loads(resp.read().decode())

        content = result.get("message", {}).get("content", "")
        if not content:
            raise ValueError("Empty response from Ollama")

        return self._extract_json(content)

    # ------------------------------------------------------------------
    # OpenAI
    # ------------------------------------------------------------------

    def _init_openai(self):
        """Initialize OpenAI client."""
        try:
            from openai import OpenAI
            self.openai_client = OpenAI(api_key=self.api_key)
            logger.info("✓ OpenAI client initialized")
        except ImportError:
            logger.warning("openai package not installed, falling back to local HF model")
            self.backend = "local"
            self._init_local()

    def _process_openai(self, text: str, user_prompt: Optional[str] = None) -> Dict[str, Any]:
        """Process with OpenAI API."""
        response = self.openai_client.chat.completions.create(
            model=self.openai_model,
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert meeting analyst. Always respond with valid JSON.",
                },
                {"role": "user", "content": user_prompt or self._build_prompt(text)},
            ],
            temperature=self.temperature,
        )
        content = response.choices[0].message.content
        if not content:
            raise ValueError("Empty response from OpenAI")
        return json.loads(content)

    # ------------------------------------------------------------------
    # HuggingFace local
    # ------------------------------------------------------------------

    def _select_best_hf_model(self) -> str:
        """Auto-select HuggingFace model based on available VRAM."""
        import torch

        if torch.cuda.is_available():
            vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
            logger.info(f"Detected VRAM: {vram_gb:.1f}GB")
            if vram_gb >= 15:
                return self.RECOMMENDED_HF_MODELS[0]
            elif vram_gb >= 7:
                return self.RECOMMENDED_HF_MODELS[1]
            else:
                return self.RECOMMENDED_HF_MODELS[2]

        logger.info("No CUDA - using smallest HF model on CPU")
        return self.RECOMMENDED_HF_MODELS[2]

    def _init_local(self):
        """Load HuggingFace model into memory."""
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
            from transformers.utils.logging import disable_progress_bar

            # Rich owns the CLI status line. Suppress Transformers' separate
            # tqdm bar so it cannot redraw or duplicate that footer.
            disable_progress_bar()

            if self.device == "auto":
                self.device = "cuda" if torch.cuda.is_available() else "cpu"

            logger.info(f"Device: {self.device}")

            if not self.local_model_name:
                self.local_model_name = self._select_best_hf_model()

            logger.info(f"Loading local model: {self.local_model_name}")
            logger.info("(This may take a few minutes on first run...)")

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.local_model_name, trust_remote_code=True
            )
            model_options = {
                "torch_dtype": torch.float16 if self.device == "cuda" else torch.float32,
                "device_map": self.device,
                "trust_remote_code": True,
                "low_cpu_mem_usage": True,
                "attn_implementation": self.local_attention_implementation,
            }
            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.local_model_name, **model_options
                )
            except ValueError as error:
                if self.local_attention_implementation != "sdpa":
                    raise
                logger.warning("SDPA is unavailable for %s (%s); using eager attention.", self.local_model_name, error)
                model_options["attn_implementation"] = "eager"
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.local_model_name, **model_options
                )
            self.pipe = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
            )
            logger.info(
                "✓ Local HF model loaded successfully (attention=%s, max_new_tokens=%d)",
                getattr(self.model.config, "_attn_implementation", model_options["attn_implementation"]),
                self.local_max_new_tokens,
            )

        except ImportError as e:
            raise RuntimeError(
                f"transformers or torch not installed: {e}\n"
                "Install with: pip install transformers torch accelerate"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load local HF model: {e}")

    def _process_local(
        self, text: str, user_prompt: Optional[str] = None, max_new_tokens: Optional[int] = None
    ) -> Dict[str, Any]:
        """Process with local HuggingFace model."""
        if self.pipe is None:
            logger.info("Lazy-loading local HF model now...")
            self._init_local()

        prompt = user_prompt or self._build_prompt(text)
        requested_tokens = max_new_tokens or self.local_max_new_tokens
        input_tokens = len(self.tokenizer(prompt, add_special_tokens=False)["input_ids"])
        try:
            import torch
            if self.device == "cuda":
                torch.cuda.synchronize()
        except Exception:
            torch = None
        started_at = time.perf_counter()
        outputs = self.pipe(
            prompt,
            max_new_tokens=requested_tokens,
            do_sample=False,
            num_beams=1,
            use_cache=True,
        )
        if "torch" in locals() and torch is not None and self.device == "cuda":
            torch.cuda.synchronize()
        generated_text = outputs[0]["generated_text"]
        response = generated_text[len(prompt):].strip()
        elapsed = time.perf_counter() - started_at
        generated_tokens = len(self.tokenizer(response, add_special_tokens=False)["input_ids"])
        logger.info(
            "Local LLM generation: input_tokens=%d generated_tokens=%d elapsed=%.2fs tokens_per_second=%.2f",
            input_tokens, generated_tokens, elapsed, generated_tokens / elapsed if elapsed else 0.0,
        )
        return self._extract_json(response)

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _build_prompt(self, text: str) -> str:
        profile = getattr(self, "_active_profile", "generic_meeting")
        profile_instruction = getattr(self, "_active_profile_instruction", PROFILE_INSTRUCTIONS[profile])
        return (
            "You are an expert meeting analyst. Analyze the following transcription "
            "and extract key information.\n\n"
            "Return your analysis in valid JSON format with these exact keys:\n"
            '- "summary": A brief executive summary (2-3 sentences)\n'
            '- "topics": A list of main topics discussed\n'
            '- "action_items": A list of tasks, each with "description", '
            '"owner" (can be null), and "priority" (High/Medium/Low)\n'
            '- "sentiment": Overall tone (Positive, Neutral, or Negative)\n\n'
            f"Formatting profile: {profile}\n"
            f"Profile instructions: {profile_instruction}\n"
            f"Write all generated fields in: {self.output_language}\n"
            "Put profile-specific fields in `profile_data` using clear snake_case keys.\n\n"
            f"Transcription:\n{text}\n\n"
            "JSON Analysis:"
        )

    def _extract_json(self, text: str) -> Dict[str, Any]:
        """Extract JSON from LLM response (handles markdown code blocks)."""
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if match:
            text = match.group(1)
        else:
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                text = match.group(0)

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            logger.warning("Failed to parse JSON, attempting manual extraction")
            return self._manual_extract(text)

    def _manual_extract(self, text: str) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "summary": "Unable to parse summary",
            "topics": [],
            "action_items": [],
            "sentiment": "Neutral",
        }
        m = re.search(r'summary["\']?\s*:\s*["\']([^"\']+)["\']', text, re.I)
        if m:
            result["summary"] = m.group(1)
        m = re.search(r'topics["\']?\s*:\s*\[(.*?)\]', text, re.I | re.DOTALL)
        if m:
            result["topics"] = [t.strip(' "\'') for t in m.group(1).split(",")]
        m = re.search(r'sentiment["\']?\s*:\s*["\']?(\w+)["\']?', text, re.I)
        if m:
            result["sentiment"] = m.group(1)
        return result

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def _split_text(self, text: str) -> List[str]:
        """Split long text at word boundaries for bounded LLM requests."""
        if len(text) <= self.chunk_size_chars:
            return [text]

        chunks = []
        remaining = text.strip()
        while remaining:
            if len(remaining) <= self.chunk_size_chars:
                chunks.append(remaining)
                break
            split_at = remaining.rfind(" ", 0, self.chunk_size_chars)
            if split_at <= 0:
                split_at = self.chunk_size_chars
            chunks.append(remaining[:split_at])
            remaining = remaining[split_at:].lstrip()
        return chunks

    def _report_progress(self, detail: str) -> None:
        callback = getattr(self, "progress_callback", None)
        if callback:
            callback(detail)

    @contextmanager
    def _llm_activity(self, operation: str):
        """Log periodic proof of life while one LLM request is in flight."""
        started_at = time.perf_counter()
        backend = getattr(self, "backend", "unknown")
        model = (
            getattr(self, "ollama_model", None)
            if backend == "ollama"
            else getattr(self, "openai_model", None)
            if backend == "openai"
            else getattr(self, "local_model_name", None)
        ) or "default"
        stopped = threading.Event()

        def heartbeat() -> None:
            while not stopped.wait(self.LLM_HEARTBEAT_SECONDS):
                elapsed = time.perf_counter() - started_at
                logger.info(
                    "LLM still processing %s after %.0fs (%s: %s)",
                    operation, elapsed, backend, model,
                )
                self._report_progress(f"LLM {operation} ({elapsed:.0f}s)")

        logger.info("LLM started %s (%s: %s)", operation, backend, model)
        self._report_progress(f"LLM {operation}")
        worker = threading.Thread(target=heartbeat, name="llm-heartbeat", daemon=True)
        worker.start()
        try:
            yield
        except Exception:
            logger.warning("LLM %s failed after %.1fs", operation, time.perf_counter() - started_at)
            raise
        else:
            logger.info("LLM completed %s in %.1fs", operation, time.perf_counter() - started_at)
        finally:
            stopped.set()
            worker.join(timeout=1)

    def _process_chunk(self, text: str) -> Dict[str, Any]:
        """Analyze one bounded text chunk with the active backend."""
        with self._llm_activity("analysis request"):
            if self.backend == "ollama":
                return self._process_ollama(text, max_length=self.chunk_max_length)
            if self.backend == "openai":
                return self._process_openai(text)
            return self._process_local(text, max_new_tokens=self.chunk_max_length)

    def _classify_with_ollama(
        self, prompt: str, response_model: type[BaseModel]
    ) -> Dict[str, Any]:
        """Adapter that lets ProfileRouter use Ollama without knowing its API."""
        return self._process_ollama(
            "",
            max_length=180,
            user_prompt=prompt,
            response_model=response_model,
        )

    def _request_structured(
        self,
        prompt: str,
        response_model: type[BaseModel],
        max_length: int,
    ) -> Dict[str, Any]:
        """Run a schema-constrained prompt through the selected LLM backend."""
        with self._llm_activity("structured request"):
            if self.backend == "ollama":
                return self._process_ollama(
                    "", max_length=max_length, user_prompt=prompt,
                    response_model=response_model,
                )
            if self.backend == "openai":
                return self._process_openai("", user_prompt=prompt)
            return self._process_local("", user_prompt=prompt, max_new_tokens=max_length)

    def suggest_speaker_names(self, segments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Suggest names from explicit dialogue evidence without changing speaker IDs."""
        excerpts: Dict[str, List[str]] = {}
        for segment in segments:
            speaker = str(segment.get("speaker", "Unknown"))
            text = str(segment.get("text", "")).strip()
            if text and speaker != "Unknown":
                excerpts.setdefault(speaker, []).append(text)

        if not excerpts:
            return {"suggestions": []}

        dialogue = "\n".join(
            f"{speaker}: {' '.join(texts)[:2000]}"
            for speaker, texts in sorted(excerpts.items())
        )
        prompt = (
            "You label diarized speakers using only explicit evidence in this dialogue. "
            "A name is allowed only when the speaker introduces themself or another "
            "speaker directly addresses or identifies them. Do not infer names from role, "
            "voice, writing style, or guesses. For unsupported labels, return null with "
            "confidence 0 and no evidence. Evidence must quote or closely paraphrase the "
            "supporting dialogue. Return only JSON matching the required schema.\n\n"
            f"Dialogue:\n{dialogue}"
        )
        try:
            result = SpeakerNameSuggestions(**self._request_structured(
                prompt, SpeakerNameSuggestions, max_length=600,
            )).model_dump()
        except (ValidationError, ValueError) as error:
            logger.warning("Speaker-name suggestion skipped: %s", error)
            return {"suggestions": []}

        known_speakers = set(excerpts)
        result["suggestions"] = [
            suggestion for suggestion in result["suggestions"]
            if suggestion["speaker"] in known_speakers
        ]
        logger.info("Generated %d text-based speaker-name suggestions", len(result["suggestions"]))
        return result

    def _consolidate_chunks(self, analyses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Merge chunk analyses into one final meeting analysis."""
        source = json.dumps(analyses, ensure_ascii=False)
        prompt = (
            "Consolidate the following partial meeting analyses into one final "
            "analysis. Deduplicate topics and action items, preserve owners and "
            "priorities, and return only the required JSON schema.\n\n"
            f"Partial analyses:\n{source}"
        )
        if self.backend == "ollama":
            return self._process_ollama(prompt, max_length=self.max_length)
        if self.backend == "openai":
            return self._process_openai(prompt)
        return self._process_local(prompt)

    def process(self, text: str, source_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze transcription text and return structured meeting analysis.

        Returns:
            Dict with keys: summary, topics, action_items, sentiment
        """
        try:
            logger.info(f"Processing with backend: {self.backend}")
            classifier = (
                self._classify_with_ollama if self.backend == "ollama" else None
            )
            profile_id = getattr(self, "profile_id", None)
            profile_prompt = getattr(self, "profile_prompt", None)
            if profile_id or profile_prompt:
                profile = profile_id or "generic_meeting"
                routing = ProfileRouting(
                    profile=profile,
                    confidence=1.0,
                    reasoning="Selected from immutable job analysis options",
                )
                self._active_profile_instruction = profile_prompt or PROFILE_INSTRUCTIONS.get(
                    profile, PROFILE_INSTRUCTIONS["generic_meeting"]
                )
            else:
                routing = self.profile_router.route(text, source_path, classifier)
                self._active_profile_instruction = PROFILE_INSTRUCTIONS[routing.profile]
            self._active_profile = routing.profile
            logger.info(
                "Selected formatting profile: %s (confidence %.2f)",
                routing.profile,
                routing.confidence,
            )

            chunks = self._split_text(text)
            if len(chunks) == 1:
                self._report_progress("LLM analysis - generating response")
                parsed = self._process_chunk(chunks[0])
            else:
                logger.info("Analyzing transcription in %d chunks", len(chunks))
                partials = []
                for index, chunk in enumerate(chunks, start=1):
                    self._report_progress(f"LLM analysis - chunk {index}/{len(chunks)}")
                    logger.info("Analyzing LLM chunk %d/%d (%d characters)", index, len(chunks), len(chunk))
                    started_at = time.perf_counter()
                    partials.append(MeetingAnalysis(**self._process_chunk(chunk)).model_dump())
                    logger.info("LLM chunk %d/%d completed in %.1fs", index, len(chunks), time.perf_counter() - started_at)
                self._report_progress(f"LLM analysis - consolidating {len(partials)} chunks")
                logger.info("Consolidating %d partial LLM analyses", len(partials))
                parsed = self._consolidate_chunks(partials)

            validated = MeetingAnalysis(**parsed)
            logger.info(f"✓ Analysis complete ({self.backend})")
            result = validated.model_dump()
            result["formatting"] = routing.model_dump()
            return result

        except ValidationError as e:
            logger.error(f"Validation failed: {e}")
            return {"error": f"Invalid response format: {str(e)}", "backend": self.backend}
        except Exception as e:
            logger.error(f"Processing failed: {e}", exc_info=True)
            return {"error": str(e), "backend": self.backend}

    def unload_model(self) -> None:
        """Release the in-process Hugging Face fallback model, if loaded."""
        if self.backend != "local":
            return
        for attribute in ("pipe", "tokenizer", "model"):
            if hasattr(self, attribute):
                setattr(self, attribute, None)
                import gc
                gc.collect()
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    logger.debug("Unable to clear CUDA cache after unloading local LLM", exc_info=True)

    def get_backend_info(self) -> Dict[str, Any]:
        """Return a summary of the active backend configuration."""
        info: Dict[str, Any] = {"backend": self.backend}

        if self.backend == "ollama":
            info["model"] = self.ollama_model
            info["host"] = self.ollama_host
            info["device"] = "local (Ollama)"
        elif self.backend == "openai":
            info["model"] = self.openai_model
            info["device"] = "cloud"
        else:
            info["model"] = self.local_model_name
            info["device"] = self.device
            try:
                import torch
                if torch.cuda.is_available():
                    info["vram_gb"] = round(
                        torch.cuda.get_device_properties(0).total_memory / 1024 ** 3, 1
                    )
            except Exception:
                pass

        return info


# Backward compatibility alias
LLMPostProcessor = HybridLLMPostProcessor
