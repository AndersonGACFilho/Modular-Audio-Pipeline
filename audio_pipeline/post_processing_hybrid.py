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

from typing import List, Optional, Dict, Any, Literal
from pydantic import BaseModel, Field, ValidationError

from .profiles import PROFILE_INSTRUCTIONS, ProfileRouter
import os
import json
import logging
import re

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
        chunk_max_length: int = 512,
        disable_thinking: bool = True,
        # HuggingFace options
        local_model: Optional[str] = None,
        device: str = "auto",
        max_length: int = 2048,
        # Shared options
        temperature: float = 0.3,
        force_local: bool = False,
        lazy_load: bool = False,
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
        self.temperature = temperature
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
            with urllib.request.urlopen(req, timeout=3) as resp:
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

        except Exception as e:
            logger.debug(f"Ollama probe failed: {e}")
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

    def _process_openai(self, text: str) -> Dict[str, Any]:
        """Process with OpenAI API."""
        response = self.openai_client.chat.completions.create(
            model=self.openai_model,
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert meeting analyst. Always respond with valid JSON.",
                },
                {"role": "user", "content": self._build_prompt(text)},
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
            self.model = AutoModelForCausalLM.from_pretrained(
                self.local_model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map=self.device,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                attn_implementation="eager",
            )
            self.pipe = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                max_new_tokens=self.max_length,
                temperature=self.temperature,
                do_sample=True,
                top_p=0.9,
                repetition_penalty=1.1,
            )
            logger.info("✓ Local HF model loaded successfully")

        except ImportError as e:
            raise RuntimeError(
                f"transformers or torch not installed: {e}\n"
                "Install with: pip install transformers torch accelerate"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load local HF model: {e}")

    def _process_local(self, text: str) -> Dict[str, Any]:
        """Process with local HuggingFace model."""
        if self.pipe is None:
            logger.info("Lazy-loading local HF model now...")
            self._init_local()

        prompt = self._build_prompt(text)
        outputs = self.pipe(prompt, max_new_tokens=self.max_length)
        generated_text = outputs[0]["generated_text"]
        response = generated_text[len(prompt):].strip()
        return self._extract_json(response)

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _build_prompt(self, text: str) -> str:
        profile = getattr(self, "_active_profile", "generic_meeting")
        profile_instruction = PROFILE_INSTRUCTIONS[profile]
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

    def _process_chunk(self, text: str) -> Dict[str, Any]:
        """Analyze one bounded text chunk with the active backend."""
        if self.backend == "ollama":
            return self._process_ollama(text, max_length=self.chunk_max_length)
        if self.backend == "openai":
            return self._process_openai(text)
        return self._process_local(text)

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
            routing = self.profile_router.route(text, source_path, classifier)
            self._active_profile = routing.profile
            logger.info(
                "Selected formatting profile: %s (confidence %.2f)",
                routing.profile,
                routing.confidence,
            )

            chunks = self._split_text(text)
            if len(chunks) == 1:
                parsed = self._process_chunk(chunks[0])
            else:
                logger.info("Analyzing transcription in %d chunks", len(chunks))
                partials = [
                    MeetingAnalysis(**self._process_chunk(chunk)).model_dump()
                    for chunk in chunks
                ]
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
