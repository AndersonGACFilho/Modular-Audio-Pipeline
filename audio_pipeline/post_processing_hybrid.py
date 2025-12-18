"""
Hybrid LLM Post-Processor with OpenAI API + Local HuggingFace fallback.

Automatically uses local LLM if OpenAI API key is not available.
"""

from typing import List, Optional, Dict, Any, Literal
from pydantic import BaseModel, Field, ValidationError
import os
import json
import logging
import re

logger = logging.getLogger(__name__)

# Schema Definition
class ActionItem(BaseModel):
    description: str
    owner: Optional[str] = Field(None)
    priority: str = Field("Medium", description="High, Medium, Low")


class MeetingAnalysis(BaseModel):
    summary: str = Field(..., description="Executive summary of the transcription")
    topics: List[str] = Field(..., description="Main topics discussed")
    action_items: List[ActionItem] = Field(..., description="Extracted tasks")
    sentiment: str = Field(..., description="Overall tone: Positive, Neutral, or Negative")


class HybridLLMPostProcessor:
    """
    Smart LLM processor that automatically selects backend:
    - OpenAI API (if key available)
    - Local HuggingFace model (fallback)
    """
    
    # Recommended local models (ordered by performance/size)
    RECOMMENDED_MODELS = [
        "mistralai/Mistral-7B-Instruct-v0.2",      # Best quality
        "microsoft/Phi-3-mini-4k-instruct",        # Good balance
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",     # Fastest/smallest
    ]
    
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        local_model: Optional[str] = None,
        device: str = "auto",
        max_length: int = 2048,
        temperature: float = 0.3,
        force_local: bool = False
    ):
        """
        Initialize hybrid processor.
        
        Args:
            model: OpenAI model name
            local_model: HuggingFace model (auto-select if None)
            device: Device for local model ('cuda', 'cpu', or 'auto')
            max_length: Max tokens for local model
            temperature: Sampling temperature
            force_local: Force local model even if API key exists
        """
        self.openai_model = model
        self.local_model_name = local_model
        self.device = device
        self.max_length = max_length
        self.temperature = temperature
        
        # Determine backend
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.backend: Literal["openai", "local"] = "local"
        
        if self.api_key and not force_local:
            self.backend = "openai"
            logger.info(f"Using OpenAI API: {self.openai_model}")
            self._init_openai()
        else:
            self.backend = "local"
            logger.info("OpenAI API key not found, using local HuggingFace model")
            self._init_local()
    
    def _init_openai(self):
        """Initialize OpenAI client."""
        try:
            from openai import OpenAI
            from openai.types.chat import (
                ChatCompletionSystemMessageParam,
                ChatCompletionUserMessageParam
            )
            
            self.openai_client = OpenAI(api_key=self.api_key)
            self.ChatCompletionSystemMessageParam = ChatCompletionSystemMessageParam
            self.ChatCompletionUserMessageParam = ChatCompletionUserMessageParam
            logger.info("✓ OpenAI client initialized")
            
        except ImportError:
            logger.warning("openai package not installed, falling back to local")
            self.backend = "local"
            self._init_local()
    
    def _init_local(self):
        """Initialize local HuggingFace model."""
        try:
            import torch
            from transformers import (
                AutoTokenizer,
                AutoModelForCausalLM,
                pipeline
            )
            
            # Auto-select device
            if self.device == "auto":
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            
            logger.info(f"Device: {self.device}")
            
            # Auto-select model if not specified
            if not self.local_model_name:
                self.local_model_name = self._select_best_model()
            
            logger.info(f"Loading local model: {self.local_model_name}")
            logger.info("(This may take a few minutes on first run...)")
            
            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.local_model_name,
                trust_remote_code=True
            )
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.local_model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map=self.device,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )

            # Create pipeline
            self.pipe = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                max_new_tokens=self.max_length,
                temperature=self.temperature,
                do_sample=True,
                top_p=0.9,
                repetition_penalty=1.1
            )

            logger.info("✓ Local model loaded successfully")
            
        except ImportError as e:
            raise RuntimeError(
                f"transformers or torch not installed: {e}\n"
                "Install with: pip install transformers torch accelerate"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load local model: {e}")
    
    def _select_best_model(self) -> str:
        """Auto-select best available model based on system resources."""
        import torch
        
        # Check VRAM if CUDA available
        if torch.cuda.is_available():
            vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logger.info(f"Detected VRAM: {vram_gb:.1f}GB")
            
            if vram_gb >= 16:
                return self.RECOMMENDED_MODELS[0]  # Mistral-7B
            elif vram_gb >= 8:
                return self.RECOMMENDED_MODELS[1]  # Phi-3-mini
            else:
                return self.RECOMMENDED_MODELS[2]  # TinyLlama
        else:
            # CPU - use smallest model
            logger.info("No CUDA detected, using CPU with smallest model")
            return self.RECOMMENDED_MODELS[2]  # TinyLlama
    
    def _build_prompt(self, text: str) -> str:
        """Build analysis prompt."""
        return f"""You are an expert meeting analyst. Analyze the following transcription and extract key information.

Return your analysis in valid JSON format with these exact keys:
- "summary": A brief executive summary (2-3 sentences)
- "topics": A list of main topics discussed
- "action_items": A list of tasks, each with "description", "owner" (can be null), and "priority" (High/Medium/Low)
- "sentiment": Overall tone (Positive, Neutral, or Negative)

Transcription:
{text}

JSON Analysis:"""
    
    def _extract_json(self, text: str) -> Dict[str, Any]:
        """Extract JSON from LLM response (handles markdown code blocks)."""
        # Try to find JSON in markdown code blocks
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
        if json_match:
            text = json_match.group(1)
        
        # Try to find raw JSON
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            text = json_match.group(0)
        
        # Parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Fallback: try to extract fields manually
            logger.warning("Failed to parse JSON, attempting manual extraction")
            return self._manual_extract(text)
    
    def _manual_extract(self, text: str) -> Dict[str, Any]:
        """Fallback: manually extract fields from unstructured text."""
        result = {
            "summary": "Unable to parse summary",
            "topics": [],
            "action_items": [],
            "sentiment": "Neutral"
        }
        
        # Try to extract summary
        summary_match = re.search(r'summary["\']?\s*:\s*["\']([^"\']+)["\']', text, re.I)
        if summary_match:
            result["summary"] = summary_match.group(1)
        
        # Try to extract topics
        topics_match = re.search(r'topics["\']?\s*:\s*\[(.*?)\]', text, re.I | re.DOTALL)
        if topics_match:
            topics_str = topics_match.group(1)
            result["topics"] = [t.strip(' "\'') for t in topics_str.split(',')]
        
        # Try to extract sentiment
        sentiment_match = re.search(r'sentiment["\']?\s*:\s*["\']?(\w+)["\']?', text, re.I)
        if sentiment_match:
            result["sentiment"] = sentiment_match.group(1)
        
        return result
    
    def _process_openai(self, text: str) -> Dict[str, Any]:
        """Process with OpenAI API."""
        system_message: self.ChatCompletionSystemMessageParam = {
            "role": "system",
            "content": "You are an expert meeting analyst. Always respond with valid JSON."
        }
        
        user_message: self.ChatCompletionUserMessageParam = {
            "role": "user",
            "content": self._build_prompt(text)
        }
        
        response = self.openai_client.chat.completions.create(
            model=self.openai_model,
            messages=[system_message, user_message],
            response_format={"type": "json_object"},
            temperature=self.temperature
        )
        
        content = response.choices[0].message.content
        if not content:
            raise ValueError("Empty response from OpenAI")
        
        return json.loads(content)
    
    def _process_local(self, text: str) -> Dict[str, Any]:
        """Process with local HuggingFace model."""
        prompt = self._build_prompt(text)
        
        # Generate
        outputs = self.pipe(prompt, max_new_tokens=self.max_length)
        generated_text = outputs[0]['generated_text']
        
        # Extract response (remove prompt)
        response = generated_text[len(prompt):].strip()
        
        # Extract and parse JSON
        return self._extract_json(response)
    
    def process(self, text: str) -> Dict[str, Any]:
        """
        Process transcription with automatic backend selection.
        
        Args:
            text: Transcription text
            
        Returns:
            Dict with analysis results
        """
        try:
            logger.info(f"Processing with {self.backend} backend...")
            
            if self.backend == "openai":
                parsed = self._process_openai(text)
            else:
                parsed = self._process_local(text)
            
            # Validate with Pydantic
            validated = MeetingAnalysis(**parsed)
            
            logger.info(f"✓ Analysis complete ({self.backend})")
            return validated.model_dump()
            
        except ValidationError as e:
            logger.error(f"Validation failed: {e}")
            return {
                "error": f"Invalid response format: {str(e)}",
                "backend": self.backend
            }
        except Exception as e:
            logger.error(f"Processing failed: {e}", exc_info=True)
            return {
                "error": str(e),
                "backend": self.backend
            }
    
    def get_backend_info(self) -> Dict[str, Any]:
        """Get information about current backend."""
        info = {
            "backend": self.backend,
            "device": self.device if self.backend == "local" else "cloud"
        }
        
        if self.backend == "openai":
            info["model"] = self.openai_model
        else:
            info["model"] = self.local_model_name
            
            import torch
            if torch.cuda.is_available():
                info["vram_gb"] = torch.cuda.get_device_properties(0).total_memory / 1024**3
        
        return info


# Backward compatibility alias
LLMPostProcessor = HybridLLMPostProcessor
