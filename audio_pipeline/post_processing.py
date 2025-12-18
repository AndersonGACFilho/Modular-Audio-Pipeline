from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
import instructor
from openai import OpenAI
from openai.types.chat import (
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam
)
import os
import logging

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
    sentiment: str = Field(..., description="Overall tone")


class LLMPostProcessor:
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o"):
        key = api_key or os.getenv("OPENAI_API_KEY")
        if not key:
            raise ValueError("API Key required for LLM Post Processor")

        self.client = instructor.from_openai(OpenAI(api_key=key))
        self.model = model

    def process(self, text: str) -> Dict[str, Any]:
        try:
            logger.info("Starting LLM analysis...")

            # Properly typed messages for OpenAI API + instructor compatibility
            system_message: ChatCompletionSystemMessageParam = {
                "role": "system",
                "content": "You are an expert meeting analyst."
            }

            user_message: ChatCompletionUserMessageParam = {
                "role": "user",
                "content": f"Analyze the following meeting transcription:\n\n{text}"
            }

            resp = self.client.chat.completions.create(
                model=self.model,
                response_model=MeetingAnalysis,
                messages=[system_message, user_message],
            )
            return resp.model_dump()
        except Exception as e:
            logger.error(f"LLM processing failed: {e}")
            return {"error": str(e)}