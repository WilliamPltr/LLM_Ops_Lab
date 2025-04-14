from pydantic import BaseModel
from typing import List, Optional


class Message(BaseModel):
    role: str  # "system", "user", "assistant"
    content: str


class ChatCompletionRequest(BaseModel):
    messages: List[Message]
    temperature: float = 0.7
    max_tokens: int = 100
    model: Optional[str] = "llama2"


class ChatCompletionResponse(BaseModel):
    content: str
