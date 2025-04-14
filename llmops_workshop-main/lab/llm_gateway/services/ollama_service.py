# llm_gateway/services/ollama_service.py
import httpx
from core.models import ChatCompletionRequest, ChatCompletionResponse
from services.base import BaseLLMService


class OllamaService(BaseLLMService):
    def __init__(self, host: str = "http://localhost:11434"):
        self.host = host

    async def get_chat_completion(
        self, request: ChatCompletionRequest
    ) -> ChatCompletionResponse:
        prompt_parts = []
        for msg in request.messages:
            if msg.role == "system":
                prompt_parts.append(f"[SYSTEM] {msg.content}")
            elif msg.role == "user":
                prompt_parts.append(f"[USER] {msg.content}")
            else:  # "assistant"
                prompt_parts.append(f"[ASSISTANT] {msg.content}")

        full_prompt = "\n".join(prompt_parts)

        payload = {
            "prompt": full_prompt,
            "temperature": request.temperature,
            "num_ctx": request.max_tokens,  # selon param Ollama
            # etc. (en fonction de la doc Ollama)
        }

        async with httpx.AsyncClient() as client:
            resp = await client.post(f"{self.host}/generate", json=payload)
            resp.raise_for_status()
            data = resp.json()

        # On suppose que data contient un champ "content" ou "generated_text"
        generated_text = data.get("content", "")

        return ChatCompletionResponse(content=generated_text)

    async def health_check(self) -> bool:
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(f"{self.host}/health")
                return resp.status_code == 200
        except Exception:
            return False
