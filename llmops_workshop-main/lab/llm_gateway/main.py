from fastapi import FastAPI, HTTPException
from core.models import ChatCompletionRequest, ChatCompletionResponse
from services.ollama_service import OllamaService

app = FastAPI(
    title="LLM Gateway - Ollama only",
    version="0.1.0",
    description="Un exemple minimal pour tester Ollama via FastAPI",
)

ollama_service = OllamaService()


@app.get("/health")
async def health():
    if await ollama_service.health_check():
        return {"status": "ok"}
    else:
        return {"status": "ollama service unreachable"}, 503


@app.post("/chat/completions", response_model=ChatCompletionResponse)
async def create_chat_completion(request: ChatCompletionRequest):
    try:
        response = await ollama_service.get_chat_completion(request)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
