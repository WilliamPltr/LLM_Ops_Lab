from abc import ABC, abstractmethod
from core.models import ChatCompletionRequest, ChatCompletionResponse


class BaseLLMService(ABC):
    @abstractmethod
    async def get_chat_completion(
        self, request: ChatCompletionRequest
    ) -> ChatCompletionResponse:
        """Obtenir une complétion chat."""
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """Vérifier la disponibilité du service."""
        pass
