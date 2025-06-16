# src/llm_adapters.py
from abc import ABC, abstractmethod
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.language_models import (
    BaseChatModel,
)  # Correct import for BaseChatModel


class AbstractLLMAdapter(ABC):
    """
    Abstract Base Class for Language Model Adapters.
    Defines the interface for interacting with different Language Models.
    """

    @abstractmethod
    def get_llm(self) -> BaseChatModel:
        """
        Returns an initialized LangChain-compatible Language Model instance.
        """
        pass


class GeminiLLMAdapter(AbstractLLMAdapter):
    """
    Concrete implementation of AbstractLLMAdapter for Google Gemini models.
    Initializes and provides a ChatGoogleGenerativeAI instance.
    """

    def __init__(self, model_name: str, temperature: float, api_key: str = None):
        """
        Initializes the GeminiLLMAdapter.

        Args:
            model_name (str): The name of the Gemini model (e.g., "gemini-pro").
            temperature (float): The model's temperature for creativity.
            api_key (str, optional): The Google Gemini API key. If not provided,
                                     it expects GOOGLE_API_KEY environment variable to be set.
        """
        if api_key:
            os.environ["GOOGLE_API_KEY"] = api_key
        elif "GOOGLE_API_KEY" not in os.environ:
            raise ValueError(
                "Google API Key not found. Please provide it via argument "
                "or set the GOOGLE_API_KEY environment variable."
            )
        self._model_name = model_name
        self._temperature = temperature
        self._llm = None  # Will be initialized on first call to get_llm

    def get_llm(self) -> ChatGoogleGenerativeAI:
        """
        Returns an initialized ChatGoogleGenerativeAI instance.
        Initializes it lazily if not already done.
        """
        if self._llm is None:
            self._llm = ChatGoogleGenerativeAI(
                model=self._model_name, temperature=self._temperature
            )
        return self._llm


# You can add other LLM adapters here in the future, e.g.:
# class OpenAILLMAdapter(AbstractLLMAdapter):
#     def __init__(self, model_name: str, temperature: float, api_key: str):
#         # ... initialization for OpenAI ...
#         pass
#     def get_llm(self) -> BaseChatModel:
#         # ... return OpenAI model ...
#         pass
