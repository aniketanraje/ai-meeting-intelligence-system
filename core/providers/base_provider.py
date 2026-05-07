"""
core/providers/base_provider.py

Abstract base class for all LLM providers.

All provider implementations must inherit from BaseLLMProvider
and implement the invoke() method. No provider-specific logic
belongs outside of the providers/ module.

Usage pattern (for agents):
    from core.providers.base_provider import BaseLLMProvider
    # receive a provider instance, call .invoke(prompt)
    # no knowledge of which provider is active
"""

from abc import ABC, abstractmethod


class BaseLLMProvider(ABC):
    """
    Provider-agnostic interface for LLM invocation.

    Agents and pipeline nodes communicate exclusively through
    this interface. The concrete provider is selected at runtime
    via config (MODEL_PROVIDER in .env).
    """

    @abstractmethod
    def invoke(self, prompt: str) -> str:
        """
        Send a prompt to the LLM and return the text response.

        Args:
            prompt: The full prompt string to send to the model.
                    Prompt construction is the responsibility of the caller (agents).

        Returns:
            The model's text response as a plain string.
            Never returns None — raises on failure instead.

        Raises:
            RuntimeError: if the provider call fails or returns an empty response.
        """
        ...

    @abstractmethod
    def get_provider_name(self) -> str:
        """
        Return the provider identifier string (e.g. 'gemini', 'openai', 'claude').
        Used for logging and diagnostics only.
        """
        ...