"""
core/providers/factory.py

Provider factory — returns the correct BaseLLMProvider instance
based on MODEL_PROVIDER in config.

All callers (Streamlit app, tests, future CLI) use this instead
of importing a specific provider directly. This is the only place
that knows which concrete provider class to instantiate.
"""

from core.config import get_config
from core.providers.base_provider import BaseLLMProvider


def get_provider() -> BaseLLMProvider:
    """
    Instantiate and return the active LLM provider.

    Provider is selected via MODEL_PROVIDER in .env.
    Raises ValueError for unsupported provider strings
    (config.py already validates this, so this is a safety net).

    Returns:
        Concrete BaseLLMProvider instance ready for use.
    """
    provider_name = get_config().model_provider

    if provider_name == "groq":
        from core.providers.groq_provider import GroqProvider
        return GroqProvider()

    if provider_name == "gemini":
        from core.providers.gemini_provider import GeminiProvider
        return GeminiProvider()

    if provider_name == "openai":
        from core.providers.openai_provider import OpenAIProvider  # future stub
        return OpenAIProvider()

    if provider_name == "claude":
        from core.providers.claude_provider import ClaudeProvider  # future stub
        return ClaudeProvider()

    raise ValueError(
        f"factory.get_provider(): unknown provider '{provider_name}'. "
        f"Check MODEL_PROVIDER in your .env file."
    )