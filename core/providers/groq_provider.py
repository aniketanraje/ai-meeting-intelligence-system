"""
core/providers/groq_provider.py

Groq LLM provider implementation.

Uses the official Groq Python SDK (groq>=0.11.0).
Groq's API is OpenAI-compatible — chat completions with
structured message format.

All configuration sourced exclusively from core.config.
No config values are hardcoded or passed by callers.

SDK reference:
    from groq import Groq
    client = Groq(api_key=...)
    response = client.chat.completions.create(
        model=..., messages=[{"role":"user","content":prompt}],
        temperature=..., max_tokens=...
    )
    response.choices[0].message.content -> str
"""

from groq import Groq

from core.config import AppConfig, get_active_api_key, get_config
from core.providers.base_provider import BaseLLMProvider


# Max tokens for structured extraction responses.
# Pipeline outputs are JSON arrays/objects — 2048 is generous.
_MAX_TOKENS = 2048


class GroqProvider(BaseLLMProvider):
    """
    LLM provider backed by Groq.

    Initialized once and reused across all agent invocations.
    Uses chat completions API with a single user message per call —
    agents construct full prompts, no system/user split needed.
    """

    def __init__(self) -> None:
        config: AppConfig = get_config()
        api_key: str = get_active_api_key(config)

        self._client = Groq(api_key=api_key)
        self._model_name: str = config.model_name
        self._temperature: float = config.llm_temperature

    def invoke(self, prompt: str) -> str:
        """
        Send prompt to Groq and return plain text response.

        Args:
            prompt: Full prompt string constructed by the calling agent.

        Returns:
            Stripped plain text response from the model.

        Raises:
            ValueError:   if prompt is empty.
            RuntimeError: if the API call fails or returns empty content.
        """
        if not prompt or not prompt.strip():
            raise ValueError("GroqProvider.invoke() received an empty prompt.")

        try:
            response = self._client.chat.completions.create(
                model=self._model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=self._temperature,
                max_tokens=_MAX_TOKENS,
            )
        except Exception as e:
            raise RuntimeError(
                f"GroqProvider: API call failed for model '{self._model_name}'. "
                f"Cause: {type(e).__name__}: {e}"
            ) from e

        text = _extract_text(response)

        if not text:
            raise RuntimeError(
                f"GroqProvider: model '{self._model_name}' returned an empty response."
            )

        return text

    def get_provider_name(self) -> str:
        return "groq"


# -----------------------------------------------
# Internal helpers
# -----------------------------------------------

def _extract_text(response: object) -> str:
    """
    Extract plain text from a Groq chat completion response.
    Returns empty string if extraction fails — caller handles empty as RuntimeError.
    """
    try:
        content = response.choices[0].message.content  # type: ignore[attr-defined]
        if isinstance(content, str) and content.strip():
            return content.strip()
    except (AttributeError, IndexError, TypeError):
        pass
    return ""