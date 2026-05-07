"""
core/providers/gemini_provider.py

Gemini LLM provider implementation.

Uses google-generativeai SDK (0.8.x).
All configuration is sourced exclusively from core.config.
No config values are hardcoded or passed by callers.

SDK reference:
    import google.generativeai as genai
    genai.configure(api_key=...)
    genai.GenerativeModel(model_name, generation_config)
    model.generate_content(prompt) -> GenerateContentResponse
    response.text -> str
"""

import google.generativeai as genai

from core.config import AppConfig, get_active_api_key, get_config
from core.providers.base_provider import BaseLLMProvider


class GeminiProvider(BaseLLMProvider):
    """
    LLM provider backed by Google Gemini.

    Initialized once and reused across all agent invocations.
    Temperature is set to 0.0 by default (config-driven) to enforce
    deterministic structured extraction outputs.
    """

    def __init__(self) -> None:
        config: AppConfig = get_config()
        api_key: str = get_active_api_key(config)

        genai.configure(api_key=api_key)

        self._model_name: str = config.model_name
        self._model = genai.GenerativeModel(
            model_name=self._model_name,
            generation_config=genai.types.GenerationConfig(
                temperature=config.llm_temperature,
                candidate_count=1,
            ),
        )

    def invoke(self, prompt: str) -> str:
        """
        Send prompt to Gemini and return plain text response.

        Args:
            prompt: Full prompt string. Constructed by the calling agent.

        Returns:
            Stripped plain text response from the model.

        Raises:
            ValueError: if prompt is empty.
            RuntimeError: if the model returns an empty or unusable response.
        """
        if not prompt or not prompt.strip():
            raise ValueError("GeminiProvider.invoke() received an empty prompt.")

        try:
            response = self._model.generate_content(prompt)
        except Exception as e:
            raise RuntimeError(
                f"GeminiProvider: API call failed for model '{self._model_name}'. "
                f"Cause: {type(e).__name__}: {e}"
            ) from e

        text = _extract_text(response)

        if not text:
            raise RuntimeError(
                f"GeminiProvider: model '{self._model_name}' returned an empty response. "
                "Check prompt quality or Gemini safety filters."
            )

        return text

    def get_provider_name(self) -> str:
        return "gemini"


# -----------------------------------------------
# Internal helpers
# -----------------------------------------------

def _extract_text(response: object) -> str:
    """
    Extract plain text from a Gemini GenerateContentResponse.

    Primary path: response.text (str | None)
    Fallback path: response.candidates[0].content.parts[0].text
    Returns empty string if both paths fail — caller decides how to handle.
    """
    # Primary path — available in google-generativeai 0.8.x
    try:
        text = response.text  # type: ignore[attr-defined]
        if isinstance(text, str) and text.strip():
            return text.strip()
    except (AttributeError, ValueError):
        pass

    # Fallback path — walk candidates manually
    try:
        candidates = response.candidates  # type: ignore[attr-defined]
        if candidates:
            parts = candidates[0].content.parts
            if parts:
                text = parts[0].text
                if isinstance(text, str) and text.strip():
                    return text.strip()
    except (AttributeError, IndexError, TypeError):
        pass

    return ""