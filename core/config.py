"""
core/config.py

Centralized configuration management.
Loads environment variables, validates required keys,
and exposes a typed AppConfig dataclass to the rest of the system.

All modules must import configuration from here.
No module should call os.environ or dotenv directly.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv

# -----------------------------------------------
# Constants
# -----------------------------------------------

SUPPORTED_PROVIDERS: set[str] = {"gemini", "openai", "claude", "groq"}

SUPPORTED_WHISPER_MODELS: set[str] = {"tiny", "base", "small", "medium", "large"}

SUPPORTED_LOG_LEVELS: set[str] = {"DEBUG", "INFO", "WARNING", "ERROR"}

PROVIDER_DEFAULT_MODELS: dict[str, str] = {
    "gemini": "gemini-1.5-pro",
    "openai": "gpt-4o",
    "claude": "claude-3-5-sonnet-20241022",
    "groq":   "llama-3.3-70b-versatile",
}

# -----------------------------------------------
# Load .env
# -----------------------------------------------

def _load_env() -> None:
    """
    Load .env from project root.
    Safe to call multiple times — dotenv skips if already loaded.
    """
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if env_path.exists():
        load_dotenv(dotenv_path=env_path, override=False)
    else:
        load_dotenv(override=False)


_load_env()


# -----------------------------------------------
# Configuration Dataclass
# -----------------------------------------------

@dataclass(frozen=True)
class AppConfig:
    """
    Typed, immutable application configuration.
    Constructed once at module load via load_config().
    """
    # Provider
    model_provider: str
    model_name: str
    llm_temperature: float

    # API Keys
    gemini_api_key: str
    openai_api_key: str
    anthropic_api_key: str
    groq_api_key: str

    # Database (env: SQLITE_DB_PATH)
    db_path: str

    # Voice
    whisper_model: str

    # Logging
    log_level: str

    # Output
    output_dir: str


# -----------------------------------------------
# Config Loader
# -----------------------------------------------

def load_config() -> AppConfig:
    """
    Read, validate, and return typed AppConfig.

    Raises:
        ValueError: if required environment variables are missing or invalid.
    """
    errors: list[str] = []

    # --- Provider ---
    model_provider = os.getenv("MODEL_PROVIDER", "groq").strip().lower()
    if model_provider not in SUPPORTED_PROVIDERS:
        errors.append(
            f"MODEL_PROVIDER='{model_provider}' is invalid. "
            f"Supported: {sorted(SUPPORTED_PROVIDERS)}"
        )

    # --- Model name --- per-provider override, fallback to defaults
    model_name = (
        os.getenv("GEMINI_MODEL_NAME")
        or os.getenv("OPENAI_MODEL_NAME")
        or os.getenv("CLAUDE_MODEL_NAME")
        or PROVIDER_DEFAULT_MODELS.get(model_provider, "gemini-1.5-pro")
    ).strip()

    # --- Temperature ---
    raw_temp = os.getenv("LLM_TEMPERATURE", "0.0").strip()
    try:
        llm_temperature = float(raw_temp)
        if not (0.0 <= llm_temperature <= 1.0):
            errors.append(
                f"LLM_TEMPERATURE='{raw_temp}' must be between 0.0 and 1.0."
            )
    except ValueError:
        llm_temperature = 0.0
        errors.append(
            f"LLM_TEMPERATURE='{raw_temp}' is not a valid float. Defaulting to 0.0."
        )

    # --- API Keys ---
    gemini_api_key = os.getenv("GEMINI_API_KEY", "").strip()
    openai_api_key = os.getenv("OPENAI_API_KEY", "").strip()
    anthropic_api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    groq_api_key = os.getenv("GROQ_API_KEY", "").strip()

    _validate_api_key(
        model_provider, gemini_api_key, openai_api_key, anthropic_api_key, groq_api_key, errors
    )

    # --- Database ---
    # Accepts SQLITE_DB_PATH (canonical) or DB_PATH (legacy fallback)
    db_path = (
        os.getenv("SQLITE_DB_PATH") or os.getenv("DB_PATH") or "db/meeting_intelligence.db"
    ).strip()

    # --- Whisper ---
    whisper_model = os.getenv("WHISPER_MODEL", "base").strip().lower()
    if whisper_model not in SUPPORTED_WHISPER_MODELS:
        errors.append(
            f"WHISPER_MODEL='{whisper_model}' is invalid. "
            f"Supported: {sorted(SUPPORTED_WHISPER_MODELS)}"
        )

    # --- Log Level --- non-fatal, safe fallback
    log_level = os.getenv("LOG_LEVEL", "INFO").strip().upper()
    if log_level not in SUPPORTED_LOG_LEVELS:
        log_level = "INFO"

    # --- Output Dir ---
    output_dir = os.getenv("OUTPUT_DIR", "outputs/").strip()

    # --- Raise all collected errors at once ---
    if errors:
        formatted = "\n  ".join(f"- {e}" for e in errors)
        raise ValueError(
            f"\n[CONFIG ERROR] Environment configuration is invalid:\n  {formatted}\n"
            f"Fix your .env file and restart."
        )

    return AppConfig(
        model_provider=model_provider,
        model_name=model_name,
        llm_temperature=llm_temperature,
        gemini_api_key=gemini_api_key,
        openai_api_key=openai_api_key,
        anthropic_api_key=anthropic_api_key,
        groq_api_key=groq_api_key,
        db_path=db_path,
        whisper_model=whisper_model,
        log_level=log_level,
        output_dir=output_dir,
    )


# -----------------------------------------------
# Internal Helpers
# -----------------------------------------------

def _validate_api_key(
    provider: str,
    gemini_key: str,
    openai_key: str,
    anthropic_key: str,
    groq_key: str,
    errors: list[str],
) -> None:
    """Validate that the active provider's API key is present and not a placeholder."""
    key_map: dict[str, tuple[str, str]] = {
        "gemini": (gemini_key,    "GEMINI_API_KEY"),
        "openai": (openai_key,    "OPENAI_API_KEY"),
        "claude": (anthropic_key, "ANTHROPIC_API_KEY"),
        "groq":   (groq_key,      "GROQ_API_KEY"),
    }

    if provider not in key_map:
        return  # provider error already captured

    key_value, key_name = key_map[provider]

    if not key_value:
        errors.append(
            f"{key_name} is required when MODEL_PROVIDER='{provider}' but is not set."
        )
        return

    placeholder_patterns = {"your_", "xxx", "changeme", "placeholder", "insert"}
    if any(p in key_value.lower() for p in placeholder_patterns):
        errors.append(
            f"{key_name} appears to contain a placeholder value. "
            f"Set a real API key in your .env file."
        )


def get_active_api_key(config: AppConfig) -> str:
    """Return the API key for the currently active provider."""
    key_map: dict[str, str] = {
        "gemini": config.gemini_api_key,
        "openai": config.openai_api_key,
        "claude": config.anthropic_api_key,
        "groq":   config.groq_api_key,
    }
    return key_map.get(config.model_provider, "")


# -----------------------------------------------
# Singleton accessor
# -----------------------------------------------

# Lazy singleton — initialized on first call to get_config().
# This avoids import-time failures during testing and module inspection.
_CONFIG_CACHE: AppConfig | None = None


def get_config() -> AppConfig:
    """
    Return the cached AppConfig singleton.
    Validates configuration on first call.

    Usage:
        from core.config import get_config
        config = get_config()

    Raises:
        ValueError: on first call if .env is misconfigured.
    """
    global _CONFIG_CACHE
    if _CONFIG_CACHE is None:
        _CONFIG_CACHE = load_config()
    return _CONFIG_CACHE