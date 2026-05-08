"""
voice/stt.py

Whisper speech-to-text module.

Accepts an uploaded audio file, transcribes it using local Whisper,
and returns a plain transcript string that feeds directly into the
existing text intelligence pipeline.

Architecture principle (from spec):
    "Single intelligence pipeline, multiple input modes."

This module exists solely to bridge audio → text.
It does NOT contain any meeting intelligence logic.

Design rules:
    - Whisper model loaded lazily on first call — not at import time
    - model size driven by config (WHISPER_MODEL)
    - audio file written to a temp file, discarded after transcription
    - returns plain str transcript — caller feeds it to run_pipeline()
    - no audio stored permanently
    - no duplicate AI logic — STT output enters the same pipeline as typed text
"""

import os
import tempfile
from typing import Union

import whisper

from core.config import get_config
from core.utils.logger import get_logger

logger = get_logger(__name__)

# Module-level model cache — loaded once, reused across calls
_whisper_model: whisper.Whisper | None = None


def _get_model() -> whisper.Whisper:
    """
    Load and cache the Whisper model.
    Uses model size from config (WHISPER_MODEL env var).
    Downloads model weights on first call if not already cached locally.
    """
    global _whisper_model
    if _whisper_model is None:
        model_size = get_config().whisper_model
        logger.info("Loading Whisper model: %s (first use — may download)", model_size)
        _whisper_model = whisper.load_model(model_size)
        logger.info("Whisper model loaded: %s", model_size)
    return _whisper_model


def transcribe(audio_source: Union[str, bytes], filename: str = "") -> str:
    """
    Transcribe audio to text using local Whisper.

    Accepts either:
        - A file path string (local path to audio file)
        - Raw audio bytes (from Streamlit file_uploader)

    The bytes path writes a temp file, transcribes it, then
    immediately deletes the temp file — no audio persisted.

    Args:
        audio_source : File path str or raw audio bytes.
        filename     : Original filename (for logging only).

    Returns:
        Transcribed text as a plain string.
        Returns "" if transcription produces no output.

    Raises:
        RuntimeError: if Whisper fails to process the audio.
        ValueError:   if audio_source is empty or invalid.
    """
    if not audio_source:
        raise ValueError("transcribe() received empty audio_source.")

    label = filename or "audio_input"
    logger.info("STT: starting transcription for '%s'", label)

    if isinstance(audio_source, str):
        # File path — transcribe directly
        transcript = _transcribe_path(audio_source)
    else:
        # Bytes — write temp file, transcribe, discard
        transcript = _transcribe_bytes(audio_source, filename)

    if not transcript or not transcript.strip():
        logger.warning("STT: Whisper returned empty transcription for '%s'", label)
        return ""

    transcript = transcript.strip()
    logger.info("STT: transcription complete (%d chars) for '%s'", len(transcript), label)
    return transcript


def _transcribe_path(path: str) -> str:
    """Transcribe a local audio file at the given path."""
    if not os.path.isfile(path):
        raise ValueError(f"Audio file not found: {path}")

    try:
        model = _get_model()
        result = model.transcribe(path, fp16=False)
        return result.get("text", "")
    except Exception as e:
        raise RuntimeError(
            f"Whisper transcription failed for '{path}': {type(e).__name__}: {e}"
        ) from e


def _transcribe_bytes(audio_bytes: bytes, filename: str = "") -> str:
    """
    Write audio bytes to a temp file, transcribe, then delete.
    Temp file is always cleaned up — even on failure.
    """
    # Infer suffix from filename for Whisper format detection
    suffix = _infer_suffix(filename)
    tmp_path = ""

    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        logger.debug("STT: temp file written to %s (%d bytes)", tmp_path, len(audio_bytes))
        return _transcribe_path(tmp_path)

    finally:
        # Always discard — audio must not persist
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
            logger.debug("STT: temp file discarded: %s", tmp_path)


def _infer_suffix(filename: str) -> str:
    """
    Extract file extension from filename for temp file naming.
    Whisper uses the extension to select the correct decoder.
    Defaults to .wav if no recognizable extension found.
    """
    supported = {".mp3", ".wav", ".m4a", ".ogg", ".flac", ".webm", ".mp4"}
    if filename:
        _, ext = os.path.splitext(filename.lower())
        if ext in supported:
            return ext
    return ".wav"