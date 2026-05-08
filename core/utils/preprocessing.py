"""
core/utils/preprocessing.py

Transcript preprocessing — cleans raw text before agents operate on it.

Real-world transcripts (especially Whisper output) are noisy:
filler words, inconsistent speaker tags, repeated punctuation,
encoding artifacts, and run-on text all degrade LLM extraction quality.

This module normalizes raw transcripts into clean, consistent text
so every downstream agent operates on the same reliable surface.

Design rules:
- pure functions only — no side effects, no state
- each cleaning concern is isolated in its own function
- clean() is the single entry point the pipeline calls
- output is a plain str — no special objects
- never truncates meaningful content
- never modifies semantic meaning
"""

import re
from core.utils.logger import get_logger

logger = get_logger(__name__)


# -----------------------------------------------
# Filler words to suppress
# -----------------------------------------------

# Common English spoken fillers. Matched as whole words only (word boundaries).
_FILLERS: tuple[str, ...] = (
    "uh", "um", "uhh", "umm", "hmm", "hm",
    "er", "err", "ah", "ahh", "oh",
    "like", "you know", "i mean", "sort of", "kind of",
    "basically", "literally", "actually", "right",
)

# Pre-compiled patterns for efficiency
_FILLER_PATTERNS: list[re.Pattern[str]] = [
    re.compile(rf"\b{re.escape(f)}\b", re.IGNORECASE)
    for f in _FILLERS
]


# -----------------------------------------------
# Public entry point
# -----------------------------------------------

def clean(raw: str) -> str:
    """
    Clean a raw meeting transcript and return normalized text.

    Applies the following transformations in order:
        1. Validate input is non-empty
        2. Normalize unicode and encoding artifacts
        3. Normalize speaker tag formatting
        4. Remove filler words
        5. Strip repeated punctuation
        6. Collapse excess whitespace
        7. Normalize sentence casing where possible
        8. Strip leading/trailing whitespace

    Args:
        raw: Raw transcript string. May come from direct text input
             or Whisper speech-to-text output.

    Returns:
        Cleaned transcript string. Always a non-empty string if raw
        was non-empty after stripping. Returns empty string if input
        is blank — caller (pipeline entry) handles that case.
    """
    if not raw or not raw.strip():
        logger.warning("preprocessing.clean() received empty input — returning empty string")
        return ""

    logger.debug("Preprocessing transcript (%d chars)", len(raw))

    text = raw
    text = _normalize_encoding(text)
    text = _normalize_speaker_tags(text)
    text = _remove_fillers(text)
    text = _strip_repeated_punctuation(text)
    text = _collapse_whitespace(text)
    text = _normalize_casing(text)
    text = text.strip()

    logger.debug("Preprocessing complete (%d chars → %d chars)", len(raw), len(text))
    return text


# -----------------------------------------------
# Cleaning steps — each handles one concern
# -----------------------------------------------

def _normalize_encoding(text: str) -> str:
    """
    Remove or replace common encoding artifacts and non-printable characters.

    Targets:
    - NULL bytes and other control characters (except tab/newline)
    - Repeated ellipsis artifacts (... ... ...)
    - Non-breaking spaces → regular spaces
    - Windows-style line endings → Unix
    """
    # Windows line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Non-breaking space
    text = text.replace("\u00a0", " ")

    # NULL bytes and control characters (keep \t and \n)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)

    # Repeated ellipsis artifacts: "... ... ..." → "..."
    text = re.sub(r"(\.\.\.)(\s*\.\.\.)+", "...", text)

    return text


def _normalize_speaker_tags(text: str) -> str:
    """
    Normalize common speaker tag formats to: "Speaker: "

    Handles:
        [Speaker Name]:  text   → Speaker Name: text
        (Speaker Name):  text   → Speaker Name: text
        SPEAKER NAME:    text   → Speaker Name: text  (all-caps → title case)
        Speaker -        text   → Speaker: text
    """
    # [Name]: or (Name): → Name:
    text = re.sub(r"[\[\(]([^\]\)]+)[\]\)]\s*:", r"\1:", text)

    # All-caps speaker tags: "JOHN DOE: " → "John Doe: "
    def _title_speaker(match: re.Match[str]) -> str:
        tag = match.group(1)
        if tag.isupper() and len(tag) > 1:
            return tag.title() + ":"
        return match.group(0)

    text = re.sub(r"^([A-Z][A-Z\s]{1,30}):", _title_speaker, text, flags=re.MULTILINE)

    # "Speaker - " → "Speaker: "
    text = re.sub(r"^(\w[\w\s]{0,25})\s+-\s+", r"\1: ", text, flags=re.MULTILINE)

    return text


def _remove_fillers(text: str) -> str:
    """
    Remove spoken filler words.

    Matches whole words only to avoid damaging real content.
    Example: "uh we should um finalize" → "we should finalize"
    """
    for pattern in _FILLER_PATTERNS:
        text = pattern.sub("", text)
    return text


def _strip_repeated_punctuation(text: str) -> str:
    """
    Collapse repeated punctuation into a single instance.

    Examples:
        "What???"   → "What?"
        "Great!!!"  → "Great!"
        ",,"        → ","
        "--"        → "—"  (em-dash normalization)
    """
    # Collapse repeated ? and !
    text = re.sub(r"\?{2,}", "?", text)
    text = re.sub(r"!{2,}", "!", text)
    text = re.sub(r",{2,}", ",", text)

    # Double-dash to em-dash
    text = re.sub(r"--+", "—", text)

    return text


def _collapse_whitespace(text: str) -> str:
    """
    Normalize all whitespace.

    - Multiple spaces → single space (within a line)
    - More than two consecutive newlines → two newlines
    - Spaces around newlines → clean newlines
    - Tab characters → single space
    """
    # Tabs → space
    text = text.replace("\t", " ")

    # Multiple spaces within a line → single space
    text = re.sub(r" {2,}", " ", text)

    # Trailing spaces on each line
    text = re.sub(r" +\n", "\n", text)
    text = re.sub(r"\n +", "\n", text)

    # More than 2 consecutive blank lines → 2
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text


def _normalize_casing(text: str) -> str:
    """
    Capitalize the first letter of each sentence where it is clearly lowercase.

    Only applies to sentence starts (after ". ", "? ", "! ", or line start).
    Does not modify speaker tags, proper nouns, or mid-sentence content.
    """
    def _cap_first(match: re.Match[str]) -> str:
        prefix = match.group(1)   # punctuation + space, or empty at line start
        letter = match.group(2)   # the first letter of the sentence
        return prefix + letter.upper()

    # After sentence-ending punctuation
    text = re.sub(r"([.?!]\s+)([a-z])", _cap_first, text)

    # At line start
    text = re.sub(r"(^|\n)([a-z])", _cap_first, text)

    return text