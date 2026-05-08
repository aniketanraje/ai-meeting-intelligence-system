"""
core/agents/topic_agent.py

Extracts key discussion topics from the cleaned transcript.
Populates: state["topics"]
Reads:     state["cleaned_transcript"]
"""

import json
from pathlib import Path

from core.providers.base_provider import BaseLLMProvider
from core.state import MeetingState
from core.utils.logger import get_logger

logger = get_logger(__name__)

_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "topic_prompt.txt"


def run(state: MeetingState, provider: BaseLLMProvider) -> MeetingState:
    """
    Topic extraction agent.

    Sends the cleaned transcript to the LLM and parses the
    returned JSON array of topic strings into state["topics"].

    Args:
        state:    Current MeetingState. Must have cleaned_transcript populated.
        provider: Active LLM provider instance.

    Returns:
        Updated MeetingState with topics populated.
        On failure, topics remains [] and error is logged.
    """
    logger.info("Topic agent: starting")

    transcript = state.get("cleaned_transcript", "").strip()
    if not transcript:
        logger.warning("Topic agent: cleaned_transcript is empty — skipping")
        return state

    prompt = _build_prompt(transcript)

    try:
        raw = provider.invoke(prompt)
        topics = _parse_topics(raw)
        state["topics"] = topics
        logger.info("Topic agent: extracted %d topic(s)", len(topics))
    except Exception as e:
        logger.error("Topic agent: failed — %s", e)
        state["topics"] = []

    return state


def _build_prompt(transcript: str) -> str:
    template = _PROMPT_PATH.read_text(encoding="utf-8")
    return template.replace("{transcript}", transcript)


def _parse_topics(raw: str) -> list[str]:
    """Parse JSON array from LLM response. Returns [] on any failure."""
    cleaned = _strip_markdown_fence(raw)
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, list):
            return [str(t).strip() for t in parsed if str(t).strip()]
    except json.JSONDecodeError:
        logger.warning("Topic agent: JSON parse failed on response: %r", raw[:120])
    return []


def _strip_markdown_fence(text: str) -> str:
    """Remove ```json ... ``` fences if the LLM wraps output despite instructions."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        # Drop first and last fence lines
        inner = lines[1:] if lines[0].startswith("```") else lines
        if inner and inner[-1].strip() == "```":
            inner = inner[:-1]
        return "\n".join(inner).strip()
    return text