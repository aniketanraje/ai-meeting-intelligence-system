"""
core/agents/summary_agent.py

Generates a concise prose summary of the meeting.
Populates: state["summary"]
Reads:     state["cleaned_transcript"]
"""

from pathlib import Path

from core.providers.base_provider import BaseLLMProvider
from core.state import MeetingState
from core.utils.logger import get_logger

logger = get_logger(__name__)

_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "summary_prompt.txt"


def run(state: MeetingState, provider: BaseLLMProvider) -> MeetingState:
    """
    Summary generation agent.

    Sends the cleaned transcript to the LLM and stores the
    returned prose paragraph in state["summary"].

    Args:
        state:    Current MeetingState. Must have cleaned_transcript populated.
        provider: Active LLM provider instance.

    Returns:
        Updated MeetingState with summary populated.
        On failure, summary remains "" and error is logged.
    """
    logger.info("Summary agent: starting")

    transcript = state.get("cleaned_transcript", "").strip()
    if not transcript:
        logger.warning("Summary agent: cleaned_transcript is empty — skipping")
        return state

    prompt = _build_prompt(transcript)

    try:
        raw = provider.invoke(prompt)
        summary = raw.strip()
        state["summary"] = summary
        logger.info("Summary agent: summary generated (%d chars)", len(summary))
    except Exception as e:
        logger.error("Summary agent: failed — %s", e)
        state["summary"] = ""

    return state


def _build_prompt(transcript: str) -> str:
    template = _PROMPT_PATH.read_text(encoding="utf-8")
    return template.replace("{transcript}", transcript)