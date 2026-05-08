"""
core/agents/priority_agent.py

Reviews and corrects priority levels on all action items.
Populates: state["action_items"][*]["priority"]  (enrichment pass)
Reads:     state["cleaned_transcript"], state["action_items"]

Sends all items to the LLM for priority validation against transcript
context. The LLM may correct items that were mis-prioritized by the
action agent.
"""

import json
from pathlib import Path

from core.providers.base_provider import BaseLLMProvider
from core.state import ActionItem, MeetingState
from core.utils.logger import get_logger

logger = get_logger(__name__)

_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "priority_prompt.txt"

_VALID_PRIORITIES = {"High", "Medium", "Low"}


def run(state: MeetingState, provider: BaseLLMProvider) -> MeetingState:
    """
    Priority classification agent.

    Sends the transcript and current action items to the LLM,
    asking it to review and correct priority levels based on
    context expressed in the meeting.

    Skips if there are no action items.

    Args:
        state:    Current MeetingState. Must have action_items populated.
        provider: Active LLM provider instance.

    Returns:
        Updated MeetingState with priorities reviewed and corrected.
        On failure, existing action_items are preserved unchanged.
    """
    logger.info("Priority agent: starting")

    items = state.get("action_items") or []
    if not items:
        logger.info("Priority agent: no action items — skipping")
        return state

    transcript = state.get("cleaned_transcript", "").strip()
    prompt = _build_prompt(transcript, items)

    try:
        raw = provider.invoke(prompt)
        reviewed = _parse_items(raw)
        if reviewed and len(reviewed) == len(items):
            state["action_items"] = reviewed
            logger.info("Priority agent: priorities reviewed for %d item(s)", len(reviewed))
        else:
            logger.warning(
                "Priority agent: returned %d items, expected %d — keeping originals",
                len(reviewed), len(items),
            )
    except Exception as e:
        logger.error("Priority agent: failed — %s", e)

    return state


def _build_prompt(transcript: str, items: list[ActionItem]) -> str:
    template = _PROMPT_PATH.read_text(encoding="utf-8")
    items_json = json.dumps(items, indent=2)
    return (
        template
        .replace("{transcript}", transcript)
        .replace("{action_items}", items_json)
    )


def _parse_items(raw: str) -> list[ActionItem]:
    cleaned = _strip_markdown_fence(raw)
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, list):
            return [
                ActionItem(
                    task=str(i.get("task", "")).strip(),
                    owner=str(i.get("owner", "Not specified")).strip(),
                    priority=_safe_priority(i.get("priority", "Medium")),
                    deadline=str(i.get("deadline", "")).strip(),
                )
                for i in parsed if isinstance(i, dict)
            ]
    except json.JSONDecodeError:
        logger.warning("Priority agent: JSON parse failed on response: %r", raw[:120])
    return []


def _safe_priority(raw: str) -> str:
    """Ensure priority is one of the three canonical values. Defaults to Medium."""
    val = str(raw).strip().capitalize()
    # Handle lowercase variants: "high" → "High"
    for canonical in _VALID_PRIORITIES:
        if val.lower() == canonical.lower():
            return canonical
    logger.debug("Priority agent: unrecognized priority '%s' — defaulting to Medium", raw)
    return "Medium"


def _strip_markdown_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        inner = lines[1:] if lines[0].startswith("```") else lines
        if inner and inner[-1].strip() == "```":
            inner = inner[:-1]
        return "\n".join(inner).strip()
    return text