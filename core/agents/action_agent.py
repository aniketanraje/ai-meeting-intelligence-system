"""
core/agents/action_agent.py

Extracts action items from the cleaned transcript.
Populates: state["action_items"] (task + initial owner/priority/deadline)
Reads:     state["cleaned_transcript"]

The owner and priority fields set here are enriched by the
owner_agent and priority_agent in subsequent pipeline steps.
"""

import json
from pathlib import Path

from core.providers.base_provider import BaseLLMProvider
from core.state import ActionItem, MeetingState, empty_action_item
from core.utils.logger import get_logger

logger = get_logger(__name__)

_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "action_prompt.txt"


def run(state: MeetingState, provider: BaseLLMProvider) -> MeetingState:
    """
    Action item extraction agent.

    Sends the cleaned transcript to the LLM and parses the
    returned JSON array of action item objects into state["action_items"].

    Args:
        state:    Current MeetingState. Must have cleaned_transcript populated.
        provider: Active LLM provider instance.

    Returns:
        Updated MeetingState with action_items populated.
        On failure, action_items remains [] and error is logged.
    """
    logger.info("Action agent: starting")

    transcript = state.get("cleaned_transcript", "").strip()
    if not transcript:
        logger.warning("Action agent: cleaned_transcript is empty — skipping")
        return state

    prompt = _build_prompt(transcript)

    try:
        raw = provider.invoke(prompt)
        items = _parse_action_items(raw)
        state["action_items"] = items
        logger.info("Action agent: extracted %d action item(s)", len(items))
    except Exception as e:
        logger.error("Action agent: failed — %s", e)
        state["action_items"] = []

    return state


def _build_prompt(transcript: str) -> str:
    template = _PROMPT_PATH.read_text(encoding="utf-8")
    return template.replace("{transcript}", transcript)


def _parse_action_items(raw: str) -> list[ActionItem]:
    """
    Parse JSON array of action item objects from LLM response.
    Returns [] on any failure. Each item is sanitized via _coerce_item().
    """
    cleaned = _strip_markdown_fence(raw)
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, list):
            return [_coerce_item(i) for i in parsed if isinstance(i, dict)]
    except json.JSONDecodeError:
        logger.warning("Action agent: JSON parse failed on response: %r", raw[:120])
    return []


def _coerce_item(raw_item: dict) -> ActionItem:
    """
    Coerce a raw dict from the LLM into a valid ActionItem.
    Uses empty_action_item() as base so all keys are always present.
    """
    base = empty_action_item()
    return ActionItem(
        task=str(raw_item.get("task") or base["task"]).strip(),
        owner=str(raw_item.get("owner") or base["owner"]).strip(),
        priority=str(raw_item.get("priority") or base["priority"]).strip(),
        deadline=str(raw_item.get("deadline") or base["deadline"]).strip(),
    )


def _strip_markdown_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        inner = lines[1:] if lines[0].startswith("```") else lines
        if inner and inner[-1].strip() == "```":
            inner = inner[:-1]
        return "\n".join(inner).strip()
    return text