"""
core/agents/owner_agent.py

Attempts to assign owners to action items marked "Not specified".
Populates: state["action_items"][*]["owner"]  (enrichment pass)
Reads:     state["cleaned_transcript"], state["action_items"]

Only items with owner == "Not specified" are sent to the LLM.
Items with existing owners are preserved as-is.
"""

import json
from pathlib import Path

from core.providers.base_provider import BaseLLMProvider
from core.state import ActionItem, MeetingState
from core.utils.logger import get_logger

logger = get_logger(__name__)

_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "owner_prompt.txt"


def run(state: MeetingState, provider: BaseLLMProvider) -> MeetingState:
    """
    Owner mapping agent.

    Sends the transcript and current action items to the LLM,
    asking it to fill in owners for unassigned items.

    Skips entirely if there are no action items or if every
    item already has an owner.

    Args:
        state:    Current MeetingState. Must have action_items populated.
        provider: Active LLM provider instance.

    Returns:
        Updated MeetingState with owners enriched where possible.
        On failure, existing action_items are preserved unchanged.
    """
    logger.info("Owner agent: starting")

    items = state.get("action_items") or []
    if not items:
        logger.info("Owner agent: no action items — skipping")
        return state

    unassigned = [i for i in items if i.get("owner", "") == "Not specified"]
    if not unassigned:
        logger.info("Owner agent: all items already have owners — skipping")
        return state

    transcript = state.get("cleaned_transcript", "").strip()
    prompt = _build_prompt(transcript, items)

    try:
        raw = provider.invoke(prompt)
        enriched = _parse_items(raw)
        if enriched and len(enriched) == len(items):
            state["action_items"] = enriched
            assigned = sum(1 for i in enriched if i["owner"] != "Not specified")
            logger.info("Owner agent: %d/%d items now have owners", assigned, len(enriched))
        else:
            logger.warning(
                "Owner agent: returned %d items, expected %d — keeping originals",
                len(enriched), len(items),
            )
    except Exception as e:
        logger.error("Owner agent: failed — %s", e)

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
                    priority=str(i.get("priority", "Medium")).strip(),
                    deadline=str(i.get("deadline", "")).strip(),
                )
                for i in parsed if isinstance(i, dict)
            ]
    except json.JSONDecodeError:
        logger.warning("Owner agent: JSON parse failed on response: %r", raw[:120])
    return []


def _strip_markdown_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        inner = lines[1:] if lines[0].startswith("```") else lines
        if inner and inner[-1].strip() == "```":
            inner = inner[:-1]
        return "\n".join(inner).strip()
    return text