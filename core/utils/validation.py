"""
core/utils/validation.py

Validation layer for the LangGraph pipeline.

Sits between the final pipeline agent and SQLite persistence.
Enforces the output contract defined in the architecture spec:

    {
      "summary": "...",
      "topics": [...],
      "action_items": [
        { "task": "...", "owner": "...", "priority": "High|Medium|Low", "deadline": "..." }
      ]
    }

Design rules:
- receives a MeetingState, returns a validated MeetingState
- never raises — errors are written to state["errors"] instead
- normalizes where possible (priority casing, missing owners)
- removes duplicates deterministically
- does not call LLMs or external services
- pure logic only — no side effects beyond state mutation
"""

import re
from core.state import MeetingState, ActionItem
from core.utils.logger import get_logger

logger = get_logger(__name__)

# -----------------------------------------------
# Priority normalization
# -----------------------------------------------

_VALID_PRIORITIES: set[str] = {"High", "Medium", "Low"}

# Tokens that map to each canonical priority
_PRIORITY_MAP: dict[str, str] = {
    # High
    "high": "High", "urgent": "High", "critical": "High",
    "asap": "High", "immediate": "High", "blocker": "High",
    # Medium
    "medium": "Medium", "normal": "Medium", "moderate": "Medium",
    "mid": "Medium", "standard": "Medium",
    # Low
    "low": "Low", "minor": "Low", "nice to have": "Low",
    "whenever": "Low", "backlog": "Low",
}


# -----------------------------------------------
# Public entry point
# -----------------------------------------------

def validate(state: MeetingState) -> MeetingState:
    """
    Validate and normalize a completed MeetingState.

    Applies in order:
        1. Summary presence check
        2. Topics list normalization (dedupe, strip empties)
        3. Action items normalization:
           a. Remove items with empty/blank tasks
           b. Normalize owner sentinel
           c. Normalize priority to High | Medium | Low
           d. Deduplicate by task text (case-insensitive)
        4. Collect all errors into state["errors"]

    Args:
        state: MeetingState after all agents have run.

    Returns:
        The same state dict with normalized fields and errors populated.
        Never raises — caller decides how to handle non-empty errors.
    """
    errors: list[str] = []

    state = _validate_summary(state, errors)
    state = _validate_topics(state, errors)
    state = _validate_action_items(state, errors)

    if errors:
        logger.warning("Validation completed with %d issue(s): %s", len(errors), errors)
    else:
        logger.debug("Validation passed cleanly")

    state["errors"] = errors
    return state


# -----------------------------------------------
# Validators
# -----------------------------------------------

def _validate_summary(state: MeetingState, errors: list[str]) -> MeetingState:
    """Check summary is present and non-trivial."""
    summary = (state.get("summary") or "").strip()

    if not summary:
        errors.append("summary: missing — summary agent produced no output")
    elif len(summary) < 20:
        errors.append(
            f"summary: suspiciously short ({len(summary)} chars) — may be incomplete"
        )
    else:
        # Write back stripped version
        state["summary"] = summary

    return state


def _validate_topics(state: MeetingState, errors: list[str]) -> MeetingState:
    """Deduplicate and strip empty topic entries."""
    raw_topics: list[str] = state.get("topics") or []

    cleaned: list[str] = []
    seen: set[str] = set()

    for topic in raw_topics:
        t = topic.strip()
        key = t.lower()
        if not t:
            continue
        if key in seen:
            logger.debug("Duplicate topic removed: '%s'", t)
            continue
        seen.add(key)
        cleaned.append(t)

    if not cleaned:
        errors.append("topics: no topics extracted — topic agent may have failed")

    state["topics"] = cleaned
    return state


def _validate_action_items(state: MeetingState, errors: list[str]) -> MeetingState:
    """
    Normalize action items:
    - remove items with blank tasks
    - normalize owner to sentinel if missing/blank
    - normalize priority to High | Medium | Low
    - deduplicate by task text (case-insensitive)
    """
    raw_items: list[ActionItem] = state.get("action_items") or []
    normalized: list[ActionItem] = []
    seen_tasks: set[str] = set()

    for item in raw_items:
        task = item.get("task", "").strip()

        # Drop items with no task description
        if not task:
            logger.debug("Action item with empty task dropped")
            continue

        # Deduplicate
        task_key = _normalize_task_key(task)
        if task_key in seen_tasks:
            logger.debug("Duplicate action item removed: '%s'", task)
            continue
        seen_tasks.add(task_key)

        # Normalize owner
        owner = item.get("owner", "").strip()
        if not owner or owner.lower() in {"unknown", "n/a", "none", "tbd", "unassigned"}:
            owner = "Not specified"

        # Normalize priority
        priority = _normalize_priority(item.get("priority", ""))

        normalized.append(ActionItem(
            task=task,
            owner=owner,
            priority=priority,
            deadline=item.get("deadline", "").strip(),
        ))

    if not normalized:
        # Not an error — some meetings genuinely have no action items
        logger.debug("No action items after validation — may be expected")

    state["action_items"] = normalized
    return state


# -----------------------------------------------
# Helpers
# -----------------------------------------------

def _normalize_priority(raw: str) -> str:
    """
    Map raw priority string to canonical High | Medium | Low.
    Falls back to "Medium" for unrecognized values.
    """
    if not raw:
        return "Medium"

    cleaned = raw.strip().lower()

    # Direct match
    if cleaned in _PRIORITY_MAP:
        return _PRIORITY_MAP[cleaned]

    # Substring match — handles "high priority", "very urgent", etc.
    for token, canonical in _PRIORITY_MAP.items():
        if token in cleaned:
            return canonical

    logger.debug("Unrecognized priority '%s' — defaulting to Medium", raw)
    return "Medium"


def _normalize_task_key(task: str) -> str:
    """
    Produce a normalized deduplication key from a task string.
    Lowercases, strips punctuation and excess whitespace.
    """
    key = task.lower()
    key = re.sub(r"[^\w\s]", "", key)   # strip punctuation
    key = re.sub(r"\s+", " ", key).strip()
    return key