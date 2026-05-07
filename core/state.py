"""
core/state.py

Shared state contract for the LangGraph multi-agent pipeline.

MeetingState is the single orchestration contract.
Every agent reads from and writes to this state.
Agents never communicate with each other directly.

Design rules:
- fields are populated progressively as the pipeline executes
- early-stage fields (raw_transcript, cleaned_transcript) are
  consumed by later agents but never mutated once written
- all fields default to empty/None — no agent assumes prior state
- no business logic lives here
- no validation logic lives here
- no provider logic lives here
- serialization-safe: only stdlib types (str, list, dict, None)
"""

from __future__ import annotations

from typing import TypedDict


# -----------------------------------------------
# Action Item
# -----------------------------------------------

class ActionItem(TypedDict):
    """
    Structured representation of a single action item.

    Produced by the Action Agent, then enriched by the Owner
    and Priority agents. All fields are present after the
    pipeline completes — missing values use sentinel strings,
    not None, to stay serialization-safe.

    Fields:
        task     : Description of the work to be done.
        owner    : Person responsible. "Not specified" if unknown.
        priority : Normalized priority level. One of: High, Medium, Low.
        deadline : Optional target date as free-form string. Empty string if absent.
    """

    task: str
    owner: str
    priority: str
    deadline: str


# -----------------------------------------------
# Pipeline Metadata
# -----------------------------------------------

class PipelineMetadata(TypedDict):
    """
    Lightweight processing metadata attached to every pipeline run.

    Not part of the meeting intelligence output. Used for
    logging, debugging, and SQLite persistence bookkeeping.

    Fields:
        run_id          : Unique identifier for this pipeline execution (UUID string).
        input_mode      : How the transcript arrived. One of: "text", "audio".
        audio_filename  : Original audio filename if input_mode is "audio". Empty string otherwise.
        processed_at    : ISO 8601 UTC timestamp of when the pipeline completed.
        model_provider  : Provider used for this run (e.g. "gemini").
        model_name      : Specific model used (e.g. "gemini-1.5-pro").
    """

    run_id: str
    input_mode: str           # "text" | "audio"
    audio_filename: str       # empty string if input was text
    processed_at: str         # ISO 8601 UTC, e.g. "2024-11-01T12:34:56Z"
    model_provider: str
    model_name: str


# -----------------------------------------------
# Meeting State — Pipeline Backbone
# -----------------------------------------------

class MeetingState(TypedDict):
    """
    Shared state object passed through every node in the LangGraph pipeline.

    Population sequence during a normal pipeline run:

        [Input Layer]
            raw_transcript      ← set at pipeline entry (text input or Whisper output)

        [Preprocessing]
            cleaned_transcript  ← set by preprocessing layer

        [Topic Agent]
            topics              ← set by topic_agent

        [Summary Agent]
            summary             ← set by summary_agent

        [Action Agent]
            action_items        ← set by action_agent (task + empty owner/priority/deadline)

        [Owner Agent]
            action_items        ← owner field populated per item

        [Priority Agent]
            action_items        ← priority field normalized per item

        [Validation Layer]
            errors              ← populated if validation finds issues

        [Final Output]
            metadata            ← set at pipeline completion

    Serialization contract:
        All field values are plain Python types (str, list, dict, None).
        The state is directly JSON-serializable once metadata is populated.

    Field reference:

        raw_transcript     : Original, unprocessed transcript text.
                             Source: direct text input or Whisper transcription.
                             Never modified after initial assignment.

        cleaned_transcript : Normalized transcript after preprocessing.
                             Filler words removed, speaker tags normalized,
                             garbage characters stripped.
                             This is what agents operate on — not raw_transcript.

        topics             : List of key discussion topics identified in the meeting.
                             Each entry is a short plain-text topic string.
                             Example: ["Q3 roadmap", "Budget approval", "Team hiring"]

        summary            : Concise prose summary of the meeting.
                             Single coherent paragraph. No bullet points.

        action_items       : List of ActionItem dicts representing tasks from the meeting.
                             Populated progressively across three agents:
                               - action_agent    sets task + empty owner/priority/deadline
                               - owner_agent     fills owner per item
                               - priority_agent  fills priority per item

        errors             : Validation error messages from the validation layer.
                             Empty list if pipeline completed cleanly.
                             Populated by validation only — agents must not write here.

        metadata           : PipelineMetadata dict attached at pipeline completion.
                             None until the pipeline finishes.
    """

    # --- Transcript ---
    raw_transcript: str
    cleaned_transcript: str

    # --- Extracted Intelligence ---
    topics: list[str]
    summary: str
    action_items: list[ActionItem]

    # --- Validation ---
    errors: list[str]

    # --- Run Metadata ---
    metadata: PipelineMetadata | None


# -----------------------------------------------
# State Factory
# -----------------------------------------------

def initial_state(
    raw_transcript: str,
    input_mode: str = "text",
    audio_filename: str = "",
) -> MeetingState:
    """
    Return a MeetingState with all fields initialized to safe empty defaults.

    This is the only sanctioned way to construct a MeetingState.
    The pipeline entry point calls this before handing state to LangGraph.

    Args:
        raw_transcript : The raw input text (or Whisper output).
        input_mode     : "text" or "audio". Defaults to "text".
        audio_filename : Original audio filename if audio input. Empty string otherwise.

    Returns:
        A fully initialized MeetingState ready for pipeline entry.
    """
    return MeetingState(
        raw_transcript=raw_transcript,
        cleaned_transcript="",
        topics=[],
        summary="",
        action_items=[],
        errors=[],
        metadata=None,
    )


def empty_action_item() -> ActionItem:
    """
    Return an ActionItem with all fields set to safe empty sentinel values.

    Agents should use this as the base when constructing new action items
    to guarantee all required keys are always present.
    """
    return ActionItem(
        task="",
        owner="Not specified",
        priority="Medium",
        deadline="",
    )