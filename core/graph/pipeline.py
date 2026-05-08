"""
core/graph/pipeline.py

LangGraph multi-agent pipeline orchestration.

Assembles the five agents into a stateful directed graph,
wires preprocessing at entry, runs validation at exit,
and returns a final structured MeetingState.

Architecture:
    START
      → preprocess
      → topic_agent
      → summary_agent
      → action_agent
      → owner_agent
      → priority_agent
      → validate
    END

Design rules:
- provider is injected once at build time, shared across all nodes
- node functions are thin closures — agent logic stays in agents/
- pipeline returns the final MeetingState dict directly
- no streaming, no async, no retries at this layer
- preprocessing runs inside the graph so state is consistent end-to-end
- validation runs inside the graph so errors are part of the state record
"""

import uuid
from datetime import datetime, timezone
from typing import Any

from langgraph.graph import END, START, StateGraph

from core.agents import (
    action_agent,
    owner_agent,
    priority_agent,
    summary_agent,
    topic_agent,
)
from core.config import get_config
from core.providers.base_provider import BaseLLMProvider
from core.state import MeetingState, PipelineMetadata, initial_state
from core.utils import preprocessing
from core.utils.logger import get_logger
from core.utils.validation import validate

logger = get_logger(__name__)


# -----------------------------------------------
# Pipeline builder
# -----------------------------------------------

def build_pipeline(provider: BaseLLMProvider) -> Any:
    """
    Compile and return the LangGraph pipeline.

    Args:
        provider: Instantiated LLM provider. Injected into every
                  agent node via closure — agents never import the
                  provider directly.

    Returns:
        A compiled LangGraph app. Call app.invoke(state) to run.
    """
    graph = StateGraph(MeetingState)

    # --- Register nodes ---
    # Each node is a closure that captures `provider`.
    # LangGraph calls node(state) and merges the returned dict.

    def node_preprocess(state: MeetingState) -> dict:
        logger.info("Pipeline: preprocessing transcript")
        cleaned = preprocessing.clean(state["raw_transcript"])
        return {"cleaned_transcript": cleaned}

    def node_topic(state: MeetingState) -> dict:
        updated = topic_agent.run(state, provider)
        return {"topics": updated["topics"]}

    def node_summary(state: MeetingState) -> dict:
        updated = summary_agent.run(state, provider)
        return {"summary": updated["summary"]}

    def node_action(state: MeetingState) -> dict:
        updated = action_agent.run(state, provider)
        return {"action_items": updated["action_items"]}

    def node_owner(state: MeetingState) -> dict:
        updated = owner_agent.run(state, provider)
        return {"action_items": updated["action_items"]}

    def node_priority(state: MeetingState) -> dict:
        updated = priority_agent.run(state, provider)
        return {"action_items": updated["action_items"]}

    def node_validate(state: MeetingState) -> dict:
        logger.info("Pipeline: running validation layer")
        validated = validate(state)
        return {
            "topics":       validated["topics"],
            "summary":      validated["summary"],
            "action_items": validated["action_items"],
            "errors":       validated["errors"],
        }

    # --- Add nodes to graph ---
    graph.add_node("preprocess",    node_preprocess)
    graph.add_node("topic_agent",   node_topic)
    graph.add_node("summary_agent", node_summary)
    graph.add_node("action_agent",  node_action)
    graph.add_node("owner_agent",   node_owner)
    graph.add_node("priority_agent",node_priority)
    graph.add_node("validate",      node_validate)

    # --- Wire edges ---
    graph.add_edge(START,            "preprocess")
    graph.add_edge("preprocess",     "topic_agent")
    graph.add_edge("topic_agent",    "summary_agent")
    graph.add_edge("summary_agent",  "action_agent")
    graph.add_edge("action_agent",   "owner_agent")
    graph.add_edge("owner_agent",    "priority_agent")
    graph.add_edge("priority_agent", "validate")
    graph.add_edge("validate",       END)

    return graph.compile()


# -----------------------------------------------
# Public run function
# -----------------------------------------------

def run_pipeline(
    raw_transcript: str,
    provider: BaseLLMProvider,
    input_mode: str = "text",
    audio_filename: str = "",
) -> MeetingState:
    """
    Run the full meeting intelligence pipeline.

    Entry point for both the Streamlit UI and any future callers.
    Constructs initial state, invokes the compiled graph,
    attaches metadata, and returns the final MeetingState.

    Args:
        raw_transcript : Raw text or Whisper-transcribed string.
        provider       : Active LLM provider instance.
        input_mode     : "text" or "audio".
        audio_filename : Original filename if audio input.

    Returns:
        Final MeetingState with all fields populated.

    Raises:
        ValueError: if raw_transcript is empty after stripping.
    """
    if not raw_transcript or not raw_transcript.strip():
        raise ValueError("run_pipeline() received an empty transcript.")

    logger.info(
        "Pipeline: starting run | mode=%s | provider=%s",
        input_mode,
        provider.get_provider_name(),
    )

    state = initial_state(
        raw_transcript=raw_transcript,
        input_mode=input_mode,
        audio_filename=audio_filename,
    )

    app = build_pipeline(provider)

    try:
        final_state: MeetingState = app.invoke(state)
    except Exception as e:
        logger.error("Pipeline: unhandled exception during execution — %s", e)
        raise

    # Attach metadata after successful completion
    config = get_config()
    final_state["metadata"] = PipelineMetadata(
        run_id=str(uuid.uuid4()),
        input_mode=input_mode,
        audio_filename=audio_filename,
        processed_at=datetime.now(timezone.utc).isoformat(),
        model_provider=config.model_provider,
        model_name=config.model_name,
    )

    errors = final_state.get("errors") or []
    if errors:
        logger.warning("Pipeline: completed with %d validation error(s)", len(errors))
    else:
        logger.info("Pipeline: completed cleanly")

    return final_state