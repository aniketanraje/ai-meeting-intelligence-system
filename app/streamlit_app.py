"""
app/streamlit_app.py

Streamlit UI — thin interaction layer over the AI pipeline.
Provider resolved via factory.get_provider() — never hardcoded here.
"""

import json

import streamlit as st

from core.config import get_config
from core.graph.pipeline import run_pipeline
from core.providers.factory import get_provider
from core.state import MeetingState
from db import sqlite_handler
from voice.stt import transcribe

# -----------------------------------------------
# Page config
# -----------------------------------------------

st.set_page_config(
    page_title="Meeting Notes Analyzer",
    page_icon="📋",
    layout="wide",
)

# -----------------------------------------------
# Init
# -----------------------------------------------

@st.cache_resource
def _init_db() -> None:
    sqlite_handler.init_db()

@st.cache_resource
def _get_provider():
    return get_provider()

_init_db()
_config = get_config()

# -----------------------------------------------
# Render helpers — defined before any call site
# -----------------------------------------------

def _render_action_table(items: list[dict]) -> None:
    icons = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}
    for item in items:
        task     = item.get("task") or ""
        owner    = item.get("owner") or "Not specified"
        priority = item.get("priority") or "Medium"
        deadline = item.get("deadline") or ""
        if not task:
            continue
        with st.container():
            c1, c2, c3, c4 = st.columns([4, 2, 1, 1])
            c1.markdown(f"**{task}**")
            c2.markdown(f"👤 {owner}")
            c3.markdown(f"{icons.get(priority, '⚪')} {priority}")
            c4.markdown(f"📅 {deadline if deadline else '—'}")
        st.divider()


def _render_results(state: MeetingState) -> None:

    # Metrics row
    topics       = state.get("topics") or []
    items        = state.get("action_items") or []
    high_count   = sum(1 for i in items if i.get("priority") == "High")

    m1, m2, m3 = st.columns(3)
    m1.metric("🗂 Topics Identified",  len(topics))
    m2.metric("✅ Action Items",        len(items))
    m3.metric("🔴 High Priority",       high_count)

    st.divider()

    # Summary
    st.subheader("📝 Summary")
    summary = state.get("summary") or ""
    if summary:
        st.write(summary)
    else:
        st.caption("No summary generated.")

    st.divider()

    # Topics
    st.subheader("🗂 Key Topics")
    if topics:
        cols = st.columns(min(len(topics), 4))
        for i, topic in enumerate(topics):
            cols[i % 4].markdown(f"**·** {topic}")
    else:
        st.caption("No topics extracted.")

    st.divider()

    # Action items
    st.subheader("✅ Action Items")
    if items:
        _render_action_table(items)
    else:
        st.caption("No action items found.")

    st.divider()

    # Agent status badges
    with st.expander("🤖 Agent Execution Status"):
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.success("✅ Topic Agent")
        col2.success("✅ Summary Agent")
        col3.success("✅ Action Agent")
        col4.success("✅ Owner Agent")
        col5.success("✅ Priority Agent")

    # Raw JSON
    with st.expander("🔍 Raw JSON Output"):
        st.code(json.dumps({
            "summary":      state.get("summary"),
            "topics":       state.get("topics"),
            "action_items": state.get("action_items"),
        }, indent=2), language="json")

    # Metadata
    meta = state.get("metadata")
    if meta:
        with st.expander("ℹ️ Run Metadata"):
            clean_meta = {k: v for k, v in meta.items() if v not in (None, "")}
            st.json(clean_meta)


# -----------------------------------------------
# Sidebar
# -----------------------------------------------

with st.sidebar:
    st.markdown("## 🧠 System Info")
    st.markdown(f"**Provider:** `{_config.model_provider.upper()}`")
    st.markdown(f"**Model:** `{_config.model_name}`")
    st.markdown(f"**Pipeline:** LangGraph Multi-Agent")
    st.markdown(f"**Temperature:** `{_config.llm_temperature}`")

    st.divider()

    st.markdown("## 👨‍💻 Developer Info")

    st.markdown(
        """
    **Developer:** Aniket Bhosale  
    **Program:** Advanced Certification in Applied Data Science, Machine Learning & AI  
    **Institution:** E&ICT Academy, IIT Guwahati  
    **Batch:** 12  
    **Capstone:** Meeting Notes Analyzer using LangGraph Multi-Agent Architecture  
    **Email:** aniketbhosale2808@gmail.com  
    **Phone:** +91-7385542808
    """
    )

    st.divider()

    st.markdown("## 🕒 Recent Runs")
    runs = sqlite_handler.get_all_runs(limit=10)

    if not runs:
        st.caption("No runs yet.")
    else:
        for run in runs:
            ts        = run.get("processed_at", "")[:19].replace("T", " ")
            mode      = run.get("input_mode", "text")
            n_actions = len(run.get("action_items") or [])
            label     = f"{ts} · {mode} · {n_actions} action(s)"

            with st.expander(label, expanded=False):
                summary = run.get("summary") or ""
                if summary:
                    st.markdown(f"**Summary:** {summary[:200]}{'...' if len(summary) > 200 else ''}")

                topics = run.get("topics") or []
                if topics:
                    st.markdown("**Topics:** " + " · ".join(topics))

                action_items = run.get("action_items") or []
                if action_items:
                    st.markdown("**Actions:**")
                    for item in action_items:
                        task     = item.get("task") or ""
                        owner    = item.get("owner") or "Not specified"
                        priority = item.get("priority") or "Medium"
                        if task:
                            st.markdown(f"- {task} — *{owner}* [{priority}]")


# -----------------------------------------------
# Main header
# -----------------------------------------------

st.title("📋 Meeting Notes Analyzer")
st.caption(
    f"Real-Time Meeting Intelligence · LangGraph + {_config.model_provider.upper()} "
    f"· `{_config.model_name}`"
)
st.divider()

# -----------------------------------------------
# Architecture expander
# -----------------------------------------------

with st.expander("⚙️ System Architecture", expanded=False):
    st.markdown("""
**Pipeline Flow:**
```
STT (Whisper) → Preprocessing → LangGraph Agents → Validation → SQLite → UI
```

| Agent | Role |
|---|---|
| 🗂 **Topic Agent** | Extracts 2–8 key discussion topics as short noun phrases |
| 📝 **Summary Agent** | Generates a concise 3–6 sentence prose summary |
| ✅ **Action Agent** | Extracts every concrete task commitment from the transcript |
| 👤 **Owner Agent** | Maps responsible owners to unassigned tasks using transcript context |
| 🔴 **Priority Agent** | Reviews and corrects priority levels (High / Medium / Low) |

All agents share a single `MeetingState` TypedDict. No agent communicates with another directly.
Every LLM output passes through a **validation layer** before entering state or being persisted.
    """)

st.divider()

# -----------------------------------------------
# Input section
# -----------------------------------------------

input_mode = st.radio(
    "Input mode",
    options=["Text transcript", "Audio upload", "Upload .txt file"],
    horizontal=True,
)

raw_transcript: str = ""
audio_filename: str = ""
mode_key: str = "text"

if input_mode == "Text transcript":
    raw_transcript = st.text_area(
        "Paste meeting transcript",
        height=280,
        placeholder=(
            "Alice: We need to finalize the Q3 roadmap by Friday.\n"
            "Bob: I can take ownership of the budget review.\n"
            "Alice: Great. Let's also schedule a follow-up next week."
        ),
    )
elif input_mode == "Upload .txt file":
    uploaded_txt = st.file_uploader(
        "Upload transcript file",
        type=["txt"],
        help="Plain text files only. File is read into the transcript field and not stored.",
    )
    if uploaded_txt is not None:
        try:
            raw_transcript = uploaded_txt.read().decode("utf-8")
            mode_key = "text"
            st.success(f"✅ Loaded '{uploaded_txt.name}' — {len(raw_transcript)} chars")
            with st.expander("Preview transcript"):
                st.text(raw_transcript[:2000] + ("..." if len(raw_transcript) > 2000 else ""))
        except UnicodeDecodeError:
            st.error("Could not decode file — make sure it is a plain UTF-8 text file.")
        except Exception as e:
            st.error(f"File read failed: {e}")
else:
    uploaded = st.file_uploader(
        "Upload audio recording",
        type=["mp3", "wav", "m4a", "ogg", "flac", "webm"],
        help="Audio is transcribed locally via Whisper — never stored.",
    )
    if uploaded is not None:
        audio_filename = uploaded.name
        mode_key = "audio"
        with st.spinner(f"Transcribing '{uploaded.name}' with Whisper..."):
            try:
                raw_transcript = transcribe(
                    audio_source=uploaded.read(),
                    filename=uploaded.name,
                )
                st.success(f"✅ Transcription complete — {len(raw_transcript)} chars")
                with st.expander("View transcript"):
                    st.text(raw_transcript)
            except Exception as e:
                st.error(f"Transcription failed: {e}")

st.divider()

# -----------------------------------------------
# Analyze button
# -----------------------------------------------

analyze_clicked = st.button(
    "⚡ Analyze Meeting",
    type="primary",
    disabled=not raw_transcript.strip(),
    use_container_width=True,
)

# -----------------------------------------------
# Pipeline execution
# -----------------------------------------------

if analyze_clicked and raw_transcript.strip():
    with st.spinner("Running intelligence pipeline..."):
        try:
            result: MeetingState = run_pipeline(
                raw_transcript=raw_transcript,
                provider=_get_provider(),
                input_mode=mode_key,
                audio_filename=audio_filename,
            )
        except Exception as e:
            st.error(f"Pipeline error: {e}")
            st.stop()

    # Persist (non-blocking)
    sqlite_handler.save_run(result)

    # Validation warnings (non-fatal)
    for err in (result.get("errors") or []):
        st.warning(f"⚠️ {err}")

    st.divider()
    _render_results(result)