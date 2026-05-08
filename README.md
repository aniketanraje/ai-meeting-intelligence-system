# 📋 AI Meeting Notes Analyzer
## Live Demo

https://multiagent-meeting-analyzer.streamlit.app/

## Features

- Multi-agent meeting analysis

- Audio transcription with Whisper

- Action item extraction

- Priority classification

- Owner assignment

- SQLite persistence

- Streamlit Cloud deployment
> **Real-Time Meeting Intelligence System with Stateful Multi-Agent LLM Pipeline**
>
> A modular AI pipeline that transforms unstructured meeting transcripts and audio recordings into validated, structured, actionable intelligence — powered by LangGraph orchestration and Groq inference.

<br/>

[![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)](https://python.org)
[![LangGraph](https://img.shields.io/badge/LangGraph-1.1.10-green)](https://github.com/langchain-ai/langgraph)
[![Groq](https://img.shields.io/badge/Groq-llama--3.3--70b-orange)](https://groq.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40.2-red?logo=streamlit)](https://streamlit.io)
[![Whisper](https://img.shields.io/badge/Whisper-local--STT-purple)](https://github.com/openai/whisper)
[![SQLite](https://img.shields.io/badge/SQLite-persistence-lightgrey?logo=sqlite)](https://sqlite.org)

---

## 🧠 What This Project Is

This is **not a chatbot**.
This is **not a generic summarizer**.
This is **not a frontend-heavy SaaS demo**.

This is a **controlled AI processing pipeline** — a multi-agent orchestration system that treats LLMs as structured reasoning components operating inside a deterministic architecture.

> *"LLMs are reasoning components, not the system itself."*

The pipeline accepts raw meeting data (text or audio), passes it through five specialized AI agents coordinated by LangGraph's state graph, validates every output, and delivers machine-readable structured intelligence.

---

## ✨ Features

### 🎯 Core Intelligence
- **Meeting Summary** — concise prose paragraph capturing decisions and outcomes
- **Key Topic Extraction** — 2–8 short noun-phrase topics from the discussion
- **Action Item Extraction** — every concrete commitment extracted as a structured task
- **Owner Mapping** — responsible person assigned per action item from transcript context
- **Priority Classification** — High / Medium / Low assigned and validated per item

### 🔊 Input Modes
- **Text transcript** — paste any raw meeting transcript directly
- **Audio upload** — upload `.mp3`, `.wav`, `.m4a`, `.ogg`, `.flac`, `.webm`; local Whisper handles transcription

### 🏗️ Engineering Quality
- **Provider-agnostic** — swap Groq, Gemini, OpenAI, or Claude via a single `.env` change
- **Validation-first** — every LLM output is schema-validated before entering state
- **Structured output contract** — deterministic JSON every time, no hallucinated fields
- **SQLite persistence** — structured intelligence stored; noise discarded
- **Lightweight preprocessing** — filler words, speaker tag normalization, encoding artifacts cleaned before agents see the transcript

### 🖥️ UI
- Clean Streamlit interface — text input, audio upload, structured output rendering
- Metrics row — topics count, action items, high priority items at a glance
- Full JSON output expander — raw structured output always accessible
- Run history in sidebar — recent runs with expandable summaries
- System architecture panel — pipeline explained inline

---

## 🏛️ Architecture

### System Flow

```
Text Input ──────────────────────────────────────────────────┐
                                                              ▼
Audio Upload → [ Whisper STT ] ──────────────────► Preprocessing Layer
                                                              │
                                                              ▼
                                                   State Initialization
                                                   (MeetingState TypedDict)
                                                              │
                                                              ▼
                                          ┌─── LangGraph StateGraph ────┐
                                          │                              │
                                          │   ┌─────────────────────┐   │
                                          │   │   Topic Agent        │   │
                                          │   └──────────┬──────────┘   │
                                          │              │               │
                                          │   ┌──────────▼──────────┐   │
                                          │   │   Summary Agent      │   │
                                          │   └──────────┬──────────┘   │
                                          │              │               │
                                          │   ┌──────────▼──────────┐   │
                                          │   │   Action Agent       │   │
                                          │   └──────────┬──────────┘   │
                                          │              │               │
                                          │   ┌──────────▼──────────┐   │
                                          │   │   Owner Agent        │   │
                                          │   └──────────┬──────────┘   │
                                          │              │               │
                                          │   ┌──────────▼──────────┐   │
                                          │   │   Priority Agent     │   │
                                          │   └──────────┬──────────┘   │
                                          │              │               │
                                          └──────────────┼───────────────┘
                                                         │
                                                         ▼
                                               Validation Layer
                                          (schema, dedup, normalization)
                                                         │
                                                         ▼
                                             SQLite Persistence
                                                         │
                                                         ▼
                                          Structured Output → Streamlit UI
```

### Agent Responsibilities

| Agent | Input | Output | Responsibility |
|---|---|---|---|
| **Topic Agent** | `cleaned_transcript` | `topics: list[str]` | Extracts 2–8 key discussion topics as noun phrases |
| **Summary Agent** | `cleaned_transcript` | `summary: str` | Generates a concise 3–6 sentence prose summary |
| **Action Agent** | `cleaned_transcript` | `action_items: list[ActionItem]` | Extracts every concrete task commitment |
| **Owner Agent** | `cleaned_transcript` + `action_items` | `action_items[*].owner` | Maps owners to unassigned tasks from context |
| **Priority Agent** | `cleaned_transcript` + `action_items` | `action_items[*].priority` | Reviews and corrects priority levels |

### State Contract

Every agent reads from and writes to a single shared `MeetingState` TypedDict. Agents never communicate with each other directly.

```python
class MeetingState(TypedDict):
    raw_transcript:     str
    cleaned_transcript: str
    topics:             list[str]
    summary:            str
    action_items:       list[ActionItem]
    errors:             list[str]
    metadata:           PipelineMetadata | None

class ActionItem(TypedDict):
    task:     str
    owner:    str      # "Not specified" if unknown
    priority: str      # "High" | "Medium" | "Low"
    deadline: str      # "" if not mentioned
```

### Output Contract

The system always produces deterministic machine-readable JSON:

```json
{
  "summary": "The team agreed to finalize the Q3 roadmap by Friday...",
  "topics": ["Q3 roadmap", "Budget planning", "Hiring timeline"],
  "action_items": [
    {
      "task": "Finalize Q3 roadmap document",
      "owner": "Alice",
      "priority": "High",
      "deadline": "Friday"
    },
    {
      "task": "Send budget estimates to finance",
      "owner": "Bob",
      "priority": "Medium",
      "deadline": ""
    }
  ]
}
```

---

## 🗂️ Project Structure

```
ai_meeting_analyzer/
│
├── app/
│   └── streamlit_app.py          # Thin UI layer — input, output rendering, history
│
├── core/
│   ├── agents/
│   │   ├── topic_agent.py        # Topic extraction
│   │   ├── summary_agent.py      # Summary generation
│   │   ├── action_agent.py       # Action item extraction
│   │   ├── owner_agent.py        # Owner mapping
│   │   └── priority_agent.py     # Priority classification
│   │
│   ├── graph/
│   │   └── pipeline.py           # LangGraph StateGraph orchestration
│   │
│   ├── prompts/
│   │   ├── topic_prompt.txt      # Externalized, editable prompt files
│   │   ├── summary_prompt.txt
│   │   ├── action_prompt.txt
│   │   ├── owner_prompt.txt
│   │   └── priority_prompt.txt
│   │
│   ├── providers/
│   │   ├── base_provider.py      # Abstract LLM interface
│   │   ├── factory.py            # Config-driven provider resolver
│   │   ├── groq_provider.py      # Groq implementation (active)
│   │   └── gemini_provider.py    # Gemini implementation (available)
│   │
│   ├── utils/
│   │   ├── preprocessing.py      # Transcript cleaning and normalization
│   │   ├── validation.py         # Schema enforcement, dedup, normalization
│   │   └── logger.py             # Centralized structured logging
│   │
│   ├── config.py                 # Typed configuration, .env loading, validation
│   └── state.py                  # MeetingState TypedDict — pipeline backbone
│
├── voice/
│   └── stt.py                    # Whisper STT — audio → text only
│
├── db/
│   └── sqlite_handler.py         # SQLite persistence layer
│
├── data/
│   └── sample_transcripts/       # Test fixtures (6 scenarios)
│
├── outputs/                      # JSON export directory
├── tests/                        # Test modules
│
├── .env.example                  # Environment template
├── .gitignore
├── requirements.txt
├── Dockerfile
├── run.sh                        # One-command local startup
└── README.md
```

---

## ⚙️ Tech Stack

| Layer | Technology | Version | Purpose |
|---|---|---|---|
| **Language** | Python | 3.12 | Core runtime |
| **Orchestration** | LangGraph | 1.1.10 | Multi-agent state graph |
| **LLM Framework** | LangChain | 0.3.7 | Provider abstractions |
| **LLM Provider** | Groq (Llama 3.3 70B) | 0.11.0 | Fast inference |
| **STT** | OpenAI Whisper | local | Audio transcription |
| **Validation** | Pydantic | 2.9.2 | Schema enforcement |
| **Frontend** | Streamlit | 1.40.2 | Thin demo UI |
| **Database** | SQLite | stdlib | Lightweight persistence |
| **Environment** | python-dotenv | 1.0.1 | Secret management |

---

## 🚀 Setup & Installation

### Prerequisites

- Python 3.12+
- `ffmpeg` installed (required by Whisper for audio processing)
  ```bash
  # macOS
  brew install ffmpeg

  # Ubuntu/Debian
  sudo apt install ffmpeg
  ```
- A [Groq API key](https://console.groq.com) (free tier available)

### Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/your-username/ai-meeting-analyzer.git
cd ai-meeting-analyzer

# 2. Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate          # macOS/Linux
# venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Open .env and set your GROQ_API_KEY

# 5. Run
bash run.sh
# → Opens at http://localhost:8501
```

### Or use the run script directly

`run.sh` handles everything — venv creation, dep install, dir setup, and launch:

```bash
bash run.sh
```

### Docker

```bash
# Build
docker build -t ai-meeting-analyzer .

# Run (pass your .env file)
docker run -p 8501:8501 --env-file .env ai-meeting-analyzer

# → Opens at http://localhost:8501
```

---

## 🔑 Environment Variables

Copy `.env.example` to `.env` and fill in your values.

```bash
# Provider selection — change this to switch LLMs
MODEL_PROVIDER=groq           # groq | gemini | openai | claude

# API Keys — only the active provider's key is required
GROQ_API_KEY=gsk_your_key_here
# GEMINI_API_KEY=your_key_here
# OPENAI_API_KEY=your_key_here
# ANTHROPIC_API_KEY=your_key_here

# Model override (optional — sensible defaults per provider)
# GROQ_MODEL_NAME=llama-3.3-70b-versatile

# LLM behavior — keep at 0.0 for deterministic extraction
LLM_TEMPERATURE=0.0

# Database path
SQLITE_DB_PATH=db/meeting_intelligence.db

# Whisper model size: tiny | base | small | medium | large
WHISPER_MODEL=base

# Logging: DEBUG | INFO | WARNING | ERROR
LOG_LEVEL=INFO
```

### Switching LLM Providers

The provider abstraction means **zero code changes** are needed to switch:

```bash
# Use Gemini
MODEL_PROVIDER=gemini
GEMINI_API_KEY=your_gemini_key

# Use OpenAI
MODEL_PROVIDER=openai
OPENAI_API_KEY=your_openai_key

# Use Groq (default — recommended for speed)
MODEL_PROVIDER=groq
GROQ_API_KEY=your_groq_key
```

---

## 🧪 Testing

Sample transcripts for testing all pipeline scenarios are provided in `data/sample_transcripts/`:

| File | Scenario |
|---|---|
| `clean_meeting.txt` | Standard well-formatted meeting |
| `noisy_meeting.txt` | Whisper-style output with fillers and artifacts |
| `no_action_items.txt` | Discussion-only, no tasks |
| `multiple_owners.txt` | Many speakers, complex ownership |
| `urgent_priorities.txt` | High-urgency language throughout |
| `missing_ownership.txt` | Tasks with no clear owner |

Run tests:
```bash
source venv/bin/activate
python -m pytest tests/ -v
```

---

## 📐 Engineering Philosophy

This project is guided by three core principles:

**1. "LLMs are reasoning components, not the system itself."**
LLMs operate inside a deterministic architecture. Every output is validated. No raw LLM output reaches the database or UI without passing through the validation layer.

**2. "The provider is a dependency. The architecture is the actual system."**
The orchestration pipeline is vendor-agnostic by design. Groq, Gemini, OpenAI, and Claude are interchangeable backends. Changing providers is a `.env` edit.

**3. "Prototype-grade professional engineering, not enterprise overengineering."**
Every module justifies its existence. Clarity over cleverness. Maintainability over abstraction addiction. No Docker Compose, no Kubernetes, no auth systems, no distributed infrastructure.

---

## 🗺️ Versioning Roadmap

### ✅ Completed

| Version | Scope |
|---|---|
| `v0.0.1` | Project skeleton, module structure, environment setup |
| `v0.0.2` | Core text pipeline — all 5 agents, LangGraph orchestration, validation, SQLite, Streamlit UI |
| `v0.0.3` | Audio input via local Whisper STT |
| `v0.0.4` | Groq provider integration, provider factory, UI polish |

### 🔭 Future Scope

#### v0.1.0 — Live Voice Input 🎙️
Real-time microphone capture with streaming transcription — no file upload needed. The meeting is transcribed as it happens and the pipeline runs post-call automatically.

- Browser microphone access via Streamlit audio component
- Streaming Whisper transcription (faster-whisper backend)
- Auto-trigger pipeline on recording stop
- Live transcript preview during recording

#### v0.2.0 — Document Intelligence 📄
Expand input modes beyond transcripts and audio. Meeting agendas, notes documents, and email threads as pipeline inputs.

- **PDF ingestion** — extract text from meeting notes PDFs, agendas, and reports
- **Word document (.docx) support** — parse structured meeting minutes
- **OCR pipeline** — scanned documents and image-based PDFs via Tesseract or EasyOCR
- **Handwriting recognition** — whiteboard photos and handwritten notes
- Multi-document context merging before pipeline entry

#### v0.3.0 — RAG Memory Layer 🧠
Cross-meeting intelligence. Ask questions across your entire meeting history.

- ChromaDB or FAISS vector store for meeting embeddings
- Semantic search across historical meeting records
- "What did we decide about X last quarter?" queries
- Action item tracking across multiple meetings — follow-up detection
- Owner history — all tasks assigned to a person across all meetings

#### v0.4.0 — Real-Time Collaborative Mode 👥
Live multi-participant meeting intelligence.

- WebSocket-based real-time transcript streaming
- Live agent output as the meeting progresses
- Shared meeting intelligence dashboard for all participants
- Mid-meeting action item surfacing

#### v0.5.0 — Analytics Dashboard 📊
Meeting intelligence over time.

- Action item completion tracking
- Meeting productivity metrics
- Topic trend analysis across meetings
- Owner workload heatmaps
- Export to PDF / Excel reports

#### v1.0.0 — Production Hardening 🔒
Enterprise-ready foundations (intentionally deferred from prototype).

- Authentication (OAuth2)
- Multi-user support with isolated workspaces
- API endpoint layer (FastAPI)
- Async pipeline execution
- Provider fallback chains (Groq → Gemini → OpenAI)
- Rate limit handling and retry logic
- Cloud deployment (containerized)

---

## ⚠️ Known Limitations (Current Scope)

- **No authentication** — single-user local prototype
- **No real-time streaming** — pipeline runs after full transcript is available
- **No cloud deployment** — local execution only
- **Whisper model download** — first audio run downloads model weights (~74MB for `base`)
- **LLM non-determinism** — `temperature=0.0` minimizes variance but doesn't eliminate it; complex transcripts may produce slightly different outputs across runs
- **No cross-meeting memory** — each pipeline run is independent

---

## 👨‍💻 Developer

**Aniket Bhosale**
Post-Graduate Program in Data Science, Machine Learning & AI
IIT Guwahati — EICT Academy | Applied DSML & AI Track

Capstone Project — AI Systems Engineering & Multi-Agent Orchestration

---

## 📄 License

This project is developed as part of an academic capstone program.
All rights reserved © Aniket Bhosale.

---

<div align="center">

**Built with engineering discipline. Not vibe-coded.**

*"Strong pipeline > feature bloat."*

</div>