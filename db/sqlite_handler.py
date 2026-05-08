"""
db/sqlite_handler.py

SQLite persistence layer.

Stores structured meeting intelligence after a successful pipeline run.
Provides lightweight retrieval for history display in the Streamlit UI.

Storage philosophy (from architecture spec):
    Store intelligence, not noise.

Stored:
    - cleaned transcript
    - summary
    - topics (JSON)
    - action items (JSON)
    - metadata (run_id, timestamps, provider, input_mode)

NOT stored:
    - raw_transcript     (noisy, unreduced — use cleaned_transcript)
    - audio files        (transient, discarded post-transcription)
    - intermediate state (only final validated output persists)
    - prompt histories   (not part of the intelligence record)
    - validation errors  (diagnostic only, not stored)

Design rules:
    - db path sourced from config (SQLITE_DB_PATH)
    - schema initialized lazily on first use via init_db()
    - all public functions are self-contained — no shared connection state
    - JSON columns store serialized list/dict fields
    - no ORM — plain sqlite3 with parameterized queries
    - never raises on read failures — returns None/[] and logs
"""

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Generator, Optional

from core.config import get_config
from core.state import MeetingState
from core.utils.logger import get_logger

logger = get_logger(__name__)


# -----------------------------------------------
# Schema
# -----------------------------------------------

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS meeting_records (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id          TEXT    NOT NULL UNIQUE,
    input_mode      TEXT    NOT NULL DEFAULT 'text',
    audio_filename  TEXT    NOT NULL DEFAULT '',
    processed_at    TEXT    NOT NULL,
    model_provider  TEXT    NOT NULL,
    model_name      TEXT    NOT NULL,
    cleaned_transcript TEXT NOT NULL,
    summary         TEXT    NOT NULL,
    topics          TEXT    NOT NULL,   -- JSON array of strings
    action_items    TEXT    NOT NULL,   -- JSON array of ActionItem dicts
    created_at      TEXT    NOT NULL    -- local insert timestamp
);
"""

_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_run_id      ON meeting_records (run_id);
CREATE INDEX IF NOT EXISTS idx_processed_at ON meeting_records (processed_at);
"""


# -----------------------------------------------
# Connection context manager
# -----------------------------------------------

@contextmanager
def _get_connection() -> Generator[sqlite3.Connection, None, None]:
    """
    Open a short-lived SQLite connection, yield it, then close.
    Commits on clean exit, rolls back on exception.
    """
    db_path = get_config().db_path
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row      # column access by name
    conn.execute("PRAGMA journal_mode=WAL")   # safer concurrent reads
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# -----------------------------------------------
# Public API
# -----------------------------------------------

def init_db() -> None:
    """
    Initialize the database schema.

    Creates the meeting_records table and indexes if they do not
    already exist. Safe to call multiple times — idempotent.

    Called once at Streamlit app startup before any pipeline run.
    """
    try:
        with _get_connection() as conn:
            conn.executescript(_CREATE_TABLE_SQL + _INDEX_SQL)
        logger.info("Database initialized: %s", get_config().db_path)
    except Exception as e:
        logger.error("Database init failed: %s", e)
        raise


def save_run(state: MeetingState) -> Optional[int]:
    """
    Persist a completed pipeline run to the database.

    Args:
        state: Final MeetingState from run_pipeline(). Must have
               metadata populated (i.e. pipeline completed successfully).

    Returns:
        The integer row id of the inserted record, or None on failure.

    Does not raise — logs error and returns None if the write fails.
    This keeps the Streamlit UI functional even if persistence fails.
    """
    meta = state.get("metadata")
    if meta is None:
        logger.warning("save_run: state has no metadata — pipeline may not have completed")
        return None

    record = {
        "run_id":            meta["run_id"],
        "input_mode":        meta["input_mode"],
        "audio_filename":    meta["audio_filename"],
        "processed_at":      meta["processed_at"],
        "model_provider":    meta["model_provider"],
        "model_name":        meta["model_name"],
        "cleaned_transcript": state.get("cleaned_transcript") or "",
        "summary":           state.get("summary") or "",
        "topics":            json.dumps(state.get("topics") or []),
        "action_items":      json.dumps(state.get("action_items") or []),
        "created_at":        datetime.now(timezone.utc).isoformat(),
    }

    sql = """
        INSERT INTO meeting_records (
            run_id, input_mode, audio_filename, processed_at,
            model_provider, model_name, cleaned_transcript,
            summary, topics, action_items, created_at
        ) VALUES (
            :run_id, :input_mode, :audio_filename, :processed_at,
            :model_provider, :model_name, :cleaned_transcript,
            :summary, :topics, :action_items, :created_at
        )
    """

    try:
        with _get_connection() as conn:
            cursor = conn.execute(sql, record)
            row_id = cursor.lastrowid
        logger.info("Run saved: run_id=%s row_id=%d", meta["run_id"], row_id)
        return row_id
    except sqlite3.IntegrityError:
        logger.warning("save_run: duplicate run_id=%s — already persisted", meta["run_id"])
        return None
    except Exception as e:
        logger.error("save_run: write failed — %s", e)
        return None


def get_all_runs(limit: int = 50) -> list[dict]:
    """
    Retrieve recent meeting records for history display.

    Args:
        limit: Maximum number of records to return. Ordered by
               most recent first (processed_at DESC).

    Returns:
        List of record dicts with topics and action_items deserialized.
        Returns [] on any read failure.
    """
    sql = """
        SELECT id, run_id, input_mode, audio_filename, processed_at,
               model_provider, model_name, summary, topics, action_items, created_at
        FROM meeting_records
        ORDER BY processed_at DESC
        LIMIT ?
    """
    try:
        with _get_connection() as conn:
            rows = conn.execute(sql, (limit,)).fetchall()
        return [_deserialize_row(dict(row)) for row in rows]
    except Exception as e:
        logger.error("get_all_runs: read failed — %s", e)
        return []


def get_run_by_id(run_id: str) -> Optional[dict]:
    """
    Retrieve a single meeting record by run_id.

    Args:
        run_id: UUID string from PipelineMetadata.

    Returns:
        Record dict with deserialized fields, or None if not found.
    """
    sql = """
        SELECT * FROM meeting_records WHERE run_id = ?
    """
    try:
        with _get_connection() as conn:
            row = conn.execute(sql, (run_id,)).fetchone()
        if row is None:
            return None
        return _deserialize_row(dict(row))
    except Exception as e:
        logger.error("get_run_by_id: read failed for run_id=%s — %s", run_id, e)
        return None


def get_run_count() -> int:
    """
    Return total number of stored meeting records.
    Returns 0 on failure.
    """
    try:
        with _get_connection() as conn:
            row = conn.execute("SELECT COUNT(*) FROM meeting_records").fetchone()
        return row[0] if row else 0
    except Exception as e:
        logger.error("get_run_count: failed — %s", e)
        return 0


# -----------------------------------------------
# Internal helpers
# -----------------------------------------------

def _deserialize_row(row: dict) -> dict:
    """
    Deserialize JSON-encoded columns back to Python objects.
    Returns the row with topics and action_items as lists.
    """
    try:
        row["topics"] = json.loads(row.get("topics") or "[]")
    except (json.JSONDecodeError, TypeError):
        row["topics"] = []

    try:
        row["action_items"] = json.loads(row.get("action_items") or "[]")
    except (json.JSONDecodeError, TypeError):
        row["action_items"] = []

    return row