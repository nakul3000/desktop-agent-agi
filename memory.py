"""
Memory store for desktop agent demo.

Currently provides:
- SQLite schema creation via `init_db`.
- Connection helpers with consistent row factory and ISO timestamps.

Embeddings/semantic recall will live in `embeddings.py`.
"""

from __future__ import annotations

import datetime as _dt
import json
import sqlite3
from pathlib import Path
from typing import Optional, Union

DEFAULT_DB_PATH = Path("memory.db")


def utc_now_iso() -> str:
    """Return an ISO-8601 UTC timestamp without microseconds."""
    return _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat() + "Z"


def get_connection(db_path: Optional[Union[str, Path]] = None) -> sqlite3.Connection:
    """
    Return a SQLite connection with Row factory enabled.

    Parameters
    ----------
    db_path : str | Path | None
        Override the database location; defaults to DEFAULT_DB_PATH.
    """
    path = Path(db_path) if db_path else DEFAULT_DB_PATH
    conn = sqlite3.connect(path, detect_types=sqlite3.PARSE_DECLTYPES)
    conn.row_factory = sqlite3.Row
    return conn


def _create_tables(conn: sqlite3.Connection) -> None:
    """Create all tables defined for the memory system if they are missing."""
    cur = conn.cursor()

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS turns (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            text TEXT NOT NULL,
            timestamp TEXT NOT NULL
        );
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS artifacts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            type TEXT NOT NULL,
            content TEXT NOT NULL,
            source_turn_id INTEGER,
            created_by TEXT,
            timestamp TEXT NOT NULL
        );
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS facts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            kind TEXT NOT NULL,
            key TEXT NOT NULL,
            value TEXT NOT NULL,
            meta_json TEXT NOT NULL,
            source_artifact_id INTEGER,
            confidence REAL NOT NULL,
            timestamp TEXT NOT NULL
        );
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS refs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            phrase TEXT NOT NULL,
            resolved_type TEXT NOT NULL,
            resolved_id INTEGER NOT NULL,
            reason TEXT NOT NULL,
            timestamp TEXT NOT NULL
        );
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS embeddings_map (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            item_type TEXT NOT NULL,
            item_id INTEGER NOT NULL,
            text_for_embedding TEXT NOT NULL,
            timestamp TEXT NOT NULL
        );
        """
    )

    conn.commit()


def init_db(db_path: Optional[str | Path] = None) -> None:
    """
    Initialize the SQLite database and create tables if they do not exist.

    Safe to call multiple times; does not drop data.
    """
    with get_connection(db_path) as conn:
        _create_tables(conn)


# --- Basic inserts --------------------------------------------------------- #

def store_turn(session_id: str, role: str, text: str, db_path: Optional[Union[str, Path]] = None) -> int:
    """Insert a conversation turn and return its id. Raises RuntimeError on failure."""
    timestamp = utc_now_iso()
    try:
        with get_connection(db_path) as conn:
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO turns (session_id, role, text, timestamp)
                VALUES (?, ?, ?, ?)
                """,
                (session_id, role, text, timestamp),
            )
            conn.commit()
            return int(cur.lastrowid)
    except sqlite3.Error as exc:
        raise RuntimeError(f"Failed to store turn for session {session_id}: {exc}") from exc


def _serialize_content(content: Union[str, dict]) -> str:
    if isinstance(content, dict):
        return json.dumps(content, ensure_ascii=False)
    return str(content)


def store_artifact(
    session_id: str,
    type: str,
    content: Union[str, dict],
    source_turn_id: Optional[int] = None,
    created_by: Optional[str] = None,
    db_path: Optional[Union[str, Path]] = None,
) -> int:
    """Insert an artifact row and return its id. Raises RuntimeError on failure."""
    timestamp = utc_now_iso()
    content_str = _serialize_content(content)
    try:
        with get_connection(db_path) as conn:
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO artifacts (session_id, type, content, source_turn_id, created_by, timestamp)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (session_id, type, content_str, source_turn_id, created_by, timestamp),
            )
            conn.commit()
            return int(cur.lastrowid)
    except sqlite3.Error as exc:
        raise RuntimeError(f"Failed to store artifact for session {session_id}: {exc}") from exc


def store_fact(
    session_id: str,
    kind: str,
    key: str,
    value: str,
    source_artifact_id: Optional[int] = None,
    confidence: float = 0.8,
    meta: Optional[dict] = None,
    db_path: Optional[Union[str, Path]] = None,
) -> int:
    """Insert a fact row and return its id. Raises RuntimeError on failure."""
    timestamp = utc_now_iso()
    meta_json = json.dumps(meta or {}, ensure_ascii=False)
    try:
        with get_connection(db_path) as conn:
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO facts (session_id, kind, key, value, meta_json, source_artifact_id, confidence, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (session_id, kind, key, value, meta_json, source_artifact_id, confidence, timestamp),
            )
            conn.commit()
            return int(cur.lastrowid)
    except sqlite3.Error as exc:
        raise RuntimeError(f"Failed to store fact for session {session_id}: {exc}") from exc


# --- Retrieval functions --------------------------------------------------- #

def get_turns_by_session(session_id: str, db_path: Optional[Union[str, Path]] = None, limit: Optional[int] = None) -> list[sqlite3.Row]:
    """Retrieve all turns for a session, optionally limited to the most recent N turns."""
    with get_connection(db_path) as conn:
        cur = conn.cursor()
        query = "SELECT * FROM turns WHERE session_id = ? ORDER BY timestamp ASC"
        params = [session_id]
        if limit:
            query += " LIMIT ?"
            params.append(limit)
        cur.execute(query, params)
        return cur.fetchall()


def get_artifacts_by_session(session_id: str, artifact_type: Optional[str] = None, db_path: Optional[Union[str, Path]] = None) -> list[sqlite3.Row]:
    """Retrieve artifacts for a session, optionally filtered by type."""
    with get_connection(db_path) as conn:
        cur = conn.cursor()
        if artifact_type:
            cur.execute(
                "SELECT * FROM artifacts WHERE session_id = ? AND type = ? ORDER BY timestamp DESC",
                (session_id, artifact_type)
            )
        else:
            cur.execute(
                "SELECT * FROM artifacts WHERE session_id = ? ORDER BY timestamp DESC",
                (session_id,)
            )
        return cur.fetchall()


def get_facts_by_session(session_id: str, fact_kind: Optional[str] = None, db_path: Optional[Union[str, Path]] = None) -> list[sqlite3.Row]:
    """Retrieve facts for a session, optionally filtered by kind."""
    with get_connection(db_path) as conn:
        cur = conn.cursor()
        if fact_kind:
            cur.execute(
                "SELECT * FROM facts WHERE session_id = ? AND kind = ? ORDER BY timestamp DESC",
                (session_id, fact_kind)
            )
        else:
            cur.execute(
                "SELECT * FROM facts WHERE session_id = ? ORDER BY timestamp DESC",
                (session_id,)
            )
        return cur.fetchall()


def search_facts_by_key(session_id: str, key_pattern: str, db_path: Optional[Union[str, Path]] = None) -> list[sqlite3.Row]:
    """Search for facts with keys matching a pattern (supports LIKE)."""
    with get_connection(db_path) as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT * FROM facts WHERE session_id = ? AND key LIKE ? ORDER BY timestamp DESC",
            (session_id, f"%{key_pattern}%")
        )
        return cur.fetchall()


__all__ = [
    "DEFAULT_DB_PATH",
    "get_connection",
    "init_db",
    "utc_now_iso",
    "store_turn",
    "store_artifact",
    "store_fact",
    "get_turns_by_session",
    "get_artifacts_by_session",
    "get_facts_by_session",
    "search_facts_by_key",
]
