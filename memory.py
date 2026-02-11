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
import uuid
from pathlib import Path
from typing import Optional, Union, Iterable, Dict, Any

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

    # Users table for mapping user identity
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT UNIQUE NOT NULL,
            name TEXT,
            email TEXT,
            meta_json TEXT,
            created_at TEXT NOT NULL
        );
        """
    )

    # Sessions table to tie session_id to a user
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT UNIQUE NOT NULL,
            user_id TEXT,
            created_at TEXT NOT NULL
        );
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS turns (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            user_id TEXT,
            role TEXT NOT NULL,
            tool_name TEXT,
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
            user_id TEXT,
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
            user_id TEXT,
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
            user_id TEXT,
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
            user_id TEXT,
            item_type TEXT NOT NULL,
            item_id INTEGER NOT NULL,
            text_for_embedding TEXT NOT NULL,
            timestamp TEXT NOT NULL
        );
        """
    )

    conn.commit()

    # Backfill helper: add user_id column if table pre-exists without it
    def _ensure_column(table: str, column: str, col_type: str) -> None:
        cur.execute(f"PRAGMA table_info({table});")
        cols = [row[1] for row in cur.fetchall()]
        if column not in cols:
            cur.execute(f"ALTER TABLE {table} ADD COLUMN {column} {col_type};")

    for tbl in ("turns", "artifacts", "facts", "refs", "embeddings_map"):
        _ensure_column(tbl, "user_id", "TEXT")
    # Ensure tool_name exists on turns
    _ensure_column("turns", "tool_name", "TEXT")


def init_db(db_path: Optional[str | Path] = None) -> None:
    """
    Initialize the SQLite database and create tables if they do not exist.

    Safe to call multiple times; does not drop data.
    """
    with get_connection(db_path) as conn:
        _create_tables(conn)


# --- User and session helpers --------------------------------------------- #

def register_user(user_id: str, name: Optional[str] = None, email: Optional[str] = None, meta: Optional[dict] = None, db_path: Optional[Union[str, Path]] = None) -> int:
    """
    Insert or ignore a user record. Returns the row id (existing or new).
    """
    created_at = utc_now_iso()
    meta_json = json.dumps(meta or {}, ensure_ascii=False)
    with get_connection(db_path) as conn:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT OR IGNORE INTO users (user_id, name, email, meta_json, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (user_id, name, email, meta_json, created_at),
        )
        # fetch id
        cur.execute("SELECT id FROM users WHERE user_id = ?", (user_id,))
        row = cur.fetchone()
        return int(row["id"]) if row else -1


def start_session(user_id: Optional[str] = None, session_id: Optional[str] = None, db_path: Optional[Union[str, Path]] = None) -> str:
    """
    Create a new session row and return the session_id. If session_id is not provided,
    a UUID4 string is generated. User_id is optional.
    """
    sid = session_id or str(uuid.uuid4())
    created_at = utc_now_iso()
    with get_connection(db_path) as conn:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT OR IGNORE INTO sessions (session_id, user_id, created_at)
            VALUES (?, ?, ?)
            """,
            (sid, user_id, created_at),
        )
    return sid


# --- Basic inserts --------------------------------------------------------- #

def store_turn(
    session_id: str,
    role: str,
    text: str,
    user_id: Optional[str] = None,
    tool_name: Optional[str] = None,
    db_path: Optional[Union[str, Path]] = None,
) -> int:
    """Insert a conversation turn and return its id. Raises RuntimeError on failure."""
    timestamp = utc_now_iso()
    try:
        with get_connection(db_path) as conn:
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO turns (session_id, user_id, role, tool_name, text, timestamp)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (session_id, user_id, role, tool_name, text, timestamp),
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
    user_id: Optional[str] = None,
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
                INSERT INTO artifacts (session_id, user_id, type, content, source_turn_id, created_by, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (session_id, user_id, type, content_str, source_turn_id, created_by, timestamp),
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
    user_id: Optional[str] = None,
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
                INSERT INTO facts (session_id, user_id, kind, key, value, meta_json, source_artifact_id, confidence, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (session_id, user_id, kind, key, value, meta_json, source_artifact_id, confidence, timestamp),
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


# --- Context assembly ------------------------------------------------------ #

def _rows_to_dict(rows: Iterable[sqlite3.Row], fields: Optional[Iterable[str]] = None) -> list[Dict[str, Any]]:
    out = []
    for row in rows:
        d = dict(row)
        if fields:
            d = {k: d[k] for k in fields if k in d}
        out.append(d)
    return out


def get_context(
    session_id: str,
    user_id: Optional[str] = None,
    k_recent_turns: int = 6,
    k_recent_artifacts: int = 6,
    k_recent_facts: int = 10,
    db_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """
    Gather recent turns, artifacts, and facts for a session (optionally filtered by user_id).
    Designed to feed into a generator prompt.
    """
    with get_connection(db_path) as conn:
        cur = conn.cursor()

        def _fetch(table: str, limit: int) -> list[sqlite3.Row]:
            query = f"SELECT * FROM {table} WHERE session_id = ?"
            params: list[Any] = [session_id]
            if user_id:
                query += " AND (user_id IS NULL OR user_id = ?)"
                params.append(user_id)
            query += " ORDER BY timestamp DESC"
            if limit:
                query += " LIMIT ?"
                params.append(limit)
            cur.execute(query, params)
            return cur.fetchall()

        turns = _fetch("turns", k_recent_turns)
        artifacts = _fetch("artifacts", k_recent_artifacts)
        facts = _fetch("facts", k_recent_facts)

    return {
        "recent_turns": _rows_to_dict(turns),
        "recent_artifacts": _rows_to_dict(artifacts),
        "recent_facts": _rows_to_dict(facts),
    }


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
    "get_context",
    "register_user",
    "start_session",
]
# ------------------------------------------------------------------ #
# OO wrapper for compatibility with code that expects `Memory()`
# ------------------------------------------------------------------ #

class Memory:
    """
    Lightweight wrapper around the functional memory module.

    This exists for compatibility with components (e.g., AgentCore/app.py)
    that expect a Memory object rather than free functions.
    """

    def __init__(self, db_path: Optional[Union[str, Path]] = None):
        self.db_path = db_path
        init_db(db_path=db_path)

    def register_user(self, user_id: str, name: Optional[str] = None, email: Optional[str] = None, meta: Optional[dict] = None) -> int:
        return register_user(user_id=user_id, name=name, email=email, meta=meta, db_path=self.db_path)

    def start_session(self, user_id: Optional[str] = None, session_id: Optional[str] = None) -> str:
        return start_session(user_id=user_id, session_id=session_id, db_path=self.db_path)

    def store_turn(self, session_id: str, role: str, text: str, user_id: Optional[str] = None, tool_name: Optional[str] = None) -> int:
        return store_turn(session_id=session_id, role=role, text=text, user_id=user_id, tool_name=tool_name, db_path=self.db_path)

    def store_artifact(self, session_id: str, type: str, content: str, source_turn_id: Optional[int] = None, created_by: Optional[str] = None, user_id: Optional[str] = None) -> int:
        return store_artifact(
            session_id=session_id,
            type=type,
            content=content,
            source_turn_id=source_turn_id,
            created_by=created_by,
            user_id=user_id,
            db_path=self.db_path,
        )

    def get_context(self, session_id: str, user_id: Optional[str] = None) -> Dict[str, Any]:
        return get_context(session_id=session_id, user_id=user_id, db_path=self.db_path)
