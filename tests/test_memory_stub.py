import os
from pathlib import Path

import pytest

import memory


def test_init_db_creates_file(tmp_path: Path) -> None:
    db_path = tmp_path / "memory.db"
    memory.init_db(db_path)
    assert db_path.exists(), "init_db should create the SQLite file"


def test_store_turn(tmp_path: Path) -> None:
    db_path = tmp_path / "memory.db"
    memory.init_db(db_path)
    
    turn_id = memory.store_turn(
        session_id="test_session",
        role="user", 
        text="Hello world",
        db_path=db_path
    )
    
    assert turn_id > 0, "store_turn should return a valid ID"
    
    # Verify the turn was stored
    conn = memory.get_connection(db_path)
    cur = conn.cursor()
    cur.execute("SELECT * FROM turns WHERE id = ?", (turn_id,))
    row = cur.fetchone()
    conn.close()
    
    assert row is not None, "Turn should be stored in database"
    assert row["session_id"] == "test_session"
    assert row["role"] == "user"
    assert row["text"] == "Hello world"


def test_store_artifact(tmp_path: Path) -> None:
    db_path = tmp_path / "memory.db"
    memory.init_db(db_path)
    
    artifact_id = memory.store_artifact(
        session_id="test_session",
        type="document",
        content={"title": "Test Doc", "content": "Test content"},
        created_by="test_user",
        db_path=db_path
    )
    
    assert artifact_id > 0, "store_artifact should return a valid ID"
    
    # Verify the artifact was stored
    conn = memory.get_connection(db_path)
    cur = conn.cursor()
    cur.execute("SELECT * FROM artifacts WHERE id = ?", (artifact_id,))
    row = cur.fetchone()
    conn.close()
    
    assert row is not None, "Artifact should be stored in database"
    assert row["session_id"] == "test_session"
    assert row["type"] == "document"
    assert row["created_by"] == "test_user"


def test_store_fact(tmp_path: Path) -> None:
    db_path = tmp_path / "memory.db"
    memory.init_db(db_path)
    
    fact_id = memory.store_fact(
        session_id="test_session",
        kind="preference",
        key="theme",
        value="dark",
        confidence=0.9,
        meta={"source": "user_input"},
        db_path=db_path
    )
    
    assert fact_id > 0, "store_fact should return a valid ID"
    
    # Verify the fact was stored
    conn = memory.get_connection(db_path)
    cur = conn.cursor()
    cur.execute("SELECT * FROM facts WHERE id = ?", (fact_id,))
    row = cur.fetchone()
    conn.close()
    
    assert row is not None, "Fact should be stored in database"
    assert row["session_id"] == "test_session"
    assert row["kind"] == "preference"
    assert row["key"] == "theme"
    assert row["value"] == "dark"
    assert row["confidence"] == 0.9


def test_get_turns_by_session(tmp_path: Path) -> None:
    db_path = tmp_path / "memory.db"
    memory.init_db(db_path)
    
    # Store some test turns
    memory.store_turn("test_session", "user", "Hello", db_path=db_path)
    memory.store_turn("test_session", "assistant", "Hi there!", db_path=db_path)
    memory.store_turn("other_session", "user", "Other message", db_path=db_path)
    
    # Get turns for test_session
    turns = memory.get_turns_by_session("test_session", db_path=db_path)
    
    assert len(turns) == 2, "Should return 2 turns for test_session"
    assert turns[0]["role"] == "user"
    assert turns[0]["text"] == "Hello"
    assert turns[1]["role"] == "assistant"
    assert turns[1]["text"] == "Hi there!"


def test_get_artifacts_by_session(tmp_path: Path) -> None:
    db_path = tmp_path / "memory.db"
    memory.init_db(db_path)
    
    # Store some test artifacts
    memory.store_artifact("test_session", "document", {"title": "Doc 1"}, db_path=db_path)
    memory.store_artifact("test_session", "image", {"url": "test.jpg"}, db_path=db_path)
    memory.store_artifact("other_session", "document", {"title": "Doc 2"}, db_path=db_path)
    
    # Get all artifacts for test_session
    artifacts = memory.get_artifacts_by_session("test_session", db_path=db_path)
    assert len(artifacts) == 2, "Should return 2 artifacts for test_session"
    
    # Get only document artifacts for test_session
    docs = memory.get_artifacts_by_session("test_session", "document", db_path=db_path)
    assert len(docs) == 1, "Should return 1 document for test_session"
    assert docs[0]["type"] == "document"


def test_get_facts_by_session(tmp_path: Path) -> None:
    db_path = tmp_path / "memory.db"
    memory.init_db(db_path)
    
    # Store some test facts
    memory.store_fact("test_session", "preference", "theme", "dark", db_path=db_path)
    memory.store_fact("test_session", "preference", "language", "en", db_path=db_path)
    memory.store_fact("test_session", "info", "name", "John", db_path=db_path)
    memory.store_fact("other_session", "preference", "theme", "light", db_path=db_path)
    
    # Get all facts for test_session
    facts = memory.get_facts_by_session("test_session", db_path=db_path)
    assert len(facts) == 3, "Should return 3 facts for test_session"
    
    # Get only preference facts for test_session
    prefs = memory.get_facts_by_session("test_session", "preference", db_path=db_path)
    assert len(prefs) == 2, "Should return 2 preference facts for test_session"


def test_search_facts_by_key(tmp_path: Path) -> None:
    db_path = tmp_path / "memory.db"
    memory.init_db(db_path)
    
    # Store some test facts
    memory.store_fact("test_session", "preference", "theme", "dark", db_path=db_path)
    memory.store_fact("test_session", "preference", "theme_color", "blue", db_path=db_path)
    memory.store_fact("test_session", "info", "name", "John", db_path=db_path)
    
    # Search for facts with key containing "theme"
    theme_facts = memory.search_facts_by_key("test_session", "theme", db_path=db_path)
    assert len(theme_facts) == 2, "Should find 2 facts with 'theme' in key"
    
    # Search for facts with key containing "name"
    name_facts = memory.search_facts_by_key("test_session", "name", db_path=db_path)
    assert len(name_facts) == 1, "Should find 1 fact with 'name' in key"


def test_store_turn_raises_runtimeerror_on_sql_error(monkeypatch) -> None:
    # Force get_connection to fail to ensure error is surfaced as RuntimeError
    def _boom(_db_path=None):
        raise memory.sqlite3.OperationalError("boom")

    monkeypatch.setattr(memory, "get_connection", _boom)

    with pytest.raises(RuntimeError):
        memory.store_turn(session_id="s", role="user", text="hi")


def test_store_artifact_raises_runtimeerror_on_sql_error(monkeypatch) -> None:
    def _boom(_db_path=None):
        raise memory.sqlite3.OperationalError("boom")

    monkeypatch.setattr(memory, "get_connection", _boom)

    with pytest.raises(RuntimeError):
        memory.store_artifact(session_id="s", type="doc", content="x")


def test_store_fact_raises_runtimeerror_on_sql_error(monkeypatch) -> None:
    def _boom(_db_path=None):
        raise memory.sqlite3.OperationalError("boom")

    monkeypatch.setattr(memory, "get_connection", _boom)

    with pytest.raises(RuntimeError):
        memory.store_fact(session_id="s", kind="k", key="k", value="v")
