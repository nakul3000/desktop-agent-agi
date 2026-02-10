import json

import memory


def test_get_context_returns_recent_items(tmp_path):
    db = tmp_path / "mem.db"
    memory.init_db(db)
    session = "s1"
    user = "u1"

    t1 = memory.store_turn(session, "user", "hello", user_id=user, db_path=db)
    a1 = memory.store_artifact(session, "linkup_research", {"q": "x"}, source_turn_id=t1, user_id=user, db_path=db)
    f1 = memory.store_fact(session, "preference", "target_company", "Acme", source_artifact_id=a1, user_id=user, db_path=db)

    ctx = memory.get_context(session, user_id=user, db_path=db, k_recent_turns=5, k_recent_artifacts=5, k_recent_facts=5)

    assert any(item["id"] == t1 for item in ctx["recent_turns"])
    assert any(item["id"] == a1 for item in ctx["recent_artifacts"])
    assert any(item["id"] == f1 for item in ctx["recent_facts"])


def test_store_turn_saves_tool_name(tmp_path):
    db = tmp_path / "mem.db"
    memory.init_db(db)
    session = "s1"
    tool_name = "job_searcher"
    tid = memory.store_turn(session, "tool", "called job searcher", user_id="u", tool_name=tool_name, db_path=db)

    rows = memory.get_turns_by_session(session, db_path=db)
    row = next(r for r in rows if r["id"] == tid)
    assert row["tool_name"] == tool_name


def test_get_context_filters_by_user_id(tmp_path):
    db = tmp_path / "mem.db"
    memory.init_db(db)
    session = "s1"
    memory.store_turn(session, "user", "u1 turn", user_id="u1", db_path=db)
    memory.store_turn(session, "user", "u2 turn", user_id="u2", db_path=db)
    memory.store_turn(session, "assistant", "no user turn", user_id=None, db_path=db)

    ctx = memory.get_context(session, user_id="u1", db_path=db, k_recent_turns=10)
    texts = [t["text"] for t in ctx["recent_turns"]]

    assert "u1 turn" in texts
    assert "no user turn" in texts  # null user_id is allowed
    assert "u2 turn" not in texts
