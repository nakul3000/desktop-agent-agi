# desktop-agent-agi
A repo for hack with DC desktop personal agent.

## Packaging notes
- This repo currently ships without a `pyproject.toml` or `setup.py`. Use `requirements.txt` for dependencies or add your own packaging config if you need installs.

## Memory component
- Files: `memory.py`, `embeddings.py`, `utils.py`, `tests/test_memory_stub.py`.
- Database: SQLite (`memory.db` by default). Call `init_db()` to create tables for turns, artifacts, facts, references, and embeddings_map.
- Persistence helpers: `store_turn`, `store_artifact`, `store_fact` write rows with ISO timestamps; they raise `RuntimeError` if a DB write fails.
- Embeddings: `embeddings.py` is an optional FAISS/sentence-transformers shim; if those heavy deps aren’t installed, it safely no-ops while exposing `EmbeddingIndex`, `SUPPORTED_ITEM_TYPES`.
- Shared helpers: `utils.py` includes `load_resume_from_env` (reads resume text from `RESUME_PATH`) and `prepare_text_for_embedding`.
- Tests: `tests/test_memory_stub.py` includes smoke tests for inserts/retrieval and asserts that insert helpers surface errors.
- Data model (all scoped by `session_id`):
  - `turns`: every user/assistant/tool message with timestamp.
  - `artifacts`: generated or extracted items (doc summaries, drafts, research outputs) with optional source turn and creator tag.
  - `facts`: structured nuggets like deadlines/meetings/tasks with confidence and meta JSON (e.g., title, source artifact).
  - `references`: records how vague phrases (“that”, “the deadline”) were resolved for deterministic follow-ups.
  - `embeddings_map`: text used for embedding per item; optional when FAISS is present.
- Optional semantic recall:
  - Set `DISABLE_EMBEDDINGS=1` to force-disable; otherwise the shim lazily activates only if FAISS + sentence-transformers are installed.
  - `SUPPORTED_ITEM_TYPES` includes resume, job descriptions, company research, cover letters, emails, recruiter profiles, user preferences, and conversation turns.
- Usage quickstart:
  1) `memory.init_db()` (once).
  2) `turn_id = memory.store_turn(session, "user", "Hi")`
  3) `artifact_id = memory.store_artifact(session, "doc_summary", {"title": "...", "content": "..."})`
  4) `fact_id = memory.store_fact(session, "deadline", "response_deadline", "2026-02-15", meta={"title": "Respond to ..."})`
  5) If embeddings installed, create `EmbeddingIndex` and add text using `prepare_text_for_embedding(content)`.
