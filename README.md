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
