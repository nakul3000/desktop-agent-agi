# desktop-agent-agi

A repo for hack with DC desktop personal agent.
This project builds a local-first desktop agent that can search, summarize, and orchestrate tools while keeping a privacy-friendly memory (SQLite) of conversations, artifacts, and facts for better follow-ups.

## Setup

Install dependencies (use the same Python you run the app with):

```bash
pip install -r requirements.txt
```

If your default `pip` is broken (e.g. SyntaxError inside pip’s truststore), use a specific Python or a venv:

```bash
# Option A: Use Python 3.11 explicitly (if installed)
python3.11 -m pip install -r requirements.txt
python3.11 -m pytest tests/test_email_handler.py -v

# Option B: Create a venv with a working Python, then install
python3.11 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
pytest tests/test_email_handler.py -v
```

Required env (see `.env.example`): `LINKUP_API_KEY`. Optional: `HF_TOKEN` (LLM), Gmail credentials for email features.

## Packaging notes
- This repo currently ships without a `pyproject.toml` or `setup.py`. Use `requirements.txt` for dependencies or add your own packaging config if you need installs.

## Memory component

### Explanation - What it does
- Captures the conversation: every user, assistant, and tool turn is saved with timestamps (so we can replay context later).
- Keeps artifacts: anything the agent produces or ingests (summaries, search results, drafts, research) is stored with who/what created it.
- Stores facts: structured nuggets like deadlines, meetings, tasks, preferences—tagged with confidence and linked back to the artifact that revealed them.
- Tracks references: when someone says “that deadline” or “the doc,” we remember what it pointed to for deterministic follow-ups.
- Optional semantic recall: if embeddings are available, we can embed artifacts/turns and do similarity search; if not, everything still works via recency/keywords.
- User + session awareness: rows are scoped by session (and optionally user_id), so different users or runs don’t collide.
- One-time init: `memory.init_db()` creates the SQLite schema; data lives on disk across runs (we ignore `memory.db` in git).

### Files and database
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

## Email handler

### Explanation - What it does
- **Recruiter lookup**: Finds recruiter contact details (name, title, email, LinkedIn) for a given company and role using the Linkup API. Uses a structured schema to request multiple contacts per query, scrapes emails and LinkedIn URLs from responses, and deduplicates by name.
- **Quality filters**: Rejects placeholder emails (e.g. `z@company.com`, `first.last@company.com`, `f.last@`), reply relays and marketing domains (e.g. `reply-xxx@reply.s12.y.mc.salesf`), and free-provider addresses when a corporate domain is expected. Prefers same-company contacts over agency/third-party when sorting.
- **Outreach package**: End-to-end flow for job outreach: parse job description and resume, find recruiter contacts via Linkup, draft a personalized outreach email (subject + body). Used by `AgentCore` for the `job_outreach` intent.
- **Gmail integration**: Optional. Read and parse emails, summarize threads, extract deadlines/tasks/entities, draft replies. Requires Google OAuth (`credentials.json` + `token.json`).

### Files and usage
- **File**: `email_handler.py` — `EmailHandler` class.
- **Recruiter lookup**: `find_recruiter_contact(company=..., role_title=..., team_or_domain=..., min_emails=3)` returns `{ "name", "title", "email", "emails", "linkedin_urls", "contacts", "confidence", "fallback_suggestion" }`. Requires `LINKUP_API_KEY`.
- **Outreach package**: `build_recruiter_outreach_package(role_title=..., job_description=..., resume_text=..., company=..., ...)` returns recruiter info + `email_subject`, `email_body`, and `analysis` (parsed JD and resume). Uses LLM for JD/resume parsing and draft (optional `HF_TOKEN`).
- **Run examples**: `python run_recruiter_examples.py` (or pass company and role as CLI args). Unit and integration tests: `pytest tests/test_email_handler.py -v`.
<<<<<<< HEAD
- **Integration**: `AgentCore.handle_message({"intent": "job_outreach", "role_title": ..., "job_description": ..., "resume_text": ..., "company": ...})` calls `build_recruiter_outreach_package` and returns the payload.
 
=======
- **Integration**: `AgentCore.handle_message({"intent": "job_outreach", "role_title": ..., "job_description": ..., "resume_text": ..., "company": ...})` calls `build_recruiter_outreach_package` and returns the payload.

