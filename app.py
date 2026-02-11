# app.py — Multiturn Conversational Job Search Agent
# Uses HuggingFace Inference API + Llama for tool-calling agent

import os
import json
import re
import uuid
import traceback
from datetime import datetime
from pathlib import Path
import requests
from dotenv import load_dotenv
try:
    from huggingface_hub import InferenceClient
except ModuleNotFoundError:
    InferenceClient = None

import memory
from linkup_client import LinkupJobSearch, normalize_search_results_to_jobs
from utils import load_resume_from_env

# One-time DB init guard
_DB_INITIALIZED = False


def _ensure_db_initialized():
    global _DB_INITIALIZED
    if _DB_INITIALIZED:
        return
    memory.init_db()
    _DB_INITIALIZED = True


def _to_json_safe(value):
    """Best-effort conversion to JSON-serializable primitives."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _to_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_json_safe(v) for v in value]

    # Pydantic v2 models (and compatible SDK models).
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        try:
            return _to_json_safe(model_dump(mode="json"))
        except Exception:
            try:
                return _to_json_safe(model_dump())
            except Exception:
                pass

    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        try:
            return _to_json_safe(to_dict())
        except Exception:
            pass

    obj_dict = getattr(value, "__dict__", None)
    if isinstance(obj_dict, dict):
        return _to_json_safe(obj_dict)

    return str(value)

load_dotenv()

# ------------------------------------------------------------------ #
# TOOL DEFINITIONS
# ------------------------------------------------------------------ #

TOOLS = [
    {
        "name": "job_searcher",
        "description": "Search for job openings at a specific company or in general. Use when the user asks to find jobs, look for positions, search openings, or mentions job hunting.",
        "parameters": {
            "role": "The job role/title to search for (e.g., Machine Learning Engineer)",
            "company": "The company name (optional, can be empty for general search)",
            "location": "Job location preference (default: United States)",
        },
        "triggers": ["find jobs", "search jobs", "job openings", "positions at", "hiring", "show me jobs", "look for roles"],
    },
    {
        "name": "company_profiler",
        "description": "Research a company's background, culture, tech stack, recent news, funding, and financials. Use when the user wants to know about a company before applying.",
        "parameters": {
            "company": "The company name to research",
        },
        "triggers": ["tell me about", "company profile", "research company", "what does", "how is", "company culture", "company background"],
    },
    {
        "name": "resume_tailor",
        "description": "Tailor the user's resume for a specific job posting. Rewrites bullet points and highlights to match the job description keywords and requirements.",
        "parameters": {
            "role": "Target job role",
            "company": "Target company",
            "job_description": "The job description to tailor resume for",
        },
        "triggers": ["tailor resume", "customize resume", "adapt resume", "resume for", "modify resume", "update resume"],
    },
    {
        "name": "cover_letter_generator",
        "description": "Generate a personalized cover letter based on the user's resume and the target job/company. Incorporates company research for personalization.",
        "parameters": {
            "role": "Target job role",
            "company": "Target company",
            "job_description": "The job description",
            "company_context": "Any company research or context to personalize the letter",
        },
        "triggers": ["cover letter", "write a letter", "draft cover", "application letter", "generate cover letter"],
    },
    {
        "name": "email_crafter",
        "description": "Draft a personalized cold outreach or follow-up email to recruiters or hiring managers. Uses company context for personalization.",
        "parameters": {
            "recipient_name": "Name of the recruiter/hiring manager",
            "recipient_role": "Their title (e.g., Senior Recruiter)",
            "company": "Company name",
            "role": "The role the user is interested in",
            "purpose": "Purpose of email: cold_outreach, follow_up, thank_you, inquiry",
        },
        "triggers": ["email", "reach out", "contact recruiter", "cold email", "follow up", "outreach", "message recruiter"],
    },
]

TOOL_NAMES = [t["name"] for t in TOOLS]

# ------------------------------------------------------------------ #
# SYSTEM PROMPT
# ------------------------------------------------------------------ #

SYSTEM_PROMPT = f"""You are JobAgent AI, an intelligent job search and application assistant. Today's date is {datetime.now().strftime("%B %d, %Y")}.

You have access to these tools:

1. **job_searcher** - Search for job openings. Use when user asks to find/search jobs.
2. **company_profiler** - Research a company's background, culture, news. Use when user asks about a company.
3. **resume_tailor** - Tailor resume for a specific role/company. Use when user wants to customize their resume.
4. **cover_letter_generator** - Write personalized cover letters. Use when user needs a cover letter.
5. **email_crafter** - Draft outreach emails to recruiters. Use when user wants to email/contact someone.

IMPORTANT RULES FOR TOOL CALLING:
- When you need to use a tool, respond ONLY with a JSON block in this exact format:
```json
{{
    "tool": "tool_name",
    "parameters": {{
        "param1": "value1",
        "param2": "value2"
    }},
    "reasoning": "Brief explanation of why you're using this tool"
}}
```
- If the user's message does NOT require any tool, respond naturally in conversation.
- If you need more info before calling a tool, ASK the user first.
- You can suggest chaining tools. For example: "Let me first search for jobs, then I can tailor your resume for the best match."
- Always be helpful, concise, and proactive in suggesting next steps.
- Remember the full conversation context — refer back to previous results when relevant.
- If the user mentions a company, proactively offer to research it.
- After any tool result, summarize key findings and suggest logical next actions.
"""

# ------------------------------------------------------------------ #
# TOOL IMPLEMENTATIONS (stubs — connect to your real modules)
# ------------------------------------------------------------------ #

def execute_job_searcher(params: dict) -> str:
    """Execute job search — connects to your linkup_client.py"""
    role = params.get("role", "Machine Learning Engineer")
    company = params.get("company", "")
    location = params.get("location", "United States")

    try:
        print(f"\n{'=' * 80}")
        print("🧭 DEBUG LOG: execute_job_searcher input")
        print(f"role={role!r} company={company!r} location={location!r}")
        print(f"{'=' * 80}\n")
        searcher = LinkupJobSearch()
        response = searcher.search_jobs(role=role, company=company or None, location=location)
        jobs = normalize_search_results_to_jobs(
            response,
            role=role,
            company=company or None,
            location=location,
            limit=12,
        )
        results_count = len(getattr(response, "results", []) or [])
        print(f"\n{'=' * 80}")
        print("🧭 DEBUG LOG: execute_job_searcher output")
        print(f"search_results_count={results_count} normalized_jobs_count={len(jobs)}")
        for idx, j in enumerate(jobs, start=1):
            jd_text = (j.get("jd_text") or "").strip()
            print(
                f"[job {idx}] title={j.get('title')} | company={j.get('company')} | location={j.get('location')} | "
                f"url={j.get('url')} | jd_len={len(jd_text)}"
            )
        print(f"{'=' * 80}\n")
        return json.dumps(
            {
                "status": "success",
                "query": {"role": role, "company": company, "location": location},
                "search_results_count": results_count,
                "jobs_found": len(jobs),
                "jobs": jobs,
                "next_steps": "Reply with the number of a job to select it. If the JD text wasn't captured, you'll be prompted to paste it.",
            },
            indent=2,
        )
    except Exception as e:
        error_type = type(e).__name__
        tb_text = traceback.format_exc().rstrip()
        print(f"\n{'=' * 80}")
        print("❌ ERROR LOG: job_searcher failed")
        print(f"Type: {error_type}")
        print(f"Message: {e}")
        print("Traceback:")
        print(tb_text)
        print(f"{'=' * 80}\n")
        return json.dumps(
            {
                "status": "error",
                "error": str(e),
                "error_type": error_type,
                "query": {"role": role, "company": company, "location": location},
            },
            indent=2,
        )


def execute_company_profiler(params: dict) -> str:
    """Execute company research — connects to your linkup_client.py"""
    context = params.get("context") or {}
    selected_job = (context.get("selected_job") or {}) if isinstance(context, dict) else {}
    job_intake_payload = (context.get("job_intake") or {}) if isinstance(context, dict) else {}

    company = (params.get("company") or "").strip() or (selected_job.get("company") or "").strip()

    warnings = []
    try:
        searcher = LinkupJobSearch()

        if (
            isinstance(job_intake_payload, dict)
            and isinstance(selected_job, dict)
            and selected_job.get("url")
            and selected_job.get("url") != "NA"
        ):
            sources = job_intake_payload.get("sources")
            has_url_source = isinstance(sources, list) and any(
                isinstance(s, dict) and (s.get("url") or "").strip() == (selected_job.get("url") or "").strip()
                for s in sources
            )
            if not has_url_source:
                merged_sources = sources if isinstance(sources, list) else []
                merged_sources.append(
                    {
                        "name": selected_job.get("title") or "Selected job posting",
                        "url": selected_job.get("url"),
                        "snippet": (selected_job.get("snippet") or "")[:300] or None,
                        "favicon": None,
                    }
                )
                job_intake_payload = {
                    "answer": (job_intake_payload.get("answer") or "").strip(),
                    "sources": merged_sources,
                }

        intake = None
        if (
            isinstance(job_intake_payload, dict)
            and (job_intake_payload.get("answer") or "").strip()
        ):
            try:
                intake = searcher.build_job_intake(job_intake_payload)
                parsed_company = (getattr(intake, "company_name", "") or "").strip()
                if parsed_company and parsed_company != "NA":
                    company = parsed_company
            except Exception:
                intake = None

        if not company or company == "NA":
            return json.dumps(
                {
                    "status": "error",
                    "error": "Missing company. Provide params.company or select a job first (with company field).",
                },
                indent=2,
            )

        # Debug trace: show which JD-aware inputs are available for profiling.
        jd_answer = (
            (job_intake_payload.get("answer") or "").strip()
            if isinstance(job_intake_payload, dict)
            else ""
        )
        jd_sources = (
            job_intake_payload.get("sources")
            if isinstance(job_intake_payload, dict)
            else []
        )
        intake_debug = {}
        if intake is not None:
            intake_debug = {
                "company_name": getattr(intake, "company_name", "NA"),
                "role_title": getattr(intake, "role_title", "NA"),
                "location": getattr(intake, "location", "NA"),
                "workplace_type": getattr(intake, "workplace_type", "NA"),
                "employment_type": getattr(intake, "employment_type", "NA"),
                "job_url": getattr(intake, "job_url", "NA"),
                "requirements_summary_len": len((getattr(intake, "requirements_summary", "") or "")),
                "preferred_summary_len": len((getattr(intake, "preferred_summary", "") or "")),
                "compensation_summary_len": len((getattr(intake, "compensation_summary", "") or "")),
            }
        elif jd_answer:
            intake_debug = {"intake_parse_error": "intake parsing failed before research"}

        print(f"\n{'=' * 80}")
        print("🧭 DEBUG LOG: company_profiler JD context snapshot")
        print(f"selected_job.company={selected_job.get('company') if isinstance(selected_job, dict) else 'NA'}")
        print(f"selected_job.url={selected_job.get('url') if isinstance(selected_job, dict) else 'NA'}")
        print(f"jd_payload_present={bool(jd_answer)}")
        print(f"jd_answer_len={len(jd_answer)}")
        print(f"jd_sources_count={len(jd_sources) if isinstance(jd_sources, list) else 0}")
        print(f"jd_sources_urls={[((s.get('url') or '').strip()) for s in jd_sources if isinstance(s, dict)] if isinstance(jd_sources, list) else []}")
        print(f"normalized_intake={json.dumps(intake_debug, indent=2)}")
        print(f"{'=' * 80}\n")

        profile = None
        sentiment = None

        if (
            isinstance(job_intake_payload, dict)
            and (job_intake_payload.get("answer") or "").strip()
            and intake is not None
        ):
            try:
                print(f"\n{'=' * 80}")
                print("🧭 DEBUG LOG: company_profiler using JD-aware path")
                print(f"job_intake_answer_len={len((job_intake_payload.get('answer') or '').strip())}")
                print(f"job_intake_sources={job_intake_payload.get('sources') or []}")
                print(f"{'=' * 80}\n")
                combined = searcher.research_from_selected_jd(
                    job_intake_payload,
                    preferred_company=company,
                )
                if isinstance(combined, dict):
                    profile = combined.get("profile")
                    sentiment = combined.get("sentiment")
                    combined_company = (combined.get("company") or "").strip()
                    if combined_company and combined_company != "NA":
                        company = combined_company
            except Exception as e:
                error_type = type(e).__name__
                tb_text = traceback.format_exc().rstrip()
                warning_message = (
                    f"JD-aware profiling failed ({error_type}): {e}. "
                    "Used fallback company/profile sentiment path."
                )
                warnings.append(warning_message)
                print(f"\n{'=' * 80}")
                print("⚠️ WARNING LOG: JD-aware company profiling path failed, using fallback path")
                print(f"Type: {error_type}")
                print(f"Message: {e}")
                print("Traceback:")
                print(tb_text)
                print(f"{'=' * 80}\n")
                profile = None
                sentiment = None
        elif isinstance(job_intake_payload, dict) and (job_intake_payload.get("answer") or "").strip() and intake is None:
            warnings.append("JD intake parsing failed before JD-aware profiling; using fallback profile path.")

        if profile is None or sentiment is None:
            print(f"\n{'=' * 80}")
            print("🧭 DEBUG LOG: company_profiler using fallback non-JD path")
            print(f"company={company}")
            print(f"{'=' * 80}\n")
            profile = searcher.get_company_profile(company, context=context if isinstance(context, dict) else None)
            sentiment = searcher.get_company_sentiment(company, context=context if isinstance(context, dict) else None)

        payload = {
            "status": "success",
            "company": company,
            "profile": _to_json_safe(profile),
            "sentiment": _to_json_safe(sentiment),
            "warnings": warnings,
            "next_steps": "Want me to tailor your resume, draft a cover letter, or craft a recruiter message for the selected job?",
        }
        return json.dumps(payload, indent=2)
    except Exception as e:
        error_type = type(e).__name__
        tb_text = traceback.format_exc().rstrip()
        print(f"\n{'=' * 80}")
        print("❌ ERROR LOG: company_profiler failed")
        print(f"Type: {error_type}")
        print(f"Message: {e}")
        print("Traceback:")
        print(tb_text)
        print(f"{'=' * 80}\n")
        return json.dumps(
            {
                "status": "error",
                "error": str(e),
                "error_type": error_type,
                "company": company or "NA",
            },
            indent=2,
        )


def execute_resume_tailor(params: dict) -> str:
    """Tailor resume for a specific role"""
    role = params.get("role", "")
    company = params.get("company", "")
    jd = params.get("job_description", "")

    return json.dumps({
        "status": "success",
        "role": role,
        "company": company,
        "tailored_sections": {
            "headline": f"{role} | ML Systems | Production AI at Scale",
            "summary": f"Applied ML Scientist with production experience building AI systems serving 150K+ users. Seeking {role} role at {company}.",
            "key_skills_highlighted": [
                "RAG systems & retrieval pipelines",
                "Production ML (sub-3s latency, 150K+ users)",
                "LLM evaluation & benchmarking (RAGAS)",
                "Python, PyTorch, LangChain",
            ],
            "bullets_rewritten": 3,
            "keywords_matched": ["machine learning", "production", "RAG", "LLM", "Python"],
        },
        "next_steps": "Want me to also generate a cover letter or draft an outreach email?",
    }, indent=2)


def execute_cover_letter_generator(params: dict) -> str:
    """Generate personalized cover letter"""
    role = params.get("role", "")
    company = params.get("company", "")

    return json.dumps({
        "status": "success",
        "cover_letter": f"""Dear Hiring Team at {company},

I am writing to express my interest in the {role} position at {company}. As an Applied ML Scientist at The Washington Post, I have built production AI systems serving 150K+ users, including multiturn RAG chatbots and enterprise AI partnerships worth $1.5M.

[This is a stub — the real implementation would use the LLM to generate a full personalized cover letter based on resume + company research + job description]

I would welcome the opportunity to discuss how my experience aligns with {company}'s mission.

Best regards,
[Your Name]""",
        "next_steps": "Want me to draft an outreach email to a recruiter at this company?",
    }, indent=2)


def execute_email_crafter(params: dict) -> str:
    """Draft outreach email"""
    recipient = params.get("recipient_name", "Hiring Manager")
    company = params.get("company", "")
    role = params.get("role", "")
    purpose = params.get("purpose", "cold_outreach")

    return json.dumps({
        "status": "success",
        "email": {
            "subject": f"Re: {role} Opportunity at {company}",
            "to": recipient,
            "body": f"""Hi {recipient},

I came across the {role} position at {company} and was excited to reach out. I'm currently an Applied ML Scientist at The Washington Post, where I've built production RAG systems serving 150K+ users.

[Stub — real version would be personalized with company research + recruiter context]

Would you have 15 minutes this week for a brief chat? I'm available [calendar slots would go here].

Best,
[Your Name]""",
        },
        "next_steps": "I can send this via Gmail API if you'd like, or refine the tone.",
    }, indent=2)


# Tool dispatcher
TOOL_EXECUTORS = {
    "job_searcher": execute_job_searcher,
    "company_profiler": execute_company_profiler,
    "resume_tailor": execute_resume_tailor,
    "cover_letter_generator": execute_cover_letter_generator,
    "email_crafter": execute_email_crafter,
}


# ------------------------------------------------------------------ #
# AGENT CORE
# ------------------------------------------------------------------ #

class JobAgent:
    def __init__(self, session_id: str | None = None, user_id: str | None = None):
        if InferenceClient is None:
            raise RuntimeError(
                "Missing dependency: `huggingface_hub`.\n"
                "Install it with: `pip install -r requirements.txt`"
            )

        self.hf_token = os.getenv("HF_TOKEN")
        if not self.hf_token:
            raise ValueError("HF_TOKEN not found in .env")

        self.linkup_api_key = os.getenv("LINKUP_API_KEY")
        if not self.linkup_api_key:
            raise ValueError("LINKUP_API_KEY not found in .env")

        # Allow overriding the HF model id via env var.
        self.hf_model = os.getenv("HF_MODEL", "meta-llama/Meta-Llama-3-70B-Instruct")
        self.client = InferenceClient(
            # Use a hosted-inference-available model id
            model=self.hf_model,
            token=self.hf_token,
        )

        # Conversation history: list of {"role": "user"/"assistant"/"system", "content": "..."}
        self.conversation_history = []
        self.conversation_history.append({
            "role": "system",
            "content": SYSTEM_PROMPT,
        })

        # Initialize memory
        # Derive user/session IDs: prefer explicit args then fallbacks
        self.user_id = user_id or "anonymous"
        self.session_id = session_id or str(uuid.uuid4())
        _ensure_db_initialized()
        memory.register_user(self.user_id)
        memory.start_session(user_id=self.user_id, session_id=self.session_id)

        # Store context from tool results for downstream use
        self.context_store = {
            "last_jobs": None,
            "company_research": None,
            "company_profile": None,
            "company_sentiment": None,
            "last_resume_tailoring": None,
            "user_resume": None,
            "selected_job": None,
            "job_intake_payload": None,
        }

        # CLI workflow state
        self.awaiting_job_selection = False
        self.awaiting_jd_text = False

        # Best-effort: load resume text for downstream stubs/agents.
        self.context_store["user_resume"] = load_resume_from_env() or None

    def _build_context_envelope(self, *, user_request: str) -> dict:
        selected_job = self.context_store.get("selected_job") or {}
        job_intake_payload = self.context_store.get("job_intake_payload") or {}
        return {
            "session": {"session_id": self.session_id, "user_id": self.user_id},
            "user_request": user_request,
            "selected_job": selected_job,
            "job_intake": job_intake_payload,
            "artifacts": {
                "last_jobs": self.context_store.get("last_jobs"),
                "company_research": self.context_store.get("company_research"),
                "company_profile": self.context_store.get("company_profile"),
                "company_sentiment": self.context_store.get("company_sentiment"),
                "last_resume_tailoring": self.context_store.get("last_resume_tailoring"),
            },
            "user": {"resume_text": self.context_store.get("user_resume")},
        }

    def _format_job_list_for_cli(self, jobs: list[dict]) -> str:
        lines = []
        for i, j in enumerate(jobs, start=1):
            title = (j.get("title") or "NA").strip()
            company = (j.get("company") or "NA").strip()
            location = (j.get("location") or "NA").strip()
            url = (j.get("url") or "NA").strip()
            lines.append(f"{i}. {title} — {company} ({location})\n   {url}")
        return "\n".join(lines)

    def _next_action_prompt(self) -> str:
        return (
            "What do you want to do next?\n"
            "- research the company\n"
            "- tailor my resume (coming next phase)\n"
            "- write a cover letter (coming next phase)\n"
            "- message a recruiter (coming next phase)"
        )

    def _build_job_intake_payload(self, *, jd_text: str, selected_job: dict | None) -> dict:
        selected_job = selected_job if isinstance(selected_job, dict) else {}
        url = (selected_job.get("url") or "").strip()
        title = (selected_job.get("title") or "Selected job posting").strip()
        snippet = (selected_job.get("snippet") or "").strip()
        source = {
            "name": title or "Selected job posting",
            "url": url or None,
            "snippet": snippet[:300] or None,
            "favicon": None,
        }
        sources = [source] if source.get("url") else []
        payload = {
            "answer": (jd_text or "").strip(),
            "sources": sources,
        }
        print(f"\n{'=' * 80}")
        print("🧭 DEBUG LOG: built job_intake_payload")
        print(f"answer_len={len(payload['answer'])}")
        print(f"sources={payload['sources']}")
        print(f"answer_full={payload['answer']}")
        print(f"{'=' * 80}\n")
        return payload

    def _extract_jd_from_selected_job_url(self, selected_job: dict | None) -> dict:
        selected_job = selected_job if isinstance(selected_job, dict) else {}
        job_url = (selected_job.get("url") or "").strip()
        existing_jd_text = (selected_job.get("jd_text") or "").strip()
        if not job_url or job_url == "NA":
            return {"status": "error", "error": "No job URL available for extraction."}
        try:
            searcher = LinkupJobSearch()
            return searcher.extract_job_description_from_url(
                job_url=job_url,
                role=(selected_job.get("title") or "").strip() or None,
                company=(selected_job.get("company") or "").strip() or None,
                existing_jd_text=existing_jd_text or None,
            )
        except Exception as e:
            return {"status": "error", "error": f"JD extraction failed: {type(e).__name__}: {e}"}

    def _format_company_research_for_cli(self, tool_result: str) -> str:
        try:
            parsed = json.loads(tool_result)
        except Exception:
            return tool_result if isinstance(tool_result, str) else str(tool_result)

        if not isinstance(parsed, dict):
            return tool_result if isinstance(tool_result, str) else str(tool_result)

        if parsed.get("status") == "error":
            error_message = parsed.get("error") or "Unknown error."
            error_type = (parsed.get("error_type") or "").strip()
            if error_type:
                return f"Company profiling failed ({error_type}): {error_message}"
            return f"Company profiling failed: {error_message}"

        company = (parsed.get("company") or "the selected company").strip()
        profile = parsed.get("profile") if isinstance(parsed.get("profile"), dict) else {}
        sentiment = parsed.get("sentiment") if isinstance(parsed.get("sentiment"), dict) else {}
        warnings = parsed.get("warnings") if isinstance(parsed.get("warnings"), list) else []

        profile_md = (profile.get("report_markdown") or "").strip()
        sentiment_md = (sentiment.get("report_markdown") or "").strip()

        if not profile_md:
            profile_md = json.dumps(profile, indent=2) if profile else "No profile report returned."
        if not sentiment_md:
            sentiment_md = json.dumps(sentiment, indent=2) if sentiment else "No sentiment report returned."

        warning_block = ""
        if warnings:
            warning_block = "Warnings:\n" + "\n".join(f"- {str(w)}" for w in warnings if str(w).strip()) + "\n\n"

        return (
            f"Here is your company profile + sentiment report for {company}.\n\n"
            f"{warning_block}"
            "=== Company Profile ===\n"
            f"{profile_md}\n\n"
            "=== Company Sentiment ===\n"
            f"{sentiment_md}\n\n"
            "Next step: say 'research another company' or select another job."
        )

    def _is_company_research_request(self, message: str) -> bool:
        lowered = (message or "").lower()
        return any(
            phrase in lowered
            for phrase in (
                "research company",
                "research the company",
                "company research",
                "company profile",
                "profile the company",
                "tell me about the company",
            )
        ) or ("research" in lowered and "company" in lowered)

    def _load_sample_jd_text(self) -> str | None:
        try:
            here = Path(__file__).resolve().parent
            return (here / "sample_jd1.txt").read_text(encoding="utf-8").strip()
        except OSError:
            return None

    def _normalize_tool_params(self, tool_name: str, params: dict | None) -> dict:
        """Normalize common parameter aliases before tool execution."""
        normalized = dict(params or {})
        if tool_name == "job_searcher":
            role_value = normalized.get("role")
            if not (isinstance(role_value, str) and role_value.strip()):
                for alias in ("job_title", "title", "position"):
                    alias_value = normalized.get(alias)
                    if isinstance(alias_value, str) and alias_value.strip():
                        normalized["role"] = alias_value.strip()
                        break
        return normalized

    def _execute_tool_and_respond(self, *, tool_name: str, params: dict, reasoning: str, user_message: str) -> str:
        if tool_name not in TOOL_EXECUTORS:
            error_msg = f"Unknown tool: {tool_name}. Available tools: {TOOL_NAMES}"
            self.conversation_history.append({"role": "assistant", "content": error_msg})
            memory.store_turn(self.session_id, role="assistant", text=error_msg, user_id=self.user_id)
            return error_msg

        params = self._normalize_tool_params(tool_name, params)
        memory.store_turn(
            self.session_id,
            role="assistant",
            text=f"[Tool call planned] {tool_name} {params}",
            user_id=self.user_id,
        )

        params = dict(params or {})
        params["context"] = self._build_context_envelope(user_request=user_message)

        tool_result = TOOL_EXECUTORS[tool_name](params)
        tool_turn_id = memory.store_turn(self.session_id, role="tool", text=str(tool_result), user_id=self.user_id, tool_name=tool_name)

        self._update_context(tool_name, tool_result)
        memory.store_artifact(
            session_id=self.session_id,
            type=tool_name,
            content=tool_result,
            source_turn_id=tool_turn_id,
            created_by="JobAgent",
            user_id=self.user_id,
        )

        self.conversation_history.append(
            {
                "role": "assistant",
                "content": f"[Called tool: {tool_name}]\n{reasoning}",
            }
        )

        # Special-case: after job search, return deterministic list for selection.
        if tool_name == "job_searcher":
            try:
                parsed = json.loads(tool_result)
            except Exception:
                parsed = None

            if isinstance(parsed, dict) and parsed.get("status") == "error":
                error_type = parsed.get("error_type") or "ToolError"
                error_text = parsed.get("error") or "Unknown job search error."
                msg = (
                    f"Job search failed with `{error_type}`: {error_text}\n\n"
                    "I logged a detailed traceback in the terminal under "
                    "`ERROR LOG: job_searcher failed`."
                )
                memory.store_turn(self.session_id, role="assistant", text=msg, user_id=self.user_id)
                return msg

            jobs = parsed.get("jobs") if isinstance(parsed, dict) else None
            if isinstance(jobs, list) and jobs:
                self.awaiting_job_selection = True
                msg = "Here are the jobs I found. Reply with a number to select one:\n\n" + self._format_job_list_for_cli(jobs)
                memory.store_turn(self.session_id, role="assistant", text=msg, user_id=self.user_id)
                return msg
            if self.client is None:
                msg = (
                    "I ran the job search, but didn’t get any structured job results back.\n"
                    "Try a slightly different query (different company spelling, broader location, or a more common role title)."
                )
                memory.store_turn(self.session_id, role="assistant", text=msg, user_id=self.user_id)
                return msg

        if tool_name == "company_profiler":
            msg = self._format_company_research_for_cli(tool_result)
            memory.store_turn(self.session_id, role="assistant", text=msg, user_id=self.user_id)
            return msg

        # In CLI router mode (no LLM), return tool output directly.
        if self.client is None:
            msg = tool_result if isinstance(tool_result, str) else str(tool_result)
            memory.store_turn(self.session_id, role="assistant", text=msg, user_id=self.user_id)
            return msg

        # Default: ask the LLM to summarize tool results.
        self.conversation_history.append(
            {
                "role": "user",
                "content": f"[Tool Result for {tool_name}]:\n{tool_result}\n\nNow summarize these results for the user in a helpful, conversational way. Highlight key findings and suggest logical next steps.",
            }
        )
        summary = self._call_llm()
        self.conversation_history.append({"role": "assistant", "content": summary})
        memory.store_turn(self.session_id, role="assistant", text=summary, user_id=self.user_id)
        return summary

    def chat(self, user_message: str) -> str:
        """Process a user message and return agent response."""

        user_message = (user_message or "").strip()

        # Always log the user turn, even for deterministic CLI intercepts.
        self.conversation_history.append({"role": "user", "content": user_message})
        memory.store_turn(self.session_id, role="user", text=user_message, user_id=self.user_id)

        # ---- Deterministic CLI workflow intercepts ----
        if self.awaiting_job_selection and user_message.isdigit():
            last_jobs = self.context_store.get("last_jobs") or {}
            jobs = last_jobs.get("jobs") if isinstance(last_jobs, dict) else None
            if not isinstance(jobs, list) or not jobs:
                self.awaiting_job_selection = False
                # fall through to normal flow
            else:
                idx = int(user_message)
                if idx < 1 or idx > len(jobs):
                    msg = f"Please reply with a number between 1 and {len(jobs)}."
                    memory.store_turn(self.session_id, role="assistant", text=msg, user_id=self.user_id)
                    return msg

                chosen = jobs[idx - 1]
                chosen_jd = (chosen.get("jd_text") or "").strip() if isinstance(chosen, dict) else ""
                print(f"\n{'=' * 80}")
                print("🧭 DEBUG LOG: job selected by user")
                print(f"user_selection_index={idx}")
                print(f"chosen_job={json.dumps(chosen, indent=2)}")
                print(f"{'=' * 80}\n")
                self.context_store["selected_job"] = {
                    "job_id": chosen.get("job_id") or str(idx),
                    "title": chosen.get("title") or "NA",
                    "company": chosen.get("company") or "NA",
                    "location": chosen.get("location") or "NA",
                    "url": chosen.get("url") or "NA",
                    "jd_text": chosen_jd if chosen_jd and chosen_jd != "NA" else "",
                }
                self.awaiting_job_selection = False

                selected_job = self.context_store.get("selected_job") or {}

                # Always try extracting from the selected link first so downstream profiling
                # uses JD text that matches the exact posting the user chose.
                extracted = self._extract_jd_from_selected_job_url(selected_job)
                extracted_jd_text = (extracted.get("jd_text") or "").strip() if isinstance(extracted, dict) else ""
                if isinstance(extracted, dict) and extracted.get("status") == "success" and extracted_jd_text:
                    print(f"\n{'=' * 80}")
                    print("🧭 DEBUG LOG: selection -> JD source decision")
                    print("source=selected_job_url_extraction")
                    print(f"extracted_source_url={(extracted.get('source_url') or '').strip()}")
                    print(f"extracted_source_name={(extracted.get('source_name') or '').strip()}")
                    print(f"extracted_jd_len={len(extracted_jd_text)}")
                    print(f"extracted_jd_full={extracted_jd_text}")
                    print(f"{'=' * 80}\n")
                    selected_job["jd_text"] = extracted_jd_text
                    extracted_source_url = (extracted.get("source_url") or "").strip()
                    if extracted_source_url:
                        selected_job["url"] = extracted_source_url
                    self.context_store["selected_job"] = selected_job
                    self.context_store["job_intake_payload"] = self._build_job_intake_payload(
                        jd_text=extracted_jd_text,
                        selected_job=selected_job,
                    )
                    msg = (
                        "Selected. I extracted the job description from the job link.\n\n"
                        + self._next_action_prompt()
                    )
                    memory.store_turn(self.session_id, role="assistant", text=msg, user_id=self.user_id)
                    return msg

                extraction_error = (
                    extracted.get("error")
                    if isinstance(extracted, dict) and extracted.get("error")
                    else "Unknown extraction error."
                )
                print(f"\n{'=' * 80}")
                print("🧭 DEBUG LOG: selection -> JD source decision")
                print("source=linkup_fetch_error_no_fallback")
                print(f"url_extraction_error={extraction_error}")
                print(f"{'=' * 80}\n")
                msg = (
                    "Selected, but JD extraction failed.\n\n"
                    f"Error: {extraction_error}"
                )
                memory.store_turn(self.session_id, role="assistant", text=msg, user_id=self.user_id)
                return msg

        if self.awaiting_jd_text:
            jd_text = user_message
            if user_message.lower() in {"sample_jd1", "use sample_jd1"}:
                jd_text = self._load_sample_jd_text() or ""

            if not jd_text:
                msg = "I didn’t receive any JD text. Please paste the full job description (or type `sample_jd1`)."
                memory.store_turn(self.session_id, role="assistant", text=msg, user_id=self.user_id)
                return msg

            selected_job = self.context_store.get("selected_job") or {}
            if isinstance(selected_job, dict):
                selected_job["jd_text"] = jd_text
                self.context_store["selected_job"] = selected_job

            self.context_store["job_intake_payload"] = self._build_job_intake_payload(
                jd_text=jd_text,
                selected_job=selected_job if isinstance(selected_job, dict) else {},
            )
            self.awaiting_jd_text = False
            print(f"\n{'=' * 80}")
            print("🧭 DEBUG LOG: manual JD intake accepted")
            print(f"manual_jd_len={len(jd_text)}")
            print(f"manual_jd_full={jd_text}")
            print(f"{'=' * 80}\n")

            msg = f"Got it. {self._next_action_prompt()}"
            memory.store_turn(self.session_id, role="assistant", text=msg, user_id=self.user_id)
            return msg

        # ---- Optional deterministic router after job selection + JD intake ----
        selected_job = self.context_store.get("selected_job") or {}
        intake_payload = self.context_store.get("job_intake_payload") or {}
        has_selected_jd = isinstance(selected_job, dict) and bool((selected_job.get("jd_text") or "").strip())
        has_intake = isinstance(intake_payload, dict) and bool((intake_payload.get("answer") or "").strip())
        if has_selected_jd or has_intake:
            lowered = user_message.lower()
            if self._is_company_research_request(user_message):
                return self._execute_tool_and_respond(
                    tool_name="company_profiler",
                    params={},
                    reasoning="User requested company research for the selected job.",
                    user_message=user_message,
                )
            if "tailor" in lowered and "resume" in lowered:
                msg = (
                    "Resume tailoring is queued for the next phase. "
                    "For now I can run company profiling on your selected job. "
                    "Say: 'research the company'."
                )
                memory.store_turn(self.session_id, role="assistant", text=msg, user_id=self.user_id)
                return msg
            if "cover letter" in lowered:
                msg = (
                    "Cover letter generation is queued for the next phase. "
                    "For now I can run company profiling on your selected job. "
                    "Say: 'research the company'."
                )
                memory.store_turn(self.session_id, role="assistant", text=msg, user_id=self.user_id)
                return msg
            if any(k in lowered for k in ("recruiter", "outreach", "cold email", "message")):
                msg = (
                    "Recruiter outreach drafting is queued for the next phase. "
                    "For now I can run company profiling on your selected job. "
                    "Say: 'research the company'."
                )
                memory.store_turn(self.session_id, role="assistant", text=msg, user_id=self.user_id)
                return msg

        # Get LLM response
        response = self._call_llm()

        # Check if the LLM wants to call a tool
        tool_call = self._parse_tool_call(response)

        if tool_call:
            tool_name = tool_call["tool"]
            params = tool_call.get("parameters", {}) or {}
            reasoning = tool_call.get("reasoning", "")

            print(f"\n🔧 Tool Selected: {tool_name}")
            print(f"📋 Parameters: {json.dumps(params, indent=2)}")
            print(f"💭 Reasoning: {reasoning}")

            # Execute the tool
            return self._execute_tool_and_respond(
                tool_name=tool_name,
                params=params,
                reasoning=reasoning,
                user_message=user_message,
            )
        else:
            # No tool call — natural language response
            self.conversation_history.append({
                "role": "assistant",
                "content": response,
            })
            memory.store_turn(self.session_id, role="assistant", text=response, user_id=self.user_id)
            return response

    def _call_llm(self) -> str:
        """Call the HuggingFace Llama model."""
        def _messages_to_prompt(msgs: list[dict]) -> str:
            # Simple, deterministic chat-to-text prompt for text-generation fallback.
            parts: list[str] = []
            for m in msgs:
                role = (m.get("role") or "").strip().upper()
                content = (m.get("content") or "").strip()
                if not role or not content:
                    continue
                parts.append(f"{role}:\n{content}\n")
            parts.append("ASSISTANT:\n")
            return "\n".join(parts)

        def _call_router_chat_completions(msgs: list[dict]) -> str:
            # OpenAI-compatible HF Router endpoint.
            url = "https://router.huggingface.co/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {self.hf_token}",
                "Content-Type": "application/json",
            }
            payload = {
                "model": self.hf_model,
                "messages": msgs,
                "max_tokens": 1024,
                "temperature": 0.7,
                "top_p": 0.9,
            }
            resp = requests.post(url, headers=headers, json=payload, timeout=60)
            if resp.status_code >= 400:
                raise RuntimeError(f"{resp.status_code} {resp.text}".strip())
            data = resp.json()
            return (data["choices"][0]["message"]["content"] or "").strip()

        messages: list[dict] = []
        try:
            ctx = memory.get_context(self.session_id, user_id=self.user_id)
            messages = self.conversation_history + [
                {
                    "role": "system",
                    "content": f"Context packet (recent turns/artifacts/facts): {json.dumps(ctx)[:4000]}"
                }
            ]
            response = self.client.chat_completion(
                messages=messages,
                max_tokens=1024,
                temperature=0.7,
                top_p=0.9,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            err = str(e)

            # If default SDK call fails (including 410 on deprecated api-inference),
            # try the current HF Router endpoint directly.
            if (
                ("api-inference.huggingface.co is no longer supported" in err)
                or ("410" in err)
                or ("Gone for url" in err)
                or ("Inference Providers" in err)
                or ("router.huggingface.co" in err)
                or ("403 Forbidden" in err)
            ):
                try:
                    return _call_router_chat_completions(messages)
                except Exception as e2:
                    err = f"{err}\nFallback (router chat) failed: {e2}"

            # Final fallback: render a plain prompt and use text-generation.
            try:
                prompt = _messages_to_prompt(messages)
                generated = self.client.text_generation(
                    prompt,
                    max_new_tokens=1024,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    stop_sequences=["\nUSER:", "\nSYSTEM:"],
                )
                if isinstance(generated, str):
                    return generated.strip()
                # If details=True ever gets enabled, handle output object.
                return (generated.generated_text or "").strip()  # type: ignore[attr-defined]
            except Exception as e3:
                return f"⚠️ LLM Error: {err}\nFallback (text_generation) failed: {e3}"

    def _parse_tool_call(self, response: str) -> dict | None:
        """Extract tool call JSON from LLM response."""
        def _validate_tool_payload(payload: object) -> dict | None:
            if not isinstance(payload, dict):
                return None
            tool_name = payload.get("tool")
            if isinstance(tool_name, str) and tool_name in TOOL_NAMES:
                return payload
            return None

        # Pattern 1: fenced JSON blocks (```json ...``` or plain ``` ... ```)
        for match in re.finditer(r'```(?:json)?\s*(\{.*?\})\s*```', response, re.DOTALL | re.IGNORECASE):
            try:
                parsed = json.loads(match.group(1))
            except json.JSONDecodeError:
                continue
            valid = _validate_tool_payload(parsed)
            if valid:
                return valid

        # Pattern 2: whole response is a JSON object
        stripped = response.strip()
        if stripped.startswith("{"):
            try:
                parsed = json.loads(stripped)
            except json.JSONDecodeError:
                parsed = None
            valid = _validate_tool_payload(parsed)
            if valid:
                return valid

        # Pattern 3: scan for the first decodable JSON object anywhere in the response
        decoder = json.JSONDecoder()
        for idx, char in enumerate(response):
            if char != "{":
                continue
            try:
                parsed, _ = decoder.raw_decode(response[idx:])
            except json.JSONDecodeError:
                continue
            valid = _validate_tool_payload(parsed)
            if valid:
                return valid

        return None

    def _update_context(self, tool_name: str, result: str):
        """Store tool results for downstream use."""
        try:
            parsed = json.loads(result)
            if tool_name == "job_searcher":
                self.context_store["last_jobs"] = parsed
            elif tool_name == "company_profiler":
                self.context_store["company_research"] = parsed
                if isinstance(parsed, dict):
                    self.context_store["company_profile"] = parsed.get("profile")
                    self.context_store["company_sentiment"] = parsed.get("sentiment")
            elif tool_name == "resume_tailor":
                self.context_store["last_resume_tailoring"] = parsed
        except json.JSONDecodeError:
            pass

    def get_history(self) -> list:
        """Return conversation history (excluding system prompt)."""
        return [m for m in self.conversation_history if m["role"] != "system"]

    def reset(self):
        """Reset conversation."""
        self.conversation_history = [self.conversation_history[0]]  # Keep system prompt
        self.context_store = {k: None for k in self.context_store}
        print("🔄 Conversation reset.")


# ------------------------------------------------------------------ #
# CLI CHAT LOOP
# ------------------------------------------------------------------ #

def main():
    print("=" * 60)
    print("🤖 JobAgent AI — Intelligent Job Search Assistant")
    print("=" * 60)
    print("\nI can help you with:")
    print("  🔍 Search for jobs    → 'Find ML engineer jobs at Google'")
    print("  🏢 Research companies → 'Tell me about Anthropic'")
    print("  📄 Tailor resume      → 'Tailor my resume for this role'")
    print("  ✉️  Write cover letter → 'Write a cover letter for Google'")
    print("  📧 Draft emails       → 'Draft an email to the recruiter'")
    print("\nType 'quit' to exit, 'reset' to start over, 'history' to see chat log.\n")

    agent = JobAgent()

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 Goodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() == "quit":
            print("👋 Goodbye! Good luck with your job search!")
            break

        if user_input.lower() == "reset":
            agent.reset()
            continue

        if user_input.lower() == "history":
            print("\n📜 Conversation History:")
            for msg in agent.get_history():
                role = "🧑 You" if msg["role"] == "user" else "🤖 Agent"
                print(f"\n{role}: {msg['content'][:200]}")
            print()
            continue

        # Get agent response
        print("\n🤖 Agent: ", end="")
        response = agent.chat(user_input)
        print(response)
        print()


if __name__ == "__main__":
    main()
