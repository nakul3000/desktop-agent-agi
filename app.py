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
from agents.resume_tailor_agent import TailorRequest, tailor_resume, _fetch_job_description_if_url
from agents.role_search_agent import search_roles, RoleSearchQuery
from job_searcher import JobSearcher
from document_handler import DocumentHandler

# from utils import load_resume_from_env

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
        "description": "Draft a personalized email to a recruiter or hiring manager. Requires job description for context and optionally resume text for personalization.",
        "parameters": {
            "job_description": "The job description text (required for context)",
            "recipient_name": "Name of the recruiter/hiring manager (optional)",
            "resume_text": "User's resume text for personalization (optional)",
            "tone": "Email tone: professional (default), casual, or friendly",
            "additional_context": "Any extra context (availability, specific achievements, etc.)",
        },
        "triggers": ["email", "reach out", "contact recruiter", "cold email", "follow up", "outreach", "message recruiter", "draft email"],
    },
]

TOOL_NAMES = [t["name"] for t in TOOLS]

# ------------------------------------------------------------------ #
# SYSTEM PROMPT
# ------------------------------------------------------------------ #

_TODAY = datetime.now().strftime("%B %d, %Y")
SYSTEM_PROMPT = f"""You are JobAgent AI, an intelligent job search and application assistant. Today's date is {_TODAY}.

You have access to these tools:

1. **job_searcher** - Search for job openings. Use when user asks to find/search jobs (e.g., "find me jobs about ML in Google" or "find ML/DS jobs").
2. **resume_tailor** - Tailor resume for a specific job. If user says "tailor my resume," prompt for a job link, then pass the link to the resume_tailor agent for parsing and tailoring.
3. **company_profiler** - Research a company's background, culture, news. Use when user asks about a company.
4. **cover_letter_generator** - Write personalized cover letters. Use when user needs a cover letter.
5. **email_crafter** - Draft outreach emails to recruiters. Use when user wants to email/contact someone.

IMPORTANT RULES FOR TOOL CALLING:
- If the query is about finding jobs (ML/DS, Google, etc.), call job_searcher.
- If the user says "tailor my resume," ask for a job link, then pass it to resume_tailor for parsing and tailoring.
- If the user selects a job from job_searcher, pass the job link to resume_tailor for tailoring.
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
    """Execute job search using JobSearcher with structured output"""
    
    # Handle both structured params (role, company, location) and generic query
    query_str = params.get("query", "")
    role = params.get("role", "")
    company = params.get("company", "")
    location = params.get("location", "United States")
    
    # If we only have a generic query, parse it for role/company/location
    if query_str and not role:
        # Extract keywords from generic query
        # e.g., "machine learning engineer jobs in google" -> role="machine learning engineer", company="google"
        query_lower = query_str.lower()
        
        # Common company names
        companies = ["google", "amazon", "microsoft", "meta", "apple", "netflix", "tesla", "stripe", "airbnb"]
        
        # Detect company
        for comp in companies:
            if comp in query_lower:
                company = comp.capitalize()
                role = query_str.replace(f"in {comp}", "").replace(f"at {comp}", "").replace("jobs", "").strip()
                if not role:
                    role = "Software Engineer"
                break
        
        # If no company detected, use the query as role
        if not role:
            role = query_str if query_str else "Software Engineer"
    
    # Fallback
    if not role:
        role = "Machine Learning Engineer"

    try:
        # Use JobSearcher with Linkup structured output
        searcher = JobSearcher()
        result_json = searcher.execute_search({
            "role": role,
            "company": company,
            "location": location,
        })
        
        # Parse the result
        result = json.loads(result_json) if isinstance(result_json, str) else result_json
        
        # Extract jobs from response
        jobs_list = []
        if result.get("status") == "success":
            response = result.get("response", {})
            
            # Handle structured output - response contains the schema output
            if hasattr(response, "model_dump"):
                response = response.model_dump()
            
            # Structured output returns jobs directly
            if isinstance(response, dict):
                jobs_list = response.get("jobs", [])
            elif isinstance(response, str):
                try:
                    response_data = json.loads(response)
                    jobs_list = response_data.get("jobs", [])
                except:
                    jobs_list = []
        
        return json.dumps(
            {
                "status": "success",
                "query": {"role": role, "company": company, "location": location},
                "search_results_count": len(jobs_list),
                "jobs_found": len(jobs_list),
                "jobs": jobs_list,
                "next_steps": "Reply with the number of a job to select it." if jobs_list else "No jobs found. Try a different search query.",
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
    """
    Real resume tailoring using agents.resume_tailor_agent.
    Requires RESUME_PATH in .env (or pass params.resume_path).
    Falls back to sample resume from samples/ folder if not found.
    Uses selected job's JD (context.selected_job.jd_text) if available.
    """
    try:
        context = params.get("context") or {}
        selected_job = (context.get("selected_job") or {}) if isinstance(context, dict) else {}

        # 1) Resume path (required) — with fallback to samples folder
        resume_path = (params.get("resume_path") or os.getenv("RESUME_PATH") or "").strip()
        
        # Fallback: look for resume in samples folder if not provided
        if not resume_path or not Path(resume_path).exists():
            samples_dir = Path(__file__).resolve().parent / "samples"
            if samples_dir.exists():
                # Look for any PDF in samples folder
                pdf_files = list(samples_dir.glob("*.pdf"))
                if pdf_files:
                    resume_path = str(pdf_files[0])
        
        if not resume_path or not Path(resume_path).exists():
            return json.dumps(
                {
                    "status": "error",
                    "error": "Missing RESUME_PATH. Add RESUME_PATH=/absolute/path/to/resume.pdf in .env "
                             "or place a resume PDF in the samples/ folder.",
                },
                indent=2,
            )

        # 2) Job description text (required)
        jd_text = ""
        if isinstance(selected_job, dict):
            jd_text = (selected_job.get("jd_text") or "").strip()

        # fallback: tool param
        if not jd_text:
            jd_text = (params.get("job_description") or "").strip()

        if not jd_text:
            return json.dumps(
                {
                    "status": "error",
                    "error": "Missing job description text. Select a job first (so jd_text is captured) "
                             "or pass job_description in the tool call.",
                },
                indent=2,
            )

        # 3) Metadata (optional)
        job_title = (params.get("role") or selected_job.get("title") or "").strip() or None
        company = (params.get("company") or selected_job.get("company") or "").strip() or None
        location = (params.get("location") or selected_job.get("location") or "").strip() or None

        # 4) Output paths (optional but recommended)
        out_dir = Path("outputs")
        out_dir.mkdir(parents=True, exist_ok=True)

        output_docx_path = (params.get("output_docx_path") or str(out_dir / "tailored_resume.docx")).strip()
        output_pdf_path = (params.get("output_pdf_path") or str(out_dir / "tailored_resume_ats.pdf")).strip()

        req = TailorRequest(
            resume_path=resume_path,
            job_description=jd_text,
            job_title=job_title,
            company=company,
            location=location,
            output_docx_path=output_docx_path,
            output_pdf_path=output_pdf_path,
        )

        result = tailor_resume(req)

        # Ensure we return a JSON string (your app expects this)
        return json.dumps(result, indent=2)

    except Exception as e:
        error_type = type(e).__name__
        tb_text = traceback.format_exc().rstrip()
        print(f"\n{'=' * 80}")
        print("❌ ERROR LOG: resume_tailor failed")
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
            },
            indent=2,
        )


def execute_cover_letter_generator(params: dict) -> str:
    """Generate personalized cover letter"""
    role = params.get("role", "")
    company = params.get("company", "")

    return json.dumps({
        "status": "success",
        "cover_letter": f"""Dear Hiring Team at Google,

I am writing to express my interest in the Software Engineer position at Google. I have experience building production AI systems and working with modern ML technologies.

[This is a stub — the real implementation would use the LLM to generate a full personalized cover letter based on resume + company research + job description]

I would welcome the opportunity to discuss how my experience aligns with Google's mission.

Best regards,
[Your Name]""",
        "next_steps": "Want me to draft an outreach email to a recruiter at this company?",
    }, indent=2)


def execute_email_crafter(params: dict) -> str:
    """Draft email to recruiter using email_handler"""
    try:
        # Get context with selected_job (if available)
        context = params.get("context") or {}
        selected_job = (context.get("selected_job") or {}) if isinstance(context, dict) else {}

        # 1) Get job description
        job_description = (params.get("job_description") or "").strip()
        
        # Fallback: use jd_text from selected_job (populated during job selection)
        if not job_description and isinstance(selected_job, dict):
            job_description = (selected_job.get("jd_text") or "").strip()

        # If still no JD, error
        if not job_description:
            return json.dumps({
                "status": "error",
                "error": "Missing job description. Select a job first or pass job_description parameter.",
            }, indent=2)

        # 2) Get recipient name
        recipient_name = (params.get("recipient_name") or "Hiring Manager").strip()

        # 3) Get resume text
        resume_text = (params.get("resume_text") or "").strip()
        
        # Fallback: load from RESUME_PATH
        if not resume_text:
            resume_path = os.getenv("RESUME_PATH")
            if resume_path and os.path.exists(resume_path):
                try:
                    doc_handler = DocumentHandler()
                    result = doc_handler.analyze(resume_path, doc_type="resume")
                    if result.get("status") == "ok":
                        resume_text = (result.get("data", {}).get("text") or "").strip()
                except Exception:
                    resume_text = ""

        # 4) Get tone and context
        tone = (params.get("tone") or "professional").strip()
        additional_context = (params.get("additional_context") or "").strip()

        # Initialize email handler
        from email_handler import EmailHandler
        email_handler = EmailHandler(hf_token=os.getenv("HF_TOKEN"))

        # Create thread_data for email_handler function
        thread_data = {
            "threadId": "",
            "messages": [],
            "latest": {
                "subject": f"Inquiry: {selected_job.get('title', 'Position')} at {selected_job.get('company', 'Company')}",
                "body": "",
                "sender": "me",
            },
            "combined_body": "",
        }

        # Draft using email_handler
        draft_body = email_handler.draft_reply_to_recruiter_thread(
            thread_data=thread_data,
            job_description=job_description,
            resume_text=resume_text,
            additional_context=additional_context,
            tone=tone,
        )

        # Format subject based on selected job info
        subject = f"Application: {selected_job.get('title', 'Position')} at {selected_job.get('company', 'Company')}"

        return json.dumps({
            "status": "success",
            "email": {
                "to": recipient_name,
                "subject": subject,
                "body": draft_body,
            },
            "next_steps": "Review the draft and let me know if you'd like me to send it or refine it.",
        }, indent=2)

    except Exception as e:
        error_type = type(e).__name__
        tb_text = traceback.format_exc().rstrip()
        print(f"\n{'=' * 80}")
        print("❌ ERROR LOG: email_crafter failed")
        print(f"Type: {error_type}")
        print(f"Message: {e}")
        print("Traceback:")
        print(tb_text)
        print(f"{'=' * 80}\n")
        return json.dumps({
            "status": "error",
            "error": f"{error_type}: {str(e)}",
            "message": "Failed to draft email. Ensure HF_TOKEN is set and email_handler is properly configured.",
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
        self.awaiting_email_jd_link = False

        # Best-effort: load resume text for downstream stubs/agents.
        self.context_store["user_resume"] = None

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
            # Handle both field name formats (jobTitle vs title, companyName vs company, etc.)
            title = (j.get("jobTitle") or j.get("title") or "N/A").strip()
            company = (j.get("companyName") or j.get("company") or "N/A").strip()
            location = (j.get("location") or "N/A").strip()
            experience = (j.get("experienceLevel") or "").strip()
            salary = (j.get("salaryRange") or "").strip()
            url = (j.get("applicationUrl") or j.get("url") or "").strip()
            
            # Build info line
            info = f"{i}. {title} — {company}"
            if location and location != "N/A":
                info += f" ({location})"
            if experience:
                info += f" [{experience}]"
            if salary:
                info += f" ${salary}"
            
            lines.append(info)
            if url:
                lines.append(f"   {url}")
        
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
        # Handle both old (url) and new (applicationUrl) field names
        job_url = (selected_job.get("url") or selected_job.get("applicationUrl") or "").strip()
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

        if tool_name == "email_crafter":
            # Show the actual drafted email, not an LLM summary
            try:
                parsed = json.loads(tool_result)
                if isinstance(parsed, dict) and parsed.get("status") == "success":
                    email_data = parsed.get("email", {})
                    to = email_data.get("to", "Hiring Manager")
                    subject = email_data.get("subject", "")
                    body = email_data.get("body", "")
                    next_steps = parsed.get("next_steps", "")

                    message_parts = [
                        "✅ Email drafted!\n",
                        f"To: {to}",
                        f"Subject: {subject}",
                        "",
                        "--- Draft Body ---",
                        body,
                        "--- End Draft ---",
                        "",
                    ]
                    if next_steps:
                        message_parts.append(next_steps)
                    message_parts.append("\nYou can also say: 'tailor my resume' for this job.")
                    msg = "\n".join(message_parts)
                elif isinstance(parsed, dict) and parsed.get("status") == "error":
                    error = parsed.get("error", "Unknown error")
                    msg = f"❌ Email drafting failed: {error}"
                else:
                    msg = tool_result if isinstance(tool_result, str) else str(tool_result)
            except json.JSONDecodeError:
                msg = tool_result if isinstance(tool_result, str) else str(tool_result)
            memory.store_turn(self.session_id, role="assistant", text=msg, user_id=self.user_id)
            return msg

        if tool_name == "resume_tailor":
            # Format resume tailor output without a second LLM call
            try:
                parsed = json.loads(tool_result)
                if isinstance(parsed, dict) and parsed.get("status") in ("success", "ok"):
                    data = parsed.get("data", {})
                    output_docx = data.get("output_docx_path") or parsed.get("output_docx_path", "N/A")
                    output_pdf = data.get("output_pdf_path") or parsed.get("output_pdf_path", "N/A")
                    message_parts = [
                        "✅ Resume tailored successfully!",
                        "",
                        f"📄 Tailored Resume: {output_docx}",
                        f"📋 ATS-Optimized PDF: {output_pdf}",
                        "",
                        "Next steps:",
                        "1. Review the tailored resume",
                        "2. Consider a cover letter (say: 'write a cover letter')",
                        "3. Research the company (say: 'research the company')",
                    ]
                    msg = "\n".join(message_parts)
                elif isinstance(parsed, dict) and parsed.get("status") == "error":
                    error = parsed.get("error", "Unknown error")
                    error_type = parsed.get("error_type", "")
                    if error_type:
                        msg = f"❌ Resume tailoring failed ({error_type}): {error}"
                    else:
                        msg = f"❌ Resume tailoring failed: {error}"
                else:
                    status = parsed.get("status", "unknown")
                    msg = f"❌ Resume tailoring failed: Status '{status}' - {parsed.get('summary', 'No summary available')}"
            except json.JSONDecodeError:
                msg = tool_result if isinstance(tool_result, str) else str(tool_result)
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
        summary = self.intent_detector_call()
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
        # Resume tailoring: prompt for job link if user mentions resume editing/tailoring
        if any(phrase in user_message.lower() for phrase in ("tailor", "edit", "customize", "adapt", "modify", "update", "resume", "cv")):
            if any(word in user_message.lower() for word in ("resume", "cv")):
                # Check if JD already exists in context (from email flow or job selection)
                existing_selected = self.context_store.get("selected_job") or {}
                existing_jd = (existing_selected.get("jd_text") or "").strip() if isinstance(existing_selected, dict) else ""
                if existing_jd:
                    tailoring_msg = self._execute_tool_and_respond(
                        tool_name="resume_tailor",
                        params={
                            "resume_path": os.getenv("RESUME_PATH", ""),
                            "job_description": existing_jd,
                            "job_title": existing_selected.get("title", ""),
                            "company": existing_selected.get("company", ""),
                            "location": existing_selected.get("location", ""),
                        },
                        reasoning="Using JD already in context from previous flow.",
                        user_message=user_message,
                    )
                    memory.store_turn(self.session_id, role="assistant", text=tailoring_msg, user_id=self.user_id)
                    return tailoring_msg
                # Check if they provided a link in the same message
                if "http" in user_message:
                    # Extract the URL from the message
                    import re as regex_module
                    url_match = regex_module.search(r'https?://[^\s]+', user_message)
                    if url_match:
                        job_link = url_match.group(0)
                        jd_text = _fetch_job_description_if_url(job_link)
                        tailoring_msg = self._execute_tool_and_respond(
                            tool_name="resume_tailor",
                            params={
                                "resume_path": os.getenv("RESUME_PATH", ""),
                                "job_description": jd_text,
                                "job_title": "",
                                "company": "",
                                "location": "",
                            },
                            reasoning="User provided job link with resume request.",
                            user_message=user_message,
                        )
                        memory.store_turn(self.session_id, role="assistant", text=tailoring_msg, user_id=self.user_id)
                        return tailoring_msg
                # No link provided and no JD in context
                self.awaiting_jd_text = True
                msg = 'Please provide a link to the job role you want to tailor your resume for.'
                memory.store_turn(self.session_id, role='assistant', text=msg, user_id=self.user_id)
                return msg

        # Handle awaiting job link after 'tailor my resume'
        if getattr(self, 'awaiting_jd_text', False):
            job_link = user_message.strip()
            # Extract job description from link using linkup
            jd_text = _fetch_job_description_if_url(job_link)
            tailoring_msg = self._execute_tool_and_respond(
                tool_name="resume_tailor",
                params={
                    "resume_path": os.getenv("RESUME_PATH", ""),
                    "job_description": jd_text,
                    "job_title": "",
                    "company": "",
                    "location": "",
                },
                reasoning="User provided job link for resume tailoring.",
                user_message=user_message,
            )
            self.awaiting_jd_text = False
            memory.store_turn(self.session_id, role="assistant", text=tailoring_msg, user_id=self.user_id)
            return tailoring_msg
        if self.awaiting_email_jd_link:
            link = user_message.strip()
            if not link or link.lower() in {"cancel", "skip"}:
                self.awaiting_email_jd_link = False
                msg = "Email drafting cancelled. What would you like to do next?"
                memory.store_turn(self.session_id, role="assistant", text=msg, user_id=self.user_id)
                return msg

            # Use the same parser function that resume_tailor uses
            try:
                jd_text = _fetch_job_description_if_url(link)
            except Exception as e:
                jd_text = ""
                print(f"\n{'=' * 80}")
                print("🧭 DEBUG LOG: email flow -> JD parse failed")
                print(f"error={type(e).__name__}: {e}")
                print(f"{'=' * 80}\n")

            if jd_text and len(jd_text) > 50:
                print(f"\n{'=' * 80}")
                print("🧭 DEBUG LOG: email flow -> JD parsed from link")
                print(f"source_url={link}")
                print(f"parsed_jd_len={len(jd_text)}")
                print(f"{'=' * 80}\n")

                self.awaiting_email_jd_link = False

                # Store JD in selected_job so downstream tools (resume_tailor, etc.) can reuse it
                self.context_store["selected_job"] = {
                    "job_id": "",
                    "title": "N/A",
                    "company": "N/A",
                    "location": "N/A",
                    "url": link,
                    "jd_text": jd_text,
                }

                # Call email_crafter with the parsed JD
                email_msg = self._execute_tool_and_respond(
                    tool_name="email_crafter",
                    params={"job_description": jd_text},
                    reasoning="User provided job link; parsed JD and drafting email.",
                    user_message=user_message,
                )
                memory.store_turn(self.session_id, role="assistant", text=email_msg, user_id=self.user_id)
                return email_msg

            msg = (
                "Could not extract a job description from that link.\n"
                "Please provide a valid job posting URL, or paste the job description text directly."
            )
            memory.store_turn(self.session_id, role="assistant", text=msg, user_id=self.user_id)
            return msg
        
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
                
                # Map both old field names (title, company, url) and new field names (jobTitle, companyName, applicationUrl)
                self.context_store["selected_job"] = {
                    "job_id": chosen.get("job_id") or str(idx),
                    "title": (chosen.get("jobTitle") or chosen.get("title") or "N/A").strip(),
                    "company": (chosen.get("companyName") or chosen.get("company") or "N/A").strip(),
                    "location": (chosen.get("location") or "N/A").strip(),
                    "url": (chosen.get("applicationUrl") or chosen.get("url") or "N/A").strip(),
                    "jd_text": chosen_jd if chosen_jd and chosen_jd != "NA" else "",
                }
                self.awaiting_job_selection = False

                selected_job = self.context_store.get("selected_job") or {}
                job_url = (selected_job.get("url") or "").strip()

                # Try to extract JD text from the job URL
                jd_text = ""

                # First attempt: _extract_jd_from_selected_job_url (Linkup structured extraction)
                extracted = self._extract_jd_from_selected_job_url(selected_job)
                if isinstance(extracted, dict) and extracted.get("status") == "success":
                    jd_text = (extracted.get("jd_text") or "").strip()

                # Fallback: use _fetch_job_description_if_url (same parser as resume tailor)
                if not jd_text and job_url and job_url != "N/A":
                    try:
                        jd_text = (_fetch_job_description_if_url(job_url) or "").strip()
                    except Exception:
                        jd_text = ""

                # Use jobDescriptionSummary from search results as last resort
                if not jd_text:
                    jd_text = (chosen.get("jobDescriptionSummary") or "").strip()

                if jd_text:
                    selected_job["jd_text"] = jd_text
                    self.context_store["selected_job"] = selected_job
                    self.context_store["job_intake_payload"] = self._build_job_intake_payload(
                        jd_text=jd_text,
                        selected_job=selected_job,
                    )

                title = selected_job.get("title", "N/A")
                company = selected_job.get("company", "N/A")
                msg_parts = [f"Selected: **{title}** at **{company}**"]
                if jd_text:
                    msg_parts.append(f"Job description extracted ({len(jd_text)} chars).")
                    msg_parts.append("")
                    msg_parts.append("What would you like to do?")
                    msg_parts.append("- 'tailor my resume' — tailor your resume for this role")
                    msg_parts.append("- 'draft an email' — draft an outreach email")
                    msg_parts.append("- 'research the company' — get company profile & sentiment")
                else:
                    msg_parts.append("Could not extract the full JD, but you can still:")
                    msg_parts.append("- 'tailor my resume' with a job link")
                    msg_parts.append("- 'draft an email' with a job link")
                    msg_parts.append("- 'research the company'")

                msg = "\n".join(msg_parts)
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

        # ---- Check for email drafting without a selected job ----
        lowered = user_message.lower()
        has_email_request = any(k in lowered for k in ("recruiter", "outreach", "cold email", "message", "email", "draft email"))
        selected_job = self.context_store.get("selected_job") or {}
        intake_payload = self.context_store.get("job_intake_payload") or {}
        has_selected_jd = isinstance(selected_job, dict) and bool((selected_job.get("jd_text") or "").strip())
        has_intake = isinstance(intake_payload, dict) and bool((intake_payload.get("answer") or "").strip())
        
        # If user asks for email but no job selected, ask for link
        if has_email_request and not (has_selected_jd or has_intake):
            self.awaiting_email_jd_link = True
            msg = "I'd be happy to help you draft an email! Please provide the job posting link so I can extract the job description and personalize the email."
            memory.store_turn(self.session_id, role="assistant", text=msg, user_id=self.user_id)
            return msg
        
        # ---- Optional deterministic router after job selection + JD intake ----
        if has_selected_jd or has_intake:
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
            if has_email_request:
                return self._execute_tool_and_respond(
                    tool_name="email_crafter",
                    params={},
                    reasoning="User requested to draft an email for the selected job.",
                    user_message=user_message,
                )

        # Get LLM response
        response = self.intent_detector_call()

        # Check if the LLM wants to call a tool
        tool_call = self._parse_tool_call(response)

        if tool_call:
            tool_name = tool_call["tool"]
            params = tool_call.get("parameters", {}) or {}
            reasoning = tool_call.get("reasoning", "")

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

    def intent_detector_call(self) -> str:
        """Intent detector and tool selector using LLM."""
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
            # Only use last 10 turns for context awareness (limit token usage)
            recent_history = self.conversation_history[-20:] if len(self.conversation_history) > 20 else self.conversation_history
            messages = recent_history + [
                {
                    "role": "system",
                    "content": f"Context packet (recent turns/artifacts/facts): {json.dumps(ctx)[:2000]}"
                }
            ]
            response = self.client.chat_completion(
                messages=messages,
                max_tokens=1024,
                temperature=0.7,
                top_p=0.9,
            )
            text = response.choices[0].message.content.strip()
            memory.store_turn(self.session_id, role="assistant", text=text, user_id=self.user_id)
            memory.store_artifact(self.session_id, "llm_response", text, user_id=self.user_id)
            return text
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
