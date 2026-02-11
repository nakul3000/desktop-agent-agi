# app.py — Multiturn Conversational Job Search Agent
# Uses HuggingFace Inference API + Llama for tool-calling agent
# ✅ Integrates your real agents.resume_tailor_agent for resume tailoring (architecture preserved)

from __future__ import annotations

import os
import json
import re
import uuid
from datetime import datetime

from dotenv import load_dotenv
from huggingface_hub import InferenceClient

import memory

# ✅ Real resume tailor agent (your codebase)
from agents.resume_tailor_agent import TailorRequest, tailor_resume

# One-time DB init guard
_DB_INITIALIZED = False


def _ensure_db_initialized() -> None:
    global _DB_INITIALIZED
    if _DB_INITIALIZED:
        return
    memory.init_db()
    _DB_INITIALIZED = True


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
        "triggers": [
            "find jobs",
            "search jobs",
            "job openings",
            "positions at",
            "hiring",
            "show me jobs",
            "look for roles",
        ],
    },
    {
        "name": "company_profiler",
        "description": "Research a company's background, culture, tech stack, recent news, funding, and financials. Use when the user wants to know about a company before applying.",
        "parameters": {
            "company": "The company name to research",
        },
        "triggers": [
            "tell me about",
            "company profile",
            "research company",
            "what does",
            "how is",
            "company culture",
            "company background",
        ],
    },
    {
        "name": "resume_tailor",
        "description": "Tailor the user's resume for a specific job posting. Rewrites bullet points and highlights to match the job description keywords and requirements.",
        "parameters": {
            "role": "Target job role",
            "company": "Target company",
            "job_description": "The job description to tailor resume for (paste raw text OR provide a URL).",
        },
        "triggers": [
            "tailor resume",
            "customize resume",
            "adapt resume",
            "resume for",
            "modify resume",
            "update resume",
            "fine tune my resume",
            "fine-tune my resume",
        ],
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
        "triggers": [
            "cover letter",
            "write a letter",
            "draft cover",
            "application letter",
            "generate cover letter",
        ],
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
        "triggers": [
            "email",
            "reach out",
            "contact recruiter",
            "cold email",
            "follow up",
            "outreach",
            "message recruiter",
        ],
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
3. **resume_tailor** - Tailor resume for a specific role/company. Use when user wants to customize/fine-tune their resume.
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
- For **resume_tailor**: you MUST request the full job description text OR a job posting URL if not provided.
- Never hallucinate job descriptions.
- After any tool result, summarize key findings and suggest logical next actions.
"""

# ------------------------------------------------------------------ #
# TOOL IMPLEMENTATIONS (stubs — connect to your real modules)
# ------------------------------------------------------------------ #


def execute_job_searcher(params: dict) -> str:
    """Stub job searcher — replace with your real Linkup job search when ready."""
    role = params.get("role", "Machine Learning Engineer")
    company = params.get("company", "")
    location = params.get("location", "United States")

    return json.dumps(
        {
            "status": "success",
            "query": f"{role} at {company} in {location}",
            "jobs_found": 3,
            "jobs": [
                {
                    "title": f"Senior {role}",
                    "company": company or "TechCorp",
                    "location": "San Francisco, CA",
                    "url": "https://careers.example.com/job/12345",
                    "salary": "$180K - $250K",
                    "posted": "2 days ago",
                },
                {
                    "title": f"{role} - AI Platform",
                    "company": company or "TechCorp",
                    "location": "Seattle, WA (Hybrid)",
                    "url": "https://careers.example.com/job/12346",
                    "salary": "$160K - $220K",
                    "posted": "1 day ago",
                },
                {
                    "title": f"Staff {role}",
                    "company": company or "TechCorp",
                    "location": "Remote US",
                    "url": "https://careers.example.com/job/12347",
                    "salary": "$200K - $300K",
                    "posted": "Today",
                },
            ],
            "next_steps": "I can research the company, tailor your resume, or draft a cover letter for any of these roles.",
        },
        indent=2,
    )


def execute_company_profiler(params: dict) -> str:
    """Stub company profiler — replace with your real company_research_agent when ready."""
    company = params.get("company", "Unknown")

    return json.dumps(
        {
            "status": "success",
            "company": company,
            "profile": {
                "overview": f"{company} is a leading technology company...",
                "industry": "Technology",
                "size": "10,000+ employees",
                "headquarters": "San Francisco, CA",
                "recent_news": [
                    f"{company} announced new AI research lab — Feb 2025",
                    f"{company} Q4 revenue beat expectations — Jan 2025",
                ],
                "tech_stack": ["Python", "PyTorch", "Kubernetes", "AWS"],
                "culture": "Fast-paced, engineering-driven, strong ML focus",
                "glassdoor_rating": "4.2/5",
                "interview_difficulty": "Hard — expect system design + ML coding",
            },
            "next_steps": "Want me to tailor your resume for this company or search for open roles?",
        },
        indent=2,
    )


def execute_resume_tailor(params: dict) -> str:
    """✅ Tailor resume using your real agents.resume_tailor_agent.

    Expected parameters:
      - role (optional but recommended)
      - company (optional)
      - job_description (required): raw JD text OR URL
    """
    role = (params.get("role") or "").strip() or None
    company = (params.get("company") or "").strip() or None
    jd = (params.get("job_description") or "").strip()

    if not jd:
        return json.dumps(
            {
                "status": "error",
                "message": "Missing job_description. Please paste the full job description text or provide the job posting URL.",
            },
            indent=2,
        )

    # Allow env overrides (keeps architecture stable)
    resume_path = os.getenv("DEFAULT_RESUME_PATH", "samples/sample_data_analyst_resume.pdf")
    out_pdf = os.getenv("TAILORED_PDF_PATH", "outputs/tailored_resume_ats.pdf")
    out_docx_env = os.getenv("TAILORED_DOCX_PATH", "").strip()
    out_docx = out_docx_env or None

    try:
        result = tailor_resume(
            TailorRequest(
                resume_path=resume_path,
                job_description=jd,  # can be text OR URL (your agent handles URL via Linkup if implemented)
                job_title=role,
                company=company,
                location=None,
                output_pdf_path=out_pdf,
                output_docx_path=out_docx,
            )
        )

        data = (result.get("data") or {}) if isinstance(result, dict) else {}
        preview = (data.get("tailored_resume_text") or "")[:1400]

        return json.dumps(
            {
                "status": "success",
                "summary": result.get("summary", "Tailored resume generated.") if isinstance(result, dict) else "Tailored resume generated.",
                "resume_path": resume_path,
                "output_pdf_path": data.get("output_pdf_path"),
                "output_docx_path": data.get("output_docx_path"),
                "preview": preview,
                "jd_keywords": (data.get("jd_keywords") or [])[:20],
            },
            indent=2,
        )
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)}, indent=2)


def execute_cover_letter_generator(params: dict) -> str:
    """Stub cover letter generator."""
    role = params.get("role", "")
    company = params.get("company", "")

    return json.dumps(
        {
            "status": "success",
            "cover_letter": f"""Dear Hiring Team at {company},

I am writing to express my interest in the {role} position at {company}.

[Stub — replace with real cover letter generation logic]

Best regards,
[Your Name]""",
            "next_steps": "Want me to draft an outreach email to a recruiter at this company?",
        },
        indent=2,
    )


def execute_email_crafter(params: dict) -> str:
    """Stub email crafter."""
    recipient = params.get("recipient_name", "Hiring Manager")
    company = params.get("company", "")
    role = params.get("role", "")
    purpose = params.get("purpose", "cold_outreach")

    return json.dumps(
        {
            "status": "success",
            "email": {
                "subject": f"Re: {role} Opportunity at {company}",
                "to": recipient,
                "body": f"""Hi {recipient},

I came across the {role} position at {company} and was excited to reach out.

[Stub — replace with real outreach email generation logic]

Best,
[Your Name]""",
            },
            "purpose": purpose,
            "next_steps": "I can refine the tone or generate variants.",
        },
        indent=2,
    )


# Tool dispatcher
TOOL_EXECUTORS = {
    "job_searcher": execute_job_searcher,
    "company_profiler": execute_company_profiler,
    "resume_tailor": execute_resume_tailor,  # ✅ real integration
    "cover_letter_generator": execute_cover_letter_generator,
    "email_crafter": execute_email_crafter,
}

# ------------------------------------------------------------------ #
# AGENT CORE
# ------------------------------------------------------------------ #


class JobAgent:
    def __init__(self, session_id: str | None = None, user_id: str | None = None):
        self.hf_token = os.getenv("HF_TOKEN")
        if not self.hf_token:
            raise ValueError("HF_TOKEN not found in .env")

        self.linkup_api_key = os.getenv("LINKUP_API_KEY")
        if not self.linkup_api_key:
            raise ValueError("LINKUP_API_KEY not found in .env")

        self.client = InferenceClient(
            model="meta-llama/Meta-Llama-3-70B-Instruct",
            token=self.hf_token,
        )

        # Conversation history: list of {"role": "user"/"assistant"/"system", "content": "..."}
        self.conversation_history: list[dict[str, str]] = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]

        # Initialize memory
        self.user_id = user_id or "anonymous"
        self.session_id = session_id or str(uuid.uuid4())
        _ensure_db_initialized()
        memory.register_user(self.user_id)
        memory.start_session(user_id=self.user_id, session_id=self.session_id)

        # Store context from tool results for downstream use
        self.context_store = {
            "last_jobs": None,
            "last_company_profile": None,
            "last_resume_tailoring": None,
            "user_resume": None,
        }

    def chat(self, user_message: str) -> str:
        """Process a user message and return agent response."""

        # Add user message to history
        self.conversation_history.append({"role": "user", "content": user_message})
        memory.store_turn(self.session_id, role="user", text=user_message, user_id=self.user_id)

        # Get LLM response
        response = self._call_llm()

        # Check if the LLM wants to call a tool
        tool_call = self._parse_tool_call(response)

        if tool_call:
            tool_name = tool_call["tool"]
            params = tool_call.get("parameters", {})
            reasoning = tool_call.get("reasoning", "")

            print(f"\n🔧 Tool Selected: {tool_name}")
            print(f"📋 Parameters: {json.dumps(params, indent=2)}")
            print(f"💭 Reasoning: {reasoning}")

            # Execute the tool
            if tool_name in TOOL_EXECUTORS:
                memory.store_turn(
                    self.session_id,
                    role="assistant",
                    text=f"[Tool call planned] {tool_name} {params}",
                    user_id=self.user_id,
                )
                tool_result = TOOL_EXECUTORS[tool_name](params)
                tool_turn_id = memory.store_turn(
                    self.session_id,
                    role="tool",
                    text=str(tool_result),
                    user_id=self.user_id,
                    tool_name=tool_name,
                )

                # Store context + persist artifact
                self._update_context(tool_name, tool_result)
                memory.store_artifact(
                    session_id=self.session_id,
                    type=tool_name,
                    content=tool_result,
                    source_turn_id=tool_turn_id,
                    created_by="JobAgent",
                    user_id=self.user_id,
                )

                # Add tool result to history and ask LLM to summarize
                self.conversation_history.append(
                    {"role": "assistant", "content": f"[Called tool: {tool_name}]\n{reasoning}"}
                )
                self.conversation_history.append(
                    {
                        "role": "user",
                        "content": (
                            f"[Tool Result for {tool_name}]:\n{tool_result}\n\n"
                            "Now summarize these results for the user in a helpful, conversational way. "
                            "Highlight key findings and suggest logical next steps."
                        ),
                    }
                )

                summary = self._call_llm()
                self.conversation_history.append({"role": "assistant", "content": summary})
                memory.store_turn(self.session_id, role="assistant", text=summary, user_id=self.user_id)
                return summary

            error_msg = f"Unknown tool: {tool_name}. Available tools: {TOOL_NAMES}"
            self.conversation_history.append({"role": "assistant", "content": error_msg})
            memory.store_turn(self.session_id, role="assistant", text=error_msg, user_id=self.user_id)
            return error_msg

        # No tool call — natural language response
        self.conversation_history.append({"role": "assistant", "content": response})
        memory.store_turn(self.session_id, role="assistant", text=response, user_id=self.user_id)
        return response

    def _call_llm(self) -> str:
        """Call the HuggingFace Llama model."""
        try:
            ctx = memory.get_context(self.session_id, user_id=self.user_id)
            messages = self.conversation_history + [
                {
                    "role": "system",
                    "content": f"Context packet (recent turns/artifacts/facts): {json.dumps(ctx)[:4000]}",
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
            return f"⚠️ LLM Error: {str(e)}"

    def _parse_tool_call(self, response: str) -> dict | None:
        """Extract tool call JSON from LLM response."""
        # Pattern 1: ```json ... ```
        json_match = re.search(r"```json\s*(\{.*?\})\s*```", response, re.DOTALL)
        if json_match:
            try:
                parsed = json.loads(json_match.group(1))
                if parsed.get("tool") in TOOL_NAMES:
                    return parsed
            except json.JSONDecodeError:
                pass

        # Pattern 2: Raw JSON
        json_match = re.search(r'(\{"tool":\s*"[^"]+?".*?\})', response, re.DOTALL)
        if json_match:
            try:
                parsed = json.loads(json_match.group(1))
                if parsed.get("tool") in TOOL_NAMES:
                    return parsed
            except json.JSONDecodeError:
                pass

        return None

    def _update_context(self, tool_name: str, result: str) -> None:
        """Store tool results for downstream use."""
        try:
            parsed = json.loads(result)
            if tool_name == "job_searcher":
                self.context_store["last_jobs"] = parsed
            elif tool_name == "company_profiler":
                self.context_store["last_company_profile"] = parsed
            elif tool_name == "resume_tailor":
                self.context_store["last_resume_tailoring"] = parsed
        except json.JSONDecodeError:
            pass

    def get_history(self) -> list:
        """Return conversation history (excluding system prompt)."""
        return [m for m in self.conversation_history if m["role"] != "system"]

    def reset(self) -> None:
        """Reset conversation."""
        self.conversation_history = [self.conversation_history[0]]  # Keep system prompt
        self.context_store = {k: None for k in self.context_store}
        print("🔄 Conversation reset.")


# ------------------------------------------------------------------ #
# CLI CHAT LOOP
# ------------------------------------------------------------------ #


def main() -> None:
    print("=" * 60)
    print("🤖 JobAgent AI — Intelligent Job Search Assistant")
    print("=" * 60)
    print("\nI can help you with:")
    print("  🔍 Search for jobs     → 'Find ML engineer jobs at Google'")
    print("  🏢 Research companies  → 'Tell me about Anthropic'")
    print("  📄 Tailor resume       → 'Tailor my resume for this role'")
    print("  ✉️  Write cover letter → 'Write a cover letter for Google'")
    print("  📧 Draft emails        → 'Draft an email to the recruiter'")
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

        print("\n🤖 Agent: ", end="")
        response = agent.chat(user_input)
        print(response)
        print()


if __name__ == "__main__":
    main()
