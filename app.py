# app.py — Multiturn Conversational Job Search Agent
# Uses HuggingFace Inference API + Llama for tool-calling agent

import os
import json
import re
from datetime import datetime
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

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

    # TODO: Replace with actual LinkupJobSearch call
    # from linkup_client import LinkupJobSearch
    # searcher = LinkupJobSearch()
    # results = searcher.search_jobs(role, company, location)

    # Stub response for testing
    return json.dumps({
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
    }, indent=2)


def execute_company_profiler(params: dict) -> str:
    """Execute company research — connects to your linkup_client.py"""
    company = params.get("company", "Unknown")

    # TODO: Replace with actual LinkupJobSearch.get_company_profile() call

    return json.dumps({
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
    }, indent=2)


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
    def __init__(self):
        self.hf_token = os.getenv("HF_TOKEN")
        if not self.hf_token:
            raise ValueError("HF_TOKEN not found in .env")

        self.linkup_api_key = os.getenv("LINKUP_API_KEY")
        if not self.linkup_api_key:
            raise ValueError("LINKUP_API_KEY not found in .env")

        self.client = InferenceClient(
            model="meta-llama/Llama-3.3-70B-Instruct",
            token=self.hf_token,
        )

        # Conversation history: list of {"role": "user"/"assistant"/"system", "content": "..."}
        self.conversation_history = []
        self.conversation_history.append({
            "role": "system",
            "content": SYSTEM_PROMPT,
        })

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
        self.conversation_history.append({
            "role": "user",
            "content": user_message,
        })

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
                tool_result = TOOL_EXECUTORS[tool_name](params)

                # Store context
                self._update_context(tool_name, tool_result)

                # Add tool call + result to conversation history
                self.conversation_history.append({
                    "role": "assistant",
                    "content": f"[Called tool: {tool_name}]\n{reasoning}",
                })
                self.conversation_history.append({
                    "role": "user",
                    "content": f"[Tool Result for {tool_name}]:\n{tool_result}\n\nNow summarize these results for the user in a helpful, conversational way. Highlight key findings and suggest logical next steps.",
                })

                # Get LLM to summarize the tool results
                summary = self._call_llm()

                self.conversation_history.append({
                    "role": "assistant",
                    "content": summary,
                })

                return summary
            else:
                error_msg = f"Unknown tool: {tool_name}. Available tools: {TOOL_NAMES}"
                self.conversation_history.append({
                    "role": "assistant",
                    "content": error_msg,
                })
                return error_msg
        else:
            # No tool call — natural language response
            self.conversation_history.append({
                "role": "assistant",
                "content": response,
            })
            return response

    def _call_llm(self) -> str:
        """Call the HuggingFace Llama model."""
        try:
            response = self.client.chat_completion(
                messages=self.conversation_history,
                max_tokens=1024,
                temperature=0.7,
                top_p=0.9,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"⚠️ LLM Error: {str(e)}"

    def _parse_tool_call(self, response: str) -> dict | None:
        """Extract tool call JSON from LLM response."""
        # Try to find JSON block in response
        # Pattern 1: ```json ... ```
        json_match = re.search(r'```json\s*(\{.*?\})\s*```', response, re.DOTALL)
        if json_match:
            try:
                parsed = json.loads(json_match.group(1))
                if "tool" in parsed and parsed["tool"] in TOOL_NAMES:
                    return parsed
            except json.JSONDecodeError:
                pass

        # Pattern 2: Raw JSON in response
        json_match = re.search(r'(\{"tool":\s*"[^"]+?".*?\})', response, re.DOTALL)
        if json_match:
            try:
                parsed = json.loads(json_match.group(1))
                if "tool" in parsed and parsed["tool"] in TOOL_NAMES:
                    return parsed
            except json.JSONDecodeError:
                pass

        # Pattern 3: Look for tool name mentions as fallback
        # (only if the response clearly indicates a tool should be called)
        return None

    def _update_context(self, tool_name: str, result: str):
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