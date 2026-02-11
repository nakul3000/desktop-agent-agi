# app.py — Multiturn Conversational Job Search Agent
# Integrated with real Resume Tailor Agent

import os
import json
import re
import uuid
from datetime import datetime
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

import memory

# 🔹 Import your real resume tailoring system
from agents.resume_tailor_agent import TailorRequest, tailor_resume

load_dotenv()

# ------------------------------------------------------------------ #
# MEMORY INIT
# ------------------------------------------------------------------ #

_DB_INITIALIZED = False


def _ensure_db_initialized():
    global _DB_INITIALIZED
    if _DB_INITIALIZED:
        return
    memory.init_db()
    _DB_INITIALIZED = True


# ------------------------------------------------------------------ #
# TOOL DEFINITIONS
# ------------------------------------------------------------------ #

TOOLS = [
    {
        "name": "resume_tailor",
        "description": "Tailor the user's resume for a specific job posting.",
        "parameters": {
            "role": "Target job role",
            "company": "Target company",
            "job_description": "Full job description text",
        },
    },
]

TOOL_NAMES = [t["name"] for t in TOOLS]

# ------------------------------------------------------------------ #
# SYSTEM PROMPT
# ------------------------------------------------------------------ #

SYSTEM_PROMPT = f"""
You are JobAgent AI, an intelligent job search assistant.
Today's date is {datetime.now().strftime("%B %d, %Y")}.

You have access to this tool:

1. resume_tailor – Use when the user wants to tailor or fine-tune their resume.

IMPORTANT TOOL RULES:
- When calling a tool, respond ONLY with valid JSON in this format:

{{
  "tool": "resume_tailor",
  "parameters": {{
    "role": "Role name",
    "company": "Company name",
    "job_description": "Full job description"
  }},
  "reasoning": "Why you're calling this tool"
}}

If required parameters are missing, ask the user first.
Never hallucinate job descriptions.
"""


# ------------------------------------------------------------------ #
# REAL TOOL IMPLEMENTATION
# ------------------------------------------------------------------ #

def execute_resume_tailor(params: dict) -> str:
    """Executes your real resume tailoring agent."""

    role = params.get("role")
    company = params.get("company")
    jd = params.get("job_description")

    if not jd:
        return json.dumps({
            "status": "error",
            "message": "Missing job_description. Please provide full job description text."
        }, indent=2)

    resume_path = os.getenv("DEFAULT_RESUME_PATH", "samples/sample_data_analyst_resume.pdf")

    try:
        result = tailor_resume(
            TailorRequest(
                resume_path=resume_path,
                job_description=jd,
                job_title=role,
                company=company,
                output_pdf_path="outputs/tailored_resume_ats.pdf"
            )
        )

        return json.dumps({
            "status": "success",
            "summary": result.get("summary"),
            "output_pdf_path": result.get("data", {}).get("output_pdf_path"),
            "preview": result.get("data", {}).get("tailored_resume_text", "")[:1200]
        }, indent=2)

    except Exception as e:
        return json.dumps({
            "status": "error",
            "message": str(e)
        }, indent=2)


TOOL_EXECUTORS = {
    "resume_tailor": execute_resume_tailor,
}


# ------------------------------------------------------------------ #
# AGENT CORE
# ------------------------------------------------------------------ #

class JobAgent:

    def __init__(self):
        self.hf_token = os.getenv("HF_TOKEN")
        if not self.hf_token:
            raise ValueError("HF_TOKEN not found in .env")

        self.client = InferenceClient(
            model="meta-llama/Meta-Llama-3-70B-Instruct",
            token=self.hf_token,
        )

        self.conversation_history = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]

        self.user_id = "anonymous"
        self.session_id = str(uuid.uuid4())

        _ensure_db_initialized()
        memory.register_user(self.user_id)
        memory.start_session(user_id=self.user_id, session_id=self.session_id)

    # ------------------------------------------------------------------ #

    def chat(self, user_message: str) -> str:

        self.conversation_history.append({
            "role": "user",
            "content": user_message
        })

        memory.store_turn(self.session_id, role="user", text=user_message, user_id=self.user_id)

        response = self._call_llm()
        tool_call = self._parse_tool_call(response)

        if tool_call:
            tool_name = tool_call["tool"]
            params = tool_call.get("parameters", {})

            if tool_name in TOOL_EXECUTORS:
                tool_result = TOOL_EXECUTORS[tool_name](params)

                memory.store_turn(
                    self.session_id,
                    role="tool",
                    text=str(tool_result),
                    user_id=self.user_id,
                    tool_name=tool_name
                )

                # Ask model to summarize tool result
                self.conversation_history.append({
                    "role": "assistant",
                    "content": f"[Tool Result]\n{tool_result}\n\nSummarize this for the user."
                })

                summary = self._call_llm()

                self.conversation_history.append({
                    "role": "assistant",
                    "content": summary
                })

                memory.store_turn(
                    self.session_id,
                    role="assistant",
                    text=summary,
                    user_id=self.user_id
                )

                return summary

        # Normal response
        self.conversation_history.append({
            "role": "assistant",
            "content": response
        })

        memory.store_turn(self.session_id, role="assistant", text=response, user_id=self.user_id)

        return response

    # ------------------------------------------------------------------ #

    def _call_llm(self) -> str:
        try:
            response = self.client.chat_completion(
                messages=self.conversation_history,
                max_tokens=1000,
                temperature=0.7,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"LLM Error: {str(e)}"

    # ------------------------------------------------------------------ #

    def _parse_tool_call(self, response: str) -> dict | None:
        json_match = re.search(r'\{.*"tool".*\}', response, re.DOTALL)
        if json_match:
            try:
                parsed = json.loads(json_match.group(0))
                if parsed.get("tool") in TOOL_NAMES:
                    return parsed
            except Exception:
                pass
        return None


# ------------------------------------------------------------------ #
# CLI LOOP
# ------------------------------------------------------------------ #

def main():
    print("=" * 60)
    print("🤖 JobAgent AI — Resume Tailoring Enabled")
    print("=" * 60)
    print("\nType 'quit' to exit.\n")

    agent = JobAgent()

    while True:
        user_input = input("You: ").strip()

        if user_input.lower() == "quit":
            print("Goodbye!")
            break

        print("\nAgent: ", end="")
        response = agent.chat(user_input)
        print(response)
        print()


if __name__ == "__main__":
    main()
