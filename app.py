# Entry point for the personalized agent app
from __future__ import annotations

from dotenv import load_dotenv
load_dotenv()

import os

from agent_core import AgentCore
from linkup_client import LinkupClient
from memory import Memory
from email_handler import EmailHandler
from document_handler import DocumentHandler
from calendar_handler import CalendarHandler

from agents.role_search_agent import recommend_jobs_for_resume
from agents.job_description_agent import JobDescriptionRequest, fetch_sample_job_descriptions
from agents.resume_tailor_agent import TailorRequest, tailor_resume


def _require_env(key: str) -> str:
    v = os.getenv(key)
    if not v:
        raise RuntimeError(f"Missing {key}. Add it to .env or export it in your shell.")
    return v


def run_resume_to_tailored_resume_flow():
    """
    Demo flow:
    1) Recommend jobs from resume
    2) User selects a job
    3) Fetch JD samples via Linkup
    4) User selects JD sample
    5) Generate tailored resume (ATS PDF + optional DOCX)
    """
    resume_path = input("Enter resume path (e.g., samples/sample_data_analyst_resume.pdf): ").strip()
    if not resume_path:
        resume_path = "samples/sample_data_analyst_resume.pdf"

    print("\nSearching jobs based on resume...\n")
    rec = recommend_jobs_for_resume(
        resume_path,
        titles=["Software Engineer", "Data Analyst", "Data Engineer"],
        location="United States",
        remote_ok=True,
        posted_within_days=60,
        max_results=15,
        top_k=3,
    )

    if rec.get("status") != "ok" or not rec["data"]["recommendations"]:
        print("No recommendations found.")
        print(rec.get("summary"))
        return

    jobs = rec["data"]["recommendations"]
    print("Top job recommendations:")
    for i, j in enumerate(jobs, start=1):
        print(f"{i}. {j['title']} — {j['company']} — {j.get('location')}")
        print(f"   URL: {j['url']}")
        print(f"   Match: {j.get('match_score')} | Matched skills: {', '.join(j.get('matched_skills', [])[:6])}")
        print("")

    choice = input(f"Pick a job (1-{len(jobs)}): ").strip()
    idx = int(choice) - 1 if choice.isdigit() else 0
    idx = max(0, min(idx, len(jobs) - 1))
    selected_job = jobs[idx]

    print("\nFetching job description samples from Linkup...\n")
    jd_samples = fetch_sample_job_descriptions(
        JobDescriptionRequest(
            title=selected_job["title"],
            company=selected_job["company"],
            location=selected_job.get("location"),
            url=selected_job["url"],
            max_variants=3,
        )
    )

    samples = jd_samples.get("data", {}).get("samples", [])
    if not samples:
        print("No job description samples found.")
        print(jd_samples.get("summary"))
        return

    print(jd_samples["summary"])
    for i, s in enumerate(samples, start=1):
        src = s.get("source") or "unknown"
        text_preview = s.get("job_description", "")[:900]
        print(f"\n--- Sample {i} (source: {src}) ---\n{text_preview}\n")

    choice2 = input(f"Pick a job description sample (1-{len(samples)}): ").strip()
    idx2 = int(choice2) - 1 if choice2.isdigit() else 0
    idx2 = max(0, min(idx2, len(samples) - 1))
    chosen_jd = samples[idx2]["job_description"]

    # Outputs
    out_pdf = input("Output ATS PDF path (enter to skip, e.g., outputs/tailored_resume_ats.pdf): ").strip() or None
    out_docx = input("Output DOCX path (enter to skip, e.g., outputs/tailored_resume.docx): ").strip() or None

    print("\nGenerating tailored resume...\n")
    tailored = tailor_resume(
        TailorRequest(
            resume_path=resume_path,
            job_description=chosen_jd,
            job_title=selected_job["title"],
            company=selected_job["company"],
            location=selected_job.get("location"),
            output_pdf_path=out_pdf,
            output_docx_path=out_docx,
        )
    )

    print(tailored["summary"])
    print("\n---- Tailored Resume (preview) ----\n")
    print(tailored["data"]["tailored_resume_text"][:2000])

    if tailored["data"].get("output_pdf_path"):
        print("\nATS PDF saved to:", tailored["data"]["output_pdf_path"])
    if tailored["data"].get("output_docx_path"):
        print("DOCX saved to:", tailored["data"]["output_docx_path"])


def main():
    api_key = _require_env("LINKUP_API_KEY")

    linkup_client = LinkupClient(api_key=api_key)
    memory = Memory()
    email_handler = EmailHandler()
    document_handler = DocumentHandler()
    calendar_handler = CalendarHandler()
    _agent = AgentCore(memory, linkup_client, email_handler, document_handler, calendar_handler)

    print("Agent is running.\n")
    run_resume_to_tailored_resume_flow()


if __name__ == "__main__":
    main()
