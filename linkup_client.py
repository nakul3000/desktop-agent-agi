# Linkup API client for agentic search
import os
from datetime import datetime
from dotenv import load_dotenv
from linkup import LinkupClient

from company_research_agent import CompanyResearchAgent, JobPostingIntake

load_dotenv()


class LinkupJobSearch:
    def __init__(self, session_id: str | None = None, user_id: str | None = None):
        api_key = os.getenv("LINKUP_API_KEY")
        if not api_key:
            raise ValueError("LINKUP_API_KEY not found in .env")
        self.client = LinkupClient(api_key=api_key)
        self.company_research_agent = CompanyResearchAgent(self.client)

    def build_job_intake(self, selected_jd_payload: dict) -> JobPostingIntake:
        """
        Normalize the "user selected one JD" payload into a stable intake object.

        Expected shape (example):
          { "answer": "...", "sources": [ { "url": "...", "snippet": "...", ... } ] }
        """
        return JobPostingIntake.from_selected_jd_payload(selected_jd_payload)

    def research_from_selected_jd(self, selected_jd_payload: dict) -> dict:
        """
        Convenience method: build the intake and run company research using it as context.
        Missing fields remain literal "NA" (no re-search / enrichment).
        """
        intake = self.build_job_intake(selected_jd_payload)

        company = intake.company_name if intake.company_name != "NA" else "NA"
        role = None if intake.role_title == "NA" else intake.role_title
        job_url = None if intake.job_url == "NA" else intake.job_url
        job_description = None if intake.answer == "NA" else intake.answer

        return self.company_research_agent.research_company(
            company=company,
            role=role,
            job_url=job_url,
            job_description=job_description,
            job_intake=intake,
        )

    def search_jobs(self, role: str, company: str = None, location: str = "United States") -> dict:
        """
        Search for job openings with detailed extraction prompt.
        Returns individual job links, titles, locations, and descriptions.
        """
        today = datetime.now().strftime("%B %d, %Y")
        company_name = company or "top tech companies"

        # Build search variant terms
        role_variants = [role]
        role_lower = role.lower()
        if "machine learning" in role_lower or "ml" in role_lower:
            role_variants.extend(["ML engineer", "data scientist machine learning", "AI researcher"])
        elif "software" in role_lower or "swe" in role_lower:
            role_variants.extend(["software developer", "backend engineer", "full stack engineer"])
        elif "data" in role_lower:
            role_variants.extend(["data analyst", "data engineer", "analytics engineer"])

        search_terms = ", ".join([f"'{company_name} {v} jobs {location}'" for v in role_variants])

        query = f"""You are a job search specialist. Your objective is to find all current {role} job openings at {company_name} in {location} that are posted today or very recently.

1) Search for {company_name} {role} jobs posted today in {location} using terms like: {search_terms}
2) Focus on official {company_name} career pages and major job boards where {company_name} posts positions.
3) For each job opening found, extract:
   - Job title
   - Location (city/state)
   - Job posting URL/link
   - Posting date (verify it's recent)
   - Brief job description or key requirements
   - Salary range (if listed)
4) Verify the positions are:
   a) Actually at {company_name} (not third-party recruiters)
   b) {role} related
   c) Located in {location}
   d) Posted today ({today}) or very recently (within last 7 days)

Return all qualifying job links and details. Prioritize official {company_name} career pages over third-party job boards."""

        print(f"🔍 Searching: {role} at {company_name} in {location}")
        print(f"📅 Date filter: {today}")

        response = self.client.search(
            query=query,
            depth="deep",
            output_type="searchResults",
            include_images=False,
        )
        return response

    def get_company_profile(self, company: str, query: str | None = None) -> dict:
        """Research company background, funding, culture, tech stack."""
        query = query or f"{company} company overview funding tech stack culture engineering team 2025"

        print(f"🏢 Researching company: {company}")
        response = self.client.search(
            query=query,
            depth="deep",
            output_type="searchResults",
            include_images=False,
        )
        return self.company_research_agent.research_profile(company)

    def get_company_sentiment(self, company: str, query: str | None = None) -> dict:
        """Get employee reviews and sentiment analysis."""
        query = query or f"{company} employee reviews glassdoor engineering culture work life balance"

        print(f"💬 Analyzing sentiment: {company}")
        response = self.client.search(
            query=query,
            depth="standard",
            output_type="searchResults",
            include_images=False,
        )
        # Return sentiment analysis report from the dedicated agent
        return self.company_research_agent.research_sentiment(company)

    def find_recruiters(self, company: str, role: str, query: str | None = None) -> dict:
        """Find recruiters and hiring managers."""
        query = query or f"{company} recruiter hiring manager {role} LinkedIn"

        print(f"👤 Finding recruiters: {company} - {role}")
        response = self.client.search(
            query=query,
            depth="standard",
            output_type="searchResults",
            include_images=False,
        )
        return response

    def full_research(self, role: str, company: str, location: str = None, user_query: str | None = None) -> dict:
        """Run the full research pipeline for a job query."""
        print(f"\n{'='*60}")
        print(f"🚀 Full Research: {role} at {company}")
        print(f"📍 Location: {location}")
        print(f"📅 Date: {datetime.now().strftime('%B %d, %Y')}")
        print(f"{'='*60}\n")

        query = user_query or self._compose_job_query(role=role, company=company, location=location)

        results = {
            "jobs": self.search_jobs(role, company, location),
            "company_profile": self.get_company_profile(company, query=query),
            "sentiment": self.get_company_sentiment(company, query=query),
            "recruiters": self.find_recruiters(company, role, query=query),
        }

        print(f"\n{'='*60}")
        print("✅ Research Complete!")
        print(f"{'='*80}")
        for key, value in results.items():
            if hasattr(value, 'results'):
                print(f"  📄 {key}: {len(value.results)} results found")
            else:
                print(f"  📄 {key}: {type(value).__name__}")

        return results


# ----- Quick test -----
if __name__ == "__main__":
    searcher = LinkupJobSearch()

    # Test: Job search with detailed extraction
    print("\n" + "=" * 60)
    print("TEST: Detailed Job Search")
    print("=" * 60)

    jobs = searcher.search_jobs(
        role="Machine Learning Engineer",
        company="Amazon",
        location="United States",
    )

    print(f"\n📋 Response type: {type(jobs)}")
    print(f"\n{'='*60}")
    print("RESULTS:")
    print(f"{'='*60}")

    # Handle different response formats
    if hasattr(jobs, "results"):
        for i, result in enumerate(jobs.results, 1):
            print(f"\n--- Result {i} ---")
            print(f"  Title:   {getattr(result, 'name', 'N/A')}")
            print(f"  URL:     {getattr(result, 'url', 'N/A')}")
            print(f"  Content: {getattr(result, 'content', 'N/A')[:300]}")
            print()
    else:
        print(jobs)

    # Uncomment to run full pipeline (4 API calls)
    # results = searcher.full_research(
    #     role="Machine Learning Engineer",
    #     company="Amazon",
    #     location="United States",
    # )
