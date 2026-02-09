# Linkup API client for agentic search

import os
from dotenv import load_dotenv
from linkup import LinkupClient

from company_research_agent import CompanyResearchAgent, JobPostingIntake

load_dotenv()


class ResultFormatter:
    """Formats Linkup search results for readable output."""
    
    @staticmethod
    def format_results(results, max_results: int = 10):
        """Format search results in a clean, readable way."""
        if not hasattr(results, 'results') or not results.results:
            print("❌ No results found")
            return
        
        print(f"\n📊 Found {len(results.results)} results (showing top {min(len(results.results), max_results)})\n")
        
        for idx, result in enumerate(results.results[:max_results], 1):
            print(f"\n{'─' * 80}")
            print(f"📌 Result #{idx}")
            print(f"{'─' * 80}")
            print(f"Title: {result.name}")
            print(f"URL: {result.url}")
            if hasattr(result, 'content') and result.content:
                content_preview = result.content[:300].strip()
                if len(result.content) > 300:
                    content_preview += "..."
                print(f"\nPreview: {content_preview}")
            print()


class LinkupJobSearch:
    def __init__(self):
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

    def search_jobs(self, role: str, company: str = None, location: str = None) -> dict:
        """Search for job openings using Linkup."""
        query_parts = [role, "job openings", "2025"]
        if company:
            query_parts.insert(0, company)
        if location:
            query_parts.append(location)
        query = " ".join(query_parts)

        print(f"🔍 Searching jobs: {query}")
        response = self.client.search(
            query=query,
            depth="deep",
            output_type="searchResults",
            include_images=False,
        )
        return response

    def get_company_profile(self, company: str) -> dict:
        """Research company background, funding, culture, tech stack."""
        print(f"🏢 Researching company: {company}")
        return self.company_research_agent.research_profile(company)

    def get_company_sentiment(self, company: str) -> dict:
        """Get employee reviews and sentiment analysis."""
        print(f"💬 Analyzing sentiment: {company}")
        return self.company_research_agent.research_sentiment(company)

    def find_recruiters(self, company: str, role: str) -> dict:
        """Find recruiters and hiring managers."""
        query = f"{company} recruiter hiring manager {role} LinkedIn"

        print(f"👤 Finding recruiters: {company} - {role}")
        response = self.client.search(
            query=query,
            depth="standard",
            output_type="searchResults",
            include_images=False,
        )
        return response

    def full_research(self, role: str, company: str, location: str = None) -> dict:
        """Run the full research pipeline for a job query."""
        print(f"\n{'='*80}")
        print(f"🚀 Full Research Pipeline: {role} at {company}")
        print(f"{'='*80}\n")

        results = {
            "jobs": self.search_jobs(role, company, location),
            "company_profile": self.get_company_profile(company),
            "sentiment": self.get_company_sentiment(company),
            "recruiters": self.find_recruiters(company, role),
        }

        # Print summary
        print(f"\n{'='*80}")
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
    formatter = ResultFormatter()

    # Test 1: Simple job search
    print("\n" + "=" * 80)
    print("TEST 1: Simple Job Search")
    print("=" * 80)
    jobs = searcher.search_jobs("Machine Learning Engineer", company="Amazon", location="USA")
    formatter.format_results(jobs, max_results=5)

    # Test 2: Full pipeline
    # Uncomment below to run full research (uses 4 API calls)
    # print("\n\nTEST 2: Full Research Pipeline")
    # results = searcher.full_research(
    #     role="Machine Learning Engineer",
    #     company="Adobe",
    #     location="USA"
    # )
    # for key, val in results.items():
    #     print(f"\n--- {key} ---")
    #     formatter.format_results(val, max_results=3)
