# Linkup API client for agentic search
import os
from datetime import datetime
from dotenv import load_dotenv
from linkup import LinkupClient

load_dotenv()


class LinkupJobSearch:
    def __init__(self):
        api_key = os.getenv("LINKUP_API_KEY")
        if not api_key:
            raise ValueError("LINKUP_API_KEY not found in .env")
        self.client = LinkupClient(api_key=api_key)

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

    def get_company_profile(self, company: str) -> dict:
        """Research company background, funding, culture, tech stack."""
        today = datetime.now().strftime("%B %d, %Y")

        query = f"""You are a company research analyst. Research {company} thoroughly as of {today}.

Extract the following:
1) Company Overview: What does {company} do? Industry, size, headquarters.
2) Recent News: Any major announcements, product launches, acquisitions in the last 30 days.
3) Financial Health: Latest revenue, funding rounds, stock performance, growth trajectory.
4) Tech Stack & AI Initiatives: What technologies does {company} use? Any AI/ML initiatives?
5) Engineering Culture: How is the engineering org structured? Key engineering leaders.
6) Growth Areas: Where is {company} investing and hiring the most?

Focus on authoritative sources: official blog, press releases, SEC filings, TechCrunch, Bloomberg."""

        print(f"🏢 Researching company: {company}")
        response = self.client.search(
            query=query,
            depth="deep",
            output_type="searchResults",
            include_images=False,
        )
        return response

    def get_company_sentiment(self, company: str) -> dict:
        """Get employee reviews and sentiment analysis."""
        query = f"""You are an employee sentiment analyst. Analyze current employee sentiment at {company}.

Research:
1) Glassdoor reviews for {company} engineering and ML teams — overall rating, pros, cons.
2) Blind app discussions about {company} work culture and compensation.
3) Recent layoffs, reorgs, or morale issues at {company}.
4) Work-life balance reputation in engineering roles.
5) Interview process difficulty and candidate experience.
6) Compensation competitiveness — how does {company} pay vs FAANG/industry?

Summarize: overall sentiment score (1-10), top 3 pros, top 3 cons, red flags if any."""

        print(f"💬 Analyzing sentiment: {company}")
        response = self.client.search(
            query=query,
            depth="standard",
            output_type="searchResults",
            include_images=False,
        )
        return response

    def find_recruiters(self, company: str, role: str) -> dict:
        """Find recruiters and hiring managers."""
        query = f"""You are a networking specialist. Find recruiters and hiring managers at {company} who hire for {role} positions.

Search for:
1) {company} technical recruiters on LinkedIn who focus on ML/AI hiring.
2) {company} engineering hiring managers for {role} teams.
3) Their LinkedIn profile URLs, names, and titles.
4) Any recent posts they've made about open roles or hiring.
5) {company} talent acquisition team members in the US.

Prioritize people who have recently posted about hiring or open positions."""

        print(f"👤 Finding recruiters: {company} - {role}")
        response = self.client.search(
            query=query,
            depth="standard",
            output_type="searchResults",
            include_images=False,
        )
        return response

    def full_research(self, role: str, company: str, location: str = "United States") -> dict:
        """Run the full research pipeline for a job query."""
        print(f"\n{'='*60}")
        print(f"🚀 Full Research: {role} at {company}")
        print(f"📍 Location: {location}")
        print(f"📅 Date: {datetime.now().strftime('%B %d, %Y')}")
        print(f"{'='*60}\n")

        results = {
            "jobs": self.search_jobs(role, company, location),
            "company_profile": self.get_company_profile(company),
            "sentiment": self.get_company_sentiment(company),
            "recruiters": self.find_recruiters(company, role),
        }

        print(f"\n{'='*60}")
        print("✅ Research Complete!")
        print(f"{'='*60}")
        for key, value in results.items():
            if hasattr(value, "results"):
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