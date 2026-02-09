# Linkup API client for agentic search

import os
from dotenv import load_dotenv
from linkup import LinkupClient

import memory

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
    def __init__(self, session_id: str | None = None, user_id: str | None = None):
        api_key = os.getenv("LINKUP_API_KEY")
        if not api_key:
            raise ValueError("LINKUP_API_KEY not found in .env")

        memory.init_db()
        self.user_id = user_id or os.getenv("USER_ID") or "anonymous"
        memory.register_user(self.user_id)
        self.session_id = session_id or memory.start_session(user_id=self.user_id)
        self.client = LinkupClient(api_key=api_key)

    # --- memory helpers ---
    def _log_turn(self, role: str, text: str) -> int:
        return memory.store_turn(self.session_id, role=role, text=text)

    def _log_artifact(self, type: str, content, source_turn_id=None, created_by="LinkupJobSearch"):
        return memory.store_artifact(
            session_id=self.session_id,
            type=type,
            content=content,
            source_turn_id=source_turn_id,
            created_by=created_by,
        )

    def _log_fact(self, kind: str, key: str, value: str, meta=None, confidence: float = 0.8):
        return memory.store_fact(
            session_id=self.session_id,
            kind=kind,
            key=key,
            value=value,
            meta=meta or {},
            confidence=confidence,
        )

    def search_jobs(self, role: str, company: str = None, location: str = None) -> dict:
        """Search for job openings using Linkup."""
        query_parts = [role, "job openings", "2025"]
        if company:
            query_parts.insert(0, company)
        if location:
            query_parts.append(location)
        query = " ".join(query_parts)

        user_turn = self._log_turn("user", f"Search jobs query: {query}")

        print(f"🔍 Searching jobs: {query}")
        response = self.client.search(
            query=query,
            depth="deep",
            output_type="searchResults",
            include_images=False,
        )

        self._log_turn("assistant", f"Linkup returned {len(getattr(response, 'results', []) or [])} results for '{query}'")
        artifact_id = self._log_artifact(
            type="linkup_research",
            content={"query": query, "location": location, "company": company},
            source_turn_id=user_turn,
        )
        if company:
            self._log_fact(
                kind="preference",
                key="target_company",
                value=company,
                meta={"source_artifact_id": artifact_id, "title": f"Target company {company}"},
                confidence=0.9,
            )
        if role:
            self._log_fact(
                kind="preference",
                key="target_role",
                value=role,
                meta={"source_artifact_id": artifact_id, "title": f"Target role {role}"},
                confidence=0.85,
            )
        return response

    def get_company_profile(self, company: str) -> dict:
        """Research company background, funding, culture, tech stack."""
        query = f"{company} company overview funding tech stack culture engineering team 2025"

        self._log_turn("user", f"Research company profile: {company}")
        print(f"🏢 Researching company: {company}")
        response = self.client.search(
            query=query,
            depth="deep",
            output_type="searchResults",
            include_images=False,
        )

        artifact_id = self._log_artifact(
            type="company_research",
            content={"query": query, "company": company},
        )
        self._log_fact(
            kind="company",
            key="company_name",
            value=company,
            meta={"source_artifact_id": artifact_id, "title": f"Company research for {company}"},
            confidence=0.75,
        )
        return response

    def get_company_sentiment(self, company: str) -> dict:
        """Get employee reviews and sentiment analysis."""
        query = f"{company} employee reviews glassdoor engineering culture work life balance"

        self._log_turn("user", f"Analyze sentiment for {company}")
        print(f"💬 Analyzing sentiment: {company}")
        response = self.client.search(
            query=query,
            depth="standard",
            output_type="searchResults",
            include_images=False,
        )

        self._log_artifact(
            type="company_sentiment",
            content={"query": query, "company": company},
        )
        return response

    def find_recruiters(self, company: str, role: str) -> dict:
        """Find recruiters and hiring managers."""
        query = f"{company} recruiter hiring manager {role} LinkedIn"

        self._log_turn("user", f"Find recruiters for {company} - {role}")
        print(f"👤 Finding recruiters: {company} - {role}")
        response = self.client.search(
            query=query,
            depth="standard",
            output_type="searchResults",
            include_images=False,
        )

        self._log_artifact(
            type="recruiter_profile",
            content={"query": query, "company": company, "role": role},
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
        print(f"{'='*60}")
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
