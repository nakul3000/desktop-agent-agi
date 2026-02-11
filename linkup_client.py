"""
linkup_client.py

Thin wrapper around LinkUp search plus helper normalization utilities.
"""

import os
import re
import sys
import json
import traceback
import urllib.parse
import urllib.request
from datetime import datetime
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

_SDK_IMPORT_ERROR: Exception | None = None
try:
    # SDK variant used in some Linkup versions
    from linkup import LinkupClient as SDKLinkupClient
except Exception as e:
    _SDK_IMPORT_ERROR = e
    SDKLinkupClient = None

from company_research_agent import CompanyResearchAgent, JobPostingIntake

load_dotenv()


def _safe_getattr(obj: Any, name: str, default: Any = None) -> Any:
    try:
        return getattr(obj, name)
    except Exception:
        return default


def _canonicalize_url(url: str) -> str:
    raw = (url or "").strip()
    if not raw:
        return ""
    try:
        parsed = urllib.parse.urlparse(raw)
        scheme = parsed.scheme.lower() or "https"
        netloc = parsed.netloc.lower()
        path = (parsed.path or "/").rstrip("/") or "/"
        return urllib.parse.urlunparse((scheme, netloc, path, "", "", ""))
    except Exception:
        return raw


def _extract_visible_text_from_html(html: str) -> str:
    if not html:
        return ""
    cleaned = re.sub(r"(?is)<(script|style|noscript|svg|iframe).*?>.*?</\1>", " ", html)
    cleaned = re.sub(r"(?is)<br\s*/?>", "\n", cleaned)
    cleaned = re.sub(r"(?is)</(p|div|li|h1|h2|h3|h4|h5|h6|section|article|tr)>", "\n", cleaned)
    cleaned = re.sub(r"(?is)<[^>]+>", " ", cleaned)
    cleaned = re.sub(r"[ \t\r\f\v]+", " ", cleaned)
    cleaned = re.sub(r"\n\s+\n", "\n\n", cleaned)
    return cleaned.strip()


def _extract_text_from_embedded_json(html: str) -> str:
    """
    Extract job-relevant text from embedded JSON payloads (JSON-LD, Next.js blobs, etc.).
    This helps with pages that hydrate content from script tags.
    """
    if not html:
        return ""

    script_chunks: List[str] = []
    script_re = re.compile(
        r"(?is)<script[^>]*?(?:type\s*=\s*[\"']application/(?:ld\+)?json[\"'][^>]*)?>(.*?)</script>"
    )
    for m in script_re.finditer(html):
        chunk = (m.group(1) or "").strip()
        if not chunk:
            continue
        # Skip obvious JavaScript code blobs; keep JSON-like chunks.
        if not (chunk.startswith("{") or chunk.startswith("[")):
            continue
        script_chunks.append(chunk)

    if not script_chunks:
        return ""

    key_weights = {
        "description": 8,
        "jobdescription": 8,
        "responsibilities": 7,
        "requirements": 7,
        "qualifications": 7,
        "preferredqualifications": 6,
        "about": 5,
        "summary": 5,
        "content": 4,
        "body": 4,
        "text": 3,
    }
    hits: List[tuple[int, str]] = []

    def _walk(node: Any, parent_key: str = "") -> None:
        if isinstance(node, dict):
            for k, v in node.items():
                _walk(v, str(k))
            return
        if isinstance(node, list):
            for item in node:
                _walk(item, parent_key)
            return
        if not isinstance(node, str):
            return

        text = _extract_visible_text_from_html(node).strip()
        if len(text) < 60:
            return
        key = (parent_key or "").replace("_", "").replace("-", "").lower()
        weight = key_weights.get(key, 1)
        hits.append((weight, text))

    for chunk in script_chunks:
        try:
            payload = json.loads(chunk)
            _walk(payload)
        except Exception:
            continue

    if not hits:
        return ""

    # Keep high-signal chunks first, then longer text, while de-duping.
    hits.sort(key=lambda x: (x[0], len(x[1])), reverse=True)
    out: List[str] = []
    seen: set[str] = set()
    for _w, text in hits:
        norm = " ".join(text.split())
        if norm in seen:
            continue
        seen.add(norm)
        out.append(text)
        if len("\n\n".join(out)) >= 16000:
            break
    return "\n\n".join(out).strip()


def _preview_text(value: Any, *, limit: int = 2200) -> str:
    text = (str(value) if value is not None else "").strip()
    if not text:
        return ""
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n...[truncated {len(text) - limit} chars]"


def _probe_linkup_search_error(client: Any, payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Best-effort diagnostic probe for SDK failures where the original response body
    is unavailable (e.g. JSONDecodeError while parsing LinkUp error payload).
    """
    out: Dict[str, Any] = {
        "ok": False,
        "url": "",
        "status_code": None,
        "content_type": "",
        "body_preview": "",
        "error": "",
    }

    try:
        import httpx
    except Exception as e:  # pragma: no cover - optional dependency path
        out["error"] = f"httpx import failed: {type(e).__name__}: {e}"
        return out

    try:
        headers_fn = _safe_getattr(client, "_headers", None)
        headers = headers_fn() if callable(headers_fn) else {}
        if not isinstance(headers, dict):
            headers = {}
        headers["Content-Type"] = "application/json"

        base_url_raw = _safe_getattr(client, "_base_url", "")
        base_url = str(base_url_raw or "").rstrip("/")
        endpoint = f"{base_url}/search" if base_url else ""
        if not endpoint:
            out["error"] = "Client base URL is unavailable for diagnostic probe."
            return out

        redacted_headers = {
            k: ("<redacted>" if k.lower() == "authorization" else v) for k, v in headers.items()
        }

        response = httpx.post(endpoint, headers=headers, json=payload, timeout=25.0)
        body_text = (response.text or "").strip()
        out.update(
            {
                "ok": True,
                "url": endpoint,
                "status_code": response.status_code,
                "content_type": (response.headers.get("content-type") or "").strip(),
                "request_headers": redacted_headers,
                "body_preview": _preview_text(body_text, limit=2200) if body_text else "<empty body>",
            }
        )
        return out
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"
        return out


class _HTTPLinkupClient:
    """
    Fallback client when the `linkup` SDK isn't importable.

    We intentionally fail fast with a clear message rather than silently misbehaving,
    because LinkUp HTTP endpoints/auth can vary by account/version.
    """

    def __init__(self, api_key: str):
        self.api_key = api_key

    def search(self, *args: Any, **kwargs: Any) -> Any:
        import_hint = (
            f"Original import error: {type(_SDK_IMPORT_ERROR).__name__}: {_SDK_IMPORT_ERROR}"
            if _SDK_IMPORT_ERROR is not None
            else "Original import error was not captured."
        )
        raise RuntimeError(
            "LinkUp SDK import failed (`from linkup import LinkupClient`). "
            f"Install/repair dependency `linkup-sdk` so `LinkupClient.search(...)` is available. "
            f"Python executable: {sys.executable}. {import_hint}"
        )


_LINE_KV_RE = re.compile(r"(?im)^\s*(company|employer|location|title|role)\s*:\s*(.+?)\s*$")


def normalize_search_results_to_jobs(
    response: Any,
    *,
    role: Optional[str] = None,
    company: Optional[str] = None,
    location: Optional[str] = None,
    limit: int = 12,
) -> List[Dict[str, Any]]:
    """
    Convert LinkUp `searchResults` response into simple "job cards" for UI/CLI.

    Since `searchResults` is not guaranteed to be structured as jobs, we do:
    - one card per result
    - best-effort extraction for company/location from result.content
    - dedupe by URL
    """

    results = None
    if isinstance(response, dict):
        results = response.get("results")
    if results is None:
        results = _safe_getattr(response, "results", None)

    if not isinstance(results, list):
        return []

    def extract_kv(content: str) -> Dict[str, str]:
        out: Dict[str, str] = {}
        if not content:
            return out
        for m in _LINE_KV_RE.finditer(content):
            k = (m.group(1) or "").strip().lower()
            v = (m.group(2) or "").strip()
            if k and v and k not in out:
                out[k] = v
        return out

    jobs: List[Dict[str, Any]] = []
    seen_urls: set[str] = set()

    for i, r in enumerate(results, start=1):
        if len(jobs) >= limit:
            break

        name = None
        url = None
        content = ""

        if isinstance(r, dict):
            name = (r.get("name") or r.get("title") or "").strip() or None
            url = (r.get("url") or "").strip() or None
            content = (r.get("content") or r.get("snippet") or "").strip()
        else:
            name = (_safe_getattr(r, "name", None) or "").strip() or None
            url = (_safe_getattr(r, "url", None) or "").strip() or None
            content = (_safe_getattr(r, "content", None) or "").strip() or ""

        if url and url in seen_urls:
            continue
        if url:
            seen_urls.add(url)

        kv = extract_kv(content)
        inferred_company = company or kv.get("company") or kv.get("employer")
        inferred_location = location or kv.get("location")
        inferred_title = kv.get("title") or kv.get("role") or name or (role or "NA")

        # For listing/search UX, keep only a short preview.
        # Full JD extraction should happen only after explicit user selection.
        jd_text = content.strip()
        if jd_text:
            jd_text = jd_text[:600]
            if len(content) > 600:
                jd_text = jd_text.rstrip() + "…"

        jobs.append(
            {
                "job_id": str(i),
                "title": inferred_title or "NA",
                "company": inferred_company or "NA",
                "location": inferred_location or "NA",
                "url": url or "NA",
                "snippet": (content[:400] + ("…" if len(content) > 400 else "")) if content else "NA",
                "jd_text": jd_text or "NA",
            }
        )

    print(f"\n{'=' * 80}")
    print("🧭 DEBUG LOG: normalize_search_results_to_jobs output")
    print(f"raw_results_count={len(results)} deduped_jobs_count={len(jobs)} limit={limit}")
    for idx, j in enumerate(jobs, start=1):
        jd = (j.get("jd_text") or "").strip()
        print(
            f"[job {idx}] title={j.get('title')} | company={j.get('company')} | location={j.get('location')} | "
            f"url={j.get('url')} | jd_len={len(jd)}"
        )
        if jd and jd != "NA":
            print(f"[job {idx}] jd_preview:\n{_preview_text(jd, limit=1200)}")
    print(f"{'=' * 80}\n")

    return jobs


class LinkupJobSearch:
    def __init__(self, session_id: str | None = None, user_id: str | None = None, api_key: str | None = None):
        api_key = api_key or os.getenv("LINKUP_API_KEY")
        if not api_key:
            raise ValueError("LINKUP_API_KEY not found in .env")
        self.api_key = api_key
        self.client = SDKLinkupClient(api_key=api_key) if SDKLinkupClient else _HTTPLinkupClient(api_key=api_key)
        self.company_research_agent = CompanyResearchAgent(self.client)

    def build_job_intake(self, selected_jd_payload: dict) -> JobPostingIntake:
        """
        Normalize the "user selected one JD" payload into a stable intake object.

        Expected shape (example):
          { "answer": "...", "sources": [ { "url": "...", "snippet": "...", ... } ] }
        """
        return JobPostingIntake.from_selected_jd_payload(selected_jd_payload)

    def research_from_selected_jd(
        self,
        selected_jd_payload: dict,
        *,
        preferred_company: str | None = None,
    ) -> dict:
        """
        Convenience method: build the intake and run company research using it as context.
        Missing fields remain literal "NA" (no re-search / enrichment).
        """
        intake = self.build_job_intake(selected_jd_payload)

        preferred_company_clean = (preferred_company or "").strip()
        company = intake.company_name if intake.company_name != "NA" else (preferred_company_clean or "NA")
        role = None if intake.role_title == "NA" else intake.role_title
        job_url = None if intake.job_url == "NA" else intake.job_url
        job_description = None if intake.answer == "NA" else intake.answer

        print(f"\n{'=' * 80}")
        print("🧭 DEBUG LOG: research_from_selected_jd input/output")
        print(f"selected_payload_keys={list((selected_jd_payload or {}).keys()) if isinstance(selected_jd_payload, dict) else []}")
        print(f"selected_payload_answer_len={len((selected_jd_payload.get('answer') or '').strip()) if isinstance(selected_jd_payload, dict) else 0}")
        print(f"selected_payload_sources={selected_jd_payload.get('sources') if isinstance(selected_jd_payload, dict) else []}")
        print(
            "normalized_intake="
            + str(
                {
                    "company": company,
                    "role": role or "NA",
                    "job_url": job_url or "NA",
                    "job_description_len": len(job_description or ""),
                    "location": intake.location,
                    "workplace_type": intake.workplace_type,
                    "employment_type": intake.employment_type,
                    "requirements_summary_len": len((intake.requirements_summary or "").strip()),
                    "preferred_summary_len": len((intake.preferred_summary or "").strip()),
                    "compensation_summary_len": len((intake.compensation_summary or "").strip()),
                }
            )
        )
        if job_description:
            print(f"normalized_job_description_preview:\n{_preview_text(job_description, limit=2400)}")
        print(f"{'=' * 80}\n")

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
        print(f"\n{'=' * 80}")
        print("🧭 DEBUG LOG: LinkUp job search request")
        print("api_call=client.search output_type=searchResults depth=deep include_images=False")
        print(f"role_variants={role_variants}")
        print(f"query:\n{query}")
        print(f"{'=' * 80}\n")

        try:
            response = self.client.search(
                query=query,
                depth="deep",
                output_type="searchResults",
                include_images=False,
            )
        except Exception as e:
            tb_text = traceback.format_exc().rstrip()
            print(f"\n{'=' * 80}")
            print("❌ ERROR LOG: LinkUp client search failed")
            print(f"Type: {type(e).__name__}")
            print(f"Message: {e}")
            print("Request context:")
            print(f"  role={role!r}")
            print(f"  company={company_name!r}")
            print(f"  location={location!r}")
            if isinstance(e, json.JSONDecodeError):
                probe_payload = {
                    "query": query,
                    "depth": "deep",
                    "outputType": "searchResults",
                    "includeImages": False,
                }
                probe = _probe_linkup_search_error(self.client, probe_payload)
                print("Diagnostics:")
                if probe.get("ok"):
                    print(f"  probe_url={probe.get('url')}")
                    print(f"  status_code={probe.get('status_code')}")
                    print(f"  content_type={probe.get('content_type')!r}")
                    print(f"  request_headers={probe.get('request_headers')}")
                    print(f"  body_preview:\n{probe.get('body_preview')}")
                else:
                    print(f"  probe_error={probe.get('error')}")
            print("Traceback:")
            print(tb_text)
            print(f"{'=' * 80}\n")
            raise
        results = response.get("results") if isinstance(response, dict) else _safe_getattr(response, "results", None)
        results = results if isinstance(results, list) else []
        print(f"\n{'=' * 80}")
        print("🧭 DEBUG LOG: LinkUp job search response")
        print(f"results_count={len(results)}")
        for idx, result in enumerate(results[:20], start=1):
            if isinstance(result, dict):
                name = (result.get("name") or result.get("title") or "").strip()
                url = (result.get("url") or "").strip()
                content = (result.get("content") or result.get("snippet") or "").strip()
            else:
                name = (_safe_getattr(result, "name", None) or "").strip()
                url = (_safe_getattr(result, "url", None) or "").strip()
                content = (_safe_getattr(result, "content", None) or "").strip()
            print(f"[result {idx}] name={name or 'NA'} | url={url or 'NA'} | content_len={len(content)}")
            if content:
                print(f"[result {idx}] content_preview:\n{_preview_text(content, limit=1200)}")
        print(f"{'=' * 80}\n")
        return response

    def extract_job_description_from_url(
        self,
        job_url: str,
        *,
        role: str | None = None,
        company: str | None = None,
        existing_jd_text: str | None = None,
    ) -> dict:
        """
        JD extraction path for selected jobs.
        This flow intentionally reuses already-extracted JD text from the
        selected job payload and does not re-fetch the URL.

        Returns:
            {
              "status": "success"|"error",
              "jd_text": "...",
              "source_url": "...",
              "source_name": "...",
              "error": "..."  # only on error
            }
        """
        url = (job_url or "").strip()
        if not url or url == "NA":
            return {
                "status": "error",
                "error": "Missing job URL for JD extraction.",
                "jd_text": "",
                "source_url": "",
                "source_name": "",
            }

        canonical_target = _canonicalize_url(url)
        existing_text = (existing_jd_text or "").strip()

        print(f"\n{'=' * 80}")
        print("🧭 DEBUG LOG: JD extraction request")
        print(f"job_url={url}")
        print(f"canonical_target={canonical_target}")
        print(f"role_hint={role or 'NA'} company_hint={company or 'NA'}")
        print(f"existing_jd_len={len(existing_text)}")
        print(f"{'=' * 80}\n")

        if existing_text:
            print(f"\n{'=' * 80}")
            print("🧭 DEBUG LOG: JD extraction reuse")
            print("reason=use_existing_jd_payload")
            print(f"job_url={url}")
            print(f"existing_jd_len={len(existing_text)}")
            print(f"{'=' * 80}\n")
            return {
                "status": "success",
                "jd_text": existing_text[:12000],
                "source_url": canonical_target or url,
                "source_name": "existing_jd_payload",
            }

        return {
            "status": "error",
            "error": (
                "Selected job did not include JD text in payload. "
                "This path is configured to use payload JD only (no URL refetch)."
            ),
            "jd_text": "",
            "source_url": canonical_target or url,
            "source_name": "",
        }

    def get_company_profile(self, company: str, query: str | None = None, *, context: Optional[Dict[str, Any]] = None) -> dict:
        """Research company background, funding, culture, tech stack."""
        query = query or f"{company} company overview funding tech stack culture engineering team 2025"

        print(f"🏢 Researching company: {company}")
        _ = self.client.search(
            query=query,
            depth="deep",
            output_type="searchResults",
            include_images=False,
        )
        return self.company_research_agent.research_profile(company, context=context)

    def get_company_sentiment(self, company: str, query: str | None = None, *, context: Optional[Dict[str, Any]] = None) -> dict:
        """Get employee reviews and sentiment analysis."""
        query = query or f"{company} employee reviews glassdoor engineering culture work life balance"

        print(f"💬 Analyzing sentiment: {company}")
        _ = self.client.search(
            query=query,
            depth="standard",
            output_type="searchResults",
            include_images=False,
        )
        # Return sentiment analysis report from the dedicated agent
        return self.company_research_agent.research_sentiment(company, context=context)

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


# Compatibility wrapper for app.py imports.
class LinkupClient(LinkupJobSearch):
    pass


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
