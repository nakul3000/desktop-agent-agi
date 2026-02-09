from __future__ import annotations

import dataclasses
import datetime as dt
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple


@dataclasses.dataclass(frozen=True)
class CompanyResearchDefaults:
    """
    Tunables for cost/latency and output consistency.

    Notes:
    - LinkUp supports `max_results` to cap retrieval context size.
    - We default to `deep` because company research is typically multi-tab work.
    """

    depth_profile: str = "deep"
    depth_sentiment: str = "deep"
    max_results: int = 8

    # If True, run fewer sub-queries (faster/cheaper, less coverage).
    quick_mode: bool = False


def _now_utc_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _safe_getattr(obj: Any, name: str, default: Any = None) -> Any:
    try:
        return getattr(obj, name)
    except Exception:
        return default


def _source_to_dict(source: Any) -> Dict[str, Any]:
    # LinkUpSource fields: name, url, snippet, favicon (per SDK docs)
    return {
        "name": _safe_getattr(source, "name"),
        "url": _safe_getattr(source, "url"),
        "snippet": _safe_getattr(source, "snippet"),
        "favicon": _safe_getattr(source, "favicon"),
    }


_CITATION_RE = re.compile(r"\[(\d{1,4})\]")


def _extract_citation_numbers(text: str) -> List[int]:
    if not text:
        return []
    nums = []
    for m in _CITATION_RE.finditer(text):
        try:
            nums.append(int(m.group(1)))
        except Exception:
            continue
    # stable order, de-duped
    seen = set()
    out: List[int] = []
    for n in nums:
        if n not in seen:
            seen.add(n)
            out.append(n)
    return out


def _replace_citations(text: str, map_local_to_global: Dict[int, int]) -> str:
    if not text or not map_local_to_global:
        return text

    def _repl(m: re.Match) -> str:
        local = int(m.group(1))
        global_id = map_local_to_global.get(local)
        if global_id is None:
            return m.group(0)
        return f"[{global_id}]"

    return _CITATION_RE.sub(_repl, text)


def _dedupe_sources_assign_ids(
    sources_by_query: List[List[Dict[str, Any]]],
) -> Tuple[List[Dict[str, Any]], List[Dict[int, int]]]:
    """
    Dedupe sources by URL across sub-queries and assign stable 1-based IDs.

    Returns:
      - merged_sources: list of sources with added `id`
      - local_to_global_maps: one dict per query mapping local_index -> global_id
    """
    merged: List[Dict[str, Any]] = []
    url_to_id: Dict[str, int] = {}
    local_maps: List[Dict[int, int]] = []

    for sources in sources_by_query:
        local_map: Dict[int, int] = {}
        for i, src in enumerate(sources, start=1):
            url = (src.get("url") or "").strip()
            if not url:
                # Keep a best-effort entry for non-url sources; dedupe by name+snippet.
                key = f"__no_url__::{src.get('name','')}::{src.get('snippet','')[:80]}"
                if key not in url_to_id:
                    url_to_id[key] = len(merged) + 1
                    merged.append({**src, "id": url_to_id[key]})
                local_map[i] = url_to_id[key]
                continue

            if url not in url_to_id:
                url_to_id[url] = len(merged) + 1
                merged.append({**src, "id": url_to_id[url]})
            local_map[i] = url_to_id[url]
        local_maps.append(local_map)

    return merged, local_maps


def _bullet_sections_to_items(markdown: str) -> List[Dict[str, Any]]:
    """
    Very lightweight extraction: pull top-level bullet lines as fact candidates.
    Each item keeps citations extracted from the line.
    """
    if not markdown:
        return []

    items: List[Dict[str, Any]] = []
    for raw_line in markdown.splitlines():
        line = raw_line.strip()
        if not line.startswith(("- ", "* ")):
            continue
        text = line[2:].strip()
        if not text:
            continue
        items.append(
            {
                "text": text,
                "citations": _extract_citation_numbers(text),
            }
        )
    return items


class CompanyResearchAgent:
    """
    A dedicated agent that orchestrates multi-step LinkUp retrieval to produce
    a structured, citation-backed company research report for resume tailoring.
    """

    def __init__(self, linkup_client: Any, *, defaults: Optional[CompanyResearchDefaults] = None):
        self._client = linkup_client
        self._defaults = defaults or CompanyResearchDefaults()

    def research_company(
        self,
        company: str,
        role: Optional[str] = None,
        job_url: Optional[str] = None,
        job_description: Optional[str] = None,
    ) -> Dict[str, Any]:
        context = {
            "company": company,
            "role": role,
            "job_url": job_url,
            "job_description": job_description,
        }
        profile = self.research_profile(company, context=context)
        sentiment = self.research_sentiment(company, context=context)
        return {
            "company": company,
            "generated_at": _now_utc_iso(),
            "profile": profile,
            "sentiment": sentiment,
            "context": context,
        }

    def research_profile(self, company: str, *, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        defaults = self._defaults
        queries: List[Tuple[str, str]] = []

        queries.append(
            (
                "official_identity",
                self._prompt_official_identity(company=company, context=context),
            )
        )
        if not defaults.quick_mode:
            queries.extend(
                [
                    ("funding_financials", self._prompt_funding_financials(company=company, context=context)),
                    ("engineering_tech_stack", self._prompt_engineering_tech_stack(company=company, context=context)),
                    ("hiring_org_leadership", self._prompt_hiring_org_leadership(company=company, context=context)),
                    ("strategy_recent_news", self._prompt_strategy_recent_news(company=company, context=context)),
                ]
            )

        results = self._run_queries(
            queries,
            depth=defaults.depth_profile,
            max_results=defaults.max_results,
        )
        report = self._compose_profile_report(company=company, results=results)
        return report

    def research_sentiment(self, company: str, *, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        defaults = self._defaults
        queries: List[Tuple[str, str]] = []

        queries.append(("reviews_summary", self._prompt_reviews_summary(company=company, context=context)))
        if not defaults.quick_mode:
            queries.extend(
                [
                    ("wlb_oncall", self._prompt_wlb_oncall(company=company, context=context)),
                    ("comp_signals", self._prompt_comp_signals(company=company, context=context)),
                    ("interview_process", self._prompt_interview_process(company=company, context=context)),
                ]
            )

        results = self._run_queries(
            queries,
            depth=defaults.depth_sentiment,
            max_results=defaults.max_results,
        )
        report = self._compose_sentiment_report(company=company, results=results)
        return report

    # -----------------------------
    # LinkUp execution + normalization
    # -----------------------------

    def _run_queries(
        self,
        named_prompts: List[Tuple[str, str]],
        *,
        depth: str,
        max_results: int,
    ) -> Dict[str, Any]:
        """
        Run multiple sourcedAnswer calls; normalize and merge sources with ID remapping.
        """
        queries_run: List[Dict[str, Any]] = []
        answers_local: List[str] = []
        sources_by_query: List[List[Dict[str, Any]]] = []

        raw_responses: Dict[str, Any] = {}

        for name, prompt in named_prompts:
            response = self._client.search(
                query=prompt,
                depth=depth,
                output_type="sourcedAnswer",
                include_inline_citations=True,
                include_images=False,
                max_results=max_results,
            )
            raw_responses[name] = response

            answer = _safe_getattr(response, "answer", "") or ""
            sources_obj = _safe_getattr(response, "sources", []) or []
            sources = [_source_to_dict(s) for s in sources_obj]

            queries_run.append(
                {
                    "name": name,
                    "query": prompt,
                    "depth": depth,
                    "output_type": "sourcedAnswer",
                    "max_results": max_results,
                    "source_count": len(sources),
                }
            )
            answers_local.append(answer)
            sources_by_query.append(sources)

        merged_sources, local_to_global_maps = _dedupe_sources_assign_ids(sources_by_query)

        # Rewrite citations in each answer to global IDs.
        answers_global: Dict[str, str] = {}
        for (name, _prompt), answer, local_map in zip(named_prompts, answers_local, local_to_global_maps):
            answers_global[name] = _replace_citations(answer, local_map)

        return {
            "answers_by_query": answers_global,
            "sources": merged_sources,
            "queries_run": queries_run,
            "raw": raw_responses,
        }

    # -----------------------------
    # Composition
    # -----------------------------

    def _compose_profile_report(self, *, company: str, results: Dict[str, Any]) -> Dict[str, Any]:
        answers = results["answers_by_query"]
        sources = results["sources"]
        queries_run = results["queries_run"]

        report_md_parts: List[str] = []
        report_md_parts.append(f"# {company} — Company Profile")
        report_md_parts.append("")
        report_md_parts.append("## Executive_summary")
        report_md_parts.append(
            "This report aggregates public information relevant to resume tailoring and interview prep. "
            "All key claims should have citations."
        )
        report_md_parts.append("")

        def add_section(title: str, key: str) -> None:
            text = (answers.get(key) or "").strip()
            if not text:
                return
            report_md_parts.append(f"## {title}")
            report_md_parts.append(text)
            report_md_parts.append("")

        add_section("What_the_company_does", "official_identity")
        add_section("Funding_and_financial_context", "funding_financials")
        add_section("Engineering_and_tech_stack", "engineering_tech_stack")
        add_section("Org_and_leadership", "hiring_org_leadership")
        add_section("Recent_news_and_strategy", "strategy_recent_news")

        # Structured signals (lightweight, derived from bullets where possible).
        sections: Dict[str, Any] = {
            "resume_tailoring_signals": _bullet_sections_to_items(
                "\n".join(
                    [
                        answers.get("engineering_tech_stack", ""),
                        answers.get("hiring_org_leadership", ""),
                        answers.get("strategy_recent_news", ""),
                    ]
                )
            ),
            "tech_stack_signals": _bullet_sections_to_items(answers.get("engineering_tech_stack", "")),
            "engineering_priorities": _bullet_sections_to_items(answers.get("hiring_org_leadership", "")),
        }

        open_questions: List[str] = []
        if not sources:
            open_questions.append("No sources returned; consider expanding max_results or refining prompts.")

        return {
            "company": company,
            "generated_at": _now_utc_iso(),
            "depth": self._defaults.depth_profile,
            "output_type": "sourcedAnswer",
            "report_markdown": "\n".join(report_md_parts).strip() + "\n",
            "sections": sections,
            "sources": sources,
            "queries_run": queries_run,
            "confidence": "medium" if sources else "low",
            "open_questions": open_questions,
            "raw": results.get("raw"),
        }

    def _compose_sentiment_report(self, *, company: str, results: Dict[str, Any]) -> Dict[str, Any]:
        answers = results["answers_by_query"]
        sources = results["sources"]
        queries_run = results["queries_run"]

        report_md_parts: List[str] = []
        report_md_parts.append(f"# {company} — Company Sentiment")
        report_md_parts.append("")
        report_md_parts.append("## Overall_sentiment")
        report_md_parts.append(
            "This section summarizes publicly reported employee sentiment themes and associated risks. "
            "Treat as directional; verify with interviews and up-to-date sources."
        )
        report_md_parts.append("")

        def add_section(title: str, key: str) -> None:
            text = (answers.get(key) or "").strip()
            if not text:
                return
            report_md_parts.append(f"## {title}")
            report_md_parts.append(text)
            report_md_parts.append("")

        add_section("Reviews_summary", "reviews_summary")
        add_section("Work_life_balance_and_oncall", "wlb_oncall")
        add_section("Compensation_signals", "comp_signals")
        add_section("Interview_process_patterns", "interview_process")

        sections: Dict[str, Any] = {
            "positive_themes": _bullet_sections_to_items(answers.get("reviews_summary", "")),
            "risk_flags": _bullet_sections_to_items(
                "\n".join([answers.get("reviews_summary", ""), answers.get("strategy_recent_news", "")])
            ),
            "questions_to_ask": _bullet_sections_to_items(
                "\n".join([answers.get("wlb_oncall", ""), answers.get("interview_process", "")])
            ),
        }

        open_questions: List[str] = []
        if not sources:
            open_questions.append("No sources returned; consider expanding max_results or refining prompts.")

        return {
            "company": company,
            "generated_at": _now_utc_iso(),
            "depth": self._defaults.depth_sentiment,
            "output_type": "sourcedAnswer",
            "report_markdown": "\n".join(report_md_parts).strip() + "\n",
            "sections": sections,
            "sources": sources,
            "queries_run": queries_run,
            "confidence": "medium" if sources else "low",
            "open_questions": open_questions,
            "raw": results.get("raw"),
        }

    # -----------------------------
    # Prompt templates
    # -----------------------------

    def _prompt_official_identity(self, *, company: str, context: Optional[Dict[str, Any]]) -> str:
        role = (context or {}).get("role")
        extra = f"\nContext role: {role}" if role else ""
        return (
            f"You are an expert company researcher.\n"
            f"Company: {company}.{extra}\n\n"
            "Objective: produce grounded facts for resume tailoring.\n\n"
            "Instructions:\n"
            "- First find the official company website and careers page.\n"
            "- Then scrape those pages.\n"
            "- Also run several searches with adjacent keywords for: product pages, about page, pricing, use cases.\n"
            "- Prefer authoritative sources (official site, filings, reputable press) over low-signal SEO pages.\n\n"
            "Return markdown with these sections (use bullets where possible, and cite every material claim):\n"
            "## What_the_company_does\n"
            "## Products_and_services\n"
            "## Ideal_customer_profile\n"
            "## Business_model\n"
            "## Differentiators\n"
        )

    def _prompt_funding_financials(self, *, company: str, context: Optional[Dict[str, Any]]) -> str:
        return (
            f"You are an expert business analyst.\n"
            f"Company: {company}.\n\n"
            "Goal: find funding and financial context relevant to hiring and strategy.\n\n"
            "Instructions:\n"
            "- Determine whether the company is public or private.\n"
            "- If private: find funding rounds, investors, valuation where available, with dates.\n"
            "- If public: find latest annual/quarterly revenue figures, segment info, and guidance.\n"
            "- Prefer filings, investor relations, earnings transcripts, and reputable press.\n"
            "- Avoid Wikipedia unless corroborated.\n\n"
            "Output: bullet list of facts with dates; cite each bullet.\n"
        )

    def _prompt_engineering_tech_stack(self, *, company: str, context: Optional[Dict[str, Any]]) -> str:
        return (
            f"You are an expert engineering researcher.\n"
            f"Company: {company}.\n\n"
            "Goal: infer engineering tech stack and priorities.\n\n"
            "Instructions:\n"
            "- Use: engineering blog, job postings, public architecture talks, GitHub/org repos, tech radar posts.\n"
            "- Extract concrete signals about: languages, cloud, data/ML stack, CI/CD, infra, observability, security/compliance.\n"
            "- If a claim is weak/inferred, label it as a signal (not a fact).\n\n"
            "Output format:\n"
            "- Provide a bullet list grouped by category (Languages, Cloud, Data/ML, Infra, Observability, Security).\n"
            "- Cite each bullet.\n"
        )

    def _prompt_hiring_org_leadership(self, *, company: str, context: Optional[Dict[str, Any]]) -> str:
        role = (context or {}).get("role")
        role_line = f"- Emphasize evidence relevant to role: {role}.\n" if role else ""
        return (
            f"You are an expert recruiter intelligence analyst.\n"
            f"Company: {company}.\n\n"
            "Goal: extract engineering org and hiring signals.\n\n"
            "Instructions:\n"
            "- Identify public engineering leadership (CTO/VP Eng) if available.\n"
            "- Extract hiring priorities from job postings and official comms.\n"
            f"{role_line}"
            "- Prefer primary sources and reputable reporting.\n\n"
            "Output: bullet list of (signal, why it matters for tailoring) with citations.\n"
        )

    def _prompt_strategy_recent_news(self, *, company: str, context: Optional[Dict[str, Any]]) -> str:
        return (
            f"You are an expert market researcher.\n"
            f"Company: {company}.\n\n"
            "Goal: summarize strategic moves and risk factors from the last 12–24 months.\n\n"
            "Instructions:\n"
            "- Find: acquisitions, major launches, partnerships, expansions, layoffs, regulatory issues, lawsuits.\n"
            "- Provide a timeline-style bullet list (date → event → implication).\n"
            "- Cite each bullet.\n"
        )

    def _prompt_reviews_summary(self, *, company: str, context: Optional[Dict[str, Any]]) -> str:
        return (
            f"You are an expert organizational psychologist focused on engineering culture.\n"
            f"Company: {company}.\n\n"
            "Goal: aggregate employee sentiment themes relevant to engineering candidates.\n\n"
            "Instructions:\n"
            "- Run several searches across: Glassdoor, Indeed, Blind/TeamBlind, Reddit, and reputable reporting.\n"
            "- Extract recurring themes. Separate what employees like vs dislike.\n"
            "- Include short quoted snippets when available.\n"
            "- Avoid overconfident conclusions; treat anecdotes as directional.\n\n"
            "Output: bullets under headings Pros_themes, Cons_themes, Engineering_culture_signals, Risk_flags. Cite each bullet.\n"
        )

    def _prompt_wlb_oncall(self, *, company: str, context: Optional[Dict[str, Any]]) -> str:
        return (
            f"You are an expert SRE/engineering manager.\n"
            f"Company: {company}.\n\n"
            "Goal: find signals about work-life balance, oncall, incident culture, pace, and burnout risk.\n\n"
            "Instructions:\n"
            "- Use employee reviews, engineering blogs, and credible reporting.\n"
            "- Output bullets with evidence. Cite each bullet.\n"
        )

    def _prompt_comp_signals(self, *, company: str, context: Optional[Dict[str, Any]]) -> str:
        return (
            f"You are an expert compensation analyst.\n"
            f"Company: {company}.\n\n"
            "Goal: find compensation signals relevant to engineering roles.\n\n"
            "Instructions:\n"
            "- Use sources like levels.fyi and reputable public datasets or reporting.\n"
            "- Only provide numeric ranges when supported by sources; otherwise describe qualitatively.\n"
            "- Cite each bullet.\n"
        )

    def _prompt_interview_process(self, *, company: str, context: Optional[Dict[str, Any]]) -> str:
        return (
            f"You are an expert interview coach.\n"
            f"Company: {company}.\n\n"
            "Goal: identify common interview process patterns for engineering candidates.\n\n"
            "Instructions:\n"
            "- Use candidate reports and reputable sources.\n"
            "- Summarize rounds, focus areas, and recurring advice.\n"
            "- Cite each bullet.\n"
        )

