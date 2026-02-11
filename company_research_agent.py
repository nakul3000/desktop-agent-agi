from __future__ import annotations

import dataclasses
import datetime as dt
import json
import os
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests


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


# -----------------------------
# JD intake (selected job) schema
# -----------------------------

_NA = "NA"


@dataclasses.dataclass
class JobPostingIntake:
    """
    Normalized intake for a single *selected* JD.

    This is designed to match the object you get back when the user chooses one JD,
    typically shaped like:
      { "answer": "<summary text>", "sources": [ { "url": "...", "snippet": "...", ... } ] }

    All missing fields are stored as the literal string "NA" (per your preference).
    """

    # Raw payload (kept for traceability)
    answer: str = _NA
    sources: List[Dict[str, Any]] = dataclasses.field(default_factory=list)

    # Commonly used extracted fields
    company_name: str = _NA
    role_title: str = _NA
    location: str = _NA
    workplace_type: str = _NA  # remote | hybrid | on-site | NA
    employment_type: str = _NA  # full-time | part-time | contract | NA

    compensation_summary: str = _NA
    requirements_summary: str = _NA
    preferred_summary: str = _NA

    job_url: str = _NA
    extracted_at: str = dataclasses.field(default_factory=_now_utc_iso)

    def to_context_dict(self) -> Dict[str, Any]:
        # Keep context compact but complete.
        return {
            "job_intake": dataclasses.asdict(self),
            "job_url": self.job_url,
            "company": self.company_name,
            "role": self.role_title,
            "location": self.location,
            "workplace_type": self.workplace_type,
            "employment_type": self.employment_type,
            "compensation_summary": self.compensation_summary,
            "requirements_summary": self.requirements_summary,
            "preferred_summary": self.preferred_summary,
        }

    @staticmethod
    def from_selected_jd_payload(payload: Any) -> "JobPostingIntake":
        """
        Accepts either:
        - dict with keys like 'answer' and 'sources'
        - an SDK object with attributes .answer and .sources
        """

        answer = _safe_getattr(payload, "answer", None)
        sources = _safe_getattr(payload, "sources", None)
        if isinstance(payload, dict):
            answer = payload.get("answer", answer)
            sources = payload.get("sources", sources)

        answer = (answer or "").strip() or _NA
        sources_list: List[Dict[str, Any]] = []
        if isinstance(sources, list):
            for s in sources:
                if isinstance(s, dict):
                    sources_list.append(
                        {
                            "name": (s.get("name") or "").strip() or None,
                            "url": (s.get("url") or "").strip() or None,
                            "snippet": (s.get("snippet") or "").strip() or None,
                            "favicon": (s.get("favicon") or "").strip() or None,
                        }
                    )
                else:
                    sources_list.append(_source_to_dict(s))

        job_url = _extract_best_job_url(sources_list)

        parse_mode = "regex"
        company, role, location = _extract_company_role_location(answer)
        workplace = _extract_workplace_type(answer)
        employment = _extract_employment_type(answer)
        comp = _extract_section_block(answer, header="Salary range")
        reqs = _extract_section_block(answer, header="Requirements")
        pref = _extract_section_block(answer, header="Preferred")

        # Raw job descriptions from selected-job intake usually do not follow
        # LinkUp sourcedAnswer formatting, so we use an LLM parser first there.
        if not _looks_like_linkup_summary(answer):
            llm_fields = _parse_jd_with_llm(answer)
            parse_mode = "llm" if _has_meaningful_llm_parse(llm_fields) else "llm_fallback_regex"
            company = _pick_non_na(llm_fields.get("company_name"), company)
            role = _pick_non_na(llm_fields.get("role_title"), role)
            location = _pick_non_na(llm_fields.get("location"), location)
            workplace = _pick_non_na(llm_fields.get("workplace_type"), workplace)
            employment = _pick_non_na(llm_fields.get("employment_type"), employment)
            comp = _pick_non_na(llm_fields.get("compensation_summary"), comp)
            reqs = _pick_non_na(llm_fields.get("requirements_summary"), reqs)
            pref = _pick_non_na(llm_fields.get("preferred_summary"), pref)

        print(f"\n{'=' * 80}")
        print("🧭 DEBUG LOG: intake parsing mode")
        print(f"mode={parse_mode}")
        print(f"answer_len={len(answer) if answer != _NA else 0}")
        print(f"company={company}")
        print(f"role={role}")
        print(f"location={location}")
        print(f"{'=' * 80}\n")

        return JobPostingIntake(
            answer=answer,
            sources=sources_list,
            company_name=company,
            role_title=role,
            location=location,
            workplace_type=workplace,
            employment_type=employment,
            compensation_summary=comp,
            requirements_summary=reqs,
            preferred_summary=pref,
            job_url=job_url,
        )


def _extract_best_job_url(sources: List[Dict[str, Any]]) -> str:
    # Prefer a URL that looks like an ATS / job posting.
    for s in sources or []:
        url = (s.get("url") or "").strip()
        if not url:
            continue
        u = url.lower()
        if any(
            token in u
            for token in (
                "workdayjobs.com",
                "/job/",
                "/jobs/",
                "greenhouse.io",
                "lever.co",
                "myworkdayjobs.com",
                "icims.com",
                "smartrecruiters.com",
            )
        ):
            return url
    # Fallback: first URL.
    for s in sources or []:
        url = (s.get("url") or "").strip()
        if url:
            return url
    return _NA


_HIRING_RE = re.compile(
    r"^\s*(?P<company>.+?)\s+is\s+hiring\s+(?:an?|the)\s+(?P<role>.+?)(?:\s+for\s+.+?)?\s+in\s+(?P<location>.+?)\.\s*",
    re.IGNORECASE | re.DOTALL,
)


def _pick_non_na(primary: Optional[str], fallback: Optional[str]) -> str:
    left = (primary or "").strip()
    if left and left.upper() != _NA:
        return left
    right = (fallback or "").strip()
    if right and right.upper() != _NA:
        return right
    return _NA


def _looks_like_linkup_summary(answer: str) -> bool:
    if not answer or answer == _NA:
        return False
    if _HIRING_RE.match(answer):
        return True
    header_hits = sum(
        1
        for header in ("Requirements", "Preferred", "Salary range")
        if _extract_section_block(answer, header=header) != _NA
    )
    return header_hits >= 2


def _extract_first_json_object(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    stripped = text.strip()

    try:
        maybe = json.loads(stripped)
        if isinstance(maybe, dict):
            return maybe
    except Exception:
        pass

    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", stripped, re.DOTALL | re.IGNORECASE)
    if fenced:
        try:
            maybe = json.loads(fenced.group(1))
            if isinstance(maybe, dict):
                return maybe
        except Exception:
            pass

    decoder = json.JSONDecoder()
    for idx, ch in enumerate(stripped):
        if ch != "{":
            continue
        try:
            obj, _end = decoder.raw_decode(stripped[idx:])
            if isinstance(obj, dict):
                return obj
        except Exception:
            continue
    return None


def _normalize_llm_field(value: Any) -> str:
    if value is None:
        return _NA
    text = str(value).strip()
    if not text:
        return _NA
    lowered = text.lower()
    if lowered in {"n/a", "na", "none", "null", "unknown", "not specified"}:
        return _NA
    return text


def _normalize_workplace_type(value: Any) -> str:
    text = _normalize_llm_field(value).lower()
    if text == _NA.lower():
        return _NA
    if "hybrid" in text:
        return "hybrid"
    if "remote" in text:
        return "remote"
    if "on-site" in text or "onsite" in text or "office" in text:
        return "on-site"
    return _NA


def _normalize_employment_type(value: Any) -> str:
    text = _normalize_llm_field(value).lower()
    if text == _NA.lower():
        return _NA
    for label in ("full-time", "part-time", "contract", "internship", "temporary"):
        if label in text:
            return label
    return _NA


def _has_meaningful_llm_parse(fields: Dict[str, str]) -> bool:
    if not isinstance(fields, dict):
        return False
    for key in ("company_name", "role_title", "requirements_summary"):
        value = (fields.get(key) or "").strip()
        if value and value != _NA:
            return True
    return False


def _parse_jd_with_llm(answer: str) -> Dict[str, str]:
    if not answer or answer == _NA:
        return {}

    hf_token = (os.getenv("HF_TOKEN") or "").strip()
    if not hf_token:
        return {}

    hf_model = (os.getenv("HF_MODEL") or "meta-llama/Meta-Llama-3-70B-Instruct").strip()
    urls = [
        "https://router.huggingface.co/v1/chat/completions",
        "https://api-inference.huggingface.co/v1/chat/completions",
    ]
    truncated = answer[:12000]
    system_prompt = (
        "You are a strict information extraction system. "
        "Extract fields from a raw job description and return JSON only."
    )
    user_prompt = (
        "Return VALID JSON ONLY with keys:\n"
        "{\n"
        '  "company_name": string,\n'
        '  "role_title": string,\n'
        '  "location": string,\n'
        '  "workplace_type": "remote"|"hybrid"|"on-site"|"NA",\n'
        '  "employment_type": "full-time"|"part-time"|"contract"|"internship"|"temporary"|"NA",\n'
        '  "compensation_summary": string,\n'
        '  "requirements_summary": string,\n'
        '  "preferred_summary": string\n'
        "}\n\n"
        "Rules:\n"
        "- Use NA when a field is absent or uncertain.\n"
        "- Keep text concise and do not invent details.\n"
        "- Requirements/preferred summaries should be compact prose, not full bullet dumps.\n\n"
        f"JOB_DESCRIPTION:\n{truncated}"
    )
    payload = {
        "model": hf_model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": 700,
        "temperature": 0.0,
        "top_p": 1.0,
    }
    headers = {
        "Authorization": f"Bearer {hf_token}",
        "Content-Type": "application/json",
    }

    llm_text = ""
    for url in urls:
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=60)
            if response.status_code >= 400:
                continue
            data = response.json()
            llm_text = (
                ((data.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
            ).strip()
            if llm_text:
                break
        except Exception:
            continue

    parsed = _extract_first_json_object(llm_text)
    if not isinstance(parsed, dict):
        return {}

    return {
        "company_name": _normalize_llm_field(parsed.get("company_name")),
        "role_title": _normalize_llm_field(parsed.get("role_title")),
        "location": _normalize_llm_field(parsed.get("location")),
        "workplace_type": _normalize_workplace_type(parsed.get("workplace_type")),
        "employment_type": _normalize_employment_type(parsed.get("employment_type")),
        "compensation_summary": _normalize_llm_field(parsed.get("compensation_summary")),
        "requirements_summary": _normalize_llm_field(parsed.get("requirements_summary")),
        "preferred_summary": _normalize_llm_field(parsed.get("preferred_summary")),
    }


def _extract_company_role_location(answer: str) -> Tuple[str, str, str]:
    """
    Best-effort extraction for the common LinkUp 'sourcedAnswer' style:
      '<Company> is hiring a <Role> ... in <Location>.'
    """
    if not answer or answer == _NA:
        return _NA, _NA, _NA

    m = _HIRING_RE.match(answer)
    if not m:
        return _NA, _NA, _NA

    company = (m.group("company") or "").strip() or _NA
    role = (m.group("role") or "").strip() or _NA
    location = (m.group("location") or "").strip() or _NA
    return company, role, location


def _extract_workplace_type(answer: str) -> str:
    if not answer or answer == _NA:
        return _NA
    lowered = answer.lower()
    # Look for explicit statement first.
    if "position is hybrid" in lowered or lowered.rstrip().endswith("hybrid."):
        return "hybrid"
    if "position is remote" in lowered or lowered.rstrip().endswith("remote."):
        return "remote"
    if "position is on-site" in lowered or "position is onsite" in lowered or lowered.rstrip().endswith("on-site."):
        return "on-site"
    return _NA


def _extract_employment_type(answer: str) -> str:
    if not answer or answer == _NA:
        return _NA
    lowered = answer.lower()
    for label in ("full-time", "part-time", "contract", "internship", "temporary"):
        if label in lowered:
            return label
    return _NA


_HEADER_RE_TEMPLATE = r"(?im)^\s*{header}\s*:\s*$"


def _extract_section_block(answer: str, *, header: str) -> str:
    """
    Extract a block that starts after a header like 'Requirements:' and ends
    before the next top-level header (e.g., 'Preferred:', 'Salary range:', etc.).
    Returns 'NA' if not found.
    """
    if not answer or answer == _NA:
        return _NA

    header_re = re.compile(_HEADER_RE_TEMPLATE.format(header=re.escape(header)))
    m = header_re.search(answer)
    if not m:
        return _NA

    start = m.end()
    rest = answer[start:]

    # Stop at the next header-like line ("Word(s):") that begins at line start.
    next_header = re.search(r"(?im)^\s*[A-Za-z][A-Za-z \-/]{0,40}\s*:\s*$", rest)
    block = rest[: next_header.start()] if next_header else rest

    block = block.strip()
    return block or _NA


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


def _to_jsonable(value: Any) -> Any:
    """
    Convert SDK/object responses into JSON-serializable primitives.
    """
    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}

    if isinstance(value, (list, tuple, set)):
        return [_to_jsonable(v) for v in value]

    if dataclasses.is_dataclass(value):
        return _to_jsonable(dataclasses.asdict(value))

    model_dump = _safe_getattr(value, "model_dump", None)
    if callable(model_dump):
        try:
            # Prefer JSON mode when available (Pydantic v2), which avoids
            # leaking SDK model instances into nested payloads.
            return _to_jsonable(model_dump(mode="json"))
        except Exception:
            try:
                return _to_jsonable(model_dump())
            except Exception:
                pass

    to_dict = _safe_getattr(value, "to_dict", None)
    if callable(to_dict):
        try:
            return _to_jsonable(to_dict())
        except Exception:
            pass

    obj_dict = _safe_getattr(value, "__dict__", None)
    if isinstance(obj_dict, dict):
        return _to_jsonable(obj_dict)

    return str(value)


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
        *,
        job_intake: Optional[JobPostingIntake] = None,
    ) -> Dict[str, Any]:
        context = {
            "company": company,
            "role": role,
            "job_url": job_url,
            "job_description": job_description,
        }
        if job_intake is not None:
            # Note: keep both the raw intake dict and the key fields at top-level for prompts.
            context.update(job_intake.to_context_dict())
        print(f"\n{'=' * 80}")
        print("🧭 DEBUG LOG: research_company context")
        print(f"company={company}")
        print(f"role={role or 'NA'}")
        print(f"job_url={job_url or 'NA'}")
        print(f"job_description_len={len((job_description or '').strip())}")
        print(f"context_keys={sorted(list(context.keys()))}")
        print(f"requirements_summary_len={len((str(context.get('requirements_summary') or '')).strip())}")
        print(f"preferred_summary_len={len((str(context.get('preferred_summary') or '')).strip())}")
        print(f"compensation_summary_len={len((str(context.get('compensation_summary') or '')).strip())}")
        print(f"{'=' * 80}\n")
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
            print(f"\n{'=' * 80}")
            print("🧭 DEBUG LOG: company research query request")
            print(f"query_name={name}")
            print(f"depth={depth} output_type=sourcedAnswer max_results={max_results}")
            print(f"prompt_len={len(prompt)}")
            print(f"prompt:\n{prompt}")
            print(f"{'=' * 80}\n")
            response = self._client.search(
                query=prompt,
                depth=depth,
                output_type="sourcedAnswer",
                include_inline_citations=True,
                include_images=False,
                max_results=max_results,
            )
            raw_responses[name] = _to_jsonable(response)

            answer = _safe_getattr(response, "answer", "") or ""
            sources_obj = _safe_getattr(response, "sources", []) or []
            sources = [_source_to_dict(s) for s in sources_obj]
            print(f"\n{'=' * 80}")
            print("🧭 DEBUG LOG: company research query response")
            print(f"query_name={name}")
            print(f"answer_len={len(answer)}")
            print(
                "answer_preview="
                + (answer[:2800] + ("...[truncated]" if len(answer) > 2800 else ""))
            )
            print(f"source_count={len(sources)}")
            for idx, src in enumerate(sources, start=1):
                print(
                    f"[source {idx}] name={(src.get('name') or '').strip() or 'NA'} | "
                    f"url={(src.get('url') or '').strip() or 'NA'}"
                )
            print(f"{'=' * 80}\n")

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

    @staticmethod
    def _non_na(value: Any) -> str:
        text = (str(value).strip() if value is not None else "")
        if not text or text.upper() == _NA:
            return ""
        return text

    def _jd_context_block(self, *, context: Optional[Dict[str, Any]], include_jd_excerpt: bool = False) -> str:
        ctx = context or {}
        role = self._non_na(ctx.get("role"))
        job_url = self._non_na(ctx.get("job_url"))
        location = self._non_na(ctx.get("location"))
        workplace_type = self._non_na(ctx.get("workplace_type"))
        employment_type = self._non_na(ctx.get("employment_type"))
        compensation = self._non_na(ctx.get("compensation_summary"))
        requirements = self._non_na(ctx.get("requirements_summary"))
        preferred = self._non_na(ctx.get("preferred_summary"))
        job_description = self._non_na(ctx.get("job_description"))

        lines: List[str] = []
        if role:
            lines.append(f"- Target role: {role}")
        if job_url:
            lines.append(f"- Selected job URL: {job_url}")
        if location:
            lines.append(f"- Job location: {location}")
        if workplace_type:
            lines.append(f"- Workplace type: {workplace_type}")
        if employment_type:
            lines.append(f"- Employment type: {employment_type}")
        if compensation:
            lines.append(f"- Compensation signal from JD: {compensation[:280]}")
        if requirements:
            lines.append(f"- Requirements signal from JD: {requirements[:420]}")
        if preferred:
            lines.append(f"- Preferred qualifications signal from JD: {preferred[:420]}")

        if include_jd_excerpt and job_description:
            lines.append(f"- JD excerpt: {job_description[:1600]}")

        if not lines:
            return ""

        return (
            "\nSelected JD context (use to bias analysis toward this specific role posting):\n"
            + "\n".join(lines)
            + "\n"
        )

    def _prompt_official_identity(self, *, company: str, context: Optional[Dict[str, Any]]) -> str:
        jd_context = self._jd_context_block(context=context, include_jd_excerpt=False)
        return (
            f"You are an expert company researcher.\n"
            f"Company: {company}."
            f"{jd_context}\n"
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
        jd_context = self._jd_context_block(context=context, include_jd_excerpt=False)
        return (
            f"You are an expert business analyst.\n"
            f"Company: {company}."
            f"{jd_context}\n"
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
        jd_context = self._jd_context_block(context=context, include_jd_excerpt=True)
        return (
            f"You are an expert engineering researcher.\n"
            f"Company: {company}."
            f"{jd_context}\n"
            "Goal: infer engineering tech stack and priorities.\n\n"
            "Instructions:\n"
            "- Treat the selected JD context as a targeting hint; prioritize evidence that maps directly to JD requirements.\n"
            "- Use: engineering blog, job postings, public architecture talks, GitHub/org repos, tech radar posts.\n"
            "- Extract concrete signals about: languages, cloud, data/ML stack, CI/CD, infra, observability, security/compliance.\n"
            "- If a claim is weak/inferred, label it as a signal (not a fact).\n\n"
            "Output format:\n"
            "- Provide a bullet list grouped by category (Languages, Cloud, Data/ML, Infra, Observability, Security).\n"
            "- Cite each bullet.\n"
        )

    def _prompt_hiring_org_leadership(self, *, company: str, context: Optional[Dict[str, Any]]) -> str:
        role = self._non_na((context or {}).get("role"))
        role_line = f"- Emphasize evidence relevant to role: {role}.\n" if role else ""
        jd_context = self._jd_context_block(context=context, include_jd_excerpt=True)
        return (
            f"You are an expert recruiter intelligence analyst.\n"
            f"Company: {company}."
            f"{jd_context}\n"
            "Goal: extract engineering org and hiring signals.\n\n"
            "Instructions:\n"
            "- Identify public engineering leadership (CTO/VP Eng) if available.\n"
            "- Extract hiring priorities from job postings and official comms.\n"
            "- Map each signal back to this selected JD's responsibilities/requirements when possible.\n"
            f"{role_line}"
            "- Prefer primary sources and reputable reporting.\n\n"
            "Output: bullet list of (signal, why it matters for tailoring) with citations.\n"
        )

    def _prompt_strategy_recent_news(self, *, company: str, context: Optional[Dict[str, Any]]) -> str:
        jd_context = self._jd_context_block(context=context, include_jd_excerpt=False)
        return (
            f"You are an expert market researcher.\n"
            f"Company: {company}."
            f"{jd_context}\n"
            "Goal: summarize strategic moves and risk factors from the last 12–24 months.\n\n"
            "Instructions:\n"
            "- Find: acquisitions, major launches, partnerships, expansions, layoffs, regulatory issues, lawsuits.\n"
            "- Prioritize events that could affect this selected role's domain, tooling, or team mandate.\n"
            "- Provide a timeline-style bullet list (date → event → implication).\n"
            "- Cite each bullet.\n"
        )

    def _prompt_reviews_summary(self, *, company: str, context: Optional[Dict[str, Any]]) -> str:
        jd_context = self._jd_context_block(context=context, include_jd_excerpt=False)
        return (
            f"You are an expert organizational psychologist focused on engineering culture.\n"
            f"Company: {company}."
            f"{jd_context}\n"
            "Goal: aggregate employee sentiment themes relevant to engineering candidates.\n\n"
            "Instructions:\n"
            "- Run several searches across: Glassdoor, Indeed, Blind/TeamBlind, Reddit, and reputable reporting.\n"
            "- Extract recurring themes. Separate what employees like vs dislike.\n"
            "- Prioritize themes that matter for this selected role's scope (e.g., AI platform ownership, regulated environment, on-call expectations).\n"
            "- Include short quoted snippets when available.\n"
            "- Avoid overconfident conclusions; treat anecdotes as directional.\n\n"
            "Output: bullets under headings Pros_themes, Cons_themes, Engineering_culture_signals, Risk_flags. Cite each bullet.\n"
        )

    def _prompt_wlb_oncall(self, *, company: str, context: Optional[Dict[str, Any]]) -> str:
        jd_context = self._jd_context_block(context=context, include_jd_excerpt=False)
        return (
            f"You are an expert SRE/engineering manager.\n"
            f"Company: {company}."
            f"{jd_context}\n"
            "Goal: find signals about work-life balance, oncall, incident culture, pace, and burnout risk.\n\n"
            "Instructions:\n"
            "- Use employee reviews, engineering blogs, and credible reporting.\n"
            "- When possible, connect on-call/operational expectations to this selected role's production responsibilities.\n"
            "- Output bullets with evidence. Cite each bullet.\n"
        )

    def _prompt_comp_signals(self, *, company: str, context: Optional[Dict[str, Any]]) -> str:
        jd_context = self._jd_context_block(context=context, include_jd_excerpt=False)
        return (
            f"You are an expert compensation analyst.\n"
            f"Company: {company}."
            f"{jd_context}\n"
            "Goal: find compensation signals relevant to engineering roles.\n\n"
            "Instructions:\n"
            "- Use sources like levels.fyi and reputable public datasets or reporting.\n"
            "- Only provide numeric ranges when supported by sources; otherwise describe qualitatively.\n"
            "- If JD compensation is provided, treat it as anchor context and compare with external benchmarks.\n"
            "- Cite each bullet.\n"
        )

    def _prompt_interview_process(self, *, company: str, context: Optional[Dict[str, Any]]) -> str:
        jd_context = self._jd_context_block(context=context, include_jd_excerpt=False)
        return (
            f"You are an expert interview coach.\n"
            f"Company: {company}."
            f"{jd_context}\n"
            "Goal: identify common interview process patterns for engineering candidates.\n\n"
            "Instructions:\n"
            "- Use candidate reports and reputable sources.\n"
            "- Summarize rounds, focus areas, and recurring advice.\n"
            "- Highlight prep focus areas most aligned with this selected role's JD requirements.\n"
            "- Cite each bullet.\n"
        )

