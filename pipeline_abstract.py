# pipeline_abstract.py
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any
import time
from urllib.parse import quote_plus
import xml.etree.ElementTree as ET

import requests


# =========================
# Data structure
# =========================
@dataclass
class Paper:
    paper_id: str
    title: str
    abstract: str
    updated: str = ""
    pdf_url: str = ""
    authors: str = ""
    url: str = ""


# =========================
# Helpers
# =========================
def _norm(s: str) -> str:
    return " ".join((s or "").strip().split())


_ARXIV_API = "http://export.arxiv.org/api/query"
_ARXIV_NS = {"atom": "http://www.w3.org/2005/Atom"}


def _fallback_fetch_arxiv(
    query: str,
    max_results: int = 50,
    start: int = 0,
    sort_by: str = "relevance",      # relevance | lastUpdatedDate | submittedDate
    sort_order: str = "descending",  # ascending | descending
    timeout_sec: int = 20,
    sleep_sec: float = 0.0,
) -> List[Paper]:
    """
    Free, no key, abstract-only via arXiv API.
    """
    query = (query or "").strip()
    if not query:
        return []

    if sleep_sec and sleep_sec > 0:
        time.sleep(sleep_sec)

    q = quote_plus(query)
    url = (
        f"{_ARXIV_API}"
        f"?search_query=all:{q}"
        f"&start={start}"
        f"&max_results={max_results}"
        f"&sortBy={sort_by}"
        f"&sortOrder={sort_order}"
    )

    headers = {"User-Agent": "autonomous-research/0.1 (abstract-only)"}
    r = requests.get(url, headers=headers, timeout=timeout_sec)
    r.raise_for_status()

    root = ET.fromstring(r.text)

    papers: List[Paper] = []
    for entry in root.findall("atom:entry", _ARXIV_NS):
        title = _norm(entry.findtext("atom:title", default="", namespaces=_ARXIV_NS))
        abstract = _norm(entry.findtext("atom:summary", default="", namespaces=_ARXIV_NS))
        entry_id = (entry.findtext("atom:id", default="", namespaces=_ARXIV_NS) or "").strip()
        updated = (entry.findtext("atom:updated", default="", namespaces=_ARXIV_NS) or "").strip()

        # authors
        author_names = []
        for a in entry.findall("atom:author", _ARXIV_NS):
            name = (a.findtext("atom:name", default="", namespaces=_ARXIV_NS) or "").strip()
            if name:
                author_names.append(name)
        authors = ", ".join(author_names)

        # paper_id
        paper_id = entry_id.rsplit("/", 1)[-1] if entry_id else (title[:32] or "unknown")

        # pdf url
        pdf_url = f"https://arxiv.org/pdf/{paper_id}.pdf" if paper_id else ""

        if title and abstract:
            papers.append(Paper(
                paper_id=paper_id,
                title=title,
                abstract=abstract,
                updated=updated,
                pdf_url=pdf_url,
                authors=authors,
                url=entry_id,
            ))

    return papers


def _try_repo_arxiv_search(
    query: str,
    max_results: int,
    sort_by: str,
    sort_order: str,
) -> Optional[List[Paper]]:
    """
    Try to use your repo's arxiv_search.py if it exists,
    but do NOT fail if its function names differ.
    """
    try:
        import arxiv_search  # your file: arxiv_search.py
    except Exception:
        return None

    # Try common function names
    candidates = [
        "search_arxiv",
        "fetch_arxiv_papers",
        "run_arxiv_search",
        "query_arxiv",
    ]

    for fn_name in candidates:
        fn = getattr(arxiv_search, fn_name, None)
        if callable(fn):
            try:
                out = fn(query=query, max_results=max_results, sort_by=sort_by, sort_order=sort_order)
                # Accept either List[Paper] or List[dict]
                papers: List[Paper] = []
                if isinstance(out, list) and out and isinstance(out[0], Paper):
                    return out
                if isinstance(out, list) and (not out or isinstance(out[0], dict)):
                    for d in out:
                        papers.append(Paper(
                            paper_id=str(d.get("paper_id") or d.get("id") or d.get("arxiv_id") or ""),
                            title=str(d.get("title") or ""),
                            abstract=str(d.get("abstract") or d.get("summary") or ""),
                            updated=str(d.get("updated") or ""),
                            pdf_url=str(d.get("pdf_url") or d.get("pdf") or ""),
                            authors=str(d.get("authors") or ""),
                            url=str(d.get("url") or d.get("link") or ""),
                        ))
                    return papers
            except Exception:
                # try next candidate
                continue

    return None


# =========================
# The function Streamlit imports (MUST EXIST)
# =========================
def run_abstract_pipeline(
    query: str,
    max_results: int = 50,
    sort_by: str = "relevance",
    sort_order: str = "descending",
    sleep_sec: float = 0.0,
) -> List[Dict[str, Any]]:
    """
    Streamlit imports this:
        from pipeline_abstract import run_abstract_pipeline

    Returns: List[dict] papers (title+abstract+meta).
    """
    query = (query or "").strip()
    if not query:
        return []

    # 1) Prefer repo arxiv_search.py if available
    papers = _try_repo_arxiv_search(query, max_results, sort_by, sort_order)

    # 2) Fallback to built-in arXiv API
    if papers is None:
        papers = _fallback_fetch_arxiv(
            query=query,
            max_results=max_results,
            sort_by=sort_by,
            sort_order=sort_order,
            sleep_sec=sleep_sec,
        )

    return [asdict(p) for p in papers]


__all__ = ["Paper", "run_abstract_pipeline"]
