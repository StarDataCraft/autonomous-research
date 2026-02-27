from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional
import time
import urllib.parse
import requests
import feedparser


@dataclass
class ArxivPaper:
    title: str
    authors: List[str]
    summary: str
    published: str
    updated: str
    arxiv_id: str
    pdf_url: str
    entry_url: str
    primary_category: Optional[str] = None


def _clean(s: str) -> str:
    return " ".join((s or "").replace("\n", " ").split())


def search_arxiv(
    query: str,
    max_results: int = 100,
    sort_by: str = "relevance",  # relevance | lastUpdatedDate | submittedDate
    sort_order: str = "descending",
    category_filter: Optional[str] = None,  # e.g. "cs.LG"
    polite_delay_sec: float = 0.2,
) -> List[ArxivPaper]:
    """
    Uses arXiv API (export.arxiv.org) via Atom feed.
    """
    q = query.strip()
    if not q:
        return []

    if category_filter:
        # arXiv syntax: cat:cs.LG AND (your query)
        q = f"cat:{category_filter} AND ({q})"

    params = {
        "search_query": q,
        "start": 0,
        "max_results": max_results,
        "sortBy": sort_by,
        "sortOrder": sort_order,
    }
    url = "http://export.arxiv.org/api/query?" + urllib.parse.urlencode(params)

    # polite delay (arXiv asks to be gentle)
    time.sleep(polite_delay_sec)

    resp = requests.get(url, timeout=30)
    resp.raise_for_status()

    feed = feedparser.parse(resp.text)
    out: List[ArxivPaper] = []

    for e in feed.entries:
        title = _clean(getattr(e, "title", ""))
        summary = _clean(getattr(e, "summary", ""))
        authors = [a.name for a in getattr(e, "authors", [])] if getattr(e, "authors", None) else []

        published = getattr(e, "published", "")
        updated = getattr(e, "updated", "")
        entry_url = getattr(e, "link", "")

        # arXiv id
        arxiv_id = ""
        if hasattr(e, "id"):
            arxiv_id = str(e.id).split("/")[-1]

        # PDF link
        pdf_url = ""
        if hasattr(e, "links"):
            for lk in e.links:
                if getattr(lk, "type", "") == "application/pdf":
                    pdf_url = lk.href
                    break
        if not pdf_url and arxiv_id:
            pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"

        primary_category = None
        if hasattr(e, "arxiv_primary_category"):
            primary_category = getattr(e.arxiv_primary_category, "term", None)

        out.append(
            ArxivPaper(
                title=title,
                authors=authors,
                summary=summary,
                published=published,
                updated=updated,
                arxiv_id=arxiv_id,
                pdf_url=pdf_url,
                entry_url=entry_url,
                primary_category=primary_category,
            )
        )

    return out
