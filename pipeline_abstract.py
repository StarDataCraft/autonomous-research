from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Tuple
import math
import numpy as np

from sklearn.cluster import KMeans

from arxiv_search import ArxivPaper, search_arxiv
from embedder import embed_texts
from mmr import mmr_select


STOPWORDS = {
    "the","a","an","and","or","to","of","in","on","for","with","without","via","by","from","is","are","be","as",
    "we","our","their","this","that","these","those","using","use","based","new","method","methods","model","models",
    "study","paper","approach","analysis","results","learning"
}


def simple_keywords(text: str, max_terms: int = 10) -> List[str]:
    # ultra-simple keyword extraction (cheap, deterministic)
    import re
    toks = re.findall(r"[A-Za-z][A-Za-z0-9_\-]{2,}", (text or "").lower())
    toks = [t for t in toks if t not in STOPWORDS]
    # keep unique, prefer longer tokens
    toks = sorted(set(toks), key=lambda x: (-len(x), x))
    return toks[:max_terms]


def build_query_variants(hypothesis: str) -> List[str]:
    """
    Generate a small set of query variants for recall.
    Not LLM-based. Focused on ML papers typical terms.
    """
    kws = simple_keywords(hypothesis, max_terms=10)
    base = " ".join(kws[:6]) if kws else hypothesis

    variants = [base]

    # add common method-family expansions if present
    hyp_lower = (hypothesis or "").lower()
    expansions = []
    if "flow" in hyp_lower:
        expansions += ["rectified flow", "flow matching"]
    if "diffusion" in hyp_lower or "ddpm" in hyp_lower:
        expansions += ["diffusion model", "ddpm", "score-based"]
    if "batch" in hyp_lower:
        expansions += ["small batch", "batch size"]

    # build a few combined queries
    for x in expansions[:6]:
        variants.append(f"({base}) AND ({x})")

    # de-dup, keep order
    seen = set()
    out = []
    for v in variants:
        v = v.strip()
        if v and v not in seen:
            seen.add(v)
            out.append(v)
    return out[:8]


@dataclass
class ClusteredResult:
    cluster_id: int
    papers: List[ArxivPaper]
    centroid_paper: ArxivPaper
    keywords: List[str]


def run_abstract_pipeline(
    hypothesis: str,
    max_results_per_query: int = 50,
    mmr_k: int = 24,
    mmr_lambda: float = 0.7,
    category_filter: str | None = "cs.LG",
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> Tuple[List[ArxivPaper], List[ClusteredResult], Dict[str, str]]:
    """
    Returns:
      - selected papers (MMR top-K)
      - clustered results
      - diagnostics
    """
    queries = build_query_variants(hypothesis)

    # 1) Recall from arXiv
    pool: List[ArxivPaper] = []
    for q in queries:
        pool.extend(
            search_arxiv(
                query=q,
                max_results=max_results_per_query,
                sort_by="relevance",
                category_filter=category_filter,
            )
        )

    # 2) De-dup by arxiv_id
    uniq: Dict[str, ArxivPaper] = {}
    for p in pool:
        key = p.arxiv_id or p.entry_url or p.title
        if key not in uniq:
            uniq[key] = p
    candidates = list(uniq.values())

    if not candidates:
        return [], [], {"queries": str(queries), "candidate_count": "0"}

    # 3) Embeddings (title + abstract)
    doc_texts = [f"{p.title}\n\n{p.summary}" for p in candidates]
    doc_vecs = embed_texts(doc_texts, model_name=embed_model)

    q_vec = embed_texts([hypothesis], model_name=embed_model)[0]

    # 4) MMR selection for relevance + diversity
    selected_idx = mmr_select(q_vec, doc_vecs, top_k=mmr_k, lambda_mult=mmr_lambda)
    selected = [candidates[i] for i in selected_idx]
    selected_vecs = doc_vecs[selected_idx, :]

    # 5) Clustering (KMeans)
    n = len(selected)
    k = max(2, min(8, int(math.sqrt(n)))) if n >= 4 else 1

    clusters: List[ClusteredResult] = []
    if k == 1:
        # single cluster
        centroid_paper = selected[0]
        clusters = [
            ClusteredResult(
                cluster_id=0,
                papers=selected,
                centroid_paper=centroid_paper,
                keywords=simple_keywords(" ".join([p.title for p in selected]), max_terms=8),
            )
        ]
    else:
        km = KMeans(n_clusters=k, n_init="auto", random_state=42)
        labels = km.fit_predict(selected_vecs)

        # compute centroid paper per cluster: nearest to cluster center
        for cid in range(k):
            idxs = [i for i, lab in enumerate(labels) if lab == cid]
            if not idxs:
                continue
            center = km.cluster_centers_[cid]
            # choose paper with max cosine sim to center
            vecs = selected_vecs[idxs]
            sims = vecs @ center
            best_local = idxs[int(np.argmax(sims))]
            centroid_paper = selected[best_local]

            title_blob = " ".join([selected[i].title for i in idxs])
            clusters.append(
                ClusteredResult(
                    cluster_id=cid,
                    papers=[selected[i] for i in idxs],
                    centroid_paper=centroid_paper,
                    keywords=simple_keywords(title_blob, max_terms=8),
                )
            )

        # sort clusters by size desc
        clusters.sort(key=lambda c: len(c.papers), reverse=True)

    diagnostics = {
        "queries": " | ".join(queries),
        "candidate_count": str(len(candidates)),
        "selected_count": str(len(selected)),
        "cluster_count": str(len(clusters)),
    }
    return selected, clusters, diagnostics
