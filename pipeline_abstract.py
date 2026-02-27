from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import math
import re
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

# Cue words for "claims / limitations / failure cases" hunting
CONTRADICTION_CUES = [
    "however", "but", "although", "despite", "nevertheless", "nonetheless", "yet",
    "limitation", "limitations", "limit", "limited",
    "fail", "fails", "failed", "failure", "unstable", "instability", "diverge", "divergence", "nan",
    "trade-off", "tradeoff", "cost", "drawback", "challenge", "hard", "difficult",
    "inconsistent", "sensitive", "sensitivity", "variance", "high variance", "bias",
    "does not", "do not", "cannot", "can't", "unlikely"
]


def simple_keywords(text: str, max_terms: int = 10) -> List[str]:
    toks = re.findall(r"[A-Za-z][A-Za-z0-9_\-]{2,}", (text or "").lower())
    toks = [t for t in toks if t not in STOPWORDS]
    toks = sorted(set(toks), key=lambda x: (-len(x), x))
    return toks[:max_terms]


def build_query_variants(hypothesis: str) -> List[str]:
    kws = simple_keywords(hypothesis, max_terms=10)
    base = " ".join(kws[:6]) if kws else hypothesis

    variants = [base]

    hyp_lower = (hypothesis or "").lower()
    expansions = []
    if "flow" in hyp_lower:
        expansions += ["rectified flow", "flow matching"]
    if "diffusion" in hyp_lower or "ddpm" in hyp_lower:
        expansions += ["diffusion model", "ddpm", "score-based"]
    if "batch" in hyp_lower:
        expansions += ["small batch", "batch size"]
    if "variance" in hyp_lower:
        expansions += ["gradient variance", "variance reduction", "variance-reduced"]

    for x in expansions[:8]:
        variants.append(f"({base}) AND ({x})")

    seen = set()
    out = []
    for v in variants:
        v = v.strip()
        if v and v not in seen:
            seen.add(v)
            out.append(v)
    return out[:10]


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    # embeddings are normalized => dot is cosine
    return float(a @ b)


def sentence_split(text: str) -> List[str]:
    """
    Very simple sentence splitter (good enough for abstracts).
    """
    t = (text or "").strip()
    if not t:
        return []
    # split on ., !, ? followed by space/capital or end
    parts = re.split(r"(?<=[.!?])\s+", t)
    # clean + filter very short
    out = []
    for p in parts:
        s = " ".join(p.split()).strip()
        if len(s) >= 30:
            out.append(s)
    return out[:24]  # keep it bounded


def has_cue(sentence: str) -> bool:
    s = (sentence or "").lower()
    return any(c in s for c in CONTRADICTION_CUES)


@dataclass
class PaperInsight:
    paper: ArxivPaper
    relevance: float          # cosine similarity to query
    novelty: float            # 1 - max similarity to other clusters (higher = more novel)
    cluster_id: int
    contradiction_sentences: List[str]  # top sentences w/ cues


@dataclass
class ClusteredResult:
    cluster_id: int
    papers: List[PaperInsight]
    centroid_paper: PaperInsight
    keywords: List[str]


def run_abstract_pipeline(
    hypothesis: str,
    max_results_per_query: int = 50,
    mmr_k: int = 24,
    mmr_lambda: float = 0.7,
    category_filter: Optional[str] = "cs.LG",
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    contradiction_top_n: int = 3,
    contradiction_candidate_n: int = 10,
) -> Tuple[List[PaperInsight], List[ClusteredResult], Dict[str, str]]:
    """
    Abstract-only autonomous research engine:
      arXiv recall -> embedding re-rank -> MMR diversify -> cluster -> novelty + contradiction hunt

    Returns:
      - selected insights (MMR diversified list)
      - clustered results (each cluster contains PaperInsights)
      - diagnostics
    """
    q = (hypothesis or "").strip()
    if not q:
        return [], [], {"queries": "", "candidate_count": "0"}

    queries = build_query_variants(q)

    # 1) Recall from arXiv
    pool: List[ArxivPaper] = []
    for qq in queries:
        pool.extend(
            search_arxiv(
                query=qq,
                max_results=max_results_per_query,
                sort_by="relevance",
                category_filter=category_filter,
            )
        )

    # 2) De-dup
    uniq: Dict[str, ArxivPaper] = {}
    for p in pool:
        key = p.arxiv_id or p.entry_url or p.title
        if key and key not in uniq:
            uniq[key] = p
    candidates = list(uniq.values())

    if not candidates:
        return [], [], {"queries": " | ".join(queries), "candidate_count": "0"}

    # 3) Embeddings (title + abstract)
    doc_texts = [f"{p.title}\n\n{p.summary}" for p in candidates]
    doc_vecs = embed_texts(doc_texts, model_name=embed_model)  # (N, d)
    q_vec = embed_texts([q], model_name=embed_model)[0]        # (d,)

    # relevance for all candidates
    rel_all = (doc_vecs @ q_vec).astype(float)  # (N,)

    # 4) MMR selection
    selected_idx = mmr_select(q_vec, doc_vecs, top_k=mmr_k, lambda_mult=mmr_lambda)
    selected_papers = [candidates[i] for i in selected_idx]
    selected_vecs = doc_vecs[selected_idx, :]
    selected_rel = rel_all[selected_idx]

    n = len(selected_papers)

    # 5) Clustering
    if n <= 3:
        labels = np.zeros((n,), dtype=int)
        centers = np.mean(selected_vecs, axis=0, keepdims=True)
        k = 1
    else:
        k = max(2, min(8, int(math.sqrt(n))))
        km = KMeans(n_clusters=k, n_init="auto", random_state=42)
        labels = km.fit_predict(selected_vecs)
        centers = km.cluster_centers_

    # 6) Novelty score: 1 - max similarity to papers in OTHER clusters
    # If only 1 cluster, novelty = 0 for all
    novelty = np.zeros((n,), dtype=float)
    if k > 1:
        # precompute sim matrix for selected
        sim_mat = selected_vecs @ selected_vecs.T  # cosine similarities
        for i in range(n):
            other = [j for j in range(n) if labels[j] != labels[i]]
            if other:
                max_other = float(np.max(sim_mat[i, other]))
                novelty[i] = 1.0 - max_other
            else:
                novelty[i] = 0.0

    # 7) Contradiction hunt: sentence-level embedding + cue filtering
    contradiction_sents: List[List[str]] = [[] for _ in range(n)]
    for i, p in enumerate(selected_papers):
        sents = sentence_split(p.summary)
        if not sents:
            continue

        # candidate ranking by embedding similarity to query
        sent_vecs = embed_texts(sents, model_name=embed_model)  # (S,d)
        sent_sim = sent_vecs @ q_vec                             # (S,)

        # take top candidate sentences by similarity
        cand_idx = np.argsort(-sent_sim)[: min(contradiction_candidate_n, len(sents))]
        # among them, keep those containing cues; rank by similarity
        filtered = [(int(j), float(sent_sim[j])) for j in cand_idx if has_cue(sents[int(j)])]

        # fallback: if none has cues, still take top 1-2 high-sim sentences (so UI not empty)
        if not filtered:
            topj = cand_idx[: min(2, len(cand_idx))]
            contradiction_sents[i] = [sents[int(j)] for j in topj]
        else:
            filtered.sort(key=lambda x: x[1], reverse=True)
            top = filtered[: min(contradiction_top_n, len(filtered))]
            contradiction_sents[i] = [sents[j] for (j, _) in top]

    # 8) Pack insights
    insights: List[PaperInsight] = []
    for i, p in enumerate(selected_papers):
        insights.append(
            PaperInsight(
                paper=p,
                relevance=float(selected_rel[i]),
                novelty=float(novelty[i]),
                cluster_id=int(labels[i]),
                contradiction_sentences=contradiction_sents[i],
            )
        )

    # 9) Build clusters with representative (closest to centroid)
    clusters: List[ClusteredResult] = []
    if k == 1:
        # representative: highest relevance
        rep = max(insights, key=lambda x: x.relevance)
        title_blob = " ".join([x.paper.title for x in insights])
        clusters.append(
            ClusteredResult(
                cluster_id=0,
                papers=sorted(insights, key=lambda x: (-x.relevance, -x.novelty)),
                centroid_paper=rep,
                keywords=simple_keywords(title_blob, max_terms=8),
            )
        )
    else:
        for cid in range(k):
            idxs = [i for i in range(n) if int(labels[i]) == cid]
            if not idxs:
                continue
            center = centers[cid]
            sims = selected_vecs[idxs] @ center
            best_local = idxs[int(np.argmax(sims))]
            rep = insights[best_local]

            title_blob = " ".join([insights[i].paper.title for i in idxs])
            cluster_insights = [insights[i] for i in idxs]
            cluster_insights.sort(key=lambda x: (-x.relevance, -x.novelty))

            clusters.append(
                ClusteredResult(
                    cluster_id=cid,
                    papers=cluster_insights,
                    centroid_paper=rep,
                    keywords=simple_keywords(title_blob, max_terms=8),
                )
            )

        clusters.sort(key=lambda c: len(c.papers), reverse=True)

    diagnostics = {
        "queries": " | ".join(queries),
        "candidate_count": str(len(candidates)),
        "selected_count": str(len(insights)),
        "cluster_count": str(len(clusters)),
        "embed_model": embed_model,
    }
    return insights, clusters, diagnostics
