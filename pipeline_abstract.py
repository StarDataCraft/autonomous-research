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

# Cues for "limitations / failure / caveats / trade-offs"
CONTRADICTION_CUES = [
    "however", "but", "although", "despite", "nevertheless", "nonetheless", "yet",
    "limitation", "limitations", "limit", "limited",
    "fail", "fails", "failed", "failure", "unstable", "instability", "diverge", "divergence", "nan", "collapse",
    "trade-off", "tradeoff", "cost", "drawback", "challenge", "hard", "difficult",
    "inconsistent", "sensitive", "sensitivity", "variance", "high variance", "bias",
    "does not", "do not", "cannot", "can't", "unlikely", "may not", "might not"
]

# Keywords for coarse type classification
TYPE_KEYWORDS = {
    "theory": [
        "theorem", "proof", "proposition", "lemma", "corollary", "bound", "upper bound", "lower bound",
        "convergence", "sample complexity", "minimax", "non-asymptotic", "asymptotic", "guarantee", "guarantees",
        "optimal", "order-optimal", "complexity", "consistency"
    ],
    "method": [
        "we propose", "we introduce", "novel method", "new method", "algorithm", "framework", "objective",
        "loss", "regularization", "variance-reduced", "variance reduction", "estimator", "training objective",
        "solver", "sampler", "path", "trajectory", "guidance"
    ],
    "empirical": [
        "experiment", "experiments", "empirical", "evaluate", "evaluation", "benchmark", "ablation",
        "we show", "we demonstrate", "we compare", "results", "fid", "imagenet", "cifar", "mnist", "runtime",
        "throughput", "latency", "hardware", "low-resource"
    ]
}

# Prototype texts for embedding-based classification
TYPE_PROTOTYPES = {
    "theory": "theoretical analysis proof theorem proposition bound convergence sample complexity minimax non-asymptotic guarantee",
    "method": "we propose a new method algorithm objective loss function regularization estimator variance reduction training procedure",
    "empirical": "empirical evaluation experiments benchmark ablation results compare performance FID CIFAR ImageNet runtime efficiency"
}


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
        expansions += ["diffusion model", "ddpm", "score-based", "sde", "ode"]
    if "batch" in hyp_lower:
        expansions += ["small batch", "batch size", "micro-batch", "gradient noise"]
    if "variance" in hyp_lower:
        expansions += ["gradient variance", "variance reduction", "variance-reduced"]
    if "stability" in hyp_lower:
        expansions += ["stability", "divergence", "training instability"]

    for x in expansions[:10]:
        variants.append(f"({base}) AND ({x})")

    seen = set()
    out = []
    for v in variants:
        v = v.strip()
        if v and v not in seen:
            seen.add(v)
            out.append(v)
    return out[:12]


def sentence_split(text: str) -> List[str]:
    t = (text or "").strip()
    if not t:
        return []
    parts = re.split(r"(?<=[.!?])\s+", t)
    out = []
    for p in parts:
        s = " ".join(p.split()).strip()
        if len(s) >= 30:
            out.append(s)
    return out[:28]


def has_cue(sentence: str) -> bool:
    s = (sentence or "").lower()
    return any(c in s for c in CONTRADICTION_CUES)


def keyword_score(text: str, keywords: List[str]) -> float:
    s = (text or "").lower()
    hits = 0
    for k in keywords:
        if k in s:
            hits += 1
    return float(hits) / max(1.0, float(len(keywords)))


def classify_paper_type(text: str, proto_vecs: Dict[str, np.ndarray]) -> Tuple[str, Dict[str, float]]:
    """
    Hybrid classification:
      - keyword score
      - embedding similarity to prototypes
    Returns (label, debug_scores)
    """
    # keyword side
    k_theory = keyword_score(text, TYPE_KEYWORDS["theory"])
    k_method = keyword_score(text, TYPE_KEYWORDS["method"])
    k_emp = keyword_score(text, TYPE_KEYWORDS["empirical"])

    # embedding side (vecs already normalized in embedder)
    v = embed_texts([text])[0]
    e_theory = float(v @ proto_vecs["theory"])
    e_method = float(v @ proto_vecs["method"])
    e_emp = float(v @ proto_vecs["empirical"])

    # combine (weights chosen to be stable)
    s_theory = 0.45 * e_theory + 0.55 * k_theory
    s_method = 0.45 * e_method + 0.55 * k_method
    s_emp = 0.45 * e_emp + 0.55 * k_emp

    scores = {"theory": s_theory, "method": s_method, "empirical": s_emp}
    label = max(scores.items(), key=lambda kv: kv[1])[0]
    return label, scores


@dataclass
class SentenceHit:
    sentence: str
    kind: str  # "cue-hit" | "relevance-fallback"
    sim: float


@dataclass
class PaperInsight:
    paper: ArxivPaper
    cluster_id: int
    relevance: float                  # cosine similarity to query
    novelty: float                    # 1 - max sim to other-cluster papers
    novelty_rank: float               # novelty adjusted by relevance
    bridge_score: float               # cross-cluster connectedness
    bridge_rank: float                # bridge score adjusted by relevance
    paper_type: str                   # method/theory/empirical
    type_scores: Dict[str, float]     # debug scores
    cluster_sims: Dict[int, float]    # similarity to each cluster centroid
    key_sentences: List[SentenceHit]  # cue-hit preferred; else relevance-fallback


@dataclass
class ClusteredResult:
    cluster_id: int
    papers: List[PaperInsight]
    centroid_paper: PaperInsight
    keywords: List[str]


def _normalize01(x: np.ndarray) -> np.ndarray:
    if x.size == 0:
        return x
    mn = float(np.min(x))
    mx = float(np.max(x))
    if abs(mx - mn) < 1e-9:
        return np.zeros_like(x)
    return (x - mn) / (mx - mn)


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

    # cluster centroids normalized (should already be near-normalized, but ensure)
    centers = centers / np.linalg.norm(centers, axis=1, keepdims=True).clip(min=1e-9)

    # 6) Novelty: 1 - max similarity to other-cluster papers
    novelty = np.zeros((n,), dtype=float)
    if k > 1:
        sim_mat = selected_vecs @ selected_vecs.T
        for i in range(n):
            other = [j for j in range(n) if labels[j] != labels[i]]
            if other:
                max_other = float(np.max(sim_mat[i, other]))
                novelty[i] = 1.0 - max_other
            else:
                novelty[i] = 0.0

    # 7) Bridge score: connectedness across clusters
    # Compute similarity to each cluster centroid.
    # A bridge paper is similar to multiple cluster centroids (not just its own).
    cluster_sims = selected_vecs @ centers.T  # (n,k)

    # bridge_score: average of top2 centroid similarities (excluding trivial low sims),
    # plus a "low gap" bonus when top1 ~ top2 (indicates true bridging).
    bridge_score = np.zeros((n,), dtype=float)
    if k > 1:
        for i in range(n):
            sims = cluster_sims[i].copy()
            # top2 over centroids
            top2 = np.sort(sims)[-2:]
            top1, top2v = float(top2[1]), float(top2[0])
            gap = max(0.0, top1 - top2v)
            # encourage small gap (bridge), penalize huge gap (purely single-cluster)
            bridge_score[i] = 0.65 * (top1 + top2v) / 2.0 + 0.35 * (1.0 - gap)
    else:
        bridge_score[:] = 0.0

    # 8) Adjusted ranks (avoid "off-topic but weird" dominating)
    rel_norm = _normalize01(selected_rel.astype(float))
    novelty_rank = novelty * (0.35 + 0.65 * rel_norm)
    bridge_rank = bridge_score * (0.35 + 0.65 * rel_norm)

    # 9) Precompute prototype vecs for type classification
    proto_texts = [TYPE_PROTOTYPES["theory"], TYPE_PROTOTYPES["method"], TYPE_PROTOTYPES["empirical"]]
    proto_embs = embed_texts(proto_texts, model_name=embed_model)
    proto_vecs = {"theory": proto_embs[0], "method": proto_embs[1], "empirical": proto_embs[2]}

    # 10) Sentence-level key sentence hunt (cue-hit preferred; else relevance-fallback)
    key_sents: List[List[SentenceHit]] = [[] for _ in range(n)]
    for i, p in enumerate(selected_papers):
        sents = sentence_split(p.summary)
        if not sents:
            continue

        sent_vecs = embed_texts(sents, model_name=embed_model)  # (S,d)
        sent_sim = (sent_vecs @ q_vec).astype(float)            # (S,)

        cand_idx = np.argsort(-sent_sim)[: min(contradiction_candidate_n, len(sents))].tolist()

        # cue-hit first
        cue_hits = [(j, float(sent_sim[j])) for j in cand_idx if has_cue(sents[j])]
        cue_hits.sort(key=lambda x: x[1], reverse=True)

        out: List[SentenceHit] = []
        for j, sim in cue_hits[: min(contradiction_top_n, len(cue_hits))]:
            out.append(SentenceHit(sentence=sents[j], kind="cue-hit", sim=sim))

        # fallback if not enough cue-hit
        if len(out) < contradiction_top_n:
            need = contradiction_top_n - len(out)
            for j in cand_idx:
                if any(h.sentence == sents[j] for h in out):
                    continue
                out.append(SentenceHit(sentence=sents[j], kind="relevance-fallback", sim=float(sent_sim[j])))
                if len(out) >= contradiction_top_n:
                    break

        key_sents[i] = out

    # 11) Pack insights
    insights: List[PaperInsight] = []
    for i, p in enumerate(selected_papers):
        text = f"{p.title}\n\n{p.summary}"
        paper_type, type_scores = classify_paper_type(text, proto_vecs)

        sims_dict = {int(cid): float(cluster_sims[i, cid]) for cid in range(k)}

        insights.append(
            PaperInsight(
                paper=p,
                cluster_id=int(labels[i]),
                relevance=float(selected_rel[i]),
                novelty=float(novelty[i]),
                novelty_rank=float(novelty_rank[i]),
                bridge_score=float(bridge_score[i]),
                bridge_rank=float(bridge_rank[i]),
                paper_type=paper_type,
                type_scores=type_scores,
                cluster_sims=sims_dict,
                key_sentences=key_sents[i],
            )
        )

    # 12) Build clusters with representative (closest to centroid)
    clusters: List[ClusteredResult] = []
    if k == 1:
        rep = max(insights, key=lambda x: x.relevance)
        title_blob = " ".join([x.paper.title for x in insights])
        clusters.append(
            ClusteredResult(
                cluster_id=0,
                papers=sorted(insights, key=lambda x: (-x.relevance, -x.novelty_rank)),
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
            cluster_insights.sort(key=lambda x: (-x.relevance, -x.novelty_rank))

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
