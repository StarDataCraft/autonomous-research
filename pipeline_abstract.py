# pipeline_abstract.py
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional
import re
import math

import numpy as np

# --- Optional: better embeddings if available ---
_HAS_ST = False
try:
    from sentence_transformers import SentenceTransformer  # type: ignore
    _HAS_ST = True
except Exception:
    _HAS_ST = False

# --- Fallback embeddings (always available if sklearn installed) ---
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans


# =========================
# Data structures
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


@dataclass
class SentenceHit:
    kind: str  # "claim" | "contra"
    sentence: str
    score: float
    cue_hit: bool


@dataclass
class ScoredPaper:
    paper: Paper
    cluster_id: int
    paper_type: str  # "method"|"theory"|"empirical"|"other"
    relevance: float
    novelty: float
    novelty_rank: float
    bridge: float
    bridge_rank: float
    key_sentences: List[SentenceHit]


# =========================
# Text utilities
# =========================
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")
_WS = re.compile(r"\s+")


def normalize_text(s: str) -> str:
    s = (s or "").strip()
    s = _WS.sub(" ", s)
    return s


def split_sentences(text: str) -> List[str]:
    text = normalize_text(text)
    if not text:
        return []
    parts = [p.strip() for p in _SENT_SPLIT.split(text) if p.strip()]
    # drop too-short junk
    out = []
    for p in parts:
        if len(p) < 20:
            continue
        out.append(p)
    return out


# =========================
# Cue dictionaries (核心改动：claim vs contra 分流)
# =========================
CLAIM_CUES = [
    "we propose", "we present", "we introduce", "we show", "we demonstrate",
    "we derive", "we develop", "we provide", "we establish", "our method",
]
LIMIT_FAIL_CUES = [
    "limitation", "limitations", "limited", "fails", "failure", "breaks down",
    "does not", "cannot", "unstable", "diverge", "diverges", "collapse", "collapses",
    "mode collapse", "nan", "sensitive to", "depends on", "requires", "assumption",
    "at the cost of", "trade-off", "tradeoff", "bias", "variance",
]
COMPARE_NEG_CUES = [
    "however", "in contrast", "yet", "nevertheless", "although",
    "worse", "inferior", "underperform", "underperforms", "at the expense of",
]


def cue_hit(sentence: str, cues: List[str]) -> bool:
    s = sentence.lower()
    return any(c in s for c in cues)


# =========================
# Paper type & axis tags
# =========================
def classify_paper_type(title: str, abstract: str) -> str:
    t = (title + " " + abstract).lower()
    # crude but effective
    theory_keys = ["theorem", "proof", "guarantee", "bound", "non-asymptotic", "minimax", "sample complexity", "convergence"]
    empirical_keys = ["experiment", "empirical", "benchmark", "cifar", "imagenet", "mnist", "fid", "ablations", "we evaluate"]
    method_keys = ["we propose", "we introduce", "algorithm", "method", "objective", "loss", "framework", "procedure"]

    score = {"theory": 0, "empirical": 0, "method": 0}
    for k in theory_keys:
        if k in t:
            score["theory"] += 1
    for k in empirical_keys:
        if k in t:
            score["empirical"] += 1
    for k in method_keys:
        if k in t:
            score["method"] += 1

    best = max(score, key=lambda x: score[x])
    if score[best] == 0:
        return "other"
    return best


def axis_tag(title: str, abstract: str) -> str:
    """Used to split novelty boards: avoid confounders hijacking breakthroughs."""
    t = (title + " " + abstract).lower()

    if any(k in t for k in ["adam", "sgd", "optimizer", "learning rate", "warmup", "gradient clipping", "micro-batch", "batch size invariant"]):
        return "optimizer/dynamics"
    if any(k in t for k in ["objective", "loss", "reweight", "generator matching", "flow matching", "rectified flow", "stochastic interpolant"]):
        return "objective/path"
    if any(k in t for k in ["theorem", "bound", "guarantee", "kl divergence", "sample complexity", "rademacher"]):
        return "theory/bounds"
    if any(k in t for k in ["benchmark", "cifar", "imagenet", "mnist", "corrupted", "robustness", "diffusion-c", "evaluation", "fid"]):
        return "evaluation/benchmark"
    return "applications"


# =========================
# Embeddings backend
# =========================
class Embedder:
    def __init__(self):
        self.mode = "tfidf"
        self._st_model = None
        self._tfidf = None

        if _HAS_ST:
            try:
                # small + good
                self._st_model = SentenceTransformer("all-MiniLM-L6-v2")
                self.mode = "sbert"
            except Exception:
                self._st_model = None
                self.mode = "tfidf"

    def fit_tfidf(self, corpus: List[str]) -> None:
        self._tfidf = TfidfVectorizer(
            lowercase=True,
            max_features=60000,
            ngram_range=(1, 2),
            stop_words="english",
        )
        self._tfidf.fit(corpus)

    def encode(self, texts: List[str]) -> np.ndarray:
        texts = [normalize_text(x) for x in texts]
        if self.mode == "sbert" and self._st_model is not None:
            X = self._st_model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
            return np.asarray(X, dtype=np.float32)
        # tfidf
        if self._tfidf is None:
            self.fit_tfidf(texts)
        X = self._tfidf.transform(texts).astype(np.float32)
        # L2 normalize (cosine-ready)
        denom = np.sqrt((X.multiply(X)).sum(axis=1)).A1 + 1e-9
        X = X.multiply(1.0 / denom[:, None])
        return X.toarray()


def cosine_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    # both assumed normalized
    return np.clip(A @ B.T, -1.0, 1.0)


# =========================
# MMR selection (sentence-level)
# =========================
def mmr_select(
    sentence_emb: np.ndarray,
    query_emb: np.ndarray,
    k: int = 5,
    lambda_rel: float = 0.65,
) -> List[int]:
    """
    sentence_emb: [n, d] normalized
    query_emb: [1, d] normalized
    """
    n = sentence_emb.shape[0]
    if n == 0:
        return []
    k = min(k, n)
    rel = (sentence_emb @ query_emb.T).reshape(-1)  # [n]

    selected: List[int] = []
    candidates = set(range(n))

    # pick best relevance first
    first = int(np.argmax(rel))
    selected.append(first)
    candidates.remove(first)

    # precompute sims
    sims = cosine_matrix(sentence_emb, sentence_emb)  # [n,n]

    while len(selected) < k and candidates:
        best_i = None
        best_score = -1e9
        for i in list(candidates):
            max_sim_to_selected = max(sims[i, j] for j in selected)
            score = lambda_rel * rel[i] - (1 - lambda_rel) * max_sim_to_selected
            if score > best_score:
                best_score = score
                best_i = i
        if best_i is None:
            break
        selected.append(best_i)
        candidates.remove(best_i)

    return selected


# =========================
# Core analysis
# =========================
def analyze_papers(
    papers: List[Paper],
    query_text: str,
    k_clusters: int = 4,
    seed: int = 42,
) -> Dict:
    """
    No LLM.
    Returns a dict with:
      - scored_papers (list[ScoredPaper as dict])
      - clusters summary
      - leaderboards: novelty_in_topic / novelty_confounder / novelty_bridge / bridge
      - synthesis signals
    """
    if not papers:
        return {"scored_papers": [], "clusters": {}, "leaderboards": {}, "meta": {"error": "no papers"}}

    embedder = Embedder()

    # --- doc embeddings ---
    docs = [f"{p.title}\n\n{p.abstract}" for p in papers]
    doc_emb = embedder.encode(docs)  # [N,d]
    q_emb = embedder.encode([query_text])  # [1,d]

    # relevance = cosine(doc, query) in [0,1] roughly
    rel = (doc_emb @ q_emb.T).reshape(-1)
    rel = np.clip(rel, -1.0, 1.0)
    rel01 = (rel + 1.0) / 2.0

    # --- clustering ---
    N = len(papers)
    k = max(2, min(k_clusters, N))
    km = KMeans(n_clusters=k, random_state=seed, n_init="auto")
    labels = km.fit_predict(doc_emb)

    # centroids (normalized)
    centroids = []
    for c in range(k):
        idx = np.where(labels == c)[0]
        vec = doc_emb[idx].mean(axis=0)
        vec = vec / (np.linalg.norm(vec) + 1e-9)
        centroids.append(vec)
    centroids = np.stack(centroids, axis=0)  # [k,d]

    # --- novelty: 1 - max similarity to other clusters ---
    cent_sim = cosine_matrix(doc_emb, centroids)  # [N,k] cosine in [-1,1]
    cent_sim01 = (cent_sim + 1.0) / 2.0

    novelty = np.zeros(N, dtype=np.float32)
    bridge = np.zeros(N, dtype=np.float32)

    for i in range(N):
        c = labels[i]
        sims = cent_sim01[i].copy()
        sims_other = np.delete(sims, c)
        novelty[i] = 1.0 - float(np.max(sims_other)) if sims_other.size else 0.0

        # bridge: high top2 centroid similarity + small gap
        top2 = np.sort(sims)[-2:]
        if len(top2) < 2:
            bridge[i] = 0.0
        else:
            gap = float(top2[-1] - top2[-2])
            bridge[i] = float(top2[-2]) * float(1.0 - gap)  # prefer two-high-close

    # --- novelty rank: avoid "离题但离群" ---
    # Use a smooth relevance weight
    def rel_weight(x: float) -> float:
        # suppress low relevance, preserve high relevance
        # in [0,1] -> [0,1]
        return float(x ** 1.8)

    novelty_rank = np.array([float(novelty[i]) * rel_weight(rel01[i]) for i in range(N)], dtype=np.float32)
    bridge_rank = np.array([float(bridge[i]) * rel_weight(rel01[i]) for i in range(N)], dtype=np.float32)

    # --- key sentences (claim vs contra hard split) ---
    scored: List[ScoredPaper] = []
    for i, p in enumerate(papers):
        sents = split_sentences(p.abstract)
        if sents:
            sent_emb = embedder.encode(sents)  # normalized
            # sentence relevance to query
            s_rel = (sent_emb @ q_emb.T).reshape(-1)
            s_rel01 = (np.clip(s_rel, -1.0, 1.0) + 1.0) / 2.0
        else:
            sent_emb = np.zeros((0, doc_emb.shape[1]), dtype=np.float32)
            s_rel01 = np.zeros((0,), dtype=np.float32)

        hits: List[SentenceHit] = []

        # Claim candidates: claim cues OR top relevant sentences
        claim_idx = []
        if sents:
            for j, s in enumerate(sents):
                if cue_hit(s, CLAIM_CUES):
                    claim_idx.append(j)
            # add top-2 relevance if not enough
            if len(claim_idx) < 2:
                topj = list(np.argsort(-s_rel01)[:2])
                claim_idx.extend([t for t in topj if t not in claim_idx])
            claim_idx = claim_idx[:3]

        for j in claim_idx:
            hits.append(SentenceHit(
                kind="claim",
                sentence=sents[j],
                score=float(s_rel01[j]),
                cue_hit=cue_hit(sents[j], CLAIM_CUES),
            ))

        # Contra candidates: MUST hit limitation/failure OR comparison/neg cues
        contra_candidates = []
        if sents:
            for j, s in enumerate(sents):
                is_contra_cue = cue_hit(s, LIMIT_FAIL_CUES) or cue_hit(s, COMPARE_NEG_CUES)
                if is_contra_cue:
                    contra_candidates.append(j)

        # If too many, keep diverse+relevant via MMR but only inside contra candidates
        if contra_candidates:
            sub_emb = sent_emb[contra_candidates]
            sub_sel = mmr_select(sub_emb, q_emb, k=min(3, len(contra_candidates)), lambda_rel=0.7)
            for pick in sub_sel:
                j = contra_candidates[pick]
                hits.append(SentenceHit(
                    kind="contra",
                    sentence=sents[j],
                    score=float(s_rel01[j]),
                    cue_hit=True,
                ))

        # paper type + axis
        ptype = classify_paper_type(p.title, p.abstract)
        _axis = axis_tag(p.title, p.abstract)

        sp = ScoredPaper(
            paper=p,
            cluster_id=int(labels[i]),
            paper_type=ptype,
            relevance=float(rel01[i]),
            novelty=float(novelty[i]),
            novelty_rank=float(novelty_rank[i]),
            bridge=float(bridge[i]),
            bridge_rank=float(bridge_rank[i]),
            key_sentences=hits,
        )
        # attach axis tag via dict expansion (keep dataclass clean)
        d = asdict(sp)
        d["axis"] = _axis
        scored.append(_dict_to_scoredpaper(d))  # keep as dataclass for further ops

    # --- cluster keywords (TF-IDF on titles+abstracts) ---
    # Always available: use fallback tfidf for interpretability
    vec = TfidfVectorizer(lowercase=True, stop_words="english", max_features=30000, ngram_range=(1, 2))
    X = vec.fit_transform(docs)
    vocab = np.array(vec.get_feature_names_out())

    clusters: Dict[int, Dict] = {}
    for c in range(k):
        idx = np.where(labels == c)[0]
        if idx.size == 0:
            continue
        # top tfidf terms in cluster mean
        mean = X[idx].mean(axis=0).A1
        top = mean.argsort()[-10:][::-1]
        keywords = [str(vocab[t]) for t in top[:8] if mean[t] > 0]

        # paper type counts
        type_counts = {"method": 0, "theory": 0, "empirical": 0, "other": 0}
        for ii in idx:
            type_counts[classify_paper_type(papers[ii].title, papers[ii].abstract)] += 1

        clusters[c] = {
            "size": int(idx.size),
            "keywords": keywords,
            "type_counts": type_counts,
        }

    # --- leaderboards: split novelty boards ---
    novelty_in_topic = []
    novelty_confounder = []
    novelty_bridge = []

    for sp in scored:
        if sp.relevance < 0.20:
            continue
        ax = getattr(sp, "axis", None)  # will be added by helper below
        # fallback: re-tag if missing
        if ax is None:
            ax = axis_tag(sp.paper.title, sp.paper.abstract)

        if ax == "optimizer/dynamics":
            novelty_confounder.append(sp)
        else:
            novelty_in_topic.append(sp)

        # bridge novelty: "bridge_rank" high and novelty decent
        novelty_bridge.append(sp)

    novelty_in_topic.sort(key=lambda x: x.novelty_rank, reverse=True)
    novelty_confounder.sort(key=lambda x: x.novelty_rank, reverse=True)
    novelty_bridge.sort(key=lambda x: (x.bridge_rank, x.novelty_rank), reverse=True)

    bridge_board = sorted(scored, key=lambda x: x.bridge_rank, reverse=True)

    # --- build return dict (serialize) ---
    scored_dicts = []
    for sp in scored:
        d = asdict(sp)
        d["axis"] = axis_tag(sp.paper.title, sp.paper.abstract)
        scored_dicts.append(d)

    return {
        "scored_papers": scored_dicts,
        "clusters": clusters,
        "leaderboards": {
            "novelty_in_topic": [asdict(x) | {"axis": axis_tag(x.paper.title, x.paper.abstract)} for x in novelty_in_topic[:20]],
            "novelty_confounder": [asdict(x) | {"axis": axis_tag(x.paper.title, x.paper.abstract)} for x in novelty_confounder[:20]],
            "novelty_bridge": [asdict(x) | {"axis": axis_tag(x.paper.title, x.paper.abstract)} for x in novelty_bridge[:20]],
            "bridge": [asdict(x) | {"axis": axis_tag(x.paper.title, x.paper.abstract)} for x in bridge_board[:20]],
        },
        "meta": {
            "embedder": "sbert(all-MiniLM-L6-v2)" if embedder.mode == "sbert" else "tfidf",
            "k_clusters": k,
        },
    }


# =========================
# Helpers for dict <-> dataclass
# =========================
def _dict_to_scoredpaper(d: Dict) -> ScoredPaper:
    # Convert nested paper + key_sentences back to dataclasses
    paper_d = d["paper"]
    paper = Paper(**paper_d)
    hits = []
    for h in d.get("key_sentences", []):
        hits.append(SentenceHit(**h))
    sp = ScoredPaper(
        paper=paper,
        cluster_id=int(d["cluster_id"]),
        paper_type=str(d["paper_type"]),
        relevance=float(d["relevance"]),
        novelty=float(d["novelty"]),
        novelty_rank=float(d["novelty_rank"]),
        bridge=float(d["bridge"]),
        bridge_rank=float(d["bridge_rank"]),
        key_sentences=hits,
    )
    # attach dynamic field
    setattr(sp, "axis", d.get("axis", axis_tag(paper.title, paper.abstract)))
    return sp
