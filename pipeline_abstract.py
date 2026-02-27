# pipeline_abstract.py
# Free / No-LLM abstract-only pipeline:
# - TF-IDF embeddings (pure python)
# - KMeans clustering (pure python, small-n friendly)
# - Relevance / novelty / bridge scoring
# - Cue-based sentence mining (claim vs limitation/failure vs comparison/negative)
# - Paper type + axis tagging (method/theory/empirical + optimizer/objective/theory/eval/applications)

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import math
import random
import re
from collections import Counter, defaultdict
from datetime import datetime


# -----------------------------
# Data structures
# -----------------------------

@dataclass
class Paper:
    pid: str
    title: str
    abstract: str
    updated: str = ""
    pdf_url: str = ""
    category: str = ""
    authors: List[str] = field(default_factory=list)

    # computed
    paper_type: str = ""   # method/theory/empirical
    axis: str = ""         # optimizer/dynamics, objective/path, theory/bounds, evaluation/benchmark, applications
    relevance: float = 0.0
    novelty: float = 0.0
    novelty_rank: float = 0.0
    bridge: float = 0.0
    bridge_rank: float = 0.0

    claim_sentences: List[str] = field(default_factory=list)
    contradiction_sentences: List[str] = field(default_factory=list)  # limitation/failure/comparison only


@dataclass
class Cluster:
    cid: int
    paper_ids: List[str]
    keywords: List[str] = field(default_factory=list)
    centroid: Dict[int, float] = field(default_factory=dict)  # sparse tfidf vector
    counts_by_type: Dict[str, int] = field(default_factory=dict)
    representative_id: str = ""


@dataclass
class PipelineResult:
    query: str
    papers: List[Paper]
    clusters: List[Cluster]
    selected_ids: List[str]  # MMR diversified list (paper ids)
    # convenience indices
    id2paper: Dict[str, Paper] = field(default_factory=dict)


# -----------------------------
# Text utilities
# -----------------------------

_WORD_RE = re.compile(r"[a-zA-Z0-9]+")
_SENT_SPLIT_RE = re.compile(r"(?<=[\.\?\!])\s+")

STOPWORDS = set("""
a an the and or of to in on for with without by as is are was were be been being from
this that these those it its we our you your they their can could would should may might
into over under between among via using use used
""".split())


def normalize_text(s: str) -> str:
    s = s.replace("\n", " ").strip()
    s = re.sub(r"\s+", " ", s)
    return s


def tokenize(s: str) -> List[str]:
    s = s.lower()
    toks = _WORD_RE.findall(s)
    toks = [t for t in toks if t not in STOPWORDS and not t.isdigit()]
    return toks


def split_sentences(text: str) -> List[str]:
    text = normalize_text(text)
    if not text:
        return []
    # Conservative sentence split; abstracts are short.
    parts = _SENT_SPLIT_RE.split(text)
    out = []
    for p in parts:
        p = p.strip()
        if len(p) < 20:
            continue
        out.append(p)
    return out


# -----------------------------
# TF-IDF embeddings (pure python)
# -----------------------------

@dataclass
class TfidfSpace:
    vocab: Dict[str, int]
    idf: List[float]


def build_tfidf_space(texts: List[str], min_df: int = 1, max_vocab: int = 8000) -> TfidfSpace:
    doc_freq = Counter()
    docs_toks = []
    for t in texts:
        toks = set(tokenize(t))
        docs_toks.append(toks)
        for w in toks:
            doc_freq[w] += 1

    # filter by min_df
    items = [(w, df) for w, df in doc_freq.items() if df >= min_df]
    # sort by df desc
    items.sort(key=lambda x: (-x[1], x[0]))
    items = items[:max_vocab]

    vocab = {w: i for i, (w, _) in enumerate(items)}
    N = max(1, len(texts))
    idf = [0.0] * len(vocab)
    for w, i in vocab.items():
        df = doc_freq[w]
        # smooth
        idf[i] = math.log((N + 1) / (df + 1)) + 1.0
    return TfidfSpace(vocab=vocab, idf=idf)


def tfidf_vector(space: TfidfSpace, text: str) -> Dict[int, float]:
    toks = tokenize(text)
    if not toks:
        return {}
    tf = Counter(toks)
    vec: Dict[int, float] = {}
    for w, c in tf.items():
        if w not in space.vocab:
            continue
        i = space.vocab[w]
        vec[i] = (c / len(toks)) * space.idf[i]
    return l2_normalize(vec)


def l2_normalize(vec: Dict[int, float]) -> Dict[int, float]:
    if not vec:
        return {}
    s = math.sqrt(sum(v * v for v in vec.values()))
    if s <= 1e-12:
        return vec
    return {k: v / s for k, v in vec.items()}


def sparse_dot(a: Dict[int, float], b: Dict[int, float]) -> float:
    if not a or not b:
        return 0.0
    # iterate smaller
    if len(a) > len(b):
        a, b = b, a
    return sum(v * b.get(k, 0.0) for k, v in a.items())


def cosine(a: Dict[int, float], b: Dict[int, float]) -> float:
    # vectors already L2-normalized
    return sparse_dot(a, b)


def sparse_add(a: Dict[int, float], b: Dict[int, float], alpha: float = 1.0) -> Dict[int, float]:
    out = dict(a)
    for k, v in b.items():
        out[k] = out.get(k, 0.0) + alpha * v
    return out


def sparse_scale(a: Dict[int, float], alpha: float) -> Dict[int, float]:
    return {k: v * alpha for k, v in a.items()}


def sparse_mean(vecs: List[Dict[int, float]]) -> Dict[int, float]:
    if not vecs:
        return {}
    out: Dict[int, float] = {}
    for v in vecs:
        for k, val in v.items():
            out[k] = out.get(k, 0.0) + val
    out = sparse_scale(out, 1.0 / max(1, len(vecs)))
    return l2_normalize(out)


# -----------------------------
# Simple KMeans (small n)
# -----------------------------

def kmeans(vectors: List[Dict[int, float]], k: int, seed: int = 13, iters: int = 25) -> Tuple[List[int], List[Dict[int, float]]]:
    n = len(vectors)
    if n == 0:
        return [], []
    k = max(1, min(k, n))
    rnd = random.Random(seed)

    # init: pick k random points
    centroids = [vectors[i] for i in rnd.sample(range(n), k)]
    labels = [0] * n

    for _ in range(iters):
        changed = 0
        # assign
        for i, v in enumerate(vectors):
            best_c, best_s = 0, -1e9
            for c, cent in enumerate(centroids):
                s = cosine(v, cent)
                if s > best_s:
                    best_s = s
                    best_c = c
            if labels[i] != best_c:
                labels[i] = best_c
                changed += 1

        # recompute
        new_centroids = []
        for c in range(k):
            members = [vectors[i] for i in range(n) if labels[i] == c]
            if not members:
                new_centroids.append(vectors[rnd.randrange(n)])
            else:
                new_centroids.append(sparse_mean(members))

        centroids = new_centroids
        if changed == 0:
            break

    return labels, centroids


# -----------------------------
# Paper type + axis tagging
# -----------------------------

TYPE_KEYWORDS = {
    "theory": ["theory", "theoretical", "guarantee", "bound", "proof", "sample complexity", "non-asymptotic", "minimax", "convergence"],
    "empirical": ["experiment", "empirical", "benchmark", "fid", "cifar", "imagenet", "mnist", "results", "evaluation", "ablation"],
    "method": ["we propose", "we introduce", "new method", "algorithm", "framework", "approach", "objective", "loss", "training"],
}

AXIS_KEYWORDS = {
    "optimizer/dynamics": ["optimizer", "adam", "sgd", "learning rate", "lr", "warmup", "gradient clipping", "micro-batch", "batch size", "edge of stability", "dynamics"],
    "objective/path": ["flow matching", "rectified flow", "stochastic interpolant", "generator matching", "bregman", "time distribution", "reweighting", "probability path", "trajectory"],
    "theory/bounds": ["kl divergence", "bound", "guarantee", "sample complexity", "rademacher", "non-asymptotic", "proof", "minimax"],
    "evaluation/benchmark": ["benchmark", "fid", "nfe", "runtime", "low-resource", "efficiency", "fidelity", "corrupted", "robust", "stress", "suite"],
    "applications": ["molecular", "inverse problem", "alignment", "guidance", "text-to-image", "stable diffusion", "plug-and-play", "deployment"],
}


def keyword_hit_score(text: str, keys: List[str]) -> int:
    t = text.lower()
    return sum(1 for k in keys if k in t)


def classify_paper_type(title: str, abstract: str) -> str:
    t = (title + " " + abstract).lower()
    scores = {k: keyword_hit_score(t, v) for k, v in TYPE_KEYWORDS.items()}
    # prefer theory if strong
    best = max(scores.items(), key=lambda x: x[1])[0]
    # tie-break: if none hit, default method
    if scores[best] == 0:
        return "method"
    return best


def classify_axis(title: str, abstract: str) -> str:
    t = (title + " " + abstract).lower()
    scores = {k: keyword_hit_score(t, v) for k, v in AXIS_KEYWORDS.items()}
    best = max(scores.items(), key=lambda x: x[1])[0]
    if scores[best] == 0:
        # fallback
        return "objective/path"
    return best


# -----------------------------
# Cue-based sentence mining
# -----------------------------

CLAIM_CUES = [
    "we propose", "we introduce", "we present", "we show", "we demonstrate", "we provide", "we develop", "we derive", "we obtain"
]

LIMIT_FAIL_CUES = [
    "limitation", "limited", "fails", "failure", "breaks down", "does not", "cannot", "unstable", "diverge", "divergence",
    "collapse", "mode collapse", "nan", "sensitive to", "depends on", "requires", "assumption", "trade-off", "at the cost of"
]

COMPARISON_NEG_CUES = [
    "however", "in contrast", "yet", "nevertheless", "although", "worse", "inferior", "underperform", "at the cost", "trade-off"
]


def sentence_bucket(sent: str) -> str:
    s = sent.lower()
    # limitation/failure
    if any(c in s for c in LIMIT_FAIL_CUES):
        return "limitation"
    # comparison/negative (only counts as contradiction if it is a real comparison-ish sentence)
    if any(c in s for c in COMPARISON_NEG_CUES):
        return "comparison"
    if any(c in s for c in CLAIM_CUES):
        return "claim"
    return "neutral"


def extract_sentences(abstract: str, max_claim: int = 3, max_contra: int = 3) -> Tuple[List[str], List[str]]:
    sents = split_sentences(abstract)
    claims: List[str] = []
    contras: List[str] = []

    for s in sents:
        b = sentence_bucket(s)
        if b == "claim":
            claims.append(s)
        elif b in ("limitation", "comparison"):
            contras.append(s)

    # keep most “cue-dense” ones (simple heuristic)
    def cue_density(s: str) -> int:
        sl = s.lower()
        return sum(int(c in sl) for c in LIMIT_FAIL_CUES + COMPARISON_NEG_CUES + CLAIM_CUES)

    claims.sort(key=cue_density, reverse=True)
    contras.sort(key=cue_density, reverse=True)

    return claims[:max_claim], contras[:max_contra]


# -----------------------------
# MMR (sentences / papers)
# -----------------------------

def mmr_select(
    items: List[str],
    item_vecs: List[Dict[int, float]],
    query_vec: Dict[int, float],
    k: int,
    lambda_rel: float = 0.65
) -> List[int]:
    """
    Select k indices via MMR:
    score(i) = lambda*sim(i,query) - (1-lambda)*max_sim(i, selected)
    """
    n = len(items)
    if n == 0 or k <= 0:
        return []
    k = min(k, n)
    selected: List[int] = []
    remaining = set(range(n))

    rel = [cosine(item_vecs[i], query_vec) for i in range(n)]

    while remaining and len(selected) < k:
        best_i, best_s = None, -1e9
        for i in remaining:
            if not selected:
                div = 0.0
            else:
                div = max(cosine(item_vecs[i], item_vecs[j]) for j in selected)
            s = lambda_rel * rel[i] - (1.0 - lambda_rel) * div
            if s > best_s:
                best_s = s
                best_i = i
        selected.append(best_i)  # type: ignore
        remaining.remove(best_i)  # type: ignore
    return selected


# -----------------------------
# Keywords for clusters
# -----------------------------

def top_keywords_for_cluster(space: TfidfSpace, paper_vecs: List[Dict[int, float]], paper_ids: List[str], id2idx: Dict[str, int], topn: int = 10) -> List[str]:
    # Sum tfidf vectors
    agg: Dict[int, float] = {}
    for pid in paper_ids:
        v = paper_vecs[id2idx[pid]]
        for k, val in v.items():
            agg[k] = agg.get(k, 0.0) + val
    # top indices
    top = sorted(agg.items(), key=lambda x: x[1], reverse=True)[:topn]
    # invert vocab
    inv_vocab = {i: w for w, i in space.vocab.items()}
    return [inv_vocab[i] for i, _ in top if i in inv_vocab]


# -----------------------------
# Penalties (tutorial / generic) for bridge
# -----------------------------

TUTORIAL_WORDS = ["tutorial", "introduction", "survey", "overview", "expository"]

def tutorial_penalty(title: str, abstract: str) -> float:
    t = (title + " " + abstract).lower()
    return 0.35 if any(w in t for w in TUTORIAL_WORDS) else 0.0

def generic_penalty(abstract: str) -> float:
    """
    Very small heuristic: penalize abstracts with too many generic verbs and too few technical nouns.
    Keeps it mild (0~0.25).
    """
    t = abstract.lower()
    generic = ["we propose", "we present", "we show", "this paper", "in this work", "novel", "significant", "various"]
    tech = ["kl", "fid", "ode", "sde", "variance", "gradient", "flow matching", "diffusion", "rectified", "generator matching"]
    g = sum(1 for x in generic if x in t)
    te = sum(1 for x in tech if x in t)
    if g <= 2:
        return 0.0
    # more generic than technical => penalty
    ratio = (g - te) / max(1, g)
    return max(0.0, min(0.25, 0.25 * ratio))


# -----------------------------
# Core pipeline
# -----------------------------

def build_pipeline(
    query: str,
    papers: List[Paper],
    k_clusters: Optional[int] = None,
    seed: int = 13,
) -> PipelineResult:
    """
    Input: papers with title+abstract already filled.
    Output: PipelineResult with clustering + scores + extracted sentences + MMR paper list.
    """
    # Enrich: type/axis + key sentences
    for p in papers:
        p.paper_type = classify_paper_type(p.title, p.abstract)
        p.axis = classify_axis(p.title, p.abstract)
        p.claim_sentences, p.contradiction_sentences = extract_sentences(p.abstract)

    id2paper = {p.pid: p for p in papers}

    # Build TF-IDF space on (title + abstract)
    texts = [f"{p.title}\n{p.abstract}" for p in papers]
    space = build_tfidf_space(texts, min_df=1, max_vocab=6000)
    vecs = [tfidf_vector(space, t) for t in texts]

    # Query vector for relevance
    qvec = tfidf_vector(space, query)

    # Decide cluster count
    n = len(papers)
    if k_clusters is None:
        # small heuristic
        k_clusters = max(2, min(8, int(round(math.sqrt(max(1, n))))))

    labels, centroids = kmeans(vecs, k=k_clusters, seed=seed, iters=30)

    # Group cluster members
    clusters_map: Dict[int, List[str]] = defaultdict(list)
    for i, p in enumerate(papers):
        clusters_map[labels[i]].append(p.pid)

    # Relevance: sim(paper, query)
    for i, p in enumerate(papers):
        p.relevance = cosine(vecs[i], qvec)

    # Cluster centroid similarities for bridge & novelty
    # novelty: 1 - max similarity to OTHER cluster centroids (weighted by relevance later)
    # bridge: high top2 centroid sim and small gap => connector
    for i, p in enumerate(papers):
        c_self = labels[i]
        sims = []
        for c, cent in enumerate(centroids):
            sims.append((c, cosine(vecs[i], cent)))
        sims.sort(key=lambda x: x[1], reverse=True)

        # novelty
        max_other = 0.0
        for c, s in sims:
            if c != c_self:
                max_other = max(max_other, s)
        p.novelty = max(0.0, 1.0 - max_other)

        # bridge
        if len(sims) >= 2:
            top1c, top1s = sims[0]
            top2c, top2s = sims[1]
            gap = max(0.0, top1s - top2s)
            # bridge prefers: high (top1+top2)/2 and small gap
            p.bridge = (top1s + top2s) / 2.0 - 0.35 * gap
            p.bridge = max(0.0, p.bridge)
        else:
            p.bridge = 0.0

        # apply penalties to bridge (tutorial/generic)
        pen = tutorial_penalty(p.title, p.abstract) + generic_penalty(p.abstract)
        p.bridge = max(0.0, p.bridge * (1.0 - pen))

        # ranks (you can tune)
        # novelty_rank: novelty * (0.35 + 0.65*relevance) to avoid off-topic outliers
        p.novelty_rank = p.novelty * (0.35 + 0.65 * p.relevance)
        # bridge_rank: bridge * (0.25 + 0.75*relevance) to keep connectors in-topic
        p.bridge_rank = p.bridge * (0.25 + 0.75 * p.relevance)

    # Build Cluster objects
    id2idx = {p.pid: i for i, p in enumerate(papers)}
    clusters: List[Cluster] = []
    for cid, pids in sorted(clusters_map.items(), key=lambda x: x[0]):
        # cluster centroid
        cent = centroids[cid] if cid < len(centroids) else {}
        # type counts
        counts = Counter(id2paper[pid].paper_type for pid in pids)
        # representative: max relevance within cluster
        rep = max(pids, key=lambda pid: id2paper[pid].relevance) if pids else ""
        kw = top_keywords_for_cluster(space, vecs, pids, id2idx, topn=10)
        clusters.append(
            Cluster(
                cid=cid,
                paper_ids=pids,
                keywords=kw,
                centroid=cent,
                counts_by_type=dict(counts),
                representative_id=rep,
            )
        )

    # MMR diversified paper selection (for "Selected papers")
    # diversify on paper vectors, relevant to query
    k_sel = min(24, max(8, int(round(math.sqrt(max(1, n))) * 6)))
    sel_idx = mmr_select([p.pid for p in papers], vecs, qvec, k=k_sel, lambda_rel=0.70)
    selected_ids = [papers[i].pid for i in sel_idx]

    return PipelineResult(
        query=query,
        papers=papers,
        clusters=clusters,
        selected_ids=selected_ids,
        id2paper=id2paper,
    )


# -----------------------------
# Leaderboards (three novelty boards)
# -----------------------------

BREAKTHROUGH_AXES = {"objective/path", "theory/bounds", "evaluation/benchmark"}
CONFOUNDER_AXES = {"optimizer/dynamics"}


def novelty_leaderboards(result: PipelineResult, topn: int = 10) -> Dict[str, List[Paper]]:
    papers = result.papers

    # In-topic breakthrough novelty: restrict axes
    breakthrough = [p for p in papers if p.axis in BREAKTHROUGH_AXES]
    breakthrough.sort(key=lambda p: p.novelty_rank, reverse=True)

    # Confounder novelty: optimizer/dynamics
    confounder = [p for p in papers if p.axis in CONFOUNDER_AXES]
    confounder.sort(key=lambda p: p.novelty_rank, reverse=True)

    # Bridge novelty: connectors that are also a bit novel (optional)
    bridge = sorted(papers, key=lambda p: p.bridge_rank, reverse=True)

    return {
        "novelty_in_topic": breakthrough[:topn],
        "novelty_confounders": confounder[:topn],
        "bridge_leaderboard": bridge[:topn],
    }


# -----------------------------
# Sentence-level headline points (MMR on sentences)
# -----------------------------

def cluster_headline_points(
    result: PipelineResult,
    cluster: Cluster,
    max_points: int = 5,
    lambda_rel: float = 0.65,
) -> List[Tuple[str, str]]:
    """
    Returns list of (sentence, paper_title) for the cluster,
    using sentence-level MMR against cluster centroid to avoid duplicates.
    """
    # Build a local tfidf space just for sentences? We reuse global by rebuilding from cluster sentences only.
    # To keep pure-python and simple, we do a small local TF-IDF for this cluster.
    papers = [result.id2paper[pid] for pid in cluster.paper_ids]
    sentences = []
    sentence_meta = []  # (paper_title, pid)
    for p in papers:
        sents = split_sentences(p.abstract)
        # keep only informative lines
        for s in sents:
            if len(s) < 30:
                continue
            sentences.append(s)
            sentence_meta.append((p.title, p.pid))

    if not sentences:
        return []

    local_texts = sentences + [f"centroid {cluster.cid}"]
    space = build_tfidf_space(local_texts, min_df=1, max_vocab=4000)
    sent_vecs = [tfidf_vector(space, s) for s in sentences]
    # centroid proxy text: use top keywords
    centroid_text = " ".join(cluster.keywords) if cluster.keywords else result.query
    centroid_vec = tfidf_vector(space, centroid_text)

    sel = mmr_select(sentences, sent_vecs, centroid_vec, k=min(max_points, len(sentences)), lambda_rel=lambda_rel)
    out: List[Tuple[str, str]] = []
    seen = set()
    for i in sel:
        sent = sentences[i].strip()
        # de-duplicate near-identical by string hashing
        key = sent.lower()[:120]
        if key in seen:
            continue
        seen.add(key)
        out.append((sent, sentence_meta[i][0]))
        if len(out) >= max_points:
            break
    return out

# -----------------------------
# Backward-compatible wrapper
# -----------------------------

def run_abstract_pipeline(
    query: str,
    papers_payload,
    k_clusters: int | None = None,
    seed: int = 13,
):
    """
    Backward-compatible entrypoint for older streamlit_app.py.

    Parameters
    ----------
    query : str
    papers_payload : list[dict] OR list[Paper]
        Each dict should contain at least: title, abstract
        Optional: id/pid, updated, pdf_url, category, authors
    k_clusters : int | None
    seed : int

    Returns
    -------
    dict with:
      - result: PipelineResult
      - papers: List[Paper]
      - clusters: List[Cluster]
      - selected_ids: List[str]
      - leaderboards: dict (novelty_in_topic / novelty_confounders / bridge_leaderboard)
    """
    # If caller already passed Paper objects
    if papers_payload and hasattr(papers_payload[0], "title") and hasattr(papers_payload[0], "abstract"):
        papers = papers_payload
        # ensure pid exists
        for i, p in enumerate(papers):
            if not getattr(p, "pid", None):
                p.pid = f"p{i+1}"
    else:
        # Convert list[dict] -> list[Paper]
        papers = []
        for i, d in enumerate(papers_payload or []):
            title = (d.get("title") or "").strip()
            abstract = (d.get("abstract") or "").strip()
            if not title or not abstract:
                # skip invalid entries
                continue

            pid = (d.get("pid") or d.get("id") or d.get("paper_id") or f"p{i+1}")
            authors = d.get("authors") or []
            if isinstance(authors, str):
                # sometimes "A, B, C"
                authors = [a.strip() for a in authors.split(",") if a.strip()]

            papers.append(
                Paper(
                    pid=str(pid),
                    title=title,
                    abstract=abstract,
                    updated=str(d.get("updated") or d.get("updated_at") or d.get("published") or ""),
                    pdf_url=str(d.get("pdf_url") or d.get("pdf") or d.get("url") or ""),
                    category=str(d.get("category") or ""),
                    authors=authors if isinstance(authors, list) else [],
                )
            )

    result = build_pipeline(query=query, papers=papers, k_clusters=k_clusters, seed=seed)

    # leaderboards
    lbs = novelty_leaderboards(result, topn=10)

    return {
        "result": result,
        "papers": result.papers,
        "clusters": result.clusters,
        "selected_ids": result.selected_ids,
        "leaderboards": lbs,
    }
