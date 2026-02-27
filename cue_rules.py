# cue_rules.py
from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Tuple

import numpy as np

# -----------------------------
# Prototypes (Semantic)
# -----------------------------
# Claim prototypes (can be extended)
_CLAIM_PROTOTYPES: List[str] = [
    "We propose a new method.",
    "We present a new approach.",
    "We introduce a novel algorithm.",
    "Our method achieves better performance.",
    "This paper provides theoretical insights.",
    "We show that our approach is effective.",
    "The main contribution of this paper is to provide a new framework.",
    "We demonstrate improved stability and efficiency.",
    "In contrast, the approach proposed here gives batch size invariance without this assumption.",
    "This approach provides improved robustness.",
]

# Limitation/failure prototypes (✅ expanded per your request)
_LIMIT_PROTOTYPES: List[str] = [
    "However, the method has limitations.",
    "The approach is unstable in practice.",
    "This remains challenging.",
    "The problem remains underexplored.",
    "Existing methods struggle in this regime.",
    "There is a lack of theoretical understanding.",
    "The results are far from state-of-the-art.",
    "The approach is limited to simple distributions.",
    "The method fails under certain distributions.",
    "There is a trade-off between efficiency and fidelity.",

    # ✅ your requested additions (exact)
    "However, the theoretical understanding remains underexplored.",
    "The theoretical understanding remains unclear.",
    "This remains an open problem.",
    "Existing analyses assume unrealistic conditions.",
    "Results are still far from state-of-the-art.",
    "The approach is limited to simple distributions.",
]

# Neutral/setup prototypes (✅ new)
_NEUTRAL_PROTOTYPES: List[str] = [
    "In this paper, we study rectified flow models.",
    "In this paper, we study the problem setting.",
    "We consider the setting where the data is generated from a distribution.",
    "This paper focuses on analyzing the method.",
    "We investigate the behavior of the algorithm.",
    "We study the properties of the proposed model.",
    "We describe the background and setup.",
    "We outline the problem formulation.",
]

# -----------------------------
# Negative evidence gate (lightweight safety rail)
# Only applied when semantic classifier predicts limitation/failure.
# -----------------------------
_NEGATIVE_EVIDENCE_PATTERNS: List[str] = [
    r"\blimit(?:ation|ations)?\b",
    r"\bunderexplored\b",
    r"\bunclear\b",
    r"\bunknown\b",
    r"\bopen problem\b",
    r"\bchallenge(?:s)?\b",
    r"\bdifficult(?:y)?\b",
    r"\bhinder(?:ed|s)?\b",
    r"\bfail(?:s|ed|ure)?\b",
    r"\bunstable\b",
    r"\bfar from\b.*\bstate[- ]of[- ]the[- ]art\b",
    r"\bassume\b.*\bunrealistic\b",
    r"\blimited to\b",
    r"\black of\b",
    r"\bdoes not\b",
    r"\bcannot\b",
    r"\bunable\b",
    r"\byet\b.*\brequires\b",
    r"\bremains\b.*\b(unclear|unknown|underexplored|challenging|difficult)\b",
]


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip().lower()


def _has_negative_evidence(s: str) -> bool:
    s = _norm(s)
    if not s:
        return False
    return any(re.search(pat, s) for pat in _NEGATIVE_EVIDENCE_PATTERNS)


# -----------------------------
# Embedding backend
# -----------------------------
def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
    return float(np.dot(a, b) / denom)


@lru_cache(maxsize=4)
def _load_encoder(model_name: str):
    """
    Cached SentenceTransformer.
    Keep this module Streamlit-free; cache via lru_cache.
    """
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


@lru_cache(maxsize=16)
def _embed_texts(model_name: str, texts_tuple: Tuple[str, ...]) -> np.ndarray:
    enc = _load_encoder(model_name)
    # normalize embeddings improves cosine stability
    embs = enc.encode(list(texts_tuple), normalize_embeddings=True, show_progress_bar=False)
    return np.asarray(embs, dtype=np.float32)


def _max_sim_to_prototypes(model_name: str, sentence: str, prototypes: List[str]) -> float:
    sent = (sentence or "").strip()
    if not sent:
        return 0.0

    sent_emb = _embed_texts(model_name, (sent,))[0]
    prot_embs = _embed_texts(model_name, tuple(prototypes))

    # since embeddings are normalized, cosine = dot
    sims = prot_embs @ sent_emb
    return float(np.max(sims)) if sims.size else 0.0


@dataclass(frozen=True)
class CueDecision:
    kind: str  # "claim" | "limitation/failure" | "relevance"
    scores: Dict[str, float]  # {"claim":..., "limit":..., "neutral":...}


def classify_cue_sentence(
    sentence: str,
    *,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    margin: float = 0.04,
    min_limit: float = 0.28,
    min_claim: float = 0.28,
) -> CueDecision:
    """
    Three-way semantic prototype classification:

      - limitation/failure if limit_score is clearly the winner (by margin) and above min_limit
      - claim if claim_score is clearly the winner (by margin) and above min_claim
      - otherwise relevance

    Then apply a lightweight negative-evidence gate ONLY for limitation:
      if predicted limitation but no negative evidence, downgrade to relevance.
    """
    sent = (sentence or "").strip()
    if not sent:
        return CueDecision(kind="relevance", scores={"claim": 0.0, "limit": 0.0, "neutral": 0.0})

    claim_score = _max_sim_to_prototypes(model_name, sent, _CLAIM_PROTOTYPES)
    limit_score = _max_sim_to_prototypes(model_name, sent, _LIMIT_PROTOTYPES)
    neutral_score = _max_sim_to_prototypes(model_name, sent, _NEUTRAL_PROTOTYPES)

    scores = {"claim": claim_score, "limit": limit_score, "neutral": neutral_score}

    # decide winner with margin
    best_label = max(scores.items(), key=lambda kv: kv[1])[0]
    best = scores[best_label]
    second = sorted(scores.values(), reverse=True)[1]

    kind = "relevance"
    if best_label == "limit" and best >= (second + margin) and best >= min_limit:
        kind = "limitation/failure"
    elif best_label == "claim" and best >= (second + margin) and best >= min_claim:
        kind = "claim"
    else:
        kind = "relevance"

    # negative-evidence gate (only for limitation)
    if kind == "limitation/failure" and not _has_negative_evidence(sent):
        kind = "relevance"

    return CueDecision(kind=kind, scores=scores)


def is_limitation_failure(
    sentence: str,
    *,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> bool:
    """
    Convenience boolean for synthesis filtering.
    Uses semantic classifier + negative-evidence gate.
    """
    d = classify_cue_sentence(sentence, model_name=model_name)
    return d.kind == "limitation/failure"
