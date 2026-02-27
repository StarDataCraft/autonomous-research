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
    "We confirm that in practice our scheme gives better performance in a larger range of scenarios.",
    "We establish an efficiency frontier.",
    "We derive an error-bounded objective.",
    "We establish bounds on the Kullback-Leibler divergence.",
    "For training, we propose a new parameterization.",
]

# ✅ Expanded limitations (A)
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
    "Existing analyses assume unrealistic conditions.",

    "However, the theoretical understanding remains underexplored.",
    "The theoretical understanding remains unclear.",
    "This remains an open problem.",
    "Existing analyses assume unrealistic conditions.",
    "Results are still far from state-of-the-art.",
    "The approach is limited to simple distributions.",
]

# ✅ Neutral / setup prototypes (B) + strengthened background patterns
_NEUTRAL_PROTOTYPES: List[str] = [
    "In this paper, we study …",
    "We consider the setting where …",
    "This paper focuses on …",
    "We investigate …",
    "We study the properties of the proposed model.",
    "We describe the background and setup.",
    "We outline the problem formulation.",
    "We present a comprehensive comparison.",
    "We provide a unified framework.",
    "Recently, this line of work has attracted attention.",
    "X has emerged as a promising approach.",
    "X has emerged as a promising tool for generating samples.",
    "Flow matching has emerged as a simulation-free alternative.",
    "Boltzmann Generators have emerged as a promising machine learning tool.",
    "Diffusion and flow-based models have become the state of the art.",
]

# -----------------------------
# Lexical signals
# -----------------------------
# These are "explicit negative evidence" markers.
# We'll use them as a hard gate for limitation to avoid false positives.
_NEGATIVE_EVIDENCE_PATTERNS: List[str] = [
    r"\bhowever\b",
    r"\byet\b",
    r"\bnevertheless\b",
    r"\bbut\b.*\b(remains|still)\b",
    r"\bunderexplored\b",
    r"\bunclear\b",
    r"\bunknown\b",
    r"\bopen problem\b",
    r"\blimitation(?:s)?\b",
    r"\blimited to\b",
    r"\black of\b",
    r"\black(?:s|ing)\b",
    r"\bstruggle(?:s|d)?\b",
    r"\bfail(?:s|ed|ure|ures)?\b",
    r"\bunstable\b",
    r"\bhinder(?:ed|s)?\b",
    r"\bfar from\b.*\bstate[- ]of[- ]the[- ]art\b",
    r"\bunrealistic\b",
    r"\bdoes not\b.*\bhold\b",
    r"\bcannot\b",
    r"\bunable\b",
    r"\bremains\b.*\b(unclear|unknown|underexplored|challenging|difficult)\b",
]

# "claim-ish" lexical cues (helps pull long technical claims out of relevance)
_CLAIM_VERB_PATTERNS: List[str] = [
    r"\bwe propose\b",
    r"\bwe present\b",
    r"\bwe introduce\b",
    r"\bwe show\b",
    r"\bwe derive\b",
    r"\bwe establish\b",
    r"\bwe provide\b",
    r"\bthe main contribution\b",
    r"\bour (method|approach|scheme)\b",
]

def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip().lower()

def _has_any(patterns: List[str], s: str) -> bool:
    s = _norm(s)
    if not s:
        return False
    return any(re.search(p, s) for p in patterns)

def _has_negative_evidence(s: str) -> bool:
    return _has_any(_NEGATIVE_EVIDENCE_PATTERNS, s)

def _has_claim_verb(s: str) -> bool:
    return _has_any(_CLAIM_VERB_PATTERNS, s)

# -----------------------------
# Embedding backend
# -----------------------------
@lru_cache(maxsize=4)
def _load_encoder(model_name: str):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_name)

@lru_cache(maxsize=32)
def _embed_texts(model_name: str, texts_tuple: Tuple[str, ...]) -> np.ndarray:
    enc = _load_encoder(model_name)
    embs = enc.encode(list(texts_tuple), normalize_embeddings=True, show_progress_bar=False)
    return np.asarray(embs, dtype=np.float32)

def _max_sim_to_prototypes(model_name: str, sentence: str, prototypes: List[str]) -> float:
    sent = (sentence or "").strip()
    if not sent:
        return 0.0
    sent_emb = _embed_texts(model_name, (sent,))[0]
    prot_embs = _embed_texts(model_name, tuple(prototypes))
    sims = prot_embs @ sent_emb  # cosine due to normalization
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
    tie_margin: float = 0.07,
    min_limit: float = 0.28,
    min_claim: float = 0.28,
    min_limit_soft: float = 0.24,
    min_claim_soft: float = 0.24,
) -> CueDecision:
    """
    Three-way semantic prototype classification + lexical tie-breakers.

    IMPORTANT CHANGE (D):
      - We DO NOT allow limitation/failure unless explicit negative evidence exists.
        This prevents false positives like "have emerged as a promising tool".

    Hard win:
      - claim if claim wins by margin and >= min_claim
      - limitation only if (negative evidence) AND limit wins by margin and >= min_limit
      - else relevance

    Soft win:
      - limitation if (negative evidence) AND limit is within tie_margin of best AND >= min_limit_soft
      - claim if (claim verb) AND claim is within tie_margin of best AND >= min_claim_soft
      - else relevance
    """
    sent = (sentence or "").strip()
    if not sent:
        return CueDecision(kind="relevance", scores={"claim": 0.0, "limit": 0.0, "neutral": 0.0})

    claim_score = _max_sim_to_prototypes(model_name, sent, _CLAIM_PROTOTYPES)
    limit_score = _max_sim_to_prototypes(model_name, sent, _LIMIT_PROTOTYPES)
    neutral_score = _max_sim_to_prototypes(model_name, sent, _NEUTRAL_PROTOTYPES)
    scores = {"claim": claim_score, "limit": limit_score, "neutral": neutral_score}

    best_label = max(scores.items(), key=lambda kv: kv[1])[0]
    best = scores[best_label]
    second = sorted(scores.values(), reverse=True)[1]

    has_neg = _has_negative_evidence(sent)

    # --- Hard wins ---
    if best_label == "claim" and best >= (second + margin) and best >= min_claim:
        return CueDecision(kind="claim", scores=scores)

    if best_label == "limit" and has_neg and best >= (second + margin) and best >= min_limit:
        return CueDecision(kind="limitation/failure", scores=scores)

    # --- Soft wins (tie-breakers) ---
    # limitation: requires explicit negative evidence
    if has_neg:
        if limit_score >= (best - tie_margin) and limit_score >= min_limit_soft:
            return CueDecision(kind="limitation/failure", scores=scores)

    # claim: can be pulled out of relevance by claim verbs
    if _has_claim_verb(sent):
        if claim_score >= (best - tie_margin) and claim_score >= min_claim_soft:
            return CueDecision(kind="claim", scores=scores)

    return CueDecision(kind="relevance", scores=scores)

def is_limitation_failure(
    sentence: str,
    *,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> bool:
    return classify_cue_sentence(sentence, model_name=model_name).kind == "limitation/failure"
