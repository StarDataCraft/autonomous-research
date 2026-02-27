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
]

# ✅ expanded limitations per your earlier request
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

    "However, the theoretical understanding remains underexplored.",
    "The theoretical understanding remains unclear.",
    "This remains an open problem.",
    "Existing analyses assume unrealistic conditions.",
    "Results are still far from state-of-the-art.",
    "The approach is limited to simple distributions.",
]

_NEUTRAL_PROTOTYPES: List[str] = [
    "In this paper, we study …",
    "In this paper, we study rectified flow models.",
    "We consider the setting where …",
    "This paper focuses on …",
    "We investigate …",
    "We study the properties of the proposed model.",
    "We describe the background and setup.",
    "We outline the problem formulation.",
]

# -----------------------------
# Lexical signals
# -----------------------------
_NEGATIVE_EVIDENCE_PATTERNS: List[str] = [
    r"\bhowever\b",
    r"\byet\b",
    r"\bstill\b",
    r"\bunderexplored\b",
    r"\bunclear\b",
    r"\bunknown\b",
    r"\bopen problem\b",
    r"\bchallenge(?:s)?\b",
    r"\bdifficult(?:y)?\b",
    r"\bhinder(?:ed|s)?\b",
    r"\bfail(?:s|ed|ure|ures)?\b",
    r"\bunstable\b",
    r"\bfar from\b.*\bstate[- ]of[- ]the[- ]art\b",
    r"\bassum(?:e|es|ed)\b.*\bunrealistic\b",
    r"\blimited to\b",
    r"\black of\b",
    r"\black(?:s|ing)\b",
    r"\bdoes not\b",
    r"\bcannot\b",
    r"\bunable\b",
    r"\bremains\b.*\b(unclear|unknown|underexplored|challenging|difficult)\b",
]

# “claim-ish” lexical cues (用于近胜判定，避免 neutral 霸榜)
_CLAIM_VERB_PATTERNS: List[str] = [
    r"\bwe propose\b",
    r"\bwe present\b",
    r"\bwe introduce\b",
    r"\bwe show\b",
    r"\bwe derive\b",
    r"\bwe establish\b",
    r"\bwe provide\b.*\binsight",
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
    sims = prot_embs @ sent_emb  # normalized => cosine
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
    # ✅ 新增：近胜判定阈值（tie-breaker）
    tie_margin: float = 0.07,
    # ✅ semantic 最低门槛（hard win）
    min_limit: float = 0.28,
    min_claim: float = 0.28,
    # ✅ lexical-tie 最低门槛（soft win）
    min_limit_soft: float = 0.24,
    min_claim_soft: float = 0.24,
) -> CueDecision:
    """
    Three-way semantic prototype classification + lexical tie-breakers.

    Hard win:
      - limitation if limit wins by margin and >= min_limit
      - claim if claim wins by margin and >= min_claim
      - else relevance

    Soft win (NEW):
      - if sentence has negative-evidence lexical cues and limit_score is close to best (within tie_margin),
        and limit_score >= min_limit_soft => limitation
      - if sentence has claim-verb cues and claim_score is close to best (within tie_margin),
        and claim_score >= min_claim_soft => claim

    This fixes:
      - "However ... remains underexplored" (limit not top but close + negative evidence)
      - "far from state-of-the-art" (limit top but wins slightly under margin)
    """
    sent = (sentence or "").strip()
    if not sent:
        return CueDecision(kind="relevance", scores={"claim": 0.0, "limit": 0.0, "neutral": 0.0})

    claim_score = _max_sim_to_prototypes(model_name, sent, _CLAIM_PROTOTYPES)
    limit_score = _max_sim_to_prototypes(model_name, sent, _LIMIT_PROTOTYPES)
    neutral_score = _max_sim_to_prototypes(model_name, sent, _NEUTRAL_PROTOTYPES)
    scores = {"claim": claim_score, "limit": limit_score, "neutral": neutral_score}

    # --- Hard win logic ---
    best_label = max(scores.items(), key=lambda kv: kv[1])[0]
    best = scores[best_label]
    second = sorted(scores.values(), reverse=True)[1]

    if best_label == "limit" and best >= (second + margin) and best >= min_limit:
        return CueDecision(kind="limitation/failure", scores=scores)
    if best_label == "claim" and best >= (second + margin) and best >= min_claim:
        return CueDecision(kind="claim", scores=scores)

    # --- Soft win tie-breakers (NEW) ---
    # 1) limitation: negative-evidence + near-best
    if _has_negative_evidence(sent):
        # near-best means: limit is within tie_margin of the best score
        if limit_score >= (best - tie_margin) and limit_score >= min_limit_soft:
            return CueDecision(kind="limitation/failure", scores=scores)

    # 2) claim: claim-verb + near-best
    if _has_claim_verb(sent):
        if claim_score >= (best - tie_margin) and claim_score >= min_claim_soft:
            return CueDecision(kind="claim", scores=scores)

    # Otherwise
    return CueDecision(kind="relevance", scores=scores)

def is_limitation_failure(
    sentence: str,
    *,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> bool:
    return classify_cue_sentence(sentence, model_name=model_name).kind == "limitation/failure"
