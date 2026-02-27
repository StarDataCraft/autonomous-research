from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, List, Tuple


# -----------------------------
# Existing helpers (keep)
# -----------------------------

def trim_block(s: str) -> str:
    return "\n".join([line.rstrip() for line in s.strip("\n").splitlines()]).strip()


def bulletify(items: Iterable[str]) -> str:
    items = [x for x in items if x]
    if not items:
        return "- (none)"
    return "\n".join([f"- {x}" for x in items])


def safe_join(parts: Iterable[str], sep: str = "\n\n") -> str:
    parts = [p.strip() for p in parts if p and p.strip()]
    return sep.join(parts)


# -----------------------------
# New: sentence splitting + cue extraction
# -----------------------------

# (A) Claim cues: highlight, NOT contradictions
_CLAIM_CUES = [
    r"\bwe\s+propose\b",
    r"\bwe\s+present\b",
    r"\bwe\s+introduce\b",
    r"\bwe\s+show\b",
    r"\bwe\s+demonstrate\b",
    r"\bwe\s+derive\b",
    r"\bwe\s+provide\b",
    r"\bwe\s+establish\b",
    r"\bthis\s+paper\s+(proposes|presents|introduces)\b",
    r"\bour\s+(method|approach|framework|analysis)\b",
]

# (B) Limitation / failure cues: contradictions/limits/failure modes
_LIMIT_CUES = [
    r"\blimitation(s)?\b",
    r"\blimited\b",
    r"\brestrict(ed|ion|ions)?\b",
    r"\bunderexplored\b",
    r"\bremains?\s+(unknown|unclear|challenging)\b",
    r"\bdoes\s+not\b",
    r"\bcannot\b",
    r"\bfails?\b",
    r"\bbreaks?\s+down\b",
    r"\bunstable\b",
    r"\bdiverg(e|es|ed|ence)\b",
    r"\bcollapse(s|d)?\b",
    r"\bmode\s+collapse\b",
    r"\bna[nN]\b",
    r"\bsensitive\s+to\b",
    r"\bdepends?\s+on\b",
    r"\brequires?\b",
    r"\bassumption(s)?\b",
    r"\bat\s+the\s+cost\s+of\b",
    r"\btrade[-\s]?off\b",
    r"\bhowever\b",
    r"\byet\b",
    r"\bnevertheless\b",
    r"\bin\s+contrast\b",
    r"\bon\s+the\s+other\s+hand\b",
    r"\bworse\b",
    r"\binferior\b",
    r"\bunderperform(s|ed)?\b",
]

# Optional: extra “comparison-ish” cue. Helps when abstract has "however" but no explicit failure word.
_COMPARISON_CUES = [
    r"\bcompared\s+to\b",
    r"\bversus\b",
    r"\bbaseline(s)?\b",
    r"\boutperform(s|ed)?\b",
    r"\bimprov(e|es|ed)\b",
]


@dataclass(frozen=True)
class CueHit:
    sentence: str
    kind: str  # "claim" or "limit"
    cue: str   # regex that matched


def _normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def split_sentences(text: str) -> List[str]:
    """
    Sentence splitter optimized for abstracts:
    - splits on . ? ! and newlines
    - keeps reasonably long segments
    """
    if not text:
        return []

    # normalize newlines/spaces
    t = text.replace("\r\n", "\n").replace("\r", "\n")

    # split by newline first (abstract often has hard breaks)
    parts = []
    for chunk in t.split("\n"):
        chunk = chunk.strip()
        if not chunk:
            continue
        parts.append(chunk)

    # further split by sentence punctuation
    sents: List[str] = []
    for p in parts:
        # Split on punctuation followed by whitespace/capital/quote/end
        # This is a heuristic; good enough for abstracts.
        raw = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])", p)
        for r0 in raw:
            r0 = _normalize_ws(r0)
            if len(r0) >= 25:   # drop tiny fragments
                sents.append(r0)

    # de-dup while keeping order
    seen = set()
    uniq = []
    for s in sents:
        key = s.lower()
        if key in seen:
            continue
        seen.add(key)
        uniq.append(s)
    return uniq


def _find_hits(sentences: List[str], patterns: List[str], kind: str) -> List[CueHit]:
    hits: List[CueHit] = []
    for s in sentences:
        low = s.lower()
        for pat in patterns:
            if re.search(pat, low):
                hits.append(CueHit(sentence=s, kind=kind, cue=pat))
                break
    return hits


def extract_claim_sentences(text: str, top_k: int = 3) -> List[str]:
    """
    Claim/highlight sentences (NOT for contradiction hunt).
    """
    sents = split_sentences(text)
    hits = _find_hits(sents, _CLAIM_CUES, kind="claim")
    # Return first top_k by appearance (abstracts usually order important info early)
    return [h.sentence for h in hits[:top_k]]


def extract_limitation_sentences(text: str, top_k: int = 5) -> List[str]:
    """
    Limitation/failure/contradiction-like sentences (USE THIS for contradiction hunt).
    """
    sents = split_sentences(text)

    # Primary: explicit limitation/failure/comparison markers
    hits = _find_hits(sents, _LIMIT_CUES, kind="limit")

    # If too few, backfill with comparison-ish sentences (weak signal)
    if len(hits) < top_k:
        comp_hits = _find_hits(sents, _COMPARISON_CUES, kind="limit")
        # append only new sentences
        seen = {h.sentence.lower() for h in hits}
        for h in comp_hits:
            if h.sentence.lower() not in seen:
                hits.append(h)
                seen.add(h.sentence.lower())
            if len(hits) >= top_k:
                break

    return [h.sentence for h in hits[:top_k]]


# -----------------------------
# Backward-compatible API
# -----------------------------

def extract_cue_sentences(text: str, top_k: int = 5) -> List[str]:
    """
    BACKWARD COMPAT:
    - Old code likely used this as "cue-hit" for contradiction hunt.
    - Now it returns limitation/failure cues only (the correct behavior).
    """
    return extract_limitation_sentences(text, top_k=top_k)


def extract_claim_and_limit(text: str, claim_k: int = 3, limit_k: int = 5) -> Tuple[List[str], List[str]]:
    """
    New convenience method:
    returns (claims, limits)
    """
    return extract_claim_sentences(text, top_k=claim_k), extract_limitation_sentences(text, top_k=limit_k)
