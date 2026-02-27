# cue_rules.py
# ------------------------------------------------------------
# Rule-based cue sentence classifier:
# - Split cue-hit sentences into: claim vs limitation/failure
# - limitation/failure = discourse_marker AND negative_semantics
# - strong claim patterns override limitation candidates
# ------------------------------------------------------------

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

CueKind = Literal["claim", "limitation/failure", "relevance-fallback"]


@dataclass(frozen=True)
class CueDecision:
    kind: CueKind
    reason: str


# -----------------------------
# Regex helpers
# -----------------------------
def _norm(s: str) -> str:
    # normalize whitespace & lowercase
    s = re.sub(r"\s+", " ", s.strip())
    return s.lower()


def _has_any(text: str, patterns: list[re.Pattern]) -> bool:
    return any(p.search(text) for p in patterns)


# -----------------------------
# Discourse markers (ONLY candidate signals)
# -----------------------------
_DISCOURSE_MARKERS: list[re.Pattern] = [
    re.compile(r"\bhowever\b", re.I),
    re.compile(r"\bdespite\b", re.I),
    re.compile(r"\bin contrast\b", re.I),
    re.compile(r"\bbut\b", re.I),
    re.compile(r"\bnevertheless\b", re.I),
    re.compile(r"\bnonetheless\b", re.I),
    re.compile(r"\byet\b", re.I),
    re.compile(r"\bstill\b", re.I),
    re.compile(r"\balthough\b", re.I),
    re.compile(r"\bwhereas\b", re.I),
]

# -----------------------------
# Negative semantics (required for limitation/failure)
# Keep this list conservative; add items only if they reduce false positives.
# -----------------------------
_NEGATIVE_SEMANTICS: list[re.Pattern] = [
    re.compile(r"\black(s|ing)?\b", re.I),
    re.compile(r"\bund(er)?explored\b", re.I),
    re.compile(r"\bhinder(ed|s|ing)?\b", re.I),
    re.compile(r"\bf(ar)? from\b", re.I),  # "far from"
    re.compile(r"\blimitation(s)?\b", re.I),
    re.compile(r"\bchallenge(s)?\b", re.I),
    re.compile(r"\bweak(ness|es)?\b", re.I),
    re.compile(r"\bfail(ure|ures|ed|ing)?\b", re.I),
    re.compile(r"\bcollapse(s|d)?\b", re.I),
    re.compile(r"\bunstable\b", re.I),
    re.compile(r"\binstabilit(y|ies)\b", re.I),
    re.compile(r"\bproblem(s)?\b", re.I),
    re.compile(r"\bdifficult(y|ies)\b", re.I),
    re.compile(r"\binsufficient\b", re.I),
    re.compile(r"\bnot (well )?understood\b", re.I),
    re.compile(r"\bremains? (an )?open\b", re.I),
    re.compile(r"\bstill (an )?open\b", re.I),
    re.compile(r"\brare(ly)?\b", re.I),
    re.compile(r"\bunderperform(s|ed|ing)?\b", re.I),
    re.compile(r"\bdegrad(e|es|ed|ing)\b", re.I),
    re.compile(r"\bpoor(ly)?\b", re.I),
    re.compile(r"\bbias(ed)?\b", re.I),
    re.compile(r"\bcomputational overhead\b", re.I),
    re.compile(r"\bexpensive\b", re.I),
    re.compile(r"\bslow\b", re.I),
]

# -----------------------------
# Strong claim patterns (override limitation candidates)
# - These indicate the sentence is likely a contribution/framing claim.
# - IMPORTANT: include your "unified framework / can be viewed" case.
# -----------------------------
_STRONG_CLAIM: list[re.Pattern] = [
    re.compile(r"\bwe (propose|present|introduce|develop|derive|prove|establish|show|demonstrate)\b", re.I),
    re.compile(r"\bthis (paper|work|study) (proposes|presents|introduces|develops|derives|proves|establishes|shows|demonstrates)\b", re.I),
    re.compile(r"\bwe (provide|offer) (a )?(theoretical|empirical|comprehensive)\b", re.I),
    re.compile(r"\bwe confirm\b", re.I),
    re.compile(r"\bwe argue\b", re.I),
    re.compile(r"\bwe find\b", re.I),
    re.compile(r"\bwe report\b", re.I),
    # framing/unification claims:
    re.compile(r"\bcan be viewed\b", re.I),
    re.compile(r"\bunified framework\b", re.I),
    re.compile(r"\bunder (a|the) unified framework\b", re.I),
    re.compile(r"\brecast(ing)?\b", re.I),
    re.compile(r"\bview(ed)? under\b", re.I),
    re.compile(r"\bsubsume(s|d)?\b", re.I),
]

# -----------------------------
# Positive-contribution verbs (guardrail to prevent misclassifying "reduces variance" etc.)
# If a sentence matches these and does NOT strongly match negative semantics, treat as claim.
# -----------------------------
_POSITIVE_CONTRIB: list[re.Pattern] = [
    re.compile(r"\breduce(s|d|ing)?\b", re.I),
    re.compile(r"\bimprove(s|d|ing)?\b", re.I),
    re.compile(r"\bachieve(s|d|ing)?\b", re.I),
    re.compile(r"\boutperform(s|ed|ing)?\b", re.I),
    re.compile(r"\bprovid(e|es|ed|ing) (guarantee|bounds|upper bound|lower bound|non-asymptotic)\b", re.I),
    re.compile(r"\bprovably\b", re.I),
    re.compile(r"\bdemonstrably\b", re.I),
    re.compile(r"\bthe main contribution\b", re.I),
]


def classify_cue_sentence(sentence: str) -> CueDecision:
    """
    Classify a sentence into:
      - "limitation/failure": must satisfy (discourse_marker AND negative_semantics)
                              OR strong negative semantics alone (rare but allowed).
      - "claim": strong claim pattern, or positive contribution (without negatives),
                 or otherwise non-negative.
      - "relevance-fallback": only if it contains neither strong claim nor limitation signals.
    """

    raw = sentence.strip()
    if not raw:
        return CueDecision(kind="relevance-fallback", reason="empty")

    s = _norm(raw)

    has_marker = _has_any(s, _DISCOURSE_MARKERS)
    has_negative = _has_any(s, _NEGATIVE_SEMANTICS)
    has_strong_claim = _has_any(s, _STRONG_CLAIM)
    has_positive = _has_any(s, _POSITIVE_CONTRIB)

    # 1) Strong claim overrides "marker-only" limitation candidates
    #    Example: "Despite X, we propose Y" should be claim.
    if has_strong_claim:
        return CueDecision(kind="claim", reason="strong-claim-pattern")

    # 2) If it's clearly a positive contribution and NOT negative -> claim
    #    This prevents misclassifying e.g. "provably reduces gradient variance" as failure.
    if has_positive and not has_negative:
        return CueDecision(kind="claim", reason="positive-contribution-no-negative")

    # 3) Limitation/failure requires BOTH marker and negative semantics
    if has_marker and has_negative:
        return CueDecision(kind="limitation/failure", reason="marker+negative")

    # 4) Allow strong negative semantics even without marker (e.g., "remains underexplored")
    #    This is common in abstracts.
    if has_negative and not has_positive:
        return CueDecision(kind="limitation/failure", reason="negative-without-positive")

    # 5) Otherwise, treat as relevance fallback (or you can default to claim if you prefer)
    return CueDecision(kind="relevance-fallback", reason="no-strong-signal")


def is_limitation_failure(sentence: str) -> bool:
    """Drop-in boolean helper if your old code expects a bool."""
    return classify_cue_sentence(sentence).kind == "limitation/failure"


def cue_label(sentence: str) -> str:
    """UI label helper: returns 'claim' or 'limitation/failure' or 'relevance-fallback'."""
    return classify_cue_sentence(sentence).kind
