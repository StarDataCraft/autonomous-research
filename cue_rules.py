# cue_rules.py
# ------------------------------------------------------------
# A lightweight, "No LLM" cue extraction + relevance scoring module.
#
# Design goals:
# 1) Claim / Limitation cues are DOMAIN-GENERAL (portable across topics).
# 2) Relevance is TOPIC-SPECIFIC but PLUGGABLE via RelevanceSpec (no hardcoding).
# 3) You can generate a RelevanceSpec from a corpus (titles/abstracts) using
#    a simple TF-IDF-like heuristic (stdlib only).
#
# Typical usage:
#   spec = RelevanceSpec.from_json("relevance_specs/my_question.json")
#   cues = extract_cues(text, spec=spec)
#   novelty_rank = compute_novelty_rank(novelty=0.52, relevance=cues.relevance)
# ------------------------------------------------------------

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Pattern, Sequence, Tuple


# -----------------------------
# Helpers: tokenization
# -----------------------------

_BASIC_STOPWORDS = {
    # English stopwords (small but decent)
    "a", "an", "the", "and", "or", "but", "if", "then", "than", "so", "because",
    "to", "of", "in", "on", "at", "for", "with", "without", "by", "from", "as",
    "is", "are", "was", "were", "be", "been", "being", "it", "this", "that",
    "these", "those", "we", "our", "you", "your", "they", "their", "he", "she",
    "i", "me", "my", "mine", "us", "them", "can", "could", "may", "might", "will",
    "would", "should", "do", "does", "did", "done", "not", "no", "yes",
    "such", "some", "any", "all", "each", "many", "much", "more", "most",
    "less", "least", "very", "also", "too", "just", "only",
    "into", "over", "under", "up", "down", "out", "about", "above", "below",
    "here", "there", "where", "when", "why", "how",
    "new", "novel", "paper", "work", "results", "method", "approach", "model",
    "models", "algorithm", "algorithms", "study", "studies", "analysis",
    "using", "use", "used", "based", "via",
}


def _normalize_text(s: str) -> str:
    # Keep it simple and robust.
    s = s.strip()
    s = s.replace("\u2019", "'")
    s = re.sub(r"\s+", " ", s)
    return s


def tokenize(text: str) -> List[str]:
    """
    A simple tokenizer suitable for TF-IDF-ish scoring:
    - lowercasing
    - keep alphanumerics and a few symbols
    - split by non-word
    - drop short tokens and stopwords
    """
    text = _normalize_text(text).lower()
    # Keep letters, numbers, underscore. Replace others with space.
    text = re.sub(r"[^a-z0-9_]+", " ", text)
    toks = [t for t in text.split() if len(t) >= 3 and t not in _BASIC_STOPWORDS]
    return toks


# -----------------------------
# Relevance spec (pluggable)
# -----------------------------

@dataclass
class RelevanceSpec:
    """
    Topic-specific relevance specification.
    - positive_terms: single-word terms that indicate relevance
    - positive_phrases: multi-word phrases (checked via substring match on normalized lower text)
    - negative_terms/phrases: downweight or gate relevance
    - weights: optional per-term weights (both terms and phrases can be weighted)
    - gate:
        - If gate_requires_any_positive is True, relevance=0 unless at least one positive hit.
        - If gate_blocks_on_negative is True, relevance=0 if any negative phrase/term hits.
    """
    positive_terms: List[str] = field(default_factory=list)
    positive_phrases: List[str] = field(default_factory=list)
    negative_terms: List[str] = field(default_factory=list)
    negative_phrases: List[str] = field(default_factory=list)
    weights: Dict[str, float] = field(default_factory=dict)

    gate_requires_any_positive: bool = True
    gate_blocks_on_negative: bool = False

    # Scoring parameters (tweakable per research question if desired)
    term_hit_weight: float = 1.0
    phrase_hit_weight: float = 1.5
    negative_penalty: float = 1.0

    @staticmethod
    def empty() -> "RelevanceSpec":
        # Domain-general extraction can run without relevance guidance.
        return RelevanceSpec(
            positive_terms=[],
            positive_phrases=[],
            negative_terms=[],
            negative_phrases=[],
            weights={},
            gate_requires_any_positive=False,  # allow relevance to be "weakly non-zero" if you later want
            gate_blocks_on_negative=False,
        )

    def to_dict(self) -> dict:
        return {
            "positive_terms": self.positive_terms,
            "positive_phrases": self.positive_phrases,
            "negative_terms": self.negative_terms,
            "negative_phrases": self.negative_phrases,
            "weights": self.weights,
            "gate_requires_any_positive": self.gate_requires_any_positive,
            "gate_blocks_on_negative": self.gate_blocks_on_negative,
            "term_hit_weight": self.term_hit_weight,
            "phrase_hit_weight": self.phrase_hit_weight,
            "negative_penalty": self.negative_penalty,
        }

    @staticmethod
    def from_dict(d: dict) -> "RelevanceSpec":
        return RelevanceSpec(
            positive_terms=list(d.get("positive_terms", [])),
            positive_phrases=list(d.get("positive_phrases", [])),
            negative_terms=list(d.get("negative_terms", [])),
            negative_phrases=list(d.get("negative_phrases", [])),
            weights=dict(d.get("weights", {})),
            gate_requires_any_positive=bool(d.get("gate_requires_any_positive", True)),
            gate_blocks_on_negative=bool(d.get("gate_blocks_on_negative", False)),
            term_hit_weight=float(d.get("term_hit_weight", 1.0)),
            phrase_hit_weight=float(d.get("phrase_hit_weight", 1.5)),
            negative_penalty=float(d.get("negative_penalty", 1.0)),
        )

    @staticmethod
    def from_json(path: str) -> "RelevanceSpec":
        with open(path, "r", encoding="utf-8") as f:
            return RelevanceSpec.from_dict(json.load(f))

    def to_json(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)


def build_relevance_spec_from_corpus(
    corpus: Sequence[str],
    seed_positive_terms: Optional[Sequence[str]] = None,
    seed_positive_phrases: Optional[Sequence[str]] = None,
    seed_negative_terms: Optional[Sequence[str]] = None,
    seed_negative_phrases: Optional[Sequence[str]] = None,
    top_k_terms: int = 40,
    min_df: int = 2,
    max_df_ratio: float = 0.6,
) -> RelevanceSpec:
    """
    No-LLM "auto-spec" builder using a simple TF-IDF-like heuristic.

    How it works:
    - Tokenize each doc, build document frequency DF(t)
    - Keep terms with DF in [min_df, max_df_ratio * N]
    - Score term by avg TF * IDF
    - Take top_k_terms as positive_terms
    - Seeds are always included (and can be weighted manually later)

    Notes:
    - This is intentionally simple and robust; it won't be perfect, but it will
      avoid topic-hardcoding in code.
    """
    if not corpus:
        return RelevanceSpec.empty()

    N = len(corpus)
    docs_tokens = [tokenize(x) for x in corpus]
    df: Dict[str, int] = {}
    tf_sum: Dict[str, float] = {}

    for toks in docs_tokens:
        seen = set()
        for t in toks:
            tf_sum[t] = tf_sum.get(t, 0.0) + 1.0
            if t not in seen:
                df[t] = df.get(t, 0) + 1
                seen.add(t)

    # Average TF per document (approx)
    for t in list(tf_sum.keys()):
        tf_sum[t] = tf_sum[t] / float(N)

    max_df = max(1, int(math.floor(max_df_ratio * N)))

    scored: List[Tuple[float, str]] = []
    for t, dft in df.items():
        if dft < min_df or dft > max_df:
            continue
        idf = math.log((N + 1.0) / (dft + 1.0)) + 1.0
        score = tf_sum.get(t, 0.0) * idf
        scored.append((score, t))

    scored.sort(reverse=True)
    auto_terms = [t for _, t in scored[:top_k_terms]]

    pos_terms = list(dict.fromkeys([*(seed_positive_terms or []), *auto_terms]))
    pos_phrases = list(dict.fromkeys(list(seed_positive_phrases or [])))
    neg_terms = list(dict.fromkeys(list(seed_negative_terms or [])))
    neg_phrases = list(dict.fromkeys(list(seed_negative_phrases or [])))

    return RelevanceSpec(
        positive_terms=pos_terms,
        positive_phrases=pos_phrases,
        negative_terms=neg_terms,
        negative_phrases=neg_phrases,
        weights={},  # you can fill in later
        gate_requires_any_positive=True,
        gate_blocks_on_negative=False,
    )


# -----------------------------
# Cue patterns (domain-general)
# -----------------------------

def _compile_patterns(patterns: Sequence[str]) -> List[Pattern[str]]:
    return [re.compile(p, flags=re.IGNORECASE) for p in patterns]


# Claim cues: "we propose/show/derive/establish/confirm/achieve..."
_CLAIM_PATTERNS = _compile_patterns([
    r"\bwe (propose|introduce|present|develop|formulate)\b",
    r"\bwe (show|demonstrate|prove|establish|derive|analyze|argue)\b",
    r"\bwe (confirm|validate|verify)\b",
    r"\bwe (achieve|obtain|attain|reach)\b",
    r"\bour (method|approach|framework|scheme|model)\b.*\b(improves?|outperforms?|reduces?|achieves?|enables?)\b",
    r"\bresults? (show|demonstrate|indicate|suggest)\b",
    r"\bwe provide\b.*\b(theoretical|empirical)\b",
    r"\bwe find\b",
    r"\bwe obtain\b",
    r"\bwe report\b",
    r"\bwe confirm\b",
    # math-ish claim: bounds/complexity/rates
    r"\b(upper|lower) bound\b",
    r"\b(sample complexity|convergence rate|minimax)\b",
    r"\bO\([^)]*\)\b",
    r"\b\tilde{O}\([^)]*\)\b",
    r"\bnon[- ]asymptotic\b",
])

# Limitation/failure cues: "however", "remains underexplored", "still far from", "limitations"
_LIMIT_PATTERNS = _compile_patterns([
    r"\bhowever\b",
    r"\bbut\b.*\b(remains?|still)\b",
    r"\bdespite\b",
    r"\bnevertheless\b",
    r"\bnonetheless\b",
    r"\byet\b",
    r"\bremains? (unclear|underexplored|unknown|challenging|open)\b",
    r"\blacks? (a|an|the)?\s*(theoretical understanding|interpretation|guarantee|analysis)\b",
    r"\blimitation(s)?\b",
    r"\bdrawback(s)?\b",
    r"\bfailure mode(s)?\b",
    r"\bdoes not\b.*\b(scale|generalize|work|hold)\b",
    r"\bfar from\b",
    r"\bcomputational (overhead|cost|burden)\b",
    r"\bunrealistic assumptions?\b",
    r"\bstrong assumptions?\b",
    r"\btrade[- ]off\b",
    r"\bunstable\b|\binstability\b",
    r"\bdiverge(s|d)?\b|\bcollapse(s|d)?\b",
])

# Additional "soft" relevance fallback cues (domain-general):
# This is NOT topic relevance; it boosts sentences that look like contributions even if spec is empty.
_SOFT_IMPORTANCE_PATTERNS = _compile_patterns([
    r"\bmain contribution\b",
    r"\bkey (idea|insight|result)\b",
    r"\bwe propose\b|\bwe present\b|\bwe show\b",
])


# -----------------------------
# Sentence splitting
# -----------------------------

_SENT_SPLIT = re.compile(r"(?<=[\.\!\?])\s+|\n+")


def split_sentences(text: str, max_sentences: int = 200) -> List[str]:
    text = _normalize_text(text)
    if not text:
        return []
    sents = [s.strip() for s in _SENT_SPLIT.split(text) if s.strip()]
    return sents[:max_sentences]


# -----------------------------
# Scoring: claim/limitation + relevance
# -----------------------------

def _pattern_hits(s: str, patterns: Sequence[Pattern[str]]) -> int:
    return sum(1 for p in patterns if p.search(s))


def classify_sentence(sentence: str) -> str:
    """
    Classify a sentence as:
      - "claim" if claim cues dominate
      - "limitation" if limitation cues dominate
      - "other" otherwise
    """
    s = sentence.strip()
    if not s:
        return "other"
    c = _pattern_hits(s, _CLAIM_PATTERNS)
    l = _pattern_hits(s, _LIMIT_PATTERNS)

    if c == 0 and l == 0:
        return "other"
    if l > c:
        return "limitation"
    if c > l:
        return "claim"
    # tie-break: prefer limitation if "however" exists
    if re.search(r"\bhowever\b", s, flags=re.IGNORECASE):
        return "limitation"
    return "claim"


def score_relevance(text: str, spec: Optional[RelevanceSpec]) -> float:
    """
    Topic relevance score in [0, 1], driven by spec.
    If spec is None: returns 0.0 (no relevance signal).
    If spec has no positives and gate_requires_any_positive=True: returns 0.0.
    """
    if spec is None:
        return 0.0

    raw = _normalize_text(text).lower()
    toks = tokenize(raw)

    pos_term_hits = 0.0
    pos_phrase_hits = 0.0
    neg_hits = 0.0

    # phrase hits
    for ph in spec.positive_phrases:
        ph_n = _normalize_text(ph).lower()
        if ph_n and ph_n in raw:
            w = spec.weights.get(ph, spec.weights.get(ph_n, 1.0))
            pos_phrase_hits += spec.phrase_hit_weight * float(w)

    for ph in spec.negative_phrases:
        ph_n = _normalize_text(ph).lower()
        if ph_n and ph_n in raw:
            w = spec.weights.get(ph, spec.weights.get(ph_n, 1.0))
            neg_hits += spec.negative_penalty * float(w)

    # term hits (count unique hits to reduce spam)
    tok_set = set(toks)
    for t in spec.positive_terms:
        t_n = _normalize_text(t).lower()
        if t_n in tok_set:
            w = spec.weights.get(t, spec.weights.get(t_n, 1.0))
            pos_term_hits += spec.term_hit_weight * float(w)

    for t in spec.negative_terms:
        t_n = _normalize_text(t).lower()
        if t_n in tok_set:
            w = spec.weights.get(t, spec.weights.get(t_n, 1.0))
            neg_hits += spec.negative_penalty * float(w)

    has_positive = (pos_term_hits + pos_phrase_hits) > 0.0
    has_negative = neg_hits > 0.0

    if spec.gate_blocks_on_negative and has_negative:
        return 0.0
    if spec.gate_requires_any_positive and not has_positive:
        return 0.0

    # Convert to [0, 1] using a saturating nonlinearity
    # Intuition: a few hits should already mean "relevant enough", many hits saturate.
    signal = (pos_term_hits + pos_phrase_hits) - neg_hits
    if signal <= 0:
        return 0.0

    # logistic-like saturation without overflow
    # r = 1 - exp(-signal / k)
    k = 4.0
    r = 1.0 - math.exp(-signal / k)
    return max(0.0, min(1.0, r))


def score_sentence_importance(sentence: str) -> float:
    """
    Domain-general sentence importance in [0, 1], unrelated to topic relevance.
    Useful if you want to rank cues even when spec is empty.
    """
    s = sentence.strip()
    if not s:
        return 0.0
    c = _pattern_hits(s, _CLAIM_PATTERNS)
    l = _pattern_hits(s, _LIMIT_PATTERNS)
    soft = _pattern_hits(s, _SOFT_IMPORTANCE_PATTERNS)
    total = c + l + soft
    # Saturate
    return 1.0 - math.exp(-float(total) / 3.0)


# -----------------------------
# Outputs
# -----------------------------

@dataclass
class CueItem:
    cue_type: str               # "claim" | "limitation"
    sentence: str
    importance: float           # domain-general
    relevance: float            # topic-specific (spec-driven)


@dataclass
class CueExtraction:
    claims: List[CueItem] = field(default_factory=list)
    limitations: List[CueItem] = field(default_factory=list)
    relevance: float = 0.0      # document-level relevance
    notes: Dict[str, float] = field(default_factory=dict)


def extract_cues(
    text: str,
    spec: Optional[RelevanceSpec] = None,
    max_claims: int = 5,
    max_limitations: int = 5,
) -> CueExtraction:
    """
    Extract top claim/limitation sentences and compute a document-level relevance score.
    """
    text = _normalize_text(text)
    sents = split_sentences(text)
    if not sents:
        return CueExtraction(relevance=0.0, notes={"num_sentences": 0.0})

    # document relevance: score on full text (more stable than per-sentence)
    doc_rel = score_relevance(text, spec)

    claims: List[CueItem] = []
    limits: List[CueItem] = []

    for s in sents:
        t = classify_sentence(s)
        if t not in ("claim", "limitation"):
            continue
        imp = score_sentence_importance(s)
        rel = score_relevance(s, spec)

        item = CueItem(cue_type=t, sentence=s, importance=imp, relevance=rel)
        if t == "claim":
            claims.append(item)
        else:
            limits.append(item)

    # Rank within type:
    # - primary: importance (domain-general)
    # - secondary: relevance (topic-specific)
    # This avoids "off-topic but loud cue words" dominating if spec is good.
    claims.sort(key=lambda x: (x.importance, x.relevance), reverse=True)
    limits.sort(key=lambda x: (x.importance, x.relevance), reverse=True)

    return CueExtraction(
        claims=claims[:max_claims],
        limitations=limits[:max_limitations],
        relevance=doc_rel,
        notes={"num_sentences": float(len(sents))},
    )


# -----------------------------
# Novelty rank: novelty × f(relevance)
# -----------------------------

def relevance_gate(relevance: float, alpha: float = 2.0, floor: float = 0.15) -> float:
    """
    f(relevance): suppress "off-topic but outlier" papers.

    - floor keeps some exploration alive (so truly novel but low-coverage terms don't go to zero)
    - alpha controls how aggressively relevance gates novelty

    f(r) = floor + (1 - floor) * r^alpha
    """
    r = max(0.0, min(1.0, relevance))
    return float(floor + (1.0 - floor) * (r ** alpha))


def compute_novelty_rank(novelty: float, relevance: float, alpha: float = 2.0, floor: float = 0.15) -> float:
    n = max(0.0, min(1.0, novelty))
    return n * relevance_gate(relevance, alpha=alpha, floor=floor)

# -----------------------------
# Backward-compatible API (for older streamlit_app.py)
# -----------------------------

def classify_cue_sentence(sentence: str) -> str:
    """
    Backward-compatible wrapper.
    Returns: "claim" | "limitation" | "other"
    """
    return classify_sentence(sentence)


def is_limitation_failure(sentence: str) -> bool:
    """
    Backward-compatible wrapper.
    True if the sentence is classified as limitation/failure.
    """
    return classify_sentence(sentence) == "limitation"

# -----------------------------
# Minimal self-test (optional)
# -----------------------------

if __name__ == "__main__":
    # Example: build spec from corpus, then extract cues
    corpus = [
        "We propose a batch size invariant version of Adam for large-scale distributed settings.",
        "However, the theoretical understanding of flow matching ODE dynamics remains underexplored.",
        "We derive a deterministic non-asymptotic upper bound on the KL divergence.",
    ]

    spec = build_relevance_spec_from_corpus(
        corpus=corpus,
        seed_positive_phrases=["batch size invariant", "flow matching", "kl divergence"],
        top_k_terms=20,
        min_df=1,
    )

    text = """
    We propose a batch size invariant version of Adam, for use in large-scale distributed settings.
    However, the approach relies on assumptions that may not hold universally.
    We confirm that in practice our scheme gives batch size invariance in a much larger range of scenarios.
    """

    cues = extract_cues(text, spec=spec)
    print("Doc relevance:", cues.relevance)
    print("\nClaims:")
    for c in cues.claims:
        print("-", c.sentence, "| imp:", round(c.importance, 3), "| rel:", round(c.relevance, 3))
    print("\nLimitations:")
    for l in cues.limitations:
        print("-", l.sentence, "| imp:", round(l.importance, 3), "| rel:", round(l.relevance, 3))

    print("\nNovelty rank example:", compute_novelty_rank(novelty=0.52, relevance=cues.relevance))
