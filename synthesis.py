from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict

from pipeline_abstract import PaperInsight, SentenceHit

# NEW: robust cue classification (two-stage: discourse marker AND negative semantics)
# plus strong-claim override (e.g., "can be viewed", "unified framework", "we propose", ...)
from cue_rules import classify_cue_sentence, is_limitation_failure


@dataclass
class ThemeSummary:
    cluster_id: int
    keywords: List[str]
    headline_points: List[str]         # 3-6 bullets
    evidence_papers: List[str]         # titles (top few)
    method_theory_empirical: Dict[str, int]


@dataclass
class SynthesisReport:
    theme_summaries: List[ThemeSummary]
    cross_cluster_bridges: List[str]   # bridge paper titles
    contradictions: List[str]          # extracted limitations/failure cues
    gaps: List[str]                    # missing experiments / controls / metrics
    new_directions: List[str]          # proposals (each includes "because papers: ...")


def _clean(s: str) -> str:
    return " ".join((s or "").split()).strip()


def _hit_kind_normalized(hit: SentenceHit) -> str:
    """
    Normalize upstream kinds to one of:
      - "limitation/failure"
      - "claim"
      - "relevance"

    We treat unknown as "relevance" (safer than polluting contradictions).
    """
    k = (getattr(hit, "kind", "") or "").strip().lower()
    if k in {"cue-limit", "cue-limitation", "cue-failure", "limitation", "failure"}:
        return "limitation/failure"
    if k in {"cue-claim", "claim"}:
        return "claim"
    if k == "relevance":
        return "relevance"
    if k == "cue-hit":
        # Backward-compat: use robust rule-based classifier on the sentence content
        decision = classify_cue_sentence(_clean(getattr(hit, "sentence", "")))
        if decision.kind == "limitation/failure":
            return "limitation/failure"
        if decision.kind == "claim":
            return "claim"
        return "relevance"
    # unknown
    return "relevance"


def _is_limitation(hit: SentenceHit) -> bool:
    """
    True only when we are confident it's a limitation/failure cue.

    IMPORTANT: This fixes the "Despite ... unified framework ..." false positives:
      - discourse marker alone is NOT enough
      - must have negative semantics unless a strong-claim override applies
    """
    return _hit_kind_normalized(hit) == "limitation/failure"


def _top_sentences(
    insights: List[PaperInsight],
    limit_only: bool,
    k: int = 4,
) -> List[str]:
    """
    Collect and rank sentences globally within a set of PaperInsight objects.

    Ranking: 0.6 * paper_relevance + 0.4 * sentence_sim (as before)
    Filtering:
      - if limit_only=True => keep only limitation/failure sentences (robust)
      - else => keep all sentences, but still de-dup
    """
    sents = []
    for it in insights:
        for h in (it.key_sentences or []):
            if limit_only and not _is_limitation(h):
                continue
            sents.append((_clean(h.sentence), it.relevance, h.sim, it.paper.title))

    sents.sort(key=lambda x: (-(0.6 * x[1] + 0.4 * x[2])))

    out: List[str] = []
    seen = set()
    for sent, _, _, title in sents:
        if not sent:
            continue
        key = sent[:120]
        if key in seen:
            continue
        seen.add(key)
        out.append(f"{sent}  — *{title}*")
        if len(out) >= k:
            break
    return out


def _type_hist(insights: List[PaperInsight]) -> Dict[str, int]:
    h = {"method": 0, "theory": 0, "empirical": 0}
    for it in insights:
        if it.paper_type in h:
            h[it.paper_type] += 1
    return h


def synthesize_from_clusters(
    clusters: List,
    insights: List[PaperInsight],
    top_bridge_n: int = 8,
) -> SynthesisReport:
    # --- Theme summaries per cluster
    theme_summaries: List[ThemeSummary] = []
    for c in clusters:
        papers = c.papers

        # headline points:
        #   - prefer true limitation/failure sentences (robust)
        #   - then add a couple of high-signal sentences regardless of kind
        limits = _top_sentences(papers, limit_only=True, k=3)
        keyinfo = _top_sentences(papers, limit_only=False, k=2)
        headline = (limits + keyinfo)[:6]

        top_titles = [p.paper.title for p in papers[:5]]
        theme_summaries.append(
            ThemeSummary(
                cluster_id=c.cluster_id,
                keywords=c.keywords,
                headline_points=headline if headline else ["(no salient sentences found)"],
                evidence_papers=top_titles,
                method_theory_empirical=_type_hist(papers),
            )
        )

    # --- Bridges (cross-cluster)
    bridge_sorted = sorted(insights, key=lambda x: (-x.bridge_rank, -x.relevance))
    cross_cluster_bridges = [
        b.paper.title for b in bridge_sorted[: min(top_bridge_n, len(bridge_sorted))]
    ]

    # --- Contradictions / limitations (global pool)
    # IMPORTANT: now truly limitation/failure only; no more "Despite ... unified framework ..." pollution.
    contradictions = _top_sentences(insights, limit_only=True, k=10)

    # --- Gaps: rule-based gap discovery (topic-specific templates)
    gaps = [
        "Define *exactly* how gradient variance is measured (global norm? per-layer? per-parameter? before/after clipping), and report it consistently across methods.",
        "Control for training-stability knobs that can erase small-batch differences: gradient clipping, EMA, LR warmup, noise/time sampling, solver choice.",
        "Run seed-robust comparisons (≥3 seeds) and report CI/Std—variance claims without uncertainty are weak.",
        "Compare under matched compute budget (same NFE / steps / wall-clock) to avoid 'efficiency' conclusions from unfair baselines.",
        "Separate three variance sources: (i) target variance (DSM/STF-style), (ii) timestep/trajectory estimator variance (TPC-style), (iii) optimizer/batch-noise variance (batch-invariant Adam).",
        "Add a harder or shifted regime beyond CIFAR/MNIST (e.g., higher-res, different modality, or corrupted data) to test whether stability/variance conclusions generalize.",
    ]

    # --- New directions: generated from bridges + novelty + gaps
    novelty_sorted = sorted(insights, key=lambda x: (-x.novelty_rank, -x.relevance))

    def cite_titles(items: List[PaperInsight], m: int = 3) -> str:
        ts = [it.paper.title for it in items[:m] if it and it.paper and it.paper.title]
        return "; ".join(ts) if ts else "(no papers)"

    new_directions: List[str] = []

    # Direction 1: unify variance taxonomy
    new_directions.append(
        "建立一个 **variance taxonomy + measurement protocol**：把 variance 分解为 target-variance（score/target）、estimator-variance（timestep coupling）、optimizer/batch-noise（micro-batch/Adam）三类，并在同一 U-Net + 同一日志定义下系统对照。"
        f"  \nBecause: {cite_titles(novelty_sorted)}"
    )

    # Direction 2: bridge-based: connect flow-matching and diffusion under generator-matching / reweighting
    new_directions.append(
        "做一个 **桥梁型统一视角实验**：以 “Generator Matching / time-dependent reweighting” 为统一框架，把 flow matching、diffusion、rectified flow 放在同一损失家族里，检验：variance 降低来自哪里（路径分布 vs 时间采样 vs loss 形式）。"
        f"  \nBecause (bridges): {', '.join(cross_cluster_bridges[:3]) if cross_cluster_bridges else '(no bridges)'}"
    )

    # Direction 3: contradiction-driven: when variance reduction trades bias
    new_directions.append(
        "研究 **variance–bias trade-off 的可控旋钮**：例如 STF 类方法用 reference batch 降 variance、引入 bias；TPC 类方法降 estimator variance；你可以提出一个统一“bias budget”指标，把这些方法放到同一 Pareto frontier 上。"
        f"  \nBecause (limitations/cues): {contradictions[0] if contradictions else '(no limitation/failure cues found)'}"
    )

    # Direction 4: small-batch focus: micro-batch invariance + FM/Diffusion
    batchish = [
        x for x in novelty_sorted
        if x.paper and x.paper.title and (
            "batch" in x.paper.title.lower() or "adam" in x.paper.title.lower() or "micro" in x.paper.title.lower()
        )
    ]
    new_directions.append(
        "针对你关心的 **small-batch regime**：引入 batch-size invariant optimizer（micro-batch 视角）作为控制变量，测试 flow matching vs diffusion 的差异是否仍存在；如果消失，说明差异主要来自 optimizer/batch-noise，而不是生成建模范式本身。"
        f"  \nBecause (novelty/bridge candidates): {cite_titles(batchish)}"
    )

    # Direction 5: failure modes (title-based anchor)
    failureish = [
        x for x in insights
        if x.paper and x.paper.title and (
            "failure" in x.paper.title.lower()
            or ("variance" in x.paper.title.lower() and "reduce" in x.paper.title.lower())
            or "unstable" in x.paper.title.lower()
            or "collapse" in x.paper.title.lower()
        )
    ]
    new_directions.append(
        "把 **failure modes** 作为主任务：用 'gradient variance reveals failure modes' 这类工作作为 anchor，定义可复现的 failure suite（loss spikes, divergence, mode depletion, corrupted-data stress），然后比较不同 variance-reduction 手段在 failure suite 上的鲁棒性。"
        f"  \nBecause (cluster evidence): {cite_titles(failureish)}"
    )

    return SynthesisReport(
        theme_summaries=theme_summaries,
        cross_cluster_bridges=cross_cluster_bridges,
        contradictions=contradictions,
        gaps=gaps,
        new_directions=new_directions,
    )
