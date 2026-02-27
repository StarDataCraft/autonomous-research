# synthesis.py
# Uses PipelineResult from pipeline_abstract.py
# Produces:
# - Evidence map (by cluster) with sentence-level MMR headline points (de-dup + coverage)
# - Bridge papers list
# - Limitations/contradictions (only limitation/failure/comparison cues)
# - Gaps (rules)
# - New directions (rules) with axis-matched "Because" evidence

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple
from collections import defaultdict, Counter

from pipeline_abstract import (
    PipelineResult, Cluster, Paper,
    novelty_leaderboards,
    cluster_headline_points,
    BREAKTHROUGH_AXES, CONFOUNDER_AXES
)


# -----------------------------
# Helpers
# -----------------------------

def md_escape(s: str) -> str:
    return s.replace("\n", " ").strip()


def paper_line(p: Paper) -> str:
    upd = f" · Updated: {p.updated}" if p.updated else ""
    pdf = f" · PDF: {p.pdf_url}" if p.pdf_url else ""
    return f"{p.title}{upd}{pdf}"


def key_sentences_block(p: Paper) -> str:
    lines = []
    if p.contradiction_sentences:
        for s in p.contradiction_sentences[:3]:
            lines.append(f"• (⚠️ cue-hit) {md_escape(s)}")
    elif p.claim_sentences:
        for s in p.claim_sentences[:3]:
            lines.append(f"• {md_escape(s)}")
    return "\n".join(lines) if lines else "• (no cue-hit sentence found in abstract)"


def axis_reasonable_for_breakthrough(axis: str) -> bool:
    return axis in BREAKTHROUGH_AXES


def axis_reasonable_for_confounder(axis: str) -> bool:
    return axis in CONFOUNDER_AXES


# -----------------------------
# Synthesis core
# -----------------------------

def synthesize_key_points_and_directions(
    result: PipelineResult,
    top_per_cluster: int = 5,
    top_bridge: int = 10,
    top_limits: int = 12,
) -> str:
    """
    Produce a markdown synthesis from title+abstract only.
    No LLM.
    """
    out: List[str] = []
    out.append("## Synthesis: key points & new directions (No LLM)")
    out.append("基于当前检索到的 papers（title+abstract），自动产出：证据地图、限制/争议、缺口、以及可执行的新研究方向。")
    out.append("")

    # A) Evidence map
    out.append("### A) Evidence map (by cluster)")
    for cl in result.clusters:
        papers = [result.id2paper[pid] for pid in cl.paper_ids]
        counts = Counter(p.paper_type for p in papers)
        kw = ", ".join(cl.keywords[:10]) if cl.keywords else "(no keywords)"

        out.append(f"**Cluster {cl.cid}** · size={len(papers)} · keywords: {kw} · (method/theory/empirical={dict(counts)})")
        # Headline points: sentence-level MMR (de-dup + coverage)
        points = cluster_headline_points(result, cl, max_points=top_per_cluster, lambda_rel=0.65)
        if points:
            out.append("")
            out.append("Headline points (MMR-selected, de-duplicated):")
            for sent, title in points:
                out.append(f"- {md_escape(sent)} — *{md_escape(title)}*")
        else:
            out.append("")
            out.append("- (No headline points extracted.)")

        # Top evidence papers: by relevance within cluster
        top_p = sorted(papers, key=lambda p: p.relevance, reverse=True)[:min(5, len(papers))]
        out.append("")
        out.append("Top evidence papers:")
        for p in top_p:
            out.append(f"- {paper_line(p)}")
        out.append("")

    # B) Leaderboards (three novelty boards)
    out.append("### B) Leaderboards")
    lbs = novelty_leaderboards(result, topn=10)

    out.append("#### B1) Novelty leaderboard — **In-topic breakthrough candidates**")
    out.append("只在 `axis ∈ {objective/path, theory/bounds, evaluation/benchmark}` 内排序，避免“离题但离群”霸榜。")
    if lbs["novelty_in_topic"]:
        for i, p in enumerate(lbs["novelty_in_topic"], 1):
            out.append(
                f"{i}. **{p.title}**\n"
                f"   - {p.paper_type} · axis={p.axis} · Rel={p.relevance:.3f} · Nov={p.novelty:.3f} · NovRank={p.novelty_rank:.3f}\n"
                f"   - {paper_line(p)}\n\n"
                f"{key_sentences_block(p)}\n"
            )
    else:
        out.append("- (No in-topic novelty candidates found.)")
    out.append("")

    out.append("#### B2) Novelty leaderboard — **Confounder novelty (optimizer/dynamics)**")
    out.append("专门给 optimizer / training dynamics 的“离群点”，它们是混杂项/控制变量，不应该混进“突破榜”。")
    if lbs["novelty_confounders"]:
        for i, p in enumerate(lbs["novelty_confounders"], 1):
            out.append(
                f"{i}. **{p.title}**\n"
                f"   - {p.paper_type} · axis={p.axis} · Rel={p.relevance:.3f} · Nov={p.novelty:.3f} · NovRank={p.novelty_rank:.3f}\n"
                f"   - {paper_line(p)}\n\n"
                f"{key_sentences_block(p)}\n"
            )
    else:
        out.append("- (No confounder novelty papers found.)")
    out.append("")

    out.append("#### B3) Bridge leaderboard — **Cross-cluster connectors**")
    out.append("Bridge：同时接近多个簇中心（top2 centroid similarity 高，且 gap 小），并对 tutorial/survey 做惩罚。")
    if lbs["bridge_leaderboard"]:
        for i, p in enumerate(lbs["bridge_leaderboard"][:10], 1):
            out.append(
                f"{i}. **{p.title}**\n"
                f"   - {p.paper_type} · axis={p.axis} · Rel={p.relevance:.3f} · Bridge={p.bridge:.3f} · BridgeRank={p.bridge_rank:.3f}\n"
                f"   - {paper_line(p)}\n\n"
                f"{key_sentences_block(p)}\n"
            )
    else:
        out.append("- (No bridge papers found.)")
    out.append("")

    # C) Limitations / contradictions (ONLY limitation/failure/comparison cues)
    out.append("### C) Limitations / contradictions (cue-hit sentences)")
    out.append("只采集 limitation/failure/comparison（不再把 “we propose/we show” 当成矛盾证据）。")
    limit_pool: List[Tuple[float, str]] = []
    for p in result.papers:
        for s in p.contradiction_sentences:
            # rank by relevance of the paper (simple but works)
            limit_pool.append((p.relevance, f"• {md_escape(s)} — *{md_escape(p.title)}*"))
    limit_pool.sort(key=lambda x: x[0], reverse=True)
    if limit_pool:
        for _, line in limit_pool[:top_limits]:
            out.append(line)
    else:
        out.append("- (No limitation/failure/comparison cue-hit found in abstracts.)")
    out.append("")

    # D) Gaps (rules)
    out.append("### D) Gaps (what’s missing)")
    gaps = [
        "定义并固定 gradient variance 的测量协议：global norm vs per-layer vs per-block；在 clipping 前/后；window size；统计量（var/trace/percentiles）。",
        "显式控制会抹平 small-batch 差异的旋钮：gradient clipping、EMA、LR warmup、time/noise sampling、solver choice（Euler/RK）。",
        "seed-robust：至少 3 seeds + 报告 mean±std/CI；否则“variance”结论不可信。",
        "公平预算：matched NFE / steps / wall-clock；否则“效率优越”可能来自 baseline 不公平。",
        "把 variance 分解成三类来源并分别做对照：target-variance（score/target）、estimator-variance（timestep coupling）、optimizer/batch-noise（micro-batch/Adam）。",
        "Beyond CIFAR/MNIST：加入更难或分布偏移/腐蚀数据（corrupted data stress）检验是否能泛化。"
    ]
    for g in gaps:
        out.append(f"- {g}")
    out.append("")

    # E) New directions (rules-based) with axis-matched evidence
    out.append("### E) New directions (rules-based proposals)")
    out.append("下面每条都给出 *axis 匹配* 的 Because（避免把“统一框架论文”硬塞进 variance-bias 论证）。")

    # Build axis -> papers sorted by relevance
    axis2papers: Dict[str, List[Paper]] = defaultdict(list)
    for p in result.papers:
        axis2papers[p.axis].append(p)
    for ax in axis2papers:
        axis2papers[ax].sort(key=lambda p: p.relevance, reverse=True)

    def because(ax: str, k: int = 3) -> str:
        ps = axis2papers.get(ax, [])[:k]
        if not ps:
            return "(no axis-matched evidence found)"
        return "; ".join(p.title for p in ps)

    proposals = [
        (
            "建立一个 **variance taxonomy + measurement protocol**：把 variance 分解为 target-variance、estimator-variance、optimizer/batch-noise 三类，并在同一 backbone + 同一日志定义下系统对照。",
            f"Because (axis-matched): optimizer/dynamics → {because('optimizer/dynamics')}; objective/path → {because('objective/path')}; theory/bounds → {because('theory/bounds')}"
        ),
        (
            "做一个 **桥梁型统一视角实验**：以 Generator Matching / time-dependent reweighting 为统一框架，把 FM / diffusion / rectified flow 放在同一损失家族里，定位 variance 降低来自：路径分布 vs 时间采样 vs loss 形式。",
            f"Because (axis-matched): objective/path → {because('objective/path')}"
        ),
        (
            "研究 **variance–bias trade-off 的可控旋钮**：STF 类用 reference batch 降 variance 引入 bias；TPC 类降 estimator variance；提出统一的 bias budget 指标，把不同方法放进同一 Pareto frontier。",
            f"Because (axis-matched): objective/path → {because('objective/path')}; evaluation/benchmark → {because('evaluation/benchmark')}"
        ),
        (
            "针对 small-batch：把 **batch-size invariant optimizer** 作为控制变量，测试 FM vs diffusion 的差异是否仍存在；若差异消失，说明差异主要来自 optimizer/batch-noise 而非范式本身。",
            f"Because (axis-matched): optimizer/dynamics → {because('optimizer/dynamics')}"
        ),
        (
            "把 **failure modes 作为主任务**：定义可复现 failure suite（loss spikes/divergence/mode depletion/corrupted-data stress），比较不同 variance-reduction 在 failure suite 上的鲁棒性。",
            f"Because (axis-matched): evaluation/benchmark → {because('evaluation/benchmark')}"
        ),
    ]

    for ptxt, btxt in proposals:
        out.append(f"- **{ptxt}**\n  - {btxt}")

    out.append("")
    out.append("### F) Selected papers (MMR diversified list)")
    if result.selected_ids:
        for pid in result.selected_ids:
            p = result.id2paper[pid]
            out.append(f"- {paper_line(p)}")
    else:
        out.append("- (No selected papers.)")

    return "\n".join(out)
