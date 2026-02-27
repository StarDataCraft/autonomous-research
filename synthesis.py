# synthesis.py
from __future__ import annotations

from typing import Dict, List
import math

from pipeline_abstract import Paper, analyze_papers, split_sentences


def _fmt_float(x: float, nd: int = 3) -> str:
    try:
        return f"{float(x):.{nd}f}"
    except Exception:
        return "n/a"


def _paper_line(p: Dict) -> str:
    pap = p["paper"]
    title = pap["title"]
    updated = pap.get("updated", "") or pap.get("updated", "")
    pdf = pap.get("pdf_url", "") or pap.get("pdf", "")
    if pdf:
        return f"{title} · Updated: {updated} · PDF: {pdf}"
    return f"{title} · Updated: {updated}"


def _render_key_sentences(p: Dict) -> str:
    hits = p.get("key_sentences", [])
    if not hits:
        return ""
    # separate claim vs contra
    claims = [h for h in hits if h.get("kind") == "claim"]
    contras = [h for h in hits if h.get("kind") == "contra"]

    out = []
    if claims:
        out.append("Key sentences (claims/highlights):")
        for h in claims[:3]:
            cue = " (⚠️ cue-hit)" if h.get("cue_hit") else ""
            out.append(f"•{cue} {h.get('sentence','').strip()}")
        out.append("")

    if contras:
        out.append("Contradiction/Limit/Fault hints (ONLY limitation/failure/comparison cues):")
        for h in contras[:3]:
            out.append(f"• (⚠️ contra-cue) {h.get('sentence','').strip()}")
        out.append("")

    return "\n".join(out).rstrip()


def render_leaderboards(result: Dict) -> str:
    lbs = result.get("leaderboards", {})
    out = []
    out.append("## Leaderboards\n")

    def render_board(title: str, items: List[Dict], show_bridge: bool = False) -> None:
        out.append(f"### {title}")
        if not items:
            out.append("(none)\n")
            return
        for i, p in enumerate(items[:10], 1):
            pap = p["paper"]
            ptype = p.get("paper_type", "other")
            cid = p.get("cluster_id", -1)
            rel = _fmt_float(p.get("relevance", 0.0))
            nov = _fmt_float(p.get("novelty", 0.0))
            novr = _fmt_float(p.get("novelty_rank", 0.0))
            br = _fmt_float(p.get("bridge", 0.0))
            brr = _fmt_float(p.get("bridge_rank", 0.0))
            axis = p.get("axis", "n/a")
            pdf = pap.get("pdf_url", "")

            if show_bridge:
                out.append(f"{i}. {pap['title']}\n{ptype} · axis={axis} · Cluster={cid} · Rel={rel} · Bridge={br} · BridgeRank={brr}")
            else:
                out.append(f"{i}. {pap['title']}\n{ptype} · axis={axis} · Cluster={cid} · Rel={rel} · Nov={nov} · NovRank={novr}")

            meta_line = f"{pap.get('authors','')} · Updated: {pap.get('updated','')}"
            if pdf:
                meta_line += f" · PDF: {pdf}"
            out.append(meta_line)

            ks = _render_key_sentences(p)
            if ks:
                out.append("\n" + ks)
            out.append("")

    render_board("Novelty leaderboard (in-topic) — 突破/异端候选（已抑制离题离群）", lbs.get("novelty_in_topic", []))
    render_board("Novelty leaderboard (confounders) — 训练/优化器离群点（单独榜）", lbs.get("novelty_confounder", []))
    render_board("Bridge leaderboard — 跨簇连接点（统一视角候选）", lbs.get("bridge", []), show_bridge=True)

    return "\n".join(out).strip()


def render_synthesis(result: Dict) -> str:
    """
    Produce:
      A) Evidence map (by cluster) with MMR headline points
      B) Bridge papers
      C) Limitations/contradictions (contra-only)
      D) Gaps
      E) New directions (rules-based, axis-aligned)
    """
    scored = result.get("scored_papers", [])
    clusters = result.get("clusters", {})
    meta = result.get("meta", {})

    if not scored:
        return "No papers to synthesize."

    # group by cluster
    by_cluster: Dict[int, List[Dict]] = {}
    for p in scored:
        cid = int(p.get("cluster_id", -1))
        by_cluster.setdefault(cid, []).append(p)

    # helper: pick top evidence papers per cluster by relevance_rank-like (relevance * sqrt(novelty+bridge))
    def evidence_score(p: Dict) -> float:
        rel = float(p.get("relevance", 0.0))
        nov = float(p.get("novelty", 0.0))
        br = float(p.get("bridge", 0.0))
        return rel * (0.70 + 0.20 * math.sqrt(max(nov, 0.0)) + 0.10 * math.sqrt(max(br, 0.0)))

    # Contra pool across all papers
    contra_lines = []
    for p in scored:
        for h in p.get("key_sentences", []):
            if h.get("kind") == "contra":
                title = p["paper"]["title"]
                contra_lines.append((float(h.get("score", 0.0)), f"• {h.get('sentence','').strip()} — {title}"))

    contra_lines.sort(key=lambda x: x[0], reverse=True)

    # Bridge list (top by bridge_rank)
    bridge_sorted = sorted(scored, key=lambda x: float(x.get("bridge_rank", 0.0)), reverse=True)
    bridge_titles = [p["paper"]["title"] for p in bridge_sorted[:12] if float(p.get("bridge_rank", 0.0)) > 0.0]

    # Build synthesis text
    out = []
    out.append("## Synthesis: key points & new directions")
    out.append(f"(No LLM) · embedder={meta.get('embedder','n/a')} · k_clusters={meta.get('k_clusters','n/a')}\n")

    # A) Evidence map
    out.append("### A) Evidence map (by cluster)")
    for cid in sorted(by_cluster.keys()):
        plist = by_cluster[cid]
        info = clusters.get(cid, {})
        keywords = info.get("keywords", [])
        type_counts = info.get("type_counts", {})

        out.append(
            f"\n**Cluster {cid}** · size={len(plist)} · keywords: {', '.join(keywords) if keywords else 'n/a'} "
            f"(method/theory/empirical = {type_counts})"
        )

        # Headline points: use only CLAIMS (highlights) but MMR-like selection already done per paper.
        # We now de-duplicate at cluster-level by picking top unique claim sentences from top papers.
        # Strategy: pick top 2 papers by evidence_score, then take their claim sentences first, then fill.
        top_papers = sorted(plist, key=evidence_score, reverse=True)[:6]

        headline = []
        seen = set()
        for p in top_papers:
            title = p["paper"]["title"]
            for h in p.get("key_sentences", []):
                if h.get("kind") != "claim":
                    continue
                s = h.get("sentence", "").strip()
                key = s.lower()
                if key in seen:
                    continue
                seen.add(key)
                headline.append(f"• {s} — {title}")
                if len(headline) >= 5:
                    break
            if len(headline) >= 5:
                break

        if headline:
            out.append("\nHeadline points (MMR-ish, de-duplicated):\n" + "\n".join(headline))
        else:
            out.append("\nHeadline points:\n(none extracted)")

        # Top evidence papers list
        evid_titles = [p["paper"]["title"] for p in top_papers[:5]]
        out.append("\nTop evidence papers:\n" + "\n".join([f"• {t}" for t in evid_titles]))

    # B) Bridge papers
    out.append("\n### B) Bridge papers (cross-cluster connectors)")
    if bridge_titles:
        out.append("\n".join([f"• {t}" for t in bridge_titles[:12]]))
    else:
        out.append("(none)")

    # C) Limitations / contradictions
    out.append("\n### C) Limitations / contradictions (contra-only)")
    if contra_lines:
        out.append("\n".join([x[1] for x in contra_lines[:12]]))
    else:
        out.append("(none found in abstracts with contra-cues)")

    # D) Gaps (generic but *axis-aligned*)
    out.append("\n### D) Gaps (what’s missing)")
    out.append("• Define *exactly* how gradient variance is measured (global norm vs per-layer vs pre/post clipping) and keep it identical across methods.")
    out.append("• Control knobs that erase small-batch differences: gradient clipping, EMA, LR warmup, time/noise sampling, solver choice.")
    out.append("• Seed-robust comparisons (≥3 seeds) + uncertainty reporting (Std/CI); variance claims without uncertainty are weak.")
    out.append("• Match compute budget (steps, NFE, wall-clock) before concluding ‘efficiency’ or ‘stability’.")
    out.append("• Separate variance sources: target-variance (STF-like), estimator variance (TPC-like), optimizer/batch-noise (batch-invariant Adam).")
    out.append("• Add a harder/shifted regime beyond CIFAR/MNIST (higher-res, corrupted data, different modality) to test generalization.")

    # E) New directions (rules-based, but cited by axis)
    out.append("\n### E) New directions (rules-based proposals, axis-aligned)")

    # Build axis index
    axis_to_titles: Dict[str, List[str]] = {}
    for p in scored:
        ax = p.get("axis", "applications")
        axis_to_titles.setdefault(ax, []).append(p["paper"]["title"])

    def because(axis: str, k: int = 3) -> str:
        titles = axis_to_titles.get(axis, [])[:k]
        if not titles:
            return "Because: (no axis-matched papers found)"
        return "Because: " + "; ".join(titles)

    out.append("• Build a **variance taxonomy + measurement protocol**: decompose variance into (i) target variance, (ii) estimator variance, (iii) optimizer/batch-noise; report all under one logging definition and one backbone.")
    out.append("  " + because("objective/path"))

    out.append("• Run a **bridge-unification experiment**: under a single ‘generator matching / time-dependent reweighting’ lens, place diffusion / FM / rectified flow in one loss family and attribute variance reduction to (path distribution vs time sampling vs loss form).")
    out.append("  " + because("objective/path"))

    out.append("• Test **optimizer confounder hypothesis** in your exact small-batch regime: include batch-size-invariant optimizer as a control; if FM vs DDPM gap shrinks, the effect is mostly optimizer/batch-noise, not paradigm.")
    out.append("  " + because("optimizer/dynamics"))

    out.append("• Create a **failure-suite benchmark** (loss spikes, divergence, mode depletion, corrupted-data stress), then compare variance-reduction knobs across the suite, not just average FID.")
    out.append("  " + because("evaluation/benchmark"))

    return "\n".join(out).strip()


def run_full_synthesis_no_llm(
    papers: List[Paper],
    query_text: str,
    k_clusters: int = 4,
) -> Dict[str, str]:
    """
    Convenience wrapper: returns markdown strings for UI.
    """
    result = analyze_papers(papers=papers, query_text=query_text, k_clusters=k_clusters)
    return {
        "leaderboards_md": render_leaderboards(result),
        "synthesis_md": render_synthesis(result),
        "meta": str(result.get("meta", {})),
    }
