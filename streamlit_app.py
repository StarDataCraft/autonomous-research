import streamlit as st

from research_agent import (
    HypothesisSpec,
    run_literature_compression,
    run_attack_phase,
    design_minimal_experiment,
    generate_conclusion_template,
    llm_available,
)

from pipeline_abstract import run_abstract_pipeline
from synthesis import run_full_synthesis_no_llm


st.set_page_config(page_title="Autonomous Research Engine", layout="wide")
st.title("Autonomous Research Engine")
st.caption(
    "主界面：arXiv 搜索 + embedding 阅读（Abstract-only，No LLM）。"
    "按钮不会清零：所有结果保存在 Streamlit session_state。"
)

# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.header("Mode (optional)")
    use_llm = st.toggle("Use LLM (OpenAI via st.secrets)", value=llm_available())
    st.caption("No-LLM 主引擎不依赖这个开关。仅影响第二个 tab 的旧 pipeline。")

    st.divider()
    st.header("Timebox")
    minutes = st.slider("Sprint length (minutes)", 10, 60, 20, step=5)
    st.write(f"当前：{minutes} 分钟 sprint")

# -----------------------------
# Tabs
# -----------------------------
tab_main, tab_pipeline = st.tabs(
    ["Search & Read Abstracts (No LLM)  ⭐", "Pipeline (Template/LLM optional)"]
)


def _badge(text: str) -> str:
    return f"`{text}`"


# =========================================================
# MAIN TAB (No LLM) — stateful, won't reset on button click
# =========================================================
with tab_main:
    st.subheader("arXiv Search + Embedding Read (Abstract-only)")
    st.caption(
        "流程：Recall(arXiv) → Embedding re-rank → MMR diversify → Cluster map → "
        "Novelty（离群但相关）→ Bridge（跨簇连接点）→ Type（method/theory/empirical）→ Key sentences（cue优先）"
    )

    # ---------- Inputs ----------
    q = st.text_area(
        "Research question / hypothesis",
        value="Flow matching vs diffusion: small-batch stability and gradient variance",
        height=90,
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        category = st.selectbox(
            "arXiv category filter",
            ["cs.LG", "cs.CV", "cs.AI", "stat.ML", "none"],
            index=0,
        )
        category_filter = None if category == "none" else category
    with col2:
        max_per_q = st.slider("Max results per query", 10, 100, 50, step=10)
    with col3:
        mmr_k = st.slider("MMR select K", 8, 60, 24, step=4)

    col4, col5, col6 = st.columns(3)
    with col4:
        mmr_lambda = st.slider("MMR λ (relevance vs diversity)", 0.1, 0.9, 0.7, step=0.1)
    with col5:
        embed_model = st.selectbox(
            "Embedding model",
            [
                "sentence-transformers/all-MiniLM-L6-v2",
                "BAAI/bge-small-en-v1.5",
            ],
            index=0,
        )
    with col6:
        key_sents_n = st.slider("Key sentences per paper", 1, 5, 3, step=1)

    # ---------- Session state init ----------
    if "insights" not in st.session_state:
        st.session_state["insights"] = None
    if "clusters" not in st.session_state:
        st.session_state["clusters"] = None
    if "diag" not in st.session_state:
        st.session_state["diag"] = None

    if "synth_report" not in st.session_state:
        st.session_state["synth_report"] = None
    if "did_synthesize" not in st.session_state:
        st.session_state["did_synthesize"] = False

    # ---------- Buttons (stable keys) ----------
    colb1, colb2, colb3 = st.columns([1, 1, 3])
    with colb1:
        go = st.button("Search & organize", type="primary", key="btn_search_v1")
    with colb2:
        clear = st.button("Clear results", key="btn_clear_v1")
    with colb3:
        st.caption("提示：点任何按钮都会 rerun，但结果都保存在 session_state，不会丢。")

    if clear:
        st.session_state["insights"] = None
        st.session_state["clusters"] = None
        st.session_state["diag"] = None
        st.session_state["synth_report"] = None
        st.session_state["did_synthesize"] = False
        st.rerun()

    if go:
        with st.status("Searching and organizing...", expanded=True) as status:
            insights, clusters, diag = run_abstract_pipeline(
                hypothesis=q.strip(),
                max_results_per_query=max_per_q,
                mmr_k=mmr_k,
                mmr_lambda=mmr_lambda,
                category_filter=category_filter,
                embed_model=embed_model,
                contradiction_top_n=key_sents_n,
            )
            status.update(label="Done", state="complete", expanded=False)

        st.session_state["insights"] = insights
        st.session_state["clusters"] = clusters
        st.session_state["diag"] = diag

        # reset synthesis
        st.session_state["synth_report"] = None
        st.session_state["did_synthesize"] = False

        st.rerun()

    # ---------- Read from state ----------
    insights = st.session_state.get("insights")
    clusters = st.session_state.get("clusters")
    diag = st.session_state.get("diag")

    if not (insights and clusters and diag):
        st.info("先点 **Search & organize** 生成结果。")
    else:
        st.success(
            f"Candidates: {diag['candidate_count']} → Selected: {diag['selected_count']} → Clusters: {diag['cluster_count']}"
        )

        with st.expander("Diagnostics (queries used)"):
            st.write(diag.get("queries"))
            st.write("Embedding model:", diag.get("embed_model"))

        # ===== Leaderboards =====
        st.markdown("## Leaderboards")
        colA, colB = st.columns(2)

        with colA:
            st.markdown("### Novelty leaderboard (异端/突破点候选)")
            st.caption("按 `novelty_rank = novelty × f(relevance)` 排序，避免“离题但离群”霸榜。")
            novelty_sorted = sorted(insights, key=lambda x: (-x.novelty_rank, -x.relevance))
            topN = novelty_sorted[: min(10, len(novelty_sorted))]

            for rank, it in enumerate(topN, start=1):
                p = it.paper
                st.markdown(
                    f"**{rank}.** [{p.title}]({p.entry_url})  \n"
                    f"{_badge(it.paper_type)}  Cluster={it.cluster_id}  ·  "
                    f"Rel={it.relevance:.3f}  ·  Nov={it.novelty:.3f}  ·  **NovRank={it.novelty_rank:.3f}**  \n"
                    f"*{', '.join(p.authors[:6])}*  ·  Updated: {p.updated}  ·  PDF: {p.pdf_url}"
                )
                if getattr(it, "key_sentences", None):
                    st.markdown("**Key sentences (cue-hit preferred):**")
                    for h in it.key_sentences:
                        tag = "⚠️ cue-hit" if h.kind == "cue-hit" else "relevance"
                        st.write(f"• ({tag}) {h.sentence}")
                st.divider()

        with colB:
            st.markdown("### Bridge papers (跨簇连接点)")
            st.caption("Bridge：同时接近多个簇中心（top2 centroid similarity 高，且 gap 小）。")
            bridge_sorted = sorted(insights, key=lambda x: (-x.bridge_rank, -x.relevance))
            topB = bridge_sorted[: min(10, len(bridge_sorted))]

            for rank, it in enumerate(topB, start=1):
                p = it.paper
                sims = sorted(it.cluster_sims.items(), key=lambda kv: kv[1], reverse=True)
                sim_str = ", ".join([f"C{cid}:{v:.2f}" for cid, v in sims[:3]])

                st.markdown(
                    f"**{rank}.** [{p.title}]({p.entry_url})  \n"
                    f"{_badge(it.paper_type)}  Cluster={it.cluster_id}  ·  "
                    f"Rel={it.relevance:.3f}  ·  Bridge={it.bridge_score:.3f}  ·  **BridgeRank={it.bridge_rank:.3f}**  \n"
                    f"Centroid sims: {sim_str}  \n"
                    f"*{', '.join(p.authors[:6])}*  ·  Updated: {p.updated}  ·  PDF: {p.pdf_url}"
                )
                if getattr(it, "key_sentences", None):
                    st.markdown("**Key sentences:**")
                    for h in it.key_sentences:
                        tag = "⚠️ cue-hit" if h.kind == "cue-hit" else "relevance"
                        st.write(f"• ({tag}) {h.sentence}")
                st.divider()

        # ===== Cluster map =====
        st.markdown("## Cluster map (what to read)")
        for c in clusters:
            st.markdown(
                f"### Cluster {c.cluster_id}  ·  size={len(c.papers)}  ·  keywords: `{', '.join(c.keywords)}`"
            )
            rep = c.centroid_paper
            rp = rep.paper

            st.markdown(
                f"**Representative:** [{rp.title}]({rp.entry_url})  \n"
                f"{_badge(rep.paper_type)}  Updated: {rp.updated}  ·  Category: {rp.primary_category or 'n/a'}  ·  "
                f"Rel={rep.relevance:.3f}  ·  Nov={rep.novelty:.3f}  ·  Bridge={rep.bridge_score:.3f}"
            )

            st.markdown("**Top papers in this cluster:**")
            for it in c.papers[:6]:
                p = it.paper
                st.markdown(
                    f"- [{p.title}]({p.entry_url})  "
                    f"{_badge(it.paper_type)}  "
                    f"(Rel={it.relevance:.3f}, NovRank={it.novelty_rank:.3f}, BridgeRank={it.bridge_rank:.3f})  \n"
                    f"  *{', '.join(p.authors[:6])}*  ·  Updated: {p.updated}"
                )

            st.markdown("**Key sentences (rep):**")
            if getattr(rep, "key_sentences", None):
                for h in rep.key_sentences:
                    tag = "⚠️ cue-hit" if h.kind == "cue-hit" else "relevance"
                    st.write(f"• ({tag}) {h.sentence}")
            else:
                st.write("(none)")

            st.markdown("**Abstract (rep):**")
            st.write(rp.summary)
            st.divider()

        # ===== MMR list =====
        st.markdown("## Selected papers (MMR diversified list)")
        mmr_sorted = sorted(insights, key=lambda x: (-x.relevance, -x.bridge_rank, -x.novelty_rank))
        for i, it in enumerate(mmr_sorted, start=1):
            p = it.paper
            st.markdown(
                f"{i}. [{p.title}]({p.entry_url})  \n"
                f"   {_badge(it.paper_type)}  Cluster={it.cluster_id}  ·  "
                f"Rel={it.relevance:.3f}  ·  NovRank={it.novelty_rank:.3f}  ·  BridgeRank={it.bridge_rank:.3f}  ·  "
                f"Updated: {p.updated}  ·  PDF: {p.pdf_url}"
            )

        # ===== Synthesis =====
        st.markdown("## Synthesis: key points & new directions")
        st.caption("基于当前检索到的 papers（title+abstract），自动产出：证据地图、限制/争议、缺口、以及可执行的新研究方向（No LLM）。")

        synth = st.button("Synthesize: Key Points & New Directions", key="btn_synth_v1")
        if synth:
            st.session_state["synth_report"] = synthesize_from_clusters(clusters, insights)
            st.session_state["did_synthesize"] = True
            st.rerun()

        if st.session_state.get("did_synthesize") and st.session_state.get("synth_report") is not None:
            report = st.session_state["synth_report"]

            st.markdown("### A) Evidence map (by cluster)")
            for t in report.theme_summaries:
                st.markdown(
                    f"#### Cluster {t.cluster_id} · keywords: `{', '.join(t.keywords)}`  "
                    f"(method/theory/empirical = {t.method_theory_empirical})"
                )
                st.markdown("**Headline points (from abstracts):**")
                for p in t.headline_points:
                    st.write("• " + p)
                st.markdown("**Top evidence papers:**")
                for title in t.evidence_papers:
                    st.write("• " + title)
                st.divider()

            st.markdown("### B) Bridge papers (cross-cluster connectors)")
            for i, title in enumerate(report.cross_cluster_bridges, start=1):
                st.write(f"{i}. {title}")

            st.markdown("### C) Limitations / contradictions (cue-hit sentences)")
            for p in report.contradictions:
                st.write("• " + p)

            st.markdown("### D) Gaps (what’s missing)")
            for g in report.gaps:
                st.write("• " + g)

            st.markdown("### E) New directions (rules-based proposals)")
            for d in report.new_directions:
                st.markdown("• " + d)


# =========================================================
# PIPELINE TAB (old template / optional LLM)
# =========================================================
with tab_pipeline:
    st.subheader("Pipeline (Template/LLM optional)")
    st.caption("保留你旧版的 Hypothesis → Literature → Attack → Experiment → Conclusion。主引擎请用第一个 tab。")

    default_h = "Flow Matching has lower gradient variance than DDPM when batch size ≤ 16."
    hypothesis = st.text_area("Hypothesis (falsifiable)", value=default_h, height=90)

    metric = st.text_area(
        "Evaluation metrics (newline separated)",
        value="Gradient norm variance\nLoss variance\nFID after 10k steps",
        height=90,
    )

    controls = st.text_area(
        "Control variables (newline separated)",
        value=(
            "Same backbone (U-Net)\n"
            "Same dataset (CIFAR-10)\n"
            "Same optimizer (AdamW)\n"
            "Same learning rate\n"
            "Same training steps"
        ),
        height=120,
    )

    failure = st.text_area(
        "Failure / reject condition (make it explicit)",
        value="Reject if variance difference < 5% OR Flow Matching FID is worse by > 3 points after 10k steps.",
        height=90,
    )

    constraints = st.text_area(
        "Constraints (compute, time, hardware, etc.)",
        value="Single GPU preferred; keep runtime under 2 hours for a first pass.",
        height=80,
    )

    spec = HypothesisSpec(
        hypothesis=hypothesis.strip(),
        metrics=[x.strip() for x in metric.splitlines() if x.strip()],
        controls=[x.strip() for x in controls.splitlines() if x.strip()],
        failure_condition=failure.strip(),
        constraints=constraints.strip(),
    )

    context = st.text_area(
        "Extra context (optional)",
        value="I care about small-batch stability and reproducibility. Prefer minimal, clean experiments.",
        height=100,
    )

    if use_llm and not llm_available():
        st.warning("LLM is ON but not available (missing key or openai package). Will fall back to template.")

    run = st.button("Run pipeline", type="primary", key="btn_run_pipeline_v1")

    if run:
        with st.status("Running pipeline...", expanded=True) as status:
            lit = run_literature_compression(spec, context=context, use_llm=use_llm)
            attack = run_attack_phase(spec, context=context, use_llm=use_llm)
            exp = design_minimal_experiment(spec, context=context, use_llm=use_llm)
            concl = generate_conclusion_template(spec)
            status.update(label="Done", state="complete", expanded=False)

        t1, t2, t3, t4 = st.tabs(["Literature", "Attack", "Experiment", "Conclusion Template"])
        with t1:
            st.markdown(lit)
        with t2:
            st.markdown(attack)
        with t3:
            st.markdown(exp)
        with t4:
            st.markdown(concl)

# --- Backward-compatible API for existing streamlit_app.py ---
# Your app expects: from synthesis import synthesize_from_clusters
# We provide it as a thin wrapper over the new pipeline.

from typing import Any, Dict, List, Tuple

def synthesize_from_clusters(
    clusters: Dict[int, Dict[str, Any]],
    scored_papers: List[Dict[str, Any]],
    query_text: str = "",
) -> str:
    """
    Backward compatible wrapper.
    Expected legacy inputs:
      - clusters: cluster metadata dict
      - scored_papers: list of scored paper dicts
    Returns:
      - markdown string synthesis
    """
    # Build a fake "result" object compatible with render_synthesis/render_leaderboards
    result = {
        "clusters": clusters or {},
        "scored_papers": scored_papers or [],
        "leaderboards": {},  # optional; legacy path may not need it
        "meta": {"embedder": "unknown", "k_clusters": len(clusters or {})},
    }
    return render_synthesis(result)
