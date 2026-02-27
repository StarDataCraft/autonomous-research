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


st.set_page_config(page_title="Autonomous Research Engine", layout="wide")
st.title("Autonomous Research Engine")
st.caption("主界面：arXiv 搜索 + embedding 阅读（Abstract-only，No LLM）。另附：旧版 Pipeline（模板/可选LLM）。")

with st.sidebar:
    st.header("Mode (optional)")
    use_llm = st.toggle("Use LLM (OpenAI via st.secrets)", value=llm_available())
    st.caption("注意：No-LLM 主引擎不依赖这个开关。仅影响第二个 tab 的旧 pipeline。")

    st.divider()
    st.header("Timebox")
    minutes = st.slider("Sprint length (minutes)", 10, 60, 20, step=5)
    st.write(f"当前：{minutes} 分钟 sprint")


# ✅ 把 Search tab 放在第一个（主界面）
tab_main, tab_pipeline = st.tabs(["Search & Read Abstracts (No LLM)  ⭐", "Pipeline (Template/LLM optional)"])


# =========================
# MAIN TAB: Abstract engine
# =========================
with tab_main:
    st.subheader("arXiv Search + Embedding Read (Abstract-only)")
    st.caption(
        "流程：Recall(arXiv) → Embedding re-rank → MMR diversify → Cluster map → "
        "Novelty score（异端/突破点）→ Contradiction hunt（claims/limits/failure sentences）"
    )

    q = st.text_area(
        "Research question / hypothesis",
        value="Flow matching vs diffusion: small-batch stability and gradient variance",
        height=90,
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        category = st.selectbox("arXiv category filter", ["cs.LG", "cs.CV", "cs.AI", "stat.ML", "none"], index=0)
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
        contradiction_top_n = st.slider("Contradiction sentences per paper", 1, 5, 3, step=1)

    go = st.button("Search & organize", type="primary")

    if go:
        with st.status("Searching and organizing...", expanded=True) as status:
            insights, clusters, diag = run_abstract_pipeline(
                hypothesis=q.strip(),
                max_results_per_query=max_per_q,
                mmr_k=mmr_k,
                mmr_lambda=mmr_lambda,
                category_filter=category_filter,
                embed_model=embed_model,
                contradiction_top_n=contradiction_top_n,
            )
            status.update(label="Done", state="complete", expanded=False)

        st.success(
            f"Candidates: {diag['candidate_count']} → Selected: {diag['selected_count']} → Clusters: {diag['cluster_count']}"
        )

        with st.expander("Diagnostics (queries used)"):
            st.write(diag["queries"])
            st.write("Embedding model:", diag["embed_model"])

        # ===== Novelty leaderboard =====
        st.markdown("## Novelty leaderboard (异端/突破点候选)")
        st.caption("Novelty = 1 - max similarity to papers in other clusters. 越高越可能是“跟别的簇都不像”的点。")

        novelty_sorted = sorted(insights, key=lambda x: (-x.novelty, -x.relevance))
        topN = novelty_sorted[: min(12, len(novelty_sorted))]
        for rank, it in enumerate(topN, start=1):
            p = it.paper
            st.markdown(
                f"**{rank}.** [{p.title}]({p.entry_url})  \n"
                f"Cluster={it.cluster_id}  ·  Relevance={it.relevance:.3f}  ·  **Novelty={it.novelty:.3f}**  \n"
                f"*{', '.join(p.authors[:6])}*  ·  Updated: {p.updated}  ·  PDF: {p.pdf_url}"
            )
            if it.contradiction_sentences:
                st.markdown("**Contradiction/Limit/Fault hints (from abstract):**")
                for s in it.contradiction_sentences:
                    st.write("• " + s)
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
                f"*{', '.join(rp.authors[:6])}*  \n"
                f"Updated: {rp.updated}  ·  Category: {rp.primary_category or 'n/a'}  ·  "
                f"Relevance={rep.relevance:.3f}  ·  Novelty={rep.novelty:.3f}"
            )

            st.markdown("**Top papers in this cluster:**")
            for it in c.papers[:6]:
                p = it.paper
                st.markdown(
                    f"- [{p.title}]({p.entry_url})  "
                    f"(Rel={it.relevance:.3f}, Nov={it.novelty:.3f})  \n"
                    f"  *{', '.join(p.authors[:6])}*  ·  Updated: {p.updated}"
                )

            st.markdown("**Contradiction hunt (rep):**")
            if rep.contradiction_sentences:
                for s in rep.contradiction_sentences:
                    st.write("• " + s)
            else:
                st.write("(none)")

            st.markdown("**Abstract (rep):**")
            st.write(rp.summary)
            st.divider()

        # ===== Diversified list =====
        st.markdown("## Selected papers (MMR diversified list)")
        mmr_sorted = sorted(insights, key=lambda x: (-x.relevance, -x.novelty))
        for i, it in enumerate(mmr_sorted, start=1):
            p = it.paper
            st.markdown(
                f"{i}. [{p.title}]({p.entry_url})  \n"
                f"   Cluster={it.cluster_id}  ·  Rel={it.relevance:.3f}  ·  Nov={it.novelty:.3f}  ·  "
                f"Updated: {p.updated}  ·  PDF: {p.pdf_url}"
            )


# =========================
# Secondary TAB: Old pipeline
# =========================
with tab_pipeline:
    st.subheader("Pipeline (Template/LLM optional)")
    st.caption("这部分保留你原来的结构（可开LLM / 或模板）。主引擎请用第一个 tab。")

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

    run = st.button("Run pipeline", type="primary")

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
