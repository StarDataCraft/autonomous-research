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


st.set_page_config(page_title="Autonomous Research Mini-Pipeline", layout="wide")
st.title("Autonomous Research Mini-Pipeline")
st.caption("LLM 可选；本页新增：arXiv 搜索 + 轻量 embedding 阅读（Abstract-only，免费/低成本）。")

with st.sidebar:
    st.header("Mode")
    use_llm = st.toggle("Use LLM (OpenAI via st.secrets)", value=llm_available())

    st.divider()
    st.header("Timebox")
    minutes = st.slider("Sprint length (minutes)", 10, 60, 20, step=5)
    st.write(f"当前：{minutes} 分钟 sprint")

tabA, tabB = st.tabs(["Pipeline (Template/LLM optional)", "Search & Read Abstracts (No LLM)"])


# =========================
# Tab A: your original pipeline
# =========================
with tabA:
    st.subheader("Hypothesis Spec")

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


# =========================
# Tab B: No-LLM abstract engine
# =========================
with tabB:
    st.subheader("arXiv Search + Embedding Read (Abstract-only)")
    st.caption("完全不调用大模型：用免费 arXiv 搜索 + 小 embedding 模型做重排、去冗余、多样化与聚类。")

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

    col4, col5 = st.columns(2)
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

    go = st.button("Search & organize", type="primary")

    if go:
        with st.status("Searching and organizing...", expanded=True) as status:
            selected, clusters, diag = run_abstract_pipeline(
                hypothesis=q.strip(),
                max_results_per_query=max_per_q,
                mmr_k=mmr_k,
                mmr_lambda=mmr_lambda,
                category_filter=category_filter,
                embed_model=embed_model,
            )
            status.update(label="Done", state="complete", expanded=False)

        st.success(
            f"Candidates: {diag['candidate_count']} → Selected: {diag['selected_count']} → Clusters: {diag['cluster_count']}"
        )

        with st.expander("Diagnostics (queries used)"):
            st.write(diag["queries"])

        st.markdown("## Cluster map (what to read)")
        for c in clusters:
            st.markdown(f"### Cluster {c.cluster_id}  ·  size={len(c.papers)}  ·  keywords: `{', '.join(c.keywords)}`")
            rep = c.centroid_paper
            st.markdown(f"**Representative:** [{rep.title}]({rep.entry_url})  \n"
                        f"*{', '.join(rep.authors[:6])}*  \n"
                        f"Updated: {rep.updated}  ·  Category: {rep.primary_category or 'n/a'}")

            st.markdown("**Top papers in this cluster:**")
            for p in c.papers[:6]:
                st.markdown(f"- [{p.title}]({p.entry_url})  \n"
                            f"  *{', '.join(p.authors[:6])}*  ·  Updated: {p.updated}")

            st.markdown("**Abstract (rep):**")
            st.write(rep.summary)
            st.divider()

        st.markdown("## Selected papers (MMR diversified list)")
        for i, p in enumerate(selected, start=1):
            st.markdown(
                f"{i}. [{p.title}]({p.entry_url})  \n"
                f"   *{', '.join(p.authors[:6])}*  ·  Updated: {p.updated}  ·  PDF: {p.pdf_url}"
            )
