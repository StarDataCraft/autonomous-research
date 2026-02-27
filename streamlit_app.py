import streamlit as st
from research_agent import (
    HypothesisSpec,
    run_literature_compression,
    run_attack_phase,
    design_minimal_experiment,
    generate_conclusion_template,
    llm_available,
)

st.set_page_config(page_title="Autonomous Research Mini-Pipeline", layout="wide")

st.title("Autonomous Research Mini-Pipeline")
st.caption(
    "20分钟结构化研究助手：Hypothesis → Literature Compression → Attack → Minimal Experiment → Conclusion.\n"
    "可选接入 LLM（OpenAI），否则使用规则模板输出。"
)

with st.sidebar:
    st.header("Mode")
    use_llm = st.toggle("Use LLM (OpenAI via st.secrets)", value=False)
    st.markdown(
        "- 如果不开 LLM：用结构化模板生成高质量 research 计划（可直接用）。\n"
        "- 如果开 LLM：会用你的输入生成更细致的文献查询关键词、攻击点、实验设计与解释。"
    )

    st.divider()
    st.header("Timebox")
    minutes = st.slider("Sprint length (minutes)", 10, 60, 20, step=5)
    st.write(f"当前：{minutes} 分钟 sprint")

st.divider()

col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("1) Hypothesis Spec（你来定义）")

    default_h = (
        "Flow Matching has lower gradient variance than DDPM when batch size ≤ 16."
    )
    hypothesis = st.text_area("Hypothesis (falsifiable)", value=default_h, height=90)

    metric = st.text_area(
        "Evaluation metrics (comma / newline separated)",
        value="Gradient norm variance\nLoss variance\nFID after 10k steps",
        height=90,
    )

    controls = st.text_area(
        "Control variables (what must be held constant)",
        value="Same backbone (U-Net)\nSame dataset (CIFAR-10)\nSame optimizer (AdamW)\nSame learning rate\nSame training steps",
        height=110,
    )

    failure = st.text_area(
        "Failure / reject condition (make it explicit)",
        value="Reject if variance difference < 5% OR Flow Matching FID is worse by > 3 points after 10k steps.",
        height=80,
    )

    constraints = st.text_area(
        "Constraints (compute, time, hardware, etc.)",
        value="Single GPU preferred; keep runtime under 2 hours for a first pass.",
        height=70,
    )

    spec = HypothesisSpec(
        hypothesis=hypothesis.strip(),
        metrics=[x.strip() for x in metric.splitlines() if x.strip()],
        controls=[x.strip() for x in controls.splitlines() if x.strip()],
        failure_condition=failure.strip(),
        constraints=constraints.strip(),
    )

    st.subheader("2) Optional: Context")
    context = st.text_area(
        "Extra context (your project / domain / assumptions). Optional.",
        value="I care about small-batch stability and reproducibility. Prefer minimal, clean experiments.",
        height=90,
    )

with col2:
    st.subheader("3) Run the pipeline")

    if use_llm and not llm_available():
        st.warning(
            "LLM mode is ON but no OpenAI key found in st.secrets.\n"
            "Add it in Streamlit Cloud → App settings → Secrets:\n"
            "OPENAI_API_KEY=\"...\""
        )

    run = st.button("Run autonomous research sprint", type="primary")

    if run:
        with st.status("Running pipeline...", expanded=True) as status:
            st.write("Step A — Literature Compression")
            lit = run_literature_compression(spec, context=context, use_llm=use_llm)
            st.write("Step B — Attack Phase (Reviewer mode)")
            attack = run_attack_phase(spec, context=context, use_llm=use_llm)
            st.write("Step C — Minimal Experiment Design")
            exp = design_minimal_experiment(spec, context=context, use_llm=use_llm)
            st.write("Step D — Conclusion Template (you write, not the agent)")
            concl = generate_conclusion_template(spec)

            status.update(label="Done", state="complete", expanded=False)

        st.success("Pipeline output generated.")

        tab1, tab2, tab3, tab4 = st.tabs(
            ["Literature Compression", "Attack Phase", "Experiment Plan", "Conclusion Template"]
        )
        with tab1:
            st.markdown(lit)
        with tab2:
            st.markdown(attack)
        with tab3:
            st.markdown(exp)
        with tab4:
            st.markdown(concl)

st.divider()
st.subheader("Export")
st.caption("你可以把输出复制到 Notion / Google Doc / issue tracker。后续我也可以帮你把它变成论文大纲。")
