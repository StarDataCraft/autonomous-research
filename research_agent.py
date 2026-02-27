from __future__ import annotations

from dataclasses import dataclass
from typing import List
import os
import streamlit as st

from prompts import (
    PROMPT_LIT_COMPRESSION,
    PROMPT_ATTACK,
    PROMPT_EXPERIMENT_PLAN,
)
from utils import bulletify, trim_block

# Optional OpenAI integration
try:
    from openai import OpenAI  # type: ignore
except Exception:
    OpenAI = None


@dataclass
class HypothesisSpec:
    hypothesis: str
    metrics: List[str]
    controls: List[str]
    failure_condition: str
    constraints: str


# ===============================
# Key management
# ===============================

def get_openai_key() -> str | None:
    """
    Priority:
    1) Streamlit Cloud secrets
    2) Environment variable
    """
    key = st.secrets.get("OPENAI_API_KEY", None)
    if key:
        return key
    key = os.getenv("OPENAI_API_KEY")
    if key:
        return key
    return None


def get_model_name() -> str:
    """
    Optional model override via secrets/env.
    """
    return (
        st.secrets.get("OPENAI_MODEL")
        or os.getenv("OPENAI_MODEL")
        or "gpt-4.1-mini"
    )


def llm_available() -> bool:
    return get_openai_key() is not None and OpenAI is not None


# ===============================
# OpenAI call
# ===============================

def _call_openai(system: str, user: str) -> str:
    api_key = get_openai_key()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not found in st.secrets or environment variable.")
    if OpenAI is None:
        raise RuntimeError("openai package not installed. Add `openai` to requirements.txt.")

    model = get_model_name()
    client = OpenAI(api_key=api_key)

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.2,
        max_tokens=1400,
    )
    return resp.choices[0].message.content or ""


# ===============================
# Formatting helper
# ===============================

def _spec_block(spec: HypothesisSpec, context: str = "") -> str:
    return trim_block(
        f"""
        Hypothesis:
        {spec.hypothesis}

        Metrics:
        {bulletify(spec.metrics)}

        Controls:
        {bulletify(spec.controls)}

        Failure condition:
        {spec.failure_condition}

        Constraints:
        {spec.constraints}

        Context:
        {context.strip() if context else "(none)"}
        """
    )


# ===============================
# Literature Compression
# ===============================

def run_literature_compression(spec: HypothesisSpec, context: str = "", use_llm: bool = False) -> str:
    spec_txt = _spec_block(spec, context=context)

    if use_llm and llm_available():
        try:
            return _call_openai(
                system="You are a meticulous research assistant. Be precise and structured.",
                user=PROMPT_LIT_COMPRESSION.format(spec=spec_txt),
            ).strip()
        except Exception as e:
            # ✅ Show real error instead of silently falling back
            return trim_block(
                f"""
                ## LLM ERROR (Literature Compression)

                The app tried to call OpenAI but failed.

                **Error message:**
                {str(e)}
                """
            )

    # Fallback template
    return trim_block(
        """
        ## Literature Compression (Template Mode)

        Focus only on evidence relevant to your hypothesis (not a general survey).

        ### Suggested search queries:
        1. "flow matching" AND diffusion AND ("small batch" OR "batch size") AND stability
        2. ("flow matching" OR "rectified flow") AND ("training stability" OR "gradient variance")
        3. DDPM AND ("small batch" OR "batch size") AND ("gradient noise" OR "variance")
        4. "rectified flow" ablation batch size CIFAR-10

        ### What to extract from each paper:
        - dataset, batch size, backbone
        - optimizer + schedule
        - instability notes (loss spikes, divergence, NaNs)
        - any variance proxy
        - negative results / limitations
        """
    )


# ===============================
# Attack Phase
# ===============================

def run_attack_phase(spec: HypothesisSpec, context: str = "", use_llm: bool = False) -> str:
    spec_txt = _spec_block(spec, context=context)

    if use_llm and llm_available():
        try:
            return _call_openai(
                system="You are a harsh NeurIPS/ICML reviewer. Be adversarial and specific.",
                user=PROMPT_ATTACK.format(spec=spec_txt),
            ).strip()
        except Exception as e:
            return trim_block(
                f"""
                ## LLM ERROR (Attack Phase)

                The app tried to call OpenAI but failed.

                **Error message:**
                {str(e)}
                """
            )

    # Fallback template
    return trim_block(
        """
        ## Reviewer Attack Checklist

        - Optimizer differences
        - LR schedule sensitivity
        - EMA masking instability
        - Dataset dependence
        - Logging definition of gradient norm
        - Random seed variance
        - Small batch instability may vanish with clipping

        Minimum demand:
        - 3 seeds
        - Mean ± std
        - Identical compute budget
        """
    )


# ===============================
# Experiment Plan
# ===============================

def design_minimal_experiment(spec: HypothesisSpec, context: str = "", use_llm: bool = False) -> str:
    spec_txt = _spec_block(spec, context=context)

    if use_llm and llm_available():
        try:
            return _call_openai(
                system="You are a practical ML researcher. Produce a minimal, controlled experiment plan.",
                user=PROMPT_EXPERIMENT_PLAN.format(spec=spec_txt),
            ).strip()
        except Exception as e:
            return trim_block(
                f"""
                ## LLM ERROR (Experiment Plan)

                The app tried to call OpenAI but failed.

                **Error message:**
                {str(e)}
                """
            )

    # Fallback template
    return trim_block(
        """
        ## Minimal Experiment Plan (Template Mode)

        Dataset: CIFAR-10
        Backbone: U-Net (same config)
        Methods: DDPM vs Flow Matching / Rectified Flow
        Batch sizes: 4, 8, 16
        Steps: 10k initial (first pass)

        Log:
        - loss per step
        - global grad norm
        - variance over 100-step windows

        Success:
        - variance_flow <= 0.95 * variance_ddpm
        - and FID not worse by >3 points after 10k steps
        """
    )


# ===============================
# Conclusion Template
# ===============================

def generate_conclusion_template(spec: HypothesisSpec) -> str:
    return trim_block(
        f"""
        ## Conclusion template (YOU write this)

        ### Hypothesis
        {spec.hypothesis}

        ### What we observed (facts only)
        - Gradient variance:
          - Batch=4: ...
          - Batch=8: ...
          - Batch=16: ...
        - Loss variance:
          - ...
        - Sample quality (FID @ 10k steps):
          - ...

        ### Does it support the hypothesis?
        - Supported / Not supported / Inconclusive
        - Why (1–2 sentences, grounded in metrics)

        ### Alternative explanations (must address)
        - Optimizer/schedule interaction
        - EMA masking instability
        - Metric definition sensitivity (where grad norms are computed)

        ### What would change your mind?
        - One extra ablation that could flip the conclusion

        ### Next decision
        - If supported: expand to bigger dataset or stronger baselines
        - If not: revise hypothesis or adjust regime
        """
    )
