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

# Try importing OpenAI safely
try:
    from openai import OpenAI
except Exception:
    OpenAI = None


# ===============================
# Data structure
# ===============================

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
    1. Streamlit Cloud secrets
    2. Environment variable
    """
    # Streamlit Cloud
    key = st.secrets.get("OPENAI_API_KEY", None)
    if key:
        return key

    # Environment variable
    key = os.getenv("OPENAI_API_KEY")
    if key:
        return key

    return None


def llm_available() -> bool:
    return get_openai_key() is not None and OpenAI is not None


# ===============================
# LLM Call
# ===============================

def _call_openai(system: str, user: str) -> str:
    api_key = get_openai_key()

    if not api_key or OpenAI is None:
        raise RuntimeError("OpenAI not available.")

    client = OpenAI(api_key=api_key)

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.2,
        max_tokens=1200,
    )

    return response.choices[0].message.content or ""


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

def run_literature_compression(
    spec: HypothesisSpec,
    context: str = "",
    use_llm: bool = False,
) -> str:

    spec_txt = _spec_block(spec, context=context)

    if use_llm and llm_available():
        try:
            return _call_openai(
                system="You are a meticulous research assistant. Be precise and structured.",
                user=PROMPT_LIT_COMPRESSION.format(spec=spec_txt),
            )
        except Exception:
            pass  # fallback

    return trim_block(
        """
        ## Literature Compression (Template Mode)

        Focus only on evidence relevant to your hypothesis.

        ### Suggested search queries:
        - "flow matching" AND "small batch" AND stability
        - diffusion AND "gradient variance"
        - rectified flow AND ablation AND CIFAR-10

        ### Extract from each paper:
        - Dataset
        - Batch size
        - Optimizer
        - Training instability notes
        - Any gradient variance metric
        - Failure cases

        Deliverable:
        A structured comparison table.
        """
    )


# ===============================
# Attack Phase
# ===============================

def run_attack_phase(
    spec: HypothesisSpec,
    context: str = "",
    use_llm: bool = False,
) -> str:

    spec_txt = _spec_block(spec, context=context)

    if use_llm and llm_available():
        try:
            return _call_openai(
                system="You are a harsh NeurIPS reviewer.",
                user=PROMPT_ATTACK.format(spec=spec_txt),
            )
        except Exception:
            pass  # fallback

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

def design_minimal_experiment(
    spec: HypothesisSpec,
    context: str = "",
    use_llm: bool = False,
) -> str:

    spec_txt = _spec_block(spec, context=context)

    if use_llm and llm_available():
        try:
            return _call_openai(
                system="You are a practical ML researcher.",
                user=PROMPT_EXPERIMENT_PLAN.format(spec=spec_txt),
            )
        except Exception:
            pass  # fallback

    return trim_block(
        """
        ## Minimal Experiment Plan

        Dataset: CIFAR-10
        Backbone: U-Net (same config)
        Methods: DDPM vs Flow Matching
        Batch sizes: 4, 8, 16
        Steps: 10k initial

        Log:
        - loss per step
        - global grad norm
        - variance over 100-step window

        Success:
        variance_flow <= 0.95 * variance_ddpm
        and FID not worse by >3
        """
    )


# ===============================
# Conclusion Template
# ===============================

def generate_conclusion_template(spec: HypothesisSpec) -> str:
    return trim_block(
        f"""
        ## Conclusion Template

        Hypothesis:
        {spec.hypothesis}

        Observations:
        - Gradient variance:
        - Loss variance:
        - FID:

        Supported?
        Why?

        Alternative explanations?

        Next decision?
        """
    )
