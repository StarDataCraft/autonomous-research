from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import streamlit as st

from prompts import (
    PROMPT_LIT_COMPRESSION,
    PROMPT_ATTACK,
    PROMPT_EXPERIMENT_PLAN,
)
from utils import bulletify, safe_join, trim_block

# Optional OpenAI integration (only used if installed + key present)
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


def llm_available() -> bool:
    return bool(st.secrets.get("OPENAI_API_KEY", "")) and OpenAI is not None


def _call_openai(system: str, user: str, model: str = "gpt-4.1-mini") -> str:
    """
    Minimal OpenAI chat call. Requires:
      - openai package in requirements.txt
      - OPENAI_API_KEY in st.secrets
    """
    if OpenAI is None:
        raise RuntimeError("openai package not installed.")
    api_key = st.secrets.get("OPENAI_API_KEY", "")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set in st.secrets.")
    client = OpenAI(api_key=api_key)
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return resp.choices[0].message.content or ""


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


def run_literature_compression(spec: HypothesisSpec, context: str = "", use_llm: bool = False) -> str:
    spec_txt = _spec_block(spec, context=context)

    if use_llm and llm_available():
        user = PROMPT_LIT_COMPRESSION.format(spec=spec_txt)
        out = _call_openai(
            system="You are a meticulous research assistant. Be precise and structured.",
            user=user,
        )
        return out.strip()

    # Fallback: rule-based template output (still useful)
    return trim_block(
        f"""
        ## Goal
        Find evidence relevant to the hypothesis (NOT a general survey).

        ## Target query axes
        - Model families: Flow Matching, Diffusion (DDPM), score-based generative modeling
        - Regime: small batch (≤16), low batch training stability
        - Signals: gradient variance, training instability, loss spikes, divergence, sensitivity to optimizer / schedule
        - Benchmarks: CIFAR-10 (preferred), ImageNet subsets

        ## Suggested search queries (copy-paste)
        1. "flow matching" AND diffusion AND ("small batch" OR "batch size") AND stability
        2. ("flow matching" OR "rectified flow") AND ("training stability" OR "gradient variance")
        3. DDPM AND ("small batch" OR "batch size") AND ("gradient noise" OR "variance")
        4. "rectified flow" ablation batch size CIFAR-10

        ## What to extract from each paper (structured)
        - Setting: dataset, batch size, model/backbone, optimizer, schedule
        - Stability notes: divergence, loss spikes, NaNs, sensitivity
        - Any variance proxy: gradient norm stats, loss variance, stability metrics
        - Negative results / limitations
        - Open questions explicitly stated by authors

        ## Deliverable
        A 1-page table:
        - Paper | Regime | Claim | Evidence | Failure cases | Relevance score (0–3)
        """
    )


def run_attack_phase(spec: HypothesisSpec, context: str = "", use_llm: bool = False) -> str:
    spec_txt = _spec_block(spec, context=context)

    if use_llm and llm_available():
        user = PROMPT_ATTACK.format(spec=spec_txt)
        out = _call_openai(
            system="You are a harsh NeurIPS/ICML reviewer. Be adversarial and specific.",
            user=user,
        )
        return out.strip()

    # Fallback attack checklist
    return trim_block(
        f"""
        ## Attack checklist (Reviewer mode)

        ### 1) Confounders
        - Optimizer & LR schedule: AdamW vs Adam; warmup; gradient clipping
        - Noise / time schedule: diffusion schedule choice vs flow time sampling
        - EMA usage: EMA can mask instability differences
        - Loss scaling / objective differences: are you comparing like-for-like?

        ### 2) Architecture sensitivity
        - Same backbone is necessary but not sufficient:
          normalization layers, attention blocks, parameter count

        ### 3) Dataset dependence
        - CIFAR-10 may be too easy; results may flip on harder datasets
        - Class imbalance / augmentation may change gradient statistics

        ### 4) Metric traps
        - Gradient norm variance depends on parameterization and logging location
        - FID at 10k steps might be too noisy—need repeats / confidence intervals

        ### 5) Failure modes where hypothesis may fail
        - Flow matching unstable with poor time sampling / stiff dynamics
        - Diffusion becomes stable with tuned noise schedule + clipping
        - Differences vanish once batch size ≥32

        ### 6) Minimum reviewer demand
        - 3 random seeds
        - Report mean ± std (or CI)
        - Ensure identical compute budget across methods
        """
    )


def design_minimal_experiment(spec: HypothesisSpec, context: str = "", use_llm: bool = False) -> str:
    spec_txt = _spec_block(spec, context=context)

    if use_llm and llm_available():
        user = PROMPT_EXPERIMENT_PLAN.format(spec=spec_txt)
        out = _call_openai(
            system="You are a practical ML researcher. Produce a minimal, controlled experiment plan.",
            user=user,
        )
        return out.strip()

    # Fallback minimal experiment plan
    return trim_block(
        f"""
        ## Minimal experiment plan (MVP)

        ### Dataset
        - CIFAR-10, standard train/val split
        - Fixed preprocessing & augmentation

        ### Models
        - Backbone: U-Net (same config for both)
        - Method A: DDPM baseline
        - Method B: Flow Matching / Rectified Flow (same backbone)

        ### Batch sizes
        - 4, 8, 16

        ### Controls (must be identical)
        - Optimizer: AdamW
        - LR: fixed (e.g., 1e-4) + same schedule (or none)
        - Steps: 10k (first pass), then 50k if signal appears
        - EMA: either ON for both or OFF for both

        ### Logging
        - Per-step: loss, grad_norm (global), grad_norm per-block (optional)
        - Compute:
          - Gradient variance: var(grad_norm) over windows of 100 steps
          - Loss variance: var(loss) over windows of 100 steps
        - Quality:
          - FID at 10k steps (plus 3 seeds)

        ### Success criteria
        - Supports hypothesis if:
          - var_grad_flow <= 0.95 * var_grad_ddpm (consistent across batch sizes)
          - AND FID_flow not worse by >3 points

        ### Next ablations (only if MVP supports hypothesis)
        - Change optimizer
        - Change schedule
        - Move to harder dataset / higher resolution
        """
    )


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
        - One extra ablation that could flip the conclusion:
          (e.g., change time sampling / add gradient clipping)

        ### Next decision
        - If supported: expand to bigger dataset or stronger baselines
        - If not: revise hypothesis or adjust regime
        """
    )
