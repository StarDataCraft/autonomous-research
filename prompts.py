PROMPT_LIT_COMPRESSION = """
You will produce a literature compression plan and a structured summary.

{spec}

Output format (markdown):

## Search strategy
- Query list (5–12 queries)
- Databases to use (arXiv, Semantic Scholar, Google Scholar)
- Inclusion/exclusion criteria

## Evidence map (fill with placeholders if uncertain)
Create a table with columns:
Paper | Year | Setting (dataset, batch, backbone) | Claim | Evidence | Failure cases | Relevance (0–3)

## What is missing?
- List 3–6 missing comparisons / gaps.

Constraints:
- Do NOT produce a generic survey.
- Focus on evidence related to small batch behavior and stability/variance.
"""

PROMPT_ATTACK = """
Act as a harsh top-tier conference reviewer.

{spec}

Output format (markdown):

## Main reasons the hypothesis may be false
(list 5–10, specific)

## Confounders to control (mandatory)
(list 8–15, concrete)

## Metrics pitfalls
(list 5–10)

## Minimum experimental bar
- seeds, CI, compute budget fairness, ablations

Tone: adversarial but actionable.
"""

PROMPT_EXPERIMENT_PLAN = """
You are designing a minimal, controlled experiment that can falsify the hypothesis quickly.

{spec}

Output format (markdown):

## MVP experiment (1–2 hours)
- dataset, models, batch sizes, steps
- exact logging plan for gradient variance
- seed count and reporting

## Controls checklist
(list)

## Minimal code plan
(files/modules you would create, functions)

## Decision rule
(when to proceed vs stop)

Be practical. Avoid extra complexity.
"""
