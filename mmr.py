from __future__ import annotations

from typing import List
import numpy as np


def cosine_sim_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    # embeddings already normalized => dot is cosine
    return a @ b.T


def mmr_select(
    query_vec: np.ndarray,
    doc_vecs: np.ndarray,
    top_k: int = 20,
    lambda_mult: float = 0.7,
) -> List[int]:
    """
    Maximal Marginal Relevance selection.
    query_vec: (d,)
    doc_vecs: (n,d)
    returns indices
    """
    n = doc_vecs.shape[0]
    if n == 0:
        return []
    top_k = min(top_k, n)

    q = query_vec.reshape(1, -1)
    sim_to_query = cosine_sim_matrix(doc_vecs, q).reshape(-1)  # (n,)
    sim_docs = cosine_sim_matrix(doc_vecs, doc_vecs)           # (n,n)

    selected: List[int] = []
    candidates = set(range(n))

    # start with best relevance
    first = int(np.argmax(sim_to_query))
    selected.append(first)
    candidates.remove(first)

    while len(selected) < top_k and candidates:
        best = None
        best_score = -1e9
        for i in list(candidates):
            relevance = sim_to_query[i]
            diversity = max(sim_docs[i, j] for j in selected) if selected else 0.0
            score = lambda_mult * relevance - (1.0 - lambda_mult) * diversity
            if score > best_score:
                best_score = score
                best = i
        selected.append(best)  # type: ignore[arg-type]
        candidates.remove(best)  # type: ignore[arg-type]

    return selected
