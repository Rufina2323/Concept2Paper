from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def _safe_roc_auc(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    """Compute ROC-AUC score considering corner cases."""
    if y_true.sum() == 0 or y_true.sum() == len(y_true):
        return float("nan")
    return float(roc_auc_score(y_true, y_scores))


def _ndcg_at_k(y_sorted: np.ndarray, k: int, n_pos: int) -> float:
    """NDCG@K with binary relevance. y_sorted is labels sorted by score desc."""
    top = y_sorted[:k]
    dcg = float(sum(rel / np.log2(i + 2) for i, rel in enumerate(top)))
    ideal_k = min(k, n_pos)
    idcg = float(sum(1.0 / np.log2(i + 2) for i in range(ideal_k)))
    return dcg / idcg if idcg > 0 else 0.0


def compute_ranking_metrics(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    k_values: Optional[List[int]] = None,
) -> Dict[str, float]:
    """
    Compute ranking metrics: ROC-AUC, MRR, P@K, NDCG@K.

    Works on any split (train / val / ranking_test) regardless of
    class balance — meaningful for both dense balanced splits and the
    highly imbalanced real-world ranking candidate set.

    Args:
        y_true: binary labels (1 = positive, 0 = negative)
        y_scores: model scores (higher = more likely positive)
        k_values: list of K values for P@K and NDCG@K

    Returns:
        dict with keys: roc_auc, mrr, p_at_{k}, ndcg_at_{k}, n_candidates, n_pos, pos_rate
    """
    if k_values is None:
        k_values = [10, 50, 100]

    y_true = np.asarray(y_true, dtype=np.int32).ravel()
    y_scores = np.asarray(y_scores, dtype=np.float64).ravel()
    n = len(y_true)
    n_pos = int(y_true.sum())

    out: Dict[str, float] = {
        "n_candidates": float(n),
        "n_pos": float(n_pos),
        "pos_rate": n_pos / max(n, 1),
        "roc_auc": _safe_roc_auc(y_true, y_scores),
    }

    order = np.argsort(y_scores)[::-1]
    y_sort = y_true[order]

    # MRR
    pos_ranks = np.where(y_sort == 1)[0]
    out["mrr"] = float(1.0 / (pos_ranks[0] + 1)) if len(pos_ranks) > 0 else 0.0

    # P@K and NDCG@K
    for k in k_values:
        top = y_sort[:k]
        out[f"p_at_{k}"] = float(top.sum()) / k if k > 0 else 0.0
        out[f"ndcg_at_{k}"] = _ndcg_at_k(y_sort, k, n_pos)

    return out


def get_top_k_pairs(
    pairs: List[Tuple[str, str]],
    y_true: np.ndarray,
    y_scores: np.ndarray,
    k: int = 20,
) -> pd.DataFrame:
    """Return a DataFrame with the top-K scored concept pairs."""
    k = min(k, len(pairs))
    top_idx = np.argsort(y_scores)[::-1][:k]
    return pd.DataFrame(
        [
            {
                "rank": rank,
                "concept_a": pairs[i][0],
                "concept_b": pairs[i][1],
                "score": float(y_scores[i]),
                "label": int(y_true[i]),
            }
            for rank, i in enumerate(top_idx, 1)
        ]
    )


def format_ranking_metrics(metrics: Dict[str, float], name: str = "") -> str:
    """Return formatted output with metrics and the top-K scored concept pairs."""
    lines = []
    if name:
        lines += [f"\n  {name}", f"  {'─' * 50}"]
    lines.append(f"  ROC-AUC : {metrics.get('roc_auc', float('nan')):.4f}")
    lines.append(f"  MRR     : {metrics.get('mrr', 0.0):.4f}")

    k_vals = sorted(int(k.split("_")[-1]) for k in metrics if k.startswith("p_at_"))
    if k_vals:
        lines.append(f"  {'K':>8}  {'P@K':>8}  {'NDCG@K':>8}")
        for k in k_vals:
            lines.append(
                f"  {k:>8d}  {metrics.get(f'p_at_{k}', 0):>8.4f}"
                f"  {metrics.get(f'ndcg_at_{k}', 0):>8.4f}"
            )

    n = int(metrics.get("n_candidates", 0))
    pos = int(metrics.get("n_pos", 0))
    lines.append(f"  N={n:,}  pos={pos:,}  rate={metrics.get('pos_rate', 0):.4%}")
    return "\n".join(lines)
