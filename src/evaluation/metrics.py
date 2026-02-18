"""Evaluation metrics for the recommendation engine.

Standard metrics:
    - Precision@K, Recall@K, NDCG@K

Novel revenue-weighted metric:
    - Revenue-Weighted Hit Rate: weights hits by game price × estimated ownership,
      since recommending a $30 game that converts is more valuable than a free one.
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)


def precision_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    """Fraction of top-K recommendations that are relevant.

    Args:
        recommended: Ordered list of recommended item indices.
        relevant: Set of truly relevant item indices.
        k: Cutoff.

    Returns:
        Precision@K in [0, 1].
    """
    top_k = recommended[:k]
    hits = sum(1 for item in top_k if item in relevant)
    return hits / k


def recall_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    """Fraction of relevant items that appear in top-K recommendations.

    Args:
        recommended: Ordered list of recommended item indices.
        relevant: Set of truly relevant item indices.
        k: Cutoff.

    Returns:
        Recall@K in [0, 1].  Returns 0 if there are no relevant items.
    """
    if not relevant:
        return 0.0
    top_k = recommended[:k]
    hits = sum(1 for item in top_k if item in relevant)
    return hits / len(relevant)


def ndcg_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    """Normalised Discounted Cumulative Gain at K.

    Uses binary relevance (1 if in relevant set, 0 otherwise).

    Args:
        recommended: Ordered list of recommended item indices.
        relevant: Set of truly relevant item indices.
        k: Cutoff.

    Returns:
        NDCG@K in [0, 1].
    """
    top_k = recommended[:k]

    # DCG
    dcg = 0.0
    for i, item in enumerate(top_k):
        if item in relevant:
            dcg += 1.0 / np.log2(i + 2)  # i+2 because rank starts at 1

    # Ideal DCG: all relevant items at the top
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_hits))

    if idcg == 0:
        return 0.0
    return dcg / idcg


def revenue_weighted_hit_rate(
    recommended: list[int],
    relevant: set[int],
    item_revenues: dict[int, float],
    k: int,
) -> float:
    """Revenue-weighted hit rate: weights each hit by the item's revenue potential.

    Formula:
        Σ(hit_i × revenue_i) / Σ(revenue_i for all relevant items)

    This measures whether the recommender is surfacing high-value items, not
    just any items.  A publisher cares more about recommending a $30 game
    that converts than a free-to-play title.

    Args:
        recommended: Ordered list of recommended item indices.
        relevant: Set of truly relevant item indices.
        item_revenues: Mapping of item_idx → estimated revenue (price × owners).
        k: Cutoff.

    Returns:
        Revenue-weighted hit rate in [0, 1].
    """
    if not relevant:
        return 0.0

    top_k = recommended[:k]

    # Numerator: revenue of relevant items that were recommended
    weighted_hits = sum(
        item_revenues.get(item, 0) for item in top_k if item in relevant
    )

    # Denominator: total revenue of all relevant items
    total_relevant_revenue = sum(item_revenues.get(item, 0) for item in relevant)

    if total_relevant_revenue == 0:
        return 0.0

    return weighted_hits / total_relevant_revenue


def evaluate_recommender(
    recommender,
    test_data: list[dict],
    item_revenues: dict[int, float] | None = None,
    k_values: list[int] | None = None,
) -> dict:
    """Evaluate a recommender across multiple users and K values.

    Args:
        recommender: Object with a ``recommend(user_idx, liked_items, n)`` method.
        test_data: List of dicts with ``user_idx``, ``train_items`` (list[int]),
            and ``test_items`` (set[int]).
        item_revenues: Optional revenue mapping for revenue-weighted metrics.
        k_values: List of K cutoffs to evaluate (default [5, 10, 20]).

    Returns:
        Dict mapping metric names to their mean values across all users.
    """
    if k_values is None:
        k_values = [5, 10, 20]

    results: dict[str, list[float]] = {}
    for k in k_values:
        results[f"precision@{k}"] = []
        results[f"recall@{k}"] = []
        results[f"ndcg@{k}"] = []
        if item_revenues:
            results[f"revenue_weighted_hr@{k}"] = []

    for entry in test_data:
        user_idx = entry["user_idx"]
        train_items = entry["train_items"]
        test_items = set(entry["test_items"])

        recs = recommender.recommend(
            user_idx, train_items, n=max(k_values), filter_already_liked=True
        )
        rec_ids = [item_idx for item_idx, _ in recs]

        for k in k_values:
            results[f"precision@{k}"].append(precision_at_k(rec_ids, test_items, k))
            results[f"recall@{k}"].append(recall_at_k(rec_ids, test_items, k))
            results[f"ndcg@{k}"].append(ndcg_at_k(rec_ids, test_items, k))
            if item_revenues:
                results[f"revenue_weighted_hr@{k}"].append(
                    revenue_weighted_hit_rate(rec_ids, test_items, item_revenues, k)
                )

    summary = {metric: round(float(np.mean(values)), 4) for metric, values in results.items()}

    logger.info("Evaluation results: %s", summary)
    return summary
