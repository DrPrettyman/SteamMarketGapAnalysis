"""Train and evaluate the hybrid recommendation engine.

Usage:
    python -m src.train
"""

import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from src.evaluation.metrics import evaluate_recommender, precision_at_k, ndcg_at_k
from src.evaluation.validation import cold_start_split, leave_one_out_split, popularity_baseline
from src.models.recommender import CollaborativeFilter, ContentBasedFilter, HybridRecommender
from src.processing.features import build_game_features, build_interaction_matrix
from src.utils import setup_logging

logger = logging.getLogger(__name__)

PROCESSED_DIR = Path("data/processed")
RESULTS_DIR = Path("results")
MODELS_DIR = Path("models")


def main() -> None:
    setup_logging()

    logger.info("=== Loading data ===")
    games = pd.read_json(PROCESSED_DIR / "games.json", lines=True)
    user_games = pd.read_csv(PROCESSED_DIR / "user_games.csv")
    logger.info("Games: %d, User-game rows: %d", len(games), len(user_games))

    # --- Feature engineering ---
    logger.info("=== Building features ===")
    features, feat_metadata = build_game_features(games)
    feature_cols = [c for c in features.columns if c != "app_id"]
    feature_matrix = features[feature_cols].values.astype(np.float32)
    logger.info("Feature matrix: %s", feature_matrix.shape)

    # --- Interaction matrix ---
    # Use min_playtime=1 to filter zero-playtime rows (54% of data)
    # These are owned-but-never-played games — not a positive signal
    interaction_matrix, user_ids, app_ids = build_interaction_matrix(
        user_games, min_playtime=1
    )
    logger.info(
        "Interaction matrix (playtime≥1): %d users × %d items, nnz=%d",
        *interaction_matrix.shape,
        interaction_matrix.nnz,
    )

    # --- Train/test split ---
    logger.info("=== Leave-one-out split ===")
    train_matrix, test_data = leave_one_out_split(interaction_matrix)
    warm_test, cold_test = cold_start_split(test_data, interaction_matrix, threshold=100)

    # --- Revenue mapping for weighted metrics ---
    app_id_to_idx = {aid: i for i, aid in enumerate(app_ids)}
    item_revenues = {}
    for _, row in games.iterrows():
        aid = row["app_id"]
        if aid in app_id_to_idx:
            item_revenues[app_id_to_idx[aid]] = row.get("estimated_revenue", 0) or 0

    # --- Popularity baseline ---
    logger.info("=== Popularity baseline ===")
    pop_items = popularity_baseline(train_matrix, n=20)

    class PopularityRecommender:
        """Wrapper to match the evaluate_recommender interface."""

        def __init__(self, pop_items):
            self._pop_items = pop_items

        def recommend(self, user_idx, liked_items, n=10, filter_already_liked=True):
            exclude = set(liked_items) if filter_already_liked else set()
            recs = [(i, 1.0 / (rank + 1)) for rank, i in enumerate(self._pop_items) if i not in exclude]
            return recs[:n]

    pop_model = PopularityRecommender(pop_items)
    pop_results = evaluate_recommender(pop_model, test_data, item_revenues, k_values=[5, 10, 20])
    logger.info("Popularity baseline: %s", pop_results)

    # --- Collaborative filtering ---
    logger.info("=== Training ALS collaborative filter ===")
    t0 = time.time()
    cf = CollaborativeFilter(factors=64, regularization=0.01, iterations=15, random_state=42)
    cf.fit(train_matrix)
    cf_time = time.time() - t0
    logger.info("ALS training took %.1f seconds", cf_time)

    # Evaluate CF alone
    logger.info("=== Evaluating CF alone ===")

    class CFWrapper:
        def __init__(self, cf_model):
            self._cf = cf_model

        def recommend(self, user_idx, liked_items, n=10, filter_already_liked=True):
            return self._cf.recommend(user_idx, n, filter_already_liked)

    cf_results = evaluate_recommender(CFWrapper(cf), test_data, item_revenues, k_values=[5, 10, 20])
    logger.info("CF results: %s", cf_results)

    # --- Content-based ---
    logger.info("=== Building content-based filter ===")
    # We need features aligned with the interaction matrix's item order
    # app_ids from build_interaction_matrix are sorted unique app_ids from user_games
    # features has app_id column aligned with games_df order
    # Create a mapping: feature matrix indexed by app_id
    features_by_appid = features.set_index("app_id")
    # Reindex to match interaction matrix item order, filling missing with zeros
    aligned_features = features_by_appid.reindex(app_ids).fillna(0).values.astype(np.float32)
    logger.info("Aligned feature matrix: %s", aligned_features.shape)

    cb = ContentBasedFilter(aligned_features)

    # Evaluate CB alone
    logger.info("=== Evaluating CB alone ===")
    cb_results_list = []
    for entry in test_data[:2000]:  # Sample for speed
        user_idx = entry["user_idx"]
        train_items = entry["train_items"]
        test_items = set(entry["test_items"])
        recs = cb.recommend_for_profile(train_items, n=20, exclude=set(train_items))
        rec_ids = [item_idx for item_idx, _ in recs]
        cb_results_list.append({
            "p@10": precision_at_k(rec_ids, test_items, 10),
            "ndcg@10": ndcg_at_k(rec_ids, test_items, 10),
        })
    cb_p10 = np.mean([r["p@10"] for r in cb_results_list])
    cb_ndcg10 = np.mean([r["ndcg@10"] for r in cb_results_list])
    logger.info("CB (sample 2K users): P@10=%.4f, NDCG@10=%.4f", cb_p10, cb_ndcg10)

    # --- Hybrid ---
    logger.info("=== Building hybrid recommender ===")
    hybrid = HybridRecommender(cf, cb, interaction_threshold=100)
    hybrid.set_interaction_counts(interaction_matrix)

    logger.info("=== Evaluating hybrid (all users) ===")
    hybrid_results = evaluate_recommender(hybrid, test_data, item_revenues, k_values=[5, 10, 20])
    logger.info("Hybrid results: %s", hybrid_results)

    # --- Warm vs cold-start evaluation ---
    logger.info("=== Evaluating warm items (≥100 interactions) ===")
    warm_results = evaluate_recommender(hybrid, warm_test, item_revenues, k_values=[5, 10, 20])
    logger.info("Warm results: %s", warm_results)

    logger.info("=== Evaluating cold-start items (<100 interactions) ===")
    cold_results = evaluate_recommender(hybrid, cold_test, item_revenues, k_values=[5, 10, 20])
    logger.info("Cold-start results: %s", cold_results)

    # --- Save results ---
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    hybrid.save(MODELS_DIR / "hybrid_recommender.pkl")

    results = {
        "popularity_baseline": pop_results,
        "collaborative_filtering": cf_results,
        "content_based_sample": {"precision@10": round(cb_p10, 4), "ndcg@10": round(cb_ndcg10, 4)},
        "hybrid_all": hybrid_results,
        "hybrid_warm": warm_results,
        "hybrid_cold": cold_results,
        "metadata": {
            "n_users": len(user_ids),
            "n_items": len(app_ids),
            "n_interactions": int(interaction_matrix.nnz),
            "n_test_users": len(test_data),
            "n_warm_test": len(warm_test),
            "n_cold_test": len(cold_test),
            "als_factors": 64,
            "als_iterations": 15,
            "feature_dims": int(aligned_features.shape[1]),
            "training_time_seconds": round(cf_time, 1),
        },
    }

    tables_dir = RESULTS_DIR / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    with open(tables_dir / "recommender_results.json", "w") as f:
        json.dump(results, f, indent=2)

    logger.info("=== Results saved to %s ===", tables_dir / "recommender_results.json")

    # Print summary
    print("\n" + "=" * 60)
    print("RECOMMENDER EVALUATION SUMMARY")
    print("=" * 60)
    for segment, res in [
        ("Popularity baseline", pop_results),
        ("Collaborative filtering", cf_results),
        ("Hybrid (all users)", hybrid_results),
        ("Hybrid (warm items)", warm_results),
        ("Hybrid (cold-start)", cold_results),
    ]:
        print(f"\n{segment}:")
        for k, v in sorted(res.items()):
            print(f"  {k}: {v:.4f}")
    print()


if __name__ == "__main__":
    main()
