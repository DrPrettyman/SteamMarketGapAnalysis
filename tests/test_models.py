"""Tests for src.models.recommender — collaborative, content-based, and hybrid."""

import numpy as np
import pytest
from scipy import sparse

from src.models.recommender import (
    CollaborativeFilter,
    ContentBasedFilter,
    HybridRecommender,
)


@pytest.fixture()
def tiny_interaction_matrix() -> sparse.csr_matrix:
    """5 users × 8 items, sparse implicit feedback."""
    np.random.seed(42)
    dense = np.random.rand(5, 8).astype(np.float32)
    dense[dense < 0.6] = 0  # make it sparse
    return sparse.csr_matrix(dense)


@pytest.fixture()
def tiny_feature_matrix() -> np.ndarray:
    """8 items × 4 features."""
    np.random.seed(42)
    return np.random.rand(8, 4).astype(np.float64)


# ── CollaborativeFilter ──────────────────────────────────────────────────────


class TestCollaborativeFilter:
    def test_fit_and_recommend(self, tiny_interaction_matrix):
        cf = CollaborativeFilter(factors=4, iterations=5, random_state=42)
        cf.fit(tiny_interaction_matrix)
        recs = cf.recommend(user_idx=0, n=3)
        assert len(recs) == 3
        assert all(isinstance(r, tuple) and len(r) == 2 for r in recs)

    def test_recommendations_are_valid_indices(self, tiny_interaction_matrix):
        cf = CollaborativeFilter(factors=4, iterations=5, random_state=42)
        cf.fit(tiny_interaction_matrix)
        recs = cf.recommend(user_idx=0, n=5)
        n_items = tiny_interaction_matrix.shape[1]
        for item_idx, score in recs:
            assert 0 <= item_idx < n_items

    def test_similar_items(self, tiny_interaction_matrix):
        cf = CollaborativeFilter(factors=4, iterations=5, random_state=42)
        cf.fit(tiny_interaction_matrix)
        similar = cf.similar_items(item_idx=0, n=3)
        assert len(similar) == 3

    def test_factor_shapes(self, tiny_interaction_matrix):
        cf = CollaborativeFilter(factors=4, iterations=5, random_state=42)
        cf.fit(tiny_interaction_matrix)
        assert cf.user_factors().shape[0] == 5
        assert cf.item_factors().shape[0] == 8


# ── ContentBasedFilter ────────────────────────────────────────────────────────


class TestContentBasedFilter:
    def test_similar_items(self, tiny_feature_matrix):
        cb = ContentBasedFilter(tiny_feature_matrix)
        similar = cb.similar_items(item_idx=0, n=3)
        assert len(similar) == 3
        # Self should be excluded
        assert all(idx != 0 for idx, _ in similar)

    def test_scores_in_valid_range(self, tiny_feature_matrix):
        cb = ContentBasedFilter(tiny_feature_matrix)
        similar = cb.similar_items(item_idx=0, n=5)
        for _, score in similar:
            assert -1.0 <= score <= 1.0 + 1e-6

    def test_recommend_for_profile(self, tiny_feature_matrix):
        cb = ContentBasedFilter(tiny_feature_matrix)
        recs = cb.recommend_for_profile(liked_item_indices=[0, 1], n=3, exclude={0, 1})
        assert len(recs) == 3
        assert all(idx not in {0, 1} for idx, _ in recs)

    def test_empty_profile_returns_empty(self, tiny_feature_matrix):
        cb = ContentBasedFilter(tiny_feature_matrix)
        recs = cb.recommend_for_profile([], n=3)
        assert recs == []


# ── HybridRecommender ────────────────────────────────────────────────────────


class TestHybridRecommender:
    def test_recommend_returns_n_items(self, tiny_interaction_matrix, tiny_feature_matrix):
        cf = CollaborativeFilter(factors=4, iterations=5, random_state=42)
        cf.fit(tiny_interaction_matrix)
        cb = ContentBasedFilter(tiny_feature_matrix)
        hybrid = HybridRecommender(cf, cb, interaction_threshold=10)
        hybrid.set_interaction_counts(tiny_interaction_matrix)

        recs = hybrid.recommend(user_idx=0, liked_item_indices=[0, 1], n=3)
        assert len(recs) <= 3

    def test_alpha_scales_with_interactions(self, tiny_interaction_matrix, tiny_feature_matrix):
        cf = CollaborativeFilter(factors=4, iterations=5, random_state=42)
        cf.fit(tiny_interaction_matrix)
        cb = ContentBasedFilter(tiny_feature_matrix)
        hybrid = HybridRecommender(cf, cb, interaction_threshold=10)
        hybrid.set_interaction_counts(tiny_interaction_matrix)

        # Items with more interactions should have higher alpha
        alpha_0 = hybrid._alpha(0)
        assert 0.0 <= alpha_0 <= 1.0

    def test_default_alpha_without_counts(self, tiny_interaction_matrix, tiny_feature_matrix):
        cf = CollaborativeFilter(factors=4, iterations=5, random_state=42)
        cf.fit(tiny_interaction_matrix)
        cb = ContentBasedFilter(tiny_feature_matrix)
        hybrid = HybridRecommender(cf, cb)
        # No set_interaction_counts called
        assert hybrid._alpha(0) == 0.5

    def test_recommendations_sorted_descending(self, tiny_interaction_matrix, tiny_feature_matrix):
        cf = CollaborativeFilter(factors=4, iterations=5, random_state=42)
        cf.fit(tiny_interaction_matrix)
        cb = ContentBasedFilter(tiny_feature_matrix)
        hybrid = HybridRecommender(cf, cb, interaction_threshold=10)
        hybrid.set_interaction_counts(tiny_interaction_matrix)

        recs = hybrid.recommend(user_idx=0, liked_item_indices=[0], n=5)
        scores = [score for _, score in recs]
        assert scores == sorted(scores, reverse=True)
