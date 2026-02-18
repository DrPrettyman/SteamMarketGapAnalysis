"""Hybrid recommendation engine: ALS collaborative filtering + content-based fallback.

The collaborative component uses Alternating Least Squares (via the ``implicit``
library) on an implicit-feedback interaction matrix (log-transformed playtime).

The content-based component computes cosine similarity between game feature
vectors (genre one-hot, tag TF-IDF, price bucket, platform flags, Metacritic).

The hybrid blends both scores with a confidence weight α that scales with the
number of interactions a game has — more data → trust collaborative more.
"""

import logging
import pickle
from pathlib import Path

import numpy as np
from implicit.als import AlternatingLeastSquares
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

logger = logging.getLogger(__name__)


class CollaborativeFilter:
    """ALS collaborative filtering for implicit feedback.

    Args:
        factors: Number of latent factors.
        regularization: L2 regularisation weight.
        iterations: Number of ALS iterations.
        random_state: Seed for reproducibility.
    """

    def __init__(
        self,
        factors: int = 64,
        regularization: float = 0.01,
        iterations: int = 15,
        random_state: int = 42,
    ) -> None:
        self.model = AlternatingLeastSquares(
            factors=factors,
            regularization=regularization,
            iterations=iterations,
            random_state=random_state,
        )
        self._user_items: sparse.csr_matrix | None = None

    def fit(self, user_items: sparse.csr_matrix) -> None:
        """Train the ALS model on a user-item interaction matrix.

        Args:
            user_items: Sparse matrix of shape (n_users, n_items) with
                implicit feedback values (e.g. log1p(playtime)).
        """
        self._user_items = user_items
        self.model.fit(user_items)
        logger.info(
            "ALS trained: %d users × %d items, %d factors",
            user_items.shape[0],
            user_items.shape[1],
            self.model.factors,
        )

    def recommend(
        self,
        user_idx: int,
        n: int = 10,
        filter_already_liked: bool = True,
    ) -> list[tuple[int, float]]:
        """Recommend top-N items for a user.

        Args:
            user_idx: Row index in the interaction matrix.
            n: Number of recommendations.
            filter_already_liked: Exclude items the user already interacted with.

        Returns:
            List of (item_idx, score) tuples sorted by descending score.
        """
        ids, scores = self.model.recommend(
            user_idx,
            self._user_items[user_idx],
            N=n,
            filter_already_liked_items=filter_already_liked,
        )
        return list(zip(ids.tolist(), scores.tolist()))

    def similar_items(self, item_idx: int, n: int = 10) -> list[tuple[int, float]]:
        """Find items similar to a given item in the latent space.

        Returns:
            List of (item_idx, score) tuples.
        """
        ids, scores = self.model.similar_items(item_idx, N=n)
        return list(zip(ids.tolist(), scores.tolist()))

    def user_factors(self) -> np.ndarray:
        """Return the learned user factor matrix (n_users × factors)."""
        return self.model.user_factors

    def item_factors(self) -> np.ndarray:
        """Return the learned item factor matrix (n_items × factors)."""
        return self.model.item_factors


class ContentBasedFilter:
    """Content-based recommender using cosine similarity on game features.

    Args:
        feature_matrix: Dense or sparse matrix of shape (n_items, n_features).
    """

    def __init__(self, feature_matrix: np.ndarray | sparse.spmatrix) -> None:
        if sparse.issparse(feature_matrix):
            self._features = normalize(feature_matrix, norm="l2")
        else:
            self._features = normalize(np.asarray(feature_matrix, dtype=np.float64), norm="l2")
        self._sim_matrix: np.ndarray | None = None

    def _ensure_similarity(self) -> None:
        if self._sim_matrix is None:
            logger.info("Computing cosine similarity matrix (%d items)...", self._features.shape[0])
            self._sim_matrix = cosine_similarity(self._features)

    def similar_items(self, item_idx: int, n: int = 10) -> list[tuple[int, float]]:
        """Find the N most similar items by content features.

        Returns:
            List of (item_idx, similarity_score) tuples, excluding the query item.
        """
        self._ensure_similarity()
        scores = self._sim_matrix[item_idx]
        # Exclude self
        scores[item_idx] = -1
        top_idx = np.argsort(scores)[::-1][:n]
        return [(int(i), float(scores[i])) for i in top_idx]

    def recommend_for_profile(
        self,
        liked_item_indices: list[int],
        n: int = 10,
        exclude: set[int] | None = None,
    ) -> list[tuple[int, float]]:
        """Recommend items based on a set of liked items.

        Computes the mean feature vector of liked items and finds the closest
        items in feature space.

        Args:
            liked_item_indices: Indices of items the user has interacted with.
            n: Number of recommendations.
            exclude: Item indices to exclude (e.g. already-owned).

        Returns:
            List of (item_idx, score) tuples.
        """
        if not liked_item_indices:
            return []

        profile = np.asarray(self._features[liked_item_indices].mean(axis=0)).flatten()
        scores = cosine_similarity(profile.reshape(1, -1), self._features).flatten()

        if exclude:
            for idx in exclude:
                scores[idx] = -1

        top_idx = np.argsort(scores)[::-1][:n]
        return [(int(i), float(scores[i])) for i in top_idx]


class HybridRecommender:
    """Hybrid recommender blending collaborative and content-based signals.

    The blending weight α is per-item:
        α(item) = min(interaction_count(item) / threshold, 1.0)

    For well-known items (many interactions), collaborative filtering dominates.
    For cold-start items (few interactions), content-based takes over.

    Args:
        collaborative: Trained CollaborativeFilter instance.
        content_based: ContentBasedFilter instance.
        interaction_threshold: Number of interactions at which α saturates to 1.0.
    """

    def __init__(
        self,
        collaborative: CollaborativeFilter,
        content_based: ContentBasedFilter,
        interaction_threshold: int = 100,
    ) -> None:
        self.cf = collaborative
        self.cb = content_based
        self._threshold = interaction_threshold
        self._item_interaction_counts: np.ndarray | None = None

    def set_interaction_counts(self, user_items: sparse.csr_matrix) -> None:
        """Precompute per-item interaction counts from the interaction matrix."""
        self._item_interaction_counts = np.asarray(
            (user_items > 0).sum(axis=0)
        ).flatten()

    def _alpha(self, item_idx: int) -> float:
        """Compute the blending weight for an item."""
        if self._item_interaction_counts is None:
            return 0.5
        count = self._item_interaction_counts[item_idx]
        return min(float(count) / self._threshold, 1.0)

    def recommend(
        self,
        user_idx: int,
        liked_item_indices: list[int],
        n: int = 10,
        filter_already_liked: bool = True,
    ) -> list[tuple[int, float]]:
        """Generate hybrid recommendations for a user.

        Args:
            user_idx: User row index in the interaction matrix.
            liked_item_indices: Indices of items this user has interacted with.
            n: Number of recommendations.
            filter_already_liked: Whether to exclude already-liked items.

        Returns:
            List of (item_idx, hybrid_score) tuples sorted by descending score.
        """
        # Get more candidates than needed from each component, then blend
        candidate_n = n * 3

        cf_recs = dict(self.cf.recommend(user_idx, candidate_n, filter_already_liked))
        exclude = set(liked_item_indices) if filter_already_liked else set()
        cb_recs = dict(self.cb.recommend_for_profile(liked_item_indices, candidate_n, exclude))

        # Union of all candidate items
        all_items = set(cf_recs.keys()) | set(cb_recs.keys())

        # Normalise scores to [0, 1] range
        cf_max = max(cf_recs.values()) if cf_recs else 1.0
        cb_max = max(cb_recs.values()) if cb_recs else 1.0
        cf_max = cf_max if cf_max > 0 else 1.0
        cb_max = cb_max if cb_max > 0 else 1.0

        scored: list[tuple[int, float]] = []
        for item_idx in all_items:
            alpha = self._alpha(item_idx)
            cf_score = cf_recs.get(item_idx, 0.0) / cf_max
            cb_score = cb_recs.get(item_idx, 0.0) / cb_max
            hybrid = alpha * cf_score + (1 - alpha) * cb_score
            scored.append((item_idx, hybrid))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:n]

    def save(self, path: str | Path) -> None:
        """Persist the hybrid recommender to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info("Hybrid recommender saved to %s", path)

    @classmethod
    def load(cls, path: str | Path) -> "HybridRecommender":
        """Load a persisted hybrid recommender."""
        with open(path, "rb") as f:
            return pickle.load(f)
