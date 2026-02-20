"""Tests for src.processing.features — feature engineering and interaction matrix."""

import numpy as np
import pandas as pd
from scipy import sparse

from src.processing.features import (
    build_game_features,
    build_interaction_matrix,
    build_niche_descriptors,
)


class TestBuildGameFeatures:
    def _games(self) -> pd.DataFrame:
        return pd.DataFrame([
            {"app_id": 1, "genres": ["Action", "RPG"], "tags": ["Action", "Adventure"],
             "price_dollars": 9.99, "metacritic": 85.0, "review_score": 0.8},
            {"app_id": 2, "genres": ["Indie"], "tags": ["Indie", "Casual"],
             "price_dollars": 0.0, "metacritic": None, "review_score": 0.6},
            {"app_id": 3, "genres": ["Action"], "tags": ["Action", "Multiplayer"],
             "price_dollars": 29.99, "metacritic": 70.0, "review_score": 0.9},
        ])

    def test_output_shape(self):
        features, meta = build_game_features(self._games())
        assert len(features) == 3
        assert features.shape[1] > 1  # app_id + at least genre/tag columns

    def test_app_id_preserved(self):
        features, _ = build_game_features(self._games())
        assert "app_id" in features.columns
        assert set(features["app_id"]) == {1, 2, 3}

    def test_genre_columns_created(self):
        features, meta = build_game_features(self._games())
        genre_cols = [c for c in features.columns if c.startswith("genre_")]
        assert len(genre_cols) > 0
        assert "genre_classes" in meta

    def test_tag_tfidf_columns_created(self):
        features, meta = build_game_features(self._games())
        tag_cols = [c for c in features.columns if c.startswith("tag_")]
        assert len(tag_cols) > 0
        assert "tfidf_vocab_size" in meta

    def test_price_bucket_columns_created(self):
        features, _ = build_game_features(self._games())
        price_cols = [c for c in features.columns if c.startswith("price_")]
        assert len(price_cols) > 0

    def test_metacritic_nan_filled(self):
        features, _ = build_game_features(self._games())
        assert not features["metacritic_norm"].isna().any()


class TestBuildInteractionMatrix:
    def _user_games(self) -> pd.DataFrame:
        return pd.DataFrame([
            {"steam_id": "u1", "app_id": 10, "playtime_forever": 120},
            {"steam_id": "u1", "app_id": 20, "playtime_forever": 60},
            {"steam_id": "u2", "app_id": 10, "playtime_forever": 300},
            {"steam_id": "u2", "app_id": 30, "playtime_forever": 0},
        ])

    def test_output_shape(self):
        matrix, users, items = build_interaction_matrix(self._user_games())
        assert matrix.shape == (2, 3)
        assert len(users) == 2
        assert len(items) == 3

    def test_sparse_format(self):
        matrix, _, _ = build_interaction_matrix(self._user_games())
        assert sparse.issparse(matrix)

    def test_log_transform_applied(self):
        matrix, _, _ = build_interaction_matrix(self._user_games())
        dense = matrix.toarray()
        # log1p(120) ≈ 4.80, not 120
        assert dense.max() < 10

    def test_min_playtime_filter(self):
        matrix, _, items = build_interaction_matrix(self._user_games(), min_playtime=100)
        # Only playtime >= 100: u1/10 (120), u2/10 (300) → 2 interactions, items {10}
        assert len(items) == 1
        assert items[0] == 10

    def test_zero_playtime_included_by_default(self):
        matrix, _, items = build_interaction_matrix(self._user_games(), min_playtime=0)
        assert 30 in items


class TestBuildNicheDescriptors:
    def _games(self) -> pd.DataFrame:
        return pd.DataFrame([
            {"app_id": i, "tags": ["Action", "RPG"], "owners_mid": 1000,
             "median_forever": 120, "review_score": 0.8, "price_dollars": 15.0,
             "estimated_revenue": 15000}
            for i in range(10)
        ])

    def test_produces_niches(self):
        result = build_niche_descriptors(self._games(), max_combo_size=2, min_games_per_niche=5)
        assert len(result) > 0
        assert "niche" in result.columns

    def test_min_games_filter(self):
        games = pd.DataFrame([
            {"app_id": i, "tags": ["Action", "RPG"], "owners_mid": 100,
             "median_forever": 60, "review_score": 0.7, "price_dollars": 10.0,
             "estimated_revenue": 1000}
            for i in range(3)  # only 3 games — below min_games_per_niche=5
        ])
        result = build_niche_descriptors(games, max_combo_size=2, min_games_per_niche=5)
        assert len(result) == 0

    def test_opportunity_score_present(self):
        result = build_niche_descriptors(self._games(), max_combo_size=2, min_games_per_niche=5)
        assert "opportunity_score" in result.columns
        assert result["opportunity_score"].iloc[0] > 0

    def test_empty_tags_handled(self):
        games = pd.DataFrame([{"app_id": 1, "tags": [], "owners_mid": 100,
                                "median_forever": 60, "review_score": 0.7,
                                "price_dollars": 10.0, "estimated_revenue": 1000}])
        result = build_niche_descriptors(games)
        assert len(result) == 0
