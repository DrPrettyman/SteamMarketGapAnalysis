"""Tests for src.processing.clean — deduplication, type casting, outlier detection."""

import pandas as pd
import pytest

from src.processing.clean import clean_rawg, clean_steamspy, clean_user_games


# ── clean_user_games ──────────────────────────────────────────────────────────


class TestCleanUserGames:
    def _raw(self, rows: list[dict]) -> pd.DataFrame:
        cols = ["steam_id", "app_id", "name", "playtime_forever", "playtime_2weeks"]
        base = {c: None for c in cols}
        return pd.DataFrame([{**base, **r} for r in rows])

    def test_dedup_same_user_game(self):
        df = self._raw([
            {"steam_id": "u1", "app_id": 10, "playtime_forever": 100},
            {"steam_id": "u1", "app_id": 10, "playtime_forever": 100},
            {"steam_id": "u1", "app_id": 20, "playtime_forever": 50},
        ])
        result = clean_user_games(df)
        assert len(result) == 2

    def test_drops_missing_app_id(self):
        df = self._raw([
            {"steam_id": "u1", "app_id": 10, "playtime_forever": 100},
            {"steam_id": "u2", "app_id": None, "playtime_forever": 50},
        ])
        result = clean_user_games(df)
        assert len(result) == 1
        assert result["app_id"].iloc[0] == 10

    def test_type_casting(self):
        df = self._raw([
            {"steam_id": "u1", "app_id": "10", "playtime_forever": "200", "playtime_2weeks": "5"},
        ])
        result = clean_user_games(df)
        assert result["app_id"].dtype == int
        assert result["playtime_forever"].dtype == int
        assert result["playtime_2weeks"].dtype == int

    def test_playtime_coercion_fills_nan(self):
        df = self._raw([
            {"steam_id": "u1", "app_id": 10, "playtime_forever": "bad", "playtime_2weeks": None},
        ])
        result = clean_user_games(df)
        assert result["playtime_forever"].iloc[0] == 0
        assert result["playtime_2weeks"].iloc[0] == 0

    def test_outlier_flag_added(self):
        df = self._raw([
            {"steam_id": f"u{i}", "app_id": i, "playtime_forever": 100}
            for i in range(100)
        ])
        result = clean_user_games(df)
        assert "playtime_outlier" in result.columns
        assert result["playtime_outlier"].dtype == bool

    def test_reset_index(self):
        df = self._raw([
            {"steam_id": "u1", "app_id": 10, "playtime_forever": 1},
            {"steam_id": "u1", "app_id": 10, "playtime_forever": 1},  # dup
            {"steam_id": "u2", "app_id": 20, "playtime_forever": 2},
        ])
        result = clean_user_games(df)
        assert list(result.index) == list(range(len(result)))


# ── clean_steamspy ────────────────────────────────────────────────────────────


class TestCleanSteamspy:
    def _raw(self, rows: list[dict]) -> pd.DataFrame:
        return pd.DataFrame(rows)

    def test_drops_no_owner_data(self):
        df = self._raw([
            {"appid": 1, "owners_mid": 1000, "price": 999},
            {"appid": 2, "owners_mid": None, "price": 500},
            {"appid": 3, "owners_mid": 0, "price": 500},
        ])
        result = clean_steamspy(df)
        assert len(result) == 1

    def test_price_cents_to_dollars(self):
        df = self._raw([{"appid": 1, "owners_mid": 100, "price": 1999}])
        result = clean_steamspy(df)
        assert result["price_dollars"].iloc[0] == pytest.approx(19.99)

    def test_review_score_calculation(self):
        df = self._raw([
            {"appid": 1, "owners_mid": 100, "positive": 80, "negative": 20},
        ])
        result = clean_steamspy(df)
        assert result["review_score"].iloc[0] == pytest.approx(0.8)

    def test_review_score_no_reviews(self):
        df = self._raw([
            {"appid": 1, "owners_mid": 100, "positive": 0, "negative": 0},
        ])
        result = clean_steamspy(df)
        assert pd.isna(result["review_score"].iloc[0])

    def test_estimated_revenue(self):
        df = self._raw([{"appid": 1, "owners_mid": 500, "price": 2000}])
        result = clean_steamspy(df)
        assert result["estimated_revenue"].iloc[0] == pytest.approx(500 * 20.0)


# ── clean_rawg ────────────────────────────────────────────────────────────────


class TestCleanRawg:
    def test_parses_release_date(self):
        df = pd.DataFrame([{"released": "2023-06-15", "metacritic": 85}])
        result = clean_rawg(df)
        assert pd.api.types.is_datetime64_any_dtype(result["released"])

    def test_list_columns_coerced(self):
        df = pd.DataFrame([{"released": None, "genres": "not_a_list", "tags": None}])
        result = clean_rawg(df)
        assert result["genres"].iloc[0] == []
        assert result["tags"].iloc[0] == []

    def test_existing_lists_preserved(self):
        df = pd.DataFrame([{"released": None, "genres": ["Action", "RPG"]}])
        result = clean_rawg(df)
        assert result["genres"].iloc[0] == ["Action", "RPG"]
