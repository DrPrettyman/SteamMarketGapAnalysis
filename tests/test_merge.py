"""Tests for src.processing.merge — three-source join and normalisation."""

import pandas as pd

from src.processing.merge import build_games_table, build_user_games_table


class TestBuildGamesTable:
    def _steamspy(self) -> pd.DataFrame:
        return pd.DataFrame([
            {"appid": 10, "name": "Game A", "owners_mid": 1000, "price_dollars": 9.99,
             "genre": "Action,RPG", "tags": {"Action": 50, "RPG": 30}},
            {"appid": 20, "name": "Game B", "owners_mid": 500, "price_dollars": 0,
             "genre": "Indie", "tags": {"Indie": 40}},
            {"appid": 30, "name": "Game C", "owners_mid": 200, "price_dollars": 14.99,
             "genre": "", "tags": {}},
        ])

    def _rawg(self) -> pd.DataFrame:
        return pd.DataFrame([
            {"steam_app_id": 10, "rawg_id": 101, "genres": ["Action"], "tags": ["Action", "Singleplayer"]},
            {"steam_app_id": 20, "rawg_id": 102, "genres": ["Indie"], "tags": ["Indie", "Casual"]},
        ])

    def test_left_join_preserves_all_steamspy_rows(self):
        result = build_games_table(self._steamspy(), self._rawg())
        assert len(result) == 3

    def test_rawg_metadata_flag(self):
        result = build_games_table(self._steamspy(), self._rawg())
        assert result.loc[result["app_id"] == 10, "has_rawg_metadata"].iloc[0] is True
        assert result.loc[result["app_id"] == 30, "has_rawg_metadata"].iloc[0] is False

    def test_tags_normalised_to_lists(self):
        result = build_games_table(self._steamspy(), self._rawg())
        for tags in result["tags"]:
            assert isinstance(tags, list)

    def test_genres_normalised_to_lists(self):
        result = build_games_table(self._steamspy(), self._rawg())
        for genres in result["genres"]:
            assert isinstance(genres, list)

    def test_steamspy_tags_dict_sorted_by_votes(self):
        result = build_games_table(self._steamspy(), self._rawg())
        game_a_tags = result.loc[result["app_id"] == 10, "tags"].iloc[0]
        assert game_a_tags[0] == "Action"  # 50 votes > 30 votes
        assert game_a_tags[1] == "RPG"

    def test_genre_fallback_to_rawg(self):
        """Game C has empty SteamSpy genre — should NOT get RAWG genres (no match)."""
        result = build_games_table(self._steamspy(), self._rawg())
        game_c_genres = result.loc[result["app_id"] == 30, "genres"].iloc[0]
        assert game_c_genres == []

    def test_appid_column_renamed(self):
        result = build_games_table(self._steamspy(), self._rawg())
        assert "app_id" in result.columns


class TestBuildUserGamesTable:
    def test_enriches_with_game_features(self):
        user_games = pd.DataFrame([
            {"steam_id": "u1", "app_id": 10, "playtime_forever": 100},
            {"steam_id": "u1", "app_id": 20, "playtime_forever": 50},
        ])
        games = pd.DataFrame([
            {"app_id": 10, "price_dollars": 9.99, "review_score": 0.8},
            {"app_id": 20, "price_dollars": 0.0, "review_score": 0.6},
        ])
        result = build_user_games_table(user_games, games)
        assert "price_dollars" in result.columns
        assert len(result) == 2
