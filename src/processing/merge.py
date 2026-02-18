"""Merge data from Steam, SteamSpy, and RAWG into unified tables."""

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def build_games_table(
    steamspy_df: pd.DataFrame,
    rawg_df: pd.DataFrame,
) -> pd.DataFrame:
    """Merge SteamSpy market data with RAWG metadata into a single games table.

    The join key is ``app_id`` (SteamSpy) ↔ ``steam_app_id`` (RAWG).
    SteamSpy is the left table — every game with market data is kept, even
    if RAWG metadata is missing.

    Args:
        steamspy_df: Cleaned SteamSpy DataFrame (must have ``appid`` or ``app_id``).
        rawg_df: Cleaned RAWG DataFrame (must have ``steam_app_id``).

    Returns:
        Merged games DataFrame with a ``has_rawg_metadata`` boolean column.
    """
    # Normalise the join key
    if "appid" in steamspy_df.columns and "app_id" not in steamspy_df.columns:
        steamspy_df = steamspy_df.rename(columns={"appid": "app_id"})
    steamspy_df["app_id"] = steamspy_df["app_id"].astype(int)

    if "steam_app_id" in rawg_df.columns:
        rawg_df = rawg_df.rename(columns={"steam_app_id": "app_id"})
    rawg_df["app_id"] = rawg_df["app_id"].astype(int)

    merged = steamspy_df.merge(rawg_df, on="app_id", how="left", suffixes=("", "_rawg"))
    merged["has_rawg_metadata"] = merged["rawg_id"].notna()

    match_rate = merged["has_rawg_metadata"].mean()
    logger.info(
        "Games table: %d rows, %.1f%% with RAWG metadata",
        len(merged),
        match_rate * 100,
    )

    return merged


def build_user_games_table(
    user_games_df: pd.DataFrame,
    games_df: pd.DataFrame,
) -> pd.DataFrame:
    """Enrich the user-game table with game-level features.

    Adds price, genre, review score, etc. to each user-game row for
    downstream modelling.

    Args:
        user_games_df: Cleaned user-game ownership DataFrame.
        games_df: Merged games DataFrame from :func:`build_games_table`.

    Returns:
        Enriched user-games DataFrame.
    """
    game_cols = [
        "app_id",
        "price_dollars",
        "review_score",
        "owners_mid",
        "genres",
        "tags",
        "metacritic",
        "released",
    ]
    available_cols = [c for c in game_cols if c in games_df.columns]
    game_features = games_df[available_cols].drop_duplicates(subset=["app_id"])

    enriched = user_games_df.merge(game_features, on="app_id", how="left")
    logger.info(
        "User-games table: %d rows, %d unique users, %d unique games",
        len(enriched),
        enriched["steam_id"].nunique(),
        enriched["app_id"].nunique(),
    )

    return enriched


def generate_data_quality_report(
    user_games_df: pd.DataFrame,
    games_df: pd.DataFrame,
) -> dict:
    """Produce a data quality summary for documentation.

    Returns:
        Dict with missingness rates, match rates, and bias indicators.
    """
    report: dict = {}

    # Game-level missingness
    for col in games_df.columns:
        missing = games_df[col].isna().mean()
        if missing > 0:
            report.setdefault("games_missing", {})[col] = round(missing, 4)

    # RAWG match rate
    if "has_rawg_metadata" in games_df.columns:
        report["rawg_match_rate"] = round(games_df["has_rawg_metadata"].mean(), 4)

    # User-level stats
    games_per_user = user_games_df.groupby("steam_id")["app_id"].count()
    report["users_total"] = int(user_games_df["steam_id"].nunique())
    report["games_total"] = int(user_games_df["app_id"].nunique())
    report["median_games_per_user"] = int(games_per_user.median())
    report["mean_games_per_user"] = round(games_per_user.mean(), 1)

    # Playtime distribution
    playtime = user_games_df["playtime_forever"]
    report["playtime_median_hrs"] = round(playtime.median() / 60, 1)
    report["playtime_mean_hrs"] = round(playtime.mean() / 60, 1)
    report["playtime_zero_pct"] = round((playtime == 0).mean(), 4)

    logger.info("Data quality report: %s", report)
    return report
