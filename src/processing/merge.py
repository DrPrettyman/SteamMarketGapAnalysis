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

    # Normalise tags and genres into clean lists for downstream use
    merged = _normalise_tags_and_genres(merged)

    match_rate = merged["has_rawg_metadata"].mean()
    logger.info(
        "Games table: %d rows, %.1f%% with RAWG metadata",
        len(merged),
        match_rate * 100,
    )

    return merged


def _normalise_tags_and_genres(df: pd.DataFrame) -> pd.DataFrame:
    """Unify tag and genre columns into clean string lists.

    SteamSpy provides:
        - ``tags``: dict {tag_name: vote_count} (40K+ games)
        - ``genre``: comma-separated string (49K+ games)

    RAWG provides:
        - ``tags_rawg``: list of strings (10K games)
        - ``genres``: list of strings (10K games)

    Strategy: use SteamSpy as primary (better coverage), RAWG as fallback.
    Output columns ``tags`` and ``genres`` are always list[str].
    """
    # --- Tags: SteamSpy dict → sorted list (by vote count desc) ---
    def _tags_to_list(val):
        if isinstance(val, dict):
            return [k for k, _ in sorted(val.items(), key=lambda x: -x[1])]
        if isinstance(val, list):
            return val
        return []

    if "tags" in df.columns:
        df["tags"] = df["tags"].apply(_tags_to_list)
    else:
        df["tags"] = [[] for _ in range(len(df))]

    # Fill empty SteamSpy tags with RAWG tags where available
    if "tags_rawg" in df.columns:
        mask = df["tags"].apply(len) == 0
        df.loc[mask, "tags"] = df.loc[mask, "tags_rawg"].apply(
            lambda x: x if isinstance(x, list) else []
        )

    # --- Genres: SteamSpy string → list, RAWG list as fallback ---
    def _genre_to_list(val):
        if isinstance(val, list):
            return val
        if isinstance(val, str) and val:
            return [g.strip() for g in val.split(",")]
        return []

    # SteamSpy genre is a string in column "genre"; RAWG genres is a list in "genres"
    if "genre" in df.columns:
        spy_genres = df["genre"].apply(_genre_to_list)
    else:
        spy_genres = pd.Series([[] for _ in range(len(df))], index=df.index)

    if "genres" in df.columns:
        rawg_genres = df["genres"].apply(lambda x: x if isinstance(x, list) else [])
    else:
        rawg_genres = pd.Series([[] for _ in range(len(df))], index=df.index)

    # Prefer SteamSpy (more coverage), fall back to RAWG
    df["genres"] = spy_genres.where(spy_genres.apply(len) > 0, rawg_genres)

    has_tags = (df["tags"].apply(len) > 0).sum()
    has_genres = (df["genres"].apply(len) > 0).sum()
    logger.info(
        "Normalised tags: %d games with tags, %d with genres",
        has_tags,
        has_genres,
    )

    return df


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
