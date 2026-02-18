"""Data cleaning: deduplication, type casting, null handling, and outlier flagging."""

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def clean_user_games(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the raw user-game ownership table.

    Steps:
        1. Drop exact duplicates (same user + same game).
        2. Cast types (app_id → int, playtime → int).
        3. Remove rows with missing app_id.
        4. Cap extreme playtime values (> 99.9th percentile) as an outlier flag.

    Args:
        df: Raw DataFrame with columns ``steam_id``, ``app_id``, ``name``,
            ``playtime_forever``, ``playtime_2weeks``.

    Returns:
        Cleaned DataFrame with an additional ``playtime_outlier`` boolean column.
    """
    initial = len(df)

    df = df.drop_duplicates(subset=["steam_id", "app_id"])
    logger.info("Dedup: %d → %d rows", initial, len(df))

    df = df.dropna(subset=["app_id"])
    df["app_id"] = df["app_id"].astype(int)
    df["playtime_forever"] = pd.to_numeric(df["playtime_forever"], errors="coerce").fillna(0).astype(int)
    df["playtime_2weeks"] = pd.to_numeric(df["playtime_2weeks"], errors="coerce").fillna(0).astype(int)

    # Flag extreme playtime (likely idle/AFK farming)
    threshold = df["playtime_forever"].quantile(0.999)
    df["playtime_outlier"] = df["playtime_forever"] > threshold
    logger.info(
        "Playtime outlier threshold: %d min (%.0f hrs), %d rows flagged",
        threshold,
        threshold / 60,
        df["playtime_outlier"].sum(),
    )

    return df.reset_index(drop=True)


def clean_steamspy(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the SteamSpy game details table.

    Steps:
        1. Drop games with no owner estimates.
        2. Cast price from cents to dollars.
        3. Compute review score: positive / (positive + negative).
        4. Compute estimated revenue midpoint: owners_mid * price_dollars.

    Args:
        df: DataFrame from SteamSpy collection.

    Returns:
        Cleaned DataFrame with derived columns.
    """
    initial = len(df)
    df = df.dropna(subset=["owners_mid"])
    df = df[df["owners_mid"] > 0]
    logger.info("SteamSpy: dropped %d games with no owner data", initial - len(df))

    df["price_dollars"] = pd.to_numeric(df.get("price", pd.Series(dtype=float)), errors="coerce").fillna(0) / 100.0
    df["average_forever"] = pd.to_numeric(df.get("average_forever", pd.Series(dtype=float)), errors="coerce").fillna(0)
    df["median_forever"] = pd.to_numeric(df.get("median_forever", pd.Series(dtype=float)), errors="coerce").fillna(0)

    pos = pd.to_numeric(df.get("positive", pd.Series(dtype=float)), errors="coerce").fillna(0)
    neg = pd.to_numeric(df.get("negative", pd.Series(dtype=float)), errors="coerce").fillna(0)
    total_reviews = pos + neg
    df["review_score"] = (pos / total_reviews).where(total_reviews > 0, other=None)
    df["total_reviews"] = total_reviews.astype(int)

    df["estimated_revenue"] = df["owners_mid"] * df["price_dollars"]

    return df.reset_index(drop=True)


def clean_rawg(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the RAWG metadata table.

    Steps:
        1. Parse release date to datetime.
        2. Ensure list columns (genres, tags, platforms) are actual lists.
        3. Fill missing Metacritic scores with NaN.

    Args:
        df: DataFrame from RAWG collection.

    Returns:
        Cleaned DataFrame.
    """
    df["released"] = pd.to_datetime(df["released"], errors="coerce")
    df["metacritic"] = pd.to_numeric(df.get("metacritic", pd.Series(dtype=float)), errors="coerce")

    for col in ("genres", "tags", "platforms", "developers", "publishers"):
        if col in df.columns:
            df[col] = df[col].apply(lambda x: x if isinstance(x, list) else [])

    return df.reset_index(drop=True)
