"""Feature engineering for the recommendation engine and market analysis.

Produces:
    - Game feature vectors (content-based): genre one-hot, tag TF-IDF,
      price bucket, platform flags, Metacritic score.
    - User-game interaction matrix (collaborative filtering): implicit
      playtime signals.
    - Niche descriptors: tag-combination features for market gap analysis.
"""

import logging
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler

logger = logging.getLogger(__name__)

# Price bucket boundaries (USD)
PRICE_BINS = [0, 0.01, 5, 10, 15, 20, 30, float("inf")]
PRICE_LABELS = ["free", "$0-5", "$5-10", "$10-15", "$15-20", "$20-30", "$30+"]


def build_game_features(games_df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Build a feature matrix for content-based similarity.

    Returns:
        Tuple of (feature_df, metadata) where feature_df has one row per game
        and metadata records the transformers used.
    """
    df = games_df.copy()
    features = pd.DataFrame(index=df.index)
    features["app_id"] = df["app_id"]
    metadata: dict = {}

    # --- Genre one-hot ---
    if "genres" in df.columns:
        mlb_genres = MultiLabelBinarizer()
        genre_matrix = mlb_genres.fit_transform(df["genres"])
        genre_cols = [f"genre_{g}" for g in mlb_genres.classes_]
        features[genre_cols] = genre_matrix
        metadata["genre_classes"] = list(mlb_genres.classes_)
        logger.info("Genre features: %d classes", len(mlb_genres.classes_))

    # --- Tag TF-IDF ---
    if "tags" in df.columns:
        tag_strings = df["tags"].apply(lambda tags: " ".join(tags) if isinstance(tags, list) else "")
        tfidf = TfidfVectorizer(max_features=200)
        tag_matrix = tfidf.fit_transform(tag_strings)
        tag_cols = [f"tag_{t}" for t in tfidf.get_feature_names_out()]
        features[tag_cols] = tag_matrix.toarray()
        metadata["tfidf_vocab_size"] = len(tfidf.get_feature_names_out())
        logger.info("Tag TF-IDF features: %d terms", len(tfidf.get_feature_names_out()))

    # --- Price bucket ---
    if "price_dollars" in df.columns:
        features["price_bucket"] = pd.cut(
            df["price_dollars"], bins=PRICE_BINS, labels=PRICE_LABELS, right=False
        )
        price_dummies = pd.get_dummies(features["price_bucket"], prefix="price")
        features = pd.concat([features.drop(columns=["price_bucket"]), price_dummies], axis=1)

    # --- Platform flags ---
    if "platforms" in df.columns:
        mlb_plat = MultiLabelBinarizer()
        plat_matrix = mlb_plat.fit_transform(df["platforms"])
        plat_cols = [f"platform_{p}" for p in mlb_plat.classes_]
        features[plat_cols] = plat_matrix

    # --- Metacritic (normalised) ---
    if "metacritic" in df.columns:
        features["metacritic_norm"] = df["metacritic"].fillna(df["metacritic"].median()) / 100.0

    # --- Review score ---
    if "review_score" in df.columns:
        features["review_score"] = df["review_score"].fillna(df["review_score"].median())

    logger.info("Game feature matrix: %d games × %d features", *features.shape)
    return features, metadata


def build_interaction_matrix(
    user_games_df: pd.DataFrame,
    min_playtime: int = 0,
) -> tuple[sparse.csr_matrix, list[str], list[int]]:
    """Build a sparse user-item interaction matrix from playtime data.

    Playtime is log-transformed to reduce the impact of extreme values:
    ``signal = log1p(playtime_forever)``.

    Args:
        user_games_df: DataFrame with ``steam_id``, ``app_id``, ``playtime_forever``.
        min_playtime: Minimum playtime (minutes) to count as an interaction.

    Returns:
        Tuple of (sparse_matrix, user_ids, app_ids) where the matrix has
        shape (n_users, n_items).
    """
    df = user_games_df[user_games_df["playtime_forever"] >= min_playtime].copy()

    user_ids = sorted(df["steam_id"].unique())
    app_ids = sorted(df["app_id"].unique())
    user_idx = {uid: i for i, uid in enumerate(user_ids)}
    app_idx = {aid: i for i, aid in enumerate(app_ids)}

    rows = df["steam_id"].map(user_idx).values
    cols = df["app_id"].map(app_idx).values
    values = np.log1p(df["playtime_forever"].values).astype(np.float32)

    matrix = sparse.csr_matrix(
        (values, (rows, cols)),
        shape=(len(user_ids), len(app_ids)),
    )

    logger.info(
        "Interaction matrix: %d users × %d items, %.4f%% non-zero",
        matrix.shape[0],
        matrix.shape[1],
        100 * matrix.nnz / (matrix.shape[0] * matrix.shape[1]),
    )

    return matrix, user_ids, app_ids


def build_niche_descriptors(
    games_df: pd.DataFrame,
    max_combo_size: int = 3,
    min_games_per_niche: int = 5,
) -> pd.DataFrame:
    """Generate tag-combination niche descriptors for market gap analysis.

    For each pairwise and triple-wise tag combination with at least
    ``min_games_per_niche`` games, compute supply, demand, engagement,
    satisfaction, and revenue metrics.

    Args:
        games_df: Merged games DataFrame with ``tags``, ``owners_mid``,
            ``median_forever``, ``review_score``, ``price_dollars``.
        max_combo_size: Maximum tag combination size (2 or 3).
        min_games_per_niche: Minimum number of games to form a valid niche.

    Returns:
        DataFrame with one row per niche and computed metrics.
    """
    if "tags" not in games_df.columns:
        logger.warning("No 'tags' column — cannot build niche descriptors")
        return pd.DataFrame()

    # Explode tags: for each game, generate all 2- and 3-tag combos
    records: list[dict] = []
    for _, row in games_df.iterrows():
        tags = row.get("tags", [])
        if not isinstance(tags, list) or len(tags) < 2:
            continue
        # Limit to top-N tags per game to keep combinatorics manageable
        top_tags = tags[:10]
        for size in range(2, max_combo_size + 1):
            for combo in combinations(sorted(top_tags), size):
                records.append(
                    {
                        "niche": " + ".join(combo),
                        "app_id": row["app_id"],
                        "owners_mid": row.get("owners_mid", 0),
                        "median_forever": row.get("median_forever", 0),
                        "review_score": row.get("review_score"),
                        "price_dollars": row.get("price_dollars", 0),
                        "estimated_revenue": row.get("estimated_revenue", 0),
                    }
                )

    if not records:
        return pd.DataFrame()

    niche_df = pd.DataFrame(records)

    # Aggregate per niche
    agg = (
        niche_df.groupby("niche")
        .agg(
            supply=("app_id", "count"),
            demand_proxy=("owners_mid", "sum"),
            engagement=("median_forever", "median"),
            satisfaction=("review_score", "median"),
            median_revenue=("estimated_revenue", "median"),
            revenue_p25=("estimated_revenue", lambda x: x.quantile(0.25)),
            revenue_p75=("estimated_revenue", lambda x: x.quantile(0.75)),
            avg_price=("price_dollars", "mean"),
        )
        .reset_index()
    )

    # Filter by minimum game count
    agg = agg[agg["supply"] >= min_games_per_niche].copy()

    # Compute opportunity score
    agg["competition_intensity"] = agg["supply"] / agg["demand_proxy"].clip(lower=1)
    agg["opportunity_score"] = (
        agg["demand_proxy"] * agg["engagement"].clip(lower=1) * agg["satisfaction"].fillna(0.5)
    ) / agg["supply"]

    agg = agg.sort_values("opportunity_score", ascending=False).reset_index(drop=True)

    logger.info(
        "Niche descriptors: %d niches (from %d tag combinations)",
        len(agg),
        len(records),
    )

    return agg
