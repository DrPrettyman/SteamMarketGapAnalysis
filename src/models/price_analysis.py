"""Price sensitivity analysis: model the relationship between price point and commercial success.

Approach:
    - Bin games by price within each genre
    - Compute median ownership, revenue, and review scores per bin
    - Fit log-linear model: log(owners) ~ price + genre + review_score + platform_count
    - Report elasticity estimates per genre

Caveats (documented honestly):
    - Observational, not causal — endogeneity from quality-based pricing
    - Cannot claim "lowering price by $5 will increase sales by X%"
    - Can say: "Games priced at $10-15 tend to have higher total revenue..."
"""

import logging

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)

PRICE_BINS = [0, 0.01, 5, 10, 15, 20, 30, float("inf")]
PRICE_LABELS = ["Free", "$0-5", "$5-10", "$10-15", "$15-20", "$20-30", "$30+"]


def compute_price_segments(games_df: pd.DataFrame) -> pd.DataFrame:
    """Compute summary statistics for each price × genre segment.

    Args:
        games_df: Merged games DataFrame with ``price_dollars``, ``genres``,
            ``owners_mid``, ``estimated_revenue``, ``review_score``.

    Returns:
        DataFrame with one row per (genre, price_bin) with median metrics.
    """
    df = games_df.copy()
    df["price_bin"] = pd.cut(df["price_dollars"], bins=PRICE_BINS, labels=PRICE_LABELS, right=False)

    # Explode genres so each game appears once per genre it belongs to
    # Drop the original SteamSpy "genre" (comma-separated string) to avoid conflict
    if "genres" in df.columns:
        if "genre" in df.columns:
            df = df.drop(columns=["genre"])
        df = df.explode("genres").rename(columns={"genres": "genre"})
    elif "genre" not in df.columns:
        df["genre"] = "Unknown"

    df = df.dropna(subset=["genre", "price_bin"])

    agg = (
        df.groupby(["genre", "price_bin"], observed=True)
        .agg(
            game_count=("app_id", "count"),
            median_owners=("owners_mid", "median"),
            median_revenue=("estimated_revenue", "median"),
            total_revenue=("estimated_revenue", "sum"),
            median_review_score=("review_score", "median"),
            median_playtime=("median_forever", "median"),
        )
        .reset_index()
    )

    # Revenue per game (density-adjusted)
    agg["revenue_per_game"] = agg["total_revenue"] / agg["game_count"].clip(lower=1)

    logger.info("Price segments: %d genre × price_bin combinations", len(agg))
    return agg


def fit_price_model(games_df: pd.DataFrame) -> dict:
    """Fit a log-linear model of ownership on price and game features.

    Model: log(owners_mid + 1) ~ price_dollars + genre_encoded + review_score + platform_count

    Args:
        games_df: Merged games DataFrame.

    Returns:
        Dict with model coefficients, R², and interpretation.
    """
    df = games_df.copy()

    # Require key columns
    required = ["price_dollars", "owners_mid", "review_score"]
    df = df.dropna(subset=required)
    df = df[df["owners_mid"] > 0]

    # Target: log ownership
    df["log_owners"] = np.log1p(df["owners_mid"])

    # Features
    features = pd.DataFrame(index=df.index)
    features["price_dollars"] = df["price_dollars"]
    features["review_score"] = df["review_score"]

    # Platform count
    if "platforms" in df.columns:
        features["platform_count"] = df["platforms"].apply(lambda x: len(x) if isinstance(x, list) else 1)
    else:
        features["platform_count"] = 1

    # Encode primary genre
    if "genres" in df.columns:
        primary_genre = df["genres"].apply(lambda x: x[0] if isinstance(x, list) and x else "Unknown")
        le = LabelEncoder()
        features["genre_encoded"] = le.fit_transform(primary_genre)
        genre_classes = list(le.classes_)
    else:
        features["genre_encoded"] = 0
        genre_classes = []

    features = features.fillna(0)
    target = df["log_owners"].values

    model = LinearRegression()
    model.fit(features.values, target)
    r_squared = model.score(features.values, target)

    coef_names = list(features.columns)
    coefficients = dict(zip(coef_names, model.coef_.tolist()))

    # Price elasticity interpretation
    # In a log-linear model, the price coefficient ≈ % change in owners per $1 increase
    price_coef = coefficients.get("price_dollars", 0)
    price_pct = (np.exp(price_coef) - 1) * 100

    result = {
        "r_squared": round(r_squared, 4),
        "coefficients": {k: round(v, 6) for k, v in coefficients.items()},
        "intercept": round(float(model.intercept_), 4),
        "n_observations": len(df),
        "genre_classes": genre_classes,
        "interpretation": {
            "price_effect": f"Each additional $1 in price is associated with a {price_pct:.2f}% change in estimated ownership",
            "caveat": "This is an observational association, not a causal estimate. "
            "Higher-quality games tend to be priced higher, creating endogeneity.",
        },
    }

    logger.info("Price model: R²=%.4f, price coef=%.6f, n=%d", r_squared, price_coef, len(df))
    return result


def compute_genre_elasticities(games_df: pd.DataFrame, min_games: int = 30) -> pd.DataFrame:
    """Fit per-genre price models and report elasticity estimates.

    Args:
        games_df: Merged games DataFrame.
        min_games: Minimum number of games in a genre to fit a model.

    Returns:
        DataFrame with one row per genre showing price coefficient and R².
    """
    df = games_df.copy()
    if "genres" not in df.columns:
        return pd.DataFrame()

    if "genre" in df.columns:
        df = df.drop(columns=["genre"])
    df = df.explode("genres").rename(columns={"genres": "genre"})
    df = df.dropna(subset=["genre", "price_dollars", "owners_mid", "review_score"])
    df = df[df["owners_mid"] > 0]

    results: list[dict] = []
    for genre, group in df.groupby("genre"):
        if len(group) < min_games:
            continue

        X = group[["price_dollars", "review_score"]].fillna(0).values
        y = np.log1p(group["owners_mid"].values)

        model = LinearRegression()
        model.fit(X, y)

        price_coef = model.coef_[0]
        price_pct = (np.exp(price_coef) - 1) * 100

        results.append(
            {
                "genre": genre,
                "n_games": len(group),
                "price_coefficient": round(price_coef, 6),
                "price_pct_per_dollar": round(price_pct, 2),
                "r_squared": round(model.score(X, y), 4),
                "interpretation": (
                    f"In {genre}, a $5 price increase is associated with a "
                    f"{abs(price_pct * 5):.1f}% {'decrease' if price_pct < 0 else 'increase'} "
                    f"in estimated ownership"
                ),
            }
        )

    result_df = pd.DataFrame(results).sort_values("price_pct_per_dollar")
    logger.info("Genre elasticities computed for %d genres", len(result_df))
    return result_df


def find_optimal_price_range(
    segments_df: pd.DataFrame,
    genre: str,
    metric: str = "median_revenue",
) -> dict:
    """Find the price range that maximises a given metric for a genre.

    Args:
        segments_df: Output of :func:`compute_price_segments`.
        genre: Genre name.
        metric: Column to maximise (e.g. ``median_revenue``, ``median_owners``).

    Returns:
        Dict with the optimal price bin and supporting data.
    """
    genre_data = segments_df[segments_df["genre"] == genre]
    if genre_data.empty:
        return {"genre": genre, "error": "No data for this genre"}

    best_row = genre_data.loc[genre_data[metric].idxmax()]

    return {
        "genre": genre,
        "optimal_price_bin": str(best_row["price_bin"]),
        f"best_{metric}": round(float(best_row[metric]), 2),
        "game_count_in_bin": int(best_row["game_count"]),
        "all_bins": genre_data[["price_bin", metric, "game_count"]].to_dict("records"),
    }
