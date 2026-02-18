"""Market landscape visualisations.

Covers:
    - Genre/tag co-occurrence heatmap
    - Revenue distribution by genre (violin/ridgeline)
    - Scatter: median playtime vs. estimated owners
    - Time series: game releases by genre over years
    - Tag network graph
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)

# Consistent colour palette across all visualisations
PALETTE = "viridis"
FIG_DIR = Path("results/figures")


def _save(fig: plt.Figure, name: str) -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved figure: %s", path)
    return path


def plot_genre_cooccurrence(games_df: pd.DataFrame) -> Path:
    """Heatmap showing how often genre pairs appear together.

    Args:
        games_df: DataFrame with a ``genres`` list column.

    Returns:
        Path to saved figure.
    """
    from sklearn.preprocessing import MultiLabelBinarizer

    genres = games_df["genres"].dropna()
    mlb = MultiLabelBinarizer()
    genre_matrix = mlb.fit_transform(genres)
    cooccurrence = np.dot(genre_matrix.T, genre_matrix)
    np.fill_diagonal(cooccurrence, 0)

    labels = mlb.classes_
    co_df = pd.DataFrame(cooccurrence, index=labels, columns=labels)

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(co_df, cmap="YlOrRd", ax=ax, linewidths=0.5, fmt="d", annot=len(labels) <= 20)
    ax.set_title("Genre Co-occurrence Matrix", fontsize=14, fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)

    return _save(fig, "genre_cooccurrence_heatmap")


def plot_revenue_by_genre(games_df: pd.DataFrame, top_n: int = 15) -> Path:
    """Violin plot of estimated revenue distribution per genre.

    Args:
        games_df: DataFrame with ``genres`` and ``estimated_revenue``.
        top_n: Number of top genres to show (by median revenue).

    Returns:
        Path to saved figure.
    """
    df = games_df.explode("genres").dropna(subset=["genres", "estimated_revenue"])
    df = df[df["estimated_revenue"] > 0].copy()
    df["log_revenue"] = np.log10(df["estimated_revenue"].clip(lower=1))

    # Top N genres by game count
    top_genres = df["genres"].value_counts().head(top_n).index.tolist()
    df = df[df["genres"].isin(top_genres)]

    # Order by median revenue
    order = df.groupby("genres")["log_revenue"].median().sort_values(ascending=False).index.tolist()

    fig, ax = plt.subplots(figsize=(14, 8))
    sns.violinplot(data=df, x="genres", y="log_revenue", order=order, palette=PALETTE, ax=ax, inner="quartile")
    ax.set_title("Revenue Distribution by Genre (log₁₀ scale)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Genre")
    ax.set_ylabel("log₁₀(Estimated Revenue $)")
    plt.xticks(rotation=45, ha="right")

    return _save(fig, "revenue_by_genre_violin")


def plot_playtime_vs_owners(games_df: pd.DataFrame) -> Path:
    """Scatter plot: median playtime vs. estimated owners, coloured by primary genre.

    Args:
        games_df: DataFrame with ``median_forever``, ``owners_mid``, ``genres``.

    Returns:
        Path to saved figure.
    """
    df = games_df.dropna(subset=["median_forever", "owners_mid"]).copy()
    df = df[(df["median_forever"] > 0) & (df["owners_mid"] > 0)]
    df["primary_genre"] = df["genres"].apply(lambda x: x[0] if isinstance(x, list) and x else "Unknown")

    # Limit to top 8 genres for readability
    top_genres = df["primary_genre"].value_counts().head(8).index.tolist()
    df["genre_display"] = df["primary_genre"].apply(lambda g: g if g in top_genres else "Other")

    fig, ax = plt.subplots(figsize=(12, 8))
    for genre in top_genres + ["Other"]:
        subset = df[df["genre_display"] == genre]
        ax.scatter(
            subset["median_forever"] / 60,
            subset["owners_mid"],
            label=genre,
            alpha=0.5,
            s=15,
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Median Playtime (hours)", fontsize=12)
    ax.set_ylabel("Estimated Owners", fontsize=12)
    ax.set_title("Playtime vs. Ownership by Genre", fontsize=14, fontweight="bold")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=9)

    return _save(fig, "playtime_vs_owners_scatter")


def plot_releases_over_time(games_df: pd.DataFrame, top_n: int = 8) -> Path:
    """Time series of game releases by genre per year.

    Args:
        games_df: DataFrame with ``released`` and ``genres``.
        top_n: Number of genres to plot.

    Returns:
        Path to saved figure.
    """
    df = games_df.dropna(subset=["released"]).copy()
    df["year"] = pd.to_datetime(df["released"], errors="coerce").dt.year
    df = df.dropna(subset=["year"])
    df = df[(df["year"] >= 2005) & (df["year"] <= 2025)]
    df = df.explode("genres").dropna(subset=["genres"])

    top_genres = df["genres"].value_counts().head(top_n).index.tolist()
    df = df[df["genres"].isin(top_genres)]

    counts = df.groupby(["year", "genres"]).size().reset_index(name="count")
    pivot = counts.pivot(index="year", columns="genres", values="count").fillna(0)

    fig, ax = plt.subplots(figsize=(14, 7))
    pivot.plot(ax=ax, linewidth=2)
    ax.set_title("Game Releases by Genre Over Time", fontsize=14, fontweight="bold")
    ax.set_xlabel("Year")
    ax.set_ylabel("Number of Releases")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    return _save(fig, "releases_over_time")
