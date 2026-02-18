"""Interactive niche opportunity charts and exploration tools.

Produces:
    - Bubble chart of top niches (demand vs. supply, sized by revenue)
    - Opportunity score distribution
    - Revenue range comparison (top niches side-by-side)
    - Heatmap: niche metrics overview
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns

logger = logging.getLogger(__name__)

FIG_DIR = Path("results/figures")


def _save_static(fig: plt.Figure, name: str) -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _save_interactive(fig: go.Figure, name: str) -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / f"{name}.html"
    fig.write_html(str(path))
    return path


def plot_niche_bubble_chart(niche_df: pd.DataFrame, top_n: int = 30) -> Path:
    """Interactive bubble chart: supply vs. demand, sized by median revenue.

    Args:
        niche_df: Scored niche DataFrame with ``supply``, ``demand_proxy``,
            ``median_revenue``, ``opportunity_score``, ``niche``.
        top_n: Number of top niches to display.

    Returns:
        Path to saved interactive HTML figure.
    """
    df = niche_df.head(top_n).copy()

    fig = px.scatter(
        df,
        x="supply",
        y="demand_proxy",
        size="median_revenue",
        color="opportunity_score",
        hover_name="niche",
        hover_data=["satisfaction", "engagement", "avg_price"],
        color_continuous_scale="Viridis",
        title="Top Market Niches: Supply vs. Demand",
        labels={
            "supply": "Number of Games (Supply)",
            "demand_proxy": "Total Estimated Owners (Demand)",
            "median_revenue": "Median Revenue ($)",
            "opportunity_score": "Opportunity Score",
        },
    )
    fig.update_layout(height=600, width=900)

    path = _save_interactive(fig, "niche_bubble_chart")
    logger.info("Saved niche bubble chart: %s", path)
    return path


def plot_opportunity_distribution(niche_df: pd.DataFrame) -> Path:
    """Histogram of opportunity scores across all niches.

    Args:
        niche_df: Scored niche DataFrame.

    Returns:
        Path to saved figure.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(niche_df["opportunity_score"], bins=50, color="#2ecc71", edgecolor="white", alpha=0.8)
    ax.axvline(
        niche_df["opportunity_score"].quantile(0.95),
        color="red",
        linestyle="--",
        label="95th percentile",
    )
    ax.set_xlabel("Opportunity Score", fontsize=12)
    ax.set_ylabel("Number of Niches", fontsize=12)
    ax.set_title("Distribution of Market Opportunity Scores", fontsize=14, fontweight="bold")
    ax.legend()

    path = _save_static(fig, "opportunity_distribution")
    logger.info("Saved opportunity distribution: %s", path)
    return path


def plot_revenue_range_comparison(niche_df: pd.DataFrame, top_n: int = 15) -> Path:
    """Horizontal bar chart comparing revenue ranges for top niches.

    Shows 25th-75th percentile range with median marked.

    Args:
        niche_df: Scored niche DataFrame.
        top_n: Number of niches to show.

    Returns:
        Path to saved figure.
    """
    df = niche_df.head(top_n).copy()
    df = df.sort_values("median_revenue", ascending=True)

    fig, ax = plt.subplots(figsize=(12, 8))

    y_pos = range(len(df))
    low = df["revenue_estimate_low"].values
    high = df["revenue_estimate_high"].values
    mid = df["median_revenue"].values

    # Range bars
    ax.barh(y_pos, high - low, left=low, height=0.6, color="#3498db", alpha=0.6, label="25th-75th percentile")
    # Median markers
    ax.scatter(mid, y_pos, color="#e74c3c", zorder=5, s=50, label="Median revenue")

    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(df["niche"].values, fontsize=9)
    ax.set_xlabel("Estimated Revenue ($)", fontsize=12)
    ax.set_title("Revenue Potential by Market Niche", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right")

    path = _save_static(fig, "revenue_range_comparison")
    logger.info("Saved revenue range comparison: %s", path)
    return path


def plot_niche_metrics_heatmap(niche_df: pd.DataFrame, top_n: int = 20) -> Path:
    """Heatmap of normalised metrics for top niches.

    Columns: supply (inv), demand, engagement, satisfaction, revenue.
    Rows: top niches.

    Args:
        niche_df: Scored niche DataFrame.
        top_n: Number of niches to show.

    Returns:
        Path to saved figure.
    """
    df = niche_df.head(top_n).copy()

    metrics = ["supply", "demand_proxy", "engagement", "satisfaction", "median_revenue"]
    available = [m for m in metrics if m in df.columns]
    plot_data = df.set_index("niche")[available].copy()

    # Normalise each column to [0, 1]
    for col in plot_data.columns:
        col_min, col_max = plot_data[col].min(), plot_data[col].max()
        rng = col_max - col_min
        if rng > 0:
            plot_data[col] = (plot_data[col] - col_min) / rng
        else:
            plot_data[col] = 0.5

    # Invert supply so lower = better
    if "supply" in plot_data.columns:
        plot_data["supply"] = 1 - plot_data["supply"]

    rename = {
        "supply": "Low Competition",
        "demand_proxy": "Demand",
        "engagement": "Engagement",
        "satisfaction": "Satisfaction",
        "median_revenue": "Revenue",
    }
    plot_data = plot_data.rename(columns={k: v for k, v in rename.items() if k in plot_data.columns})

    fig, ax = plt.subplots(figsize=(10, max(8, top_n * 0.4)))
    sns.heatmap(plot_data, cmap="YlGn", annot=True, fmt=".2f", ax=ax, linewidths=0.5)
    ax.set_title("Niche Quality Scorecard (normalised)", fontsize=14, fontweight="bold")

    path = _save_static(fig, "niche_metrics_heatmap")
    logger.info("Saved niche metrics heatmap: %s", path)
    return path
