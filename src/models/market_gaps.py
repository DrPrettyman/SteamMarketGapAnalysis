"""Market gap analysis: identify underserved niches with quantified revenue opportunity.

This is the headline deliverable — the part no existing portfolio project does.

Method:
    1. Define the tag-combination space (pairwise + triple-wise).
    2. For each niche, compute supply, demand, engagement, satisfaction, revenue.
    3. Score niches by opportunity = (demand × engagement × satisfaction) / supply.
    4. Estimate revenue potential for a hypothetical new entrant.
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def score_niches(niche_df: pd.DataFrame) -> pd.DataFrame:
    """Score and rank niches by market opportunity.

    Expects a DataFrame from ``features.build_niche_descriptors`` with columns:
        niche, supply, demand_proxy, engagement, satisfaction, median_revenue,
        revenue_p25, revenue_p75, avg_price.

    Adds:
        - ``opportunity_score``: composite ranking metric
        - ``rank``: 1-indexed rank
        - ``revenue_estimate_low``: conservative new-entrant estimate (p25)
        - ``revenue_estimate_high``: optimistic new-entrant estimate (p75)
        - ``recency_trend``: whether newer games outperform older ones in this niche

    Returns:
        Ranked DataFrame sorted by opportunity score descending.
    """
    df = niche_df.copy()

    # Normalise each component to [0, 1] for fair weighting
    for col in ("demand_proxy", "engagement", "satisfaction"):
        if col in df.columns:
            col_min = df[col].min()
            col_max = df[col].max()
            rng = col_max - col_min
            if rng > 0:
                df[f"{col}_norm"] = (df[col] - col_min) / rng
            else:
                df[f"{col}_norm"] = 0.5

    # Inverse supply normalisation (lower supply = higher opportunity)
    s_min, s_max = df["supply"].min(), df["supply"].max()
    s_rng = s_max - s_min
    if s_rng > 0:
        df["supply_inv_norm"] = 1 - (df["supply"] - s_min) / s_rng
    else:
        df["supply_inv_norm"] = 0.5

    # Composite opportunity score (equal weights — can be tuned)
    df["opportunity_score"] = (
        df.get("demand_proxy_norm", 0.5)
        * df.get("engagement_norm", 0.5)
        * df.get("satisfaction_norm", 0.5)
        * df["supply_inv_norm"]
    )

    df = df.sort_values("opportunity_score", ascending=False).reset_index(drop=True)
    df["rank"] = df.index + 1

    # Revenue estimates for a new entrant
    df["revenue_estimate_low"] = df["revenue_p25"]
    df["revenue_estimate_high"] = df["revenue_p75"]

    logger.info("Scored %d niches; top niche: %s (score %.4f)", len(df), df.iloc[0]["niche"], df.iloc[0]["opportunity_score"])

    return df


def estimate_new_entrant_revenue(
    niche_row: pd.Series,
    recency_multiplier: float = 1.0,
) -> dict:
    """Produce a revenue estimate for a hypothetical new game in this niche.

    Args:
        niche_row: A single row from the scored niche DataFrame.
        recency_multiplier: Adjustment factor if recent games in this niche
            outperform (>1) or underperform (<1) the historical median.

    Returns:
        Dict with estimate range, assumptions, and the niche description.
    """
    low = niche_row.get("revenue_estimate_low", 0) * recency_multiplier
    mid = niche_row.get("median_revenue", 0) * recency_multiplier
    high = niche_row.get("revenue_estimate_high", 0) * recency_multiplier

    return {
        "niche": niche_row["niche"],
        "num_existing_games": int(niche_row["supply"]),
        "total_estimated_players": int(niche_row.get("demand_proxy", 0)),
        "median_engagement_min": float(niche_row.get("engagement", 0)),
        "median_satisfaction": float(niche_row.get("satisfaction", 0)),
        "revenue_low": round(low, 2),
        "revenue_mid": round(mid, 2),
        "revenue_high": round(high, 2),
        "avg_price": round(float(niche_row.get("avg_price", 0)), 2),
        "recency_multiplier": recency_multiplier,
        "assumptions": [
            "Revenue estimated from SteamSpy owner midpoints × current price",
            "Owner estimates carry ±20-30% uncertainty for smaller titles",
            "Price reflects current listing, not historical sale prices",
            f"Recency adjustment: {recency_multiplier:.2f}x",
            "A new entrant is assumed to achieve between the 25th and 75th percentile of existing titles",
        ],
    }


def build_opportunity_table(
    niche_df: pd.DataFrame,
    top_n: int = 25,
) -> pd.DataFrame:
    """Build the final ranked opportunity table for the README/notebook.

    Args:
        niche_df: Scored niche DataFrame from :func:`score_niches`.
        top_n: Number of top niches to include.

    Returns:
        Clean DataFrame formatted for presentation.
    """
    cols = [
        "rank",
        "niche",
        "supply",
        "demand_proxy",
        "engagement",
        "satisfaction",
        "median_revenue",
        "opportunity_score",
        "revenue_estimate_low",
        "revenue_estimate_high",
        "avg_price",
    ]
    available = [c for c in cols if c in niche_df.columns]
    table = niche_df[available].head(top_n).copy()

    # Rename for readability
    rename_map = {
        "supply": "# Games",
        "demand_proxy": "Est. Total Owners",
        "engagement": "Median Playtime (min)",
        "satisfaction": "Median Review Score",
        "median_revenue": "Median Revenue ($)",
        "opportunity_score": "Opportunity Score",
        "revenue_estimate_low": "Est. Revenue Low ($)",
        "revenue_estimate_high": "Est. Revenue High ($)",
        "avg_price": "Avg Price ($)",
    }
    table = table.rename(columns={k: v for k, v in rename_map.items() if k in table.columns})

    return table


def compute_recency_trend(
    games_df: pd.DataFrame,
    niche_tags: list[str],
    recent_years: int = 3,
) -> float:
    """Compute whether recent games in a niche outperform older ones.

    Args:
        games_df: Full games DataFrame with ``tags``, ``released``, ``estimated_revenue``.
        niche_tags: List of tags defining the niche.
        recent_years: How many years count as "recent".

    Returns:
        Ratio of median revenue (recent) / median revenue (older).  >1.0 means
        newer games are doing better.
    """
    if "tags" not in games_df.columns or "released" not in games_df.columns:
        return 1.0

    # Filter to games that have all niche tags
    mask = games_df["tags"].apply(
        lambda tags: all(t in tags for t in niche_tags) if isinstance(tags, list) else False
    )
    niche_games = games_df[mask].copy()

    if len(niche_games) < 4:
        return 1.0

    niche_games["released"] = pd.to_datetime(niche_games["released"], errors="coerce")
    cutoff = pd.Timestamp.now() - pd.DateOffset(years=recent_years)

    recent = niche_games[niche_games["released"] >= cutoff]["estimated_revenue"]
    older = niche_games[niche_games["released"] < cutoff]["estimated_revenue"]

    if len(recent) < 2 or len(older) < 2:
        return 1.0

    older_median = older.median()
    if older_median <= 0:
        return 1.0

    return float(recent.median() / older_median)
