"""Generate all visualisations.

Usage:
    python -m src.visualise
"""

import logging
from pathlib import Path

import pandas as pd

from src.visualisation.market_map import (
    plot_genre_cooccurrence,
    plot_playtime_vs_owners,
    plot_releases_over_time,
    plot_revenue_by_genre,
)
from src.visualisation.niche_explorer import (
    plot_niche_bubble_chart,
    plot_niche_metrics_heatmap,
    plot_opportunity_distribution,
    plot_revenue_range_comparison,
)
from src.utils import setup_logging

logger = logging.getLogger(__name__)

PROCESSED_DIR = Path("data/processed")
RESULTS_DIR = Path("results")


def main() -> None:
    setup_logging()

    logger.info("=== Loading data ===")
    games = pd.read_json(PROCESSED_DIR / "games.json", lines=True)
    scored_path = RESULTS_DIR / "tables" / "all_niches_scored.csv"
    scored = pd.read_csv(scored_path)
    logger.info("Games: %d, Scored niches: %d", len(games), len(scored))

    # --- Market landscape charts ---
    logger.info("=== Generating market landscape charts ===")

    logger.info("Genre co-occurrence heatmap...")
    plot_genre_cooccurrence(games)

    logger.info("Revenue by genre violin...")
    plot_revenue_by_genre(games, top_n=15)

    logger.info("Playtime vs owners scatter...")
    plot_playtime_vs_owners(games)

    logger.info("Releases over time...")
    plot_releases_over_time(games, top_n=8)

    # --- Niche analysis charts ---
    logger.info("=== Generating niche analysis charts ===")

    # Ensure non-zero median_revenue for bubble chart sizing
    scored_plot = scored[scored["median_revenue"] > 0].copy()

    logger.info("Niche bubble chart...")
    plot_niche_bubble_chart(scored_plot, top_n=30)

    logger.info("Opportunity distribution...")
    plot_opportunity_distribution(scored)

    logger.info("Revenue range comparison...")
    plot_revenue_range_comparison(scored, top_n=15)

    logger.info("Niche metrics heatmap...")
    plot_niche_metrics_heatmap(scored, top_n=20)

    logger.info("=== All visualisations saved to results/figures/ ===")


if __name__ == "__main__":
    main()
