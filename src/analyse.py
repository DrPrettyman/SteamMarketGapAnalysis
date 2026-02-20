"""Run market gap analysis and price sensitivity modelling.

Usage:
    python -m src.analyse
"""

import json
import logging
import time
from pathlib import Path

import pandas as pd

from src.models.market_gaps import (
    build_opportunity_table,
    compute_recency_trend,
    estimate_new_entrant_revenue,
    score_niches,
)
from src.models.price_analysis import (
    compute_genre_elasticities,
    compute_price_segments,
    find_optimal_price_range,
    fit_price_model,
)
from src.processing.features import build_niche_descriptors
from src.utils import setup_logging

logger = logging.getLogger(__name__)

PROCESSED_DIR = Path("data/processed")
RESULTS_DIR = Path("results")


def main() -> None:
    setup_logging()

    logger.info("=== Loading data ===")
    games = pd.read_json(PROCESSED_DIR / "games.json", lines=True)
    logger.info("Games: %d rows", len(games))

    # ================================================================
    # Part 1: Market Gap / Niche Analysis
    # ================================================================
    logger.info("=== Building niche descriptors (this may take a few minutes) ===")
    t0 = time.time()
    niche_df = build_niche_descriptors(games, max_combo_size=3, min_games_per_niche=5)
    logger.info("Niche descriptors built in %.1f seconds: %d niches", time.time() - t0, len(niche_df))

    if niche_df.empty:
        logger.error("No niches found — check tags column")
        return

    # Score and rank
    scored = score_niches(niche_df)
    logger.info("Top 10 niches:")
    for _, row in scored.head(10).iterrows():
        logger.info(
            "  #%d %s — supply=%d, demand=%d, opportunity=%.4f",
            row["rank"],
            row["niche"],
            row["supply"],
            row["demand_proxy"],
            row["opportunity_score"],
        )

    # Revenue estimates for top niches
    estimates = []
    for _, row in scored.head(15).iterrows():
        tags = row["niche"].split(" + ")
        recency = compute_recency_trend(games, tags)
        est = estimate_new_entrant_revenue(row, recency_multiplier=recency)
        estimates.append(est)

    # ================================================================
    # Part 2: Price Sensitivity
    # ================================================================
    logger.info("=== Computing price segments ===")
    segments = compute_price_segments(games)
    logger.info("Price segments: %d rows", len(segments))

    logger.info("=== Fitting global price model ===")
    model_result = fit_price_model(games)
    logger.info("R²=%.4f, n=%d", model_result["r_squared"], model_result["n_observations"])
    logger.info("Interpretation: %s", model_result["interpretation"]["price_effect"])

    logger.info("=== Computing genre elasticities ===")
    elasticities = compute_genre_elasticities(games)
    logger.info("Elasticities for %d genres", len(elasticities))

    # Optimal price ranges for top genres
    top_genres = ["Action", "Indie", "RPG", "Strategy", "Simulation", "Adventure", "Casual"]
    optimal_prices = {}
    for genre in top_genres:
        opt = find_optimal_price_range(segments, genre, "median_revenue")
        optimal_prices[genre] = opt
        if "error" not in opt:
            logger.info("  %s: optimal price bin = %s (median revenue $%.0f)",
                        genre, opt["optimal_price_bin"], opt[f"best_median_revenue"])

    # ================================================================
    # Save all results
    # ================================================================
    tables_dir = RESULTS_DIR / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    # Opportunity table
    opportunity = build_opportunity_table(scored, top_n=25)
    opportunity.to_csv(tables_dir / "top_niches.csv", index=False)
    logger.info("Saved top_niches.csv (%d rows)", len(opportunity))

    # Full niche data (for visualisations)
    scored.to_csv(tables_dir / "all_niches_scored.csv", index=False)
    logger.info("Saved all_niches_scored.csv (%d rows)", len(scored))

    # Price segments
    segments.to_csv(tables_dir / "price_segments.csv", index=False)

    # Elasticities
    elasticities.to_csv(tables_dir / "genre_elasticities.csv", index=False)

    # Revenue estimates
    with open(tables_dir / "revenue_estimates.json", "w") as f:
        json.dump(estimates, f, indent=2)

    # Price model
    with open(tables_dir / "price_model.json", "w") as f:
        json.dump(model_result, f, indent=2)

    # Optimal prices
    with open(tables_dir / "optimal_prices.json", "w") as f:
        json.dump(optimal_prices, f, indent=2)

    logger.info("=== All analysis results saved to %s ===", tables_dir)

    # Print summary
    print("\n" + "=" * 60)
    print("MARKET GAP ANALYSIS SUMMARY")
    print("=" * 60)
    print(f"\nTotal niches identified: {len(scored):,}")
    print(f"\nTop 10 Market Opportunities:")
    for est in estimates[:10]:
        print(f"\n  {est['niche']}")
        print(f"    Games: {est['num_existing_games']}, Players: {est['total_estimated_players']:,}")
        print(f"    Revenue: ${est['revenue_low']:,.0f} – ${est['revenue_high']:,.0f}")
        print(f"    Avg price: ${est['avg_price']:.2f}, Recency: {est['recency_multiplier']:.2f}x")

    print(f"\n{'=' * 60}")
    print("PRICE SENSITIVITY SUMMARY")
    print("=" * 60)
    print(f"\nGlobal model R²: {model_result['r_squared']:.4f}")
    print(f"{model_result['interpretation']['price_effect']}")
    print(f"\nMost price-sensitive genres (top 5):")
    for _, row in elasticities.head(5).iterrows():
        print(f"  {row['genre']}: {row['price_pct_per_dollar']:.2f}% per $1 (R²={row['r_squared']:.3f})")
    print()


if __name__ == "__main__":
    main()
