"""Generate lightweight sample data for the Streamlit dashboard.

Creates data/sample/ files that allow the dashboard to run without
the full data pipeline. Uses seeded randomness for reproducibility.

Usage:
    python scripts/create_sample_data.py              # from scratch (synthetic)
    python scripts/create_sample_data.py --from-full  # downsample real data

The --from-full flag reads from data/processed/ and downsamples.
Without it, generates synthetic data matching the project's actual findings.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SAMPLE_DIR = Path("data/sample")
PROCESSED_DIR = Path("data/processed")

# Actual project statistics (from notes.md / results)
GENRES = [
    "Action", "Adventure", "Casual", "Early Access", "Free To Play",
    "Indie", "Massively Multiplayer", "RPG", "Racing", "Simulation",
    "Sports", "Strategy", "Violent", "Gore", "Nudity", "Sexual Content",
    "Animation & Modeling", "Audio Production", "Design & Illustration",
    "Education", "Photo Editing", "Software Training", "Utilities",
    "Video Production", "Web Publishing", "Accounting", "Game Development",
]

TAGS = [
    "Singleplayer", "Multiplayer", "Co-op", "Online Co-Op", "Open World",
    "FPS", "Shooter", "Action", "Adventure", "RPG", "Strategy",
    "Simulation", "Casual", "Indie", "Puzzle", "Platformer",
    "Story Rich", "Atmospheric", "Horror", "Survival", "Sandbox",
    "PvP", "Tactical", "e-sports", "Base-Building", "Crafting",
    "First-Person", "Third Person", "2D", "3D", "Pixel Graphics",
    "Retro", "Sci-fi", "Fantasy", "Historical", "Military",
]

RNG = np.random.RandomState(42)


def generate_synthetic_games(n: int = 500) -> pd.DataFrame:
    """Generate synthetic game records matching dashboard schema."""
    rows = []
    for i in range(n):
        n_genres = RNG.randint(1, 4)
        n_tags = RNG.randint(2, 8)
        price = RNG.choice([0, 0, 0] + list(range(1, 60)))  # ~15% F2P
        owners_exp = RNG.uniform(3.5, 7)  # log10 of owners
        owners = int(10 ** owners_exp)
        revenue = owners * price if price > 0 else 0

        rows.append({
            "app_id": 100000 + i,
            "name": f"Game_{i:04d}",
            "genres": list(RNG.choice(GENRES, n_genres, replace=False)),
            "tags": list(RNG.choice(TAGS, n_tags, replace=False)),
            "price": float(price),
            "owners_estimate": owners,
            "estimated_revenue": float(revenue),
            "positive_reviews": int(RNG.uniform(10, 50000)),
            "negative_reviews": int(RNG.uniform(1, 10000)),
            "average_playtime": int(RNG.exponential(300)),
            "median_playtime": int(RNG.exponential(120)),
            "has_rawg_metadata": bool(RNG.random() < 0.199),
        })

    return pd.DataFrame(rows)


def generate_quality_report() -> dict:
    """Generate data quality report matching actual project statistics."""
    return {
        "users_total": 10000,
        "games_total": 50005,
        "median_games_per_user": 194,
        "mean_games_per_user": 862.5,
        "playtime_median_hrs": 2.2,
        "playtime_mean_hrs": 43.8,
        "playtime_zero_pct": 0.539,
        "rawg_match_rate": 0.199,
        "interaction_rows": 8625136,
        "playtime_outlier_threshold_hrs": 2565,
        "playtime_outlier_count": 8626,
        "games_missing": {
            "price": 0.062,
            "tags": 0.183,
            "genres": 0.078,
            "metacritic": 0.941,
            "rawg_id": 0.801,
            "average_playtime": 0.0,
            "owners_estimate": 0.0,
        },
    }


def downsample_full(n: int = 500) -> tuple[pd.DataFrame, dict]:
    """Downsample from full processed data."""
    games_path = PROCESSED_DIR / "games.json"
    report_path = PROCESSED_DIR / "data_quality_report.json"

    if not games_path.exists():
        print(f"ERROR: {games_path} not found. Use without --from-full.")
        sys.exit(1)

    games = pd.read_json(games_path, lines=True)
    sample = games.sample(n=min(n, len(games)), random_state=42)

    report = {}
    if report_path.exists():
        with open(report_path) as f:
            report = json.load(f)

    return sample, report


def main() -> None:
    parser = argparse.ArgumentParser(description="Create sample data for dashboard")
    parser.add_argument(
        "--from-full", action="store_true",
        help="Downsample from data/processed/ instead of generating synthetic",
    )
    parser.add_argument(
        "-n", type=int, default=500,
        help="Number of sample games (default: 500)",
    )
    args = parser.parse_args()

    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)

    if args.from_full:
        print("Downsampling from full processed data...")
        games_df, quality_report = downsample_full(args.n)
    else:
        print("Generating synthetic sample data...")
        games_df = generate_synthetic_games(args.n)
        quality_report = generate_quality_report()

    # Write games sample
    games_path = SAMPLE_DIR / "games_sample.json"
    games_df.to_json(games_path, orient="records", lines=True)
    print(f"  {games_path}: {len(games_df)} games ({games_path.stat().st_size / 1024:.0f} KB)")

    # Write quality report
    report_path = SAMPLE_DIR / "data_quality_report.json"
    with open(report_path, "w") as f:
        json.dump(quality_report, f, indent=2)
    print(f"  {report_path}: {report_path.stat().st_size / 1024:.0f} KB")

    # Copy top niches (already small, but ensure they're in sample too)
    for name in ["top_niches.csv", "recommender_results.json",
                  "revenue_estimates.json", "price_segments.csv"]:
        src = Path("results/tables") / name
        dst = SAMPLE_DIR / name
        if src.exists():
            dst.write_bytes(src.read_bytes())
            print(f"  {dst}: copied from results/tables/ ({dst.stat().st_size / 1024:.0f} KB)")

    print(f"\nSample data created in {SAMPLE_DIR}/")
    print("Dashboard can now run with: streamlit run src/visualisation/dashboard.py")


if __name__ == "__main__":
    main()
