"""Collection orchestrator — coordinates all three API clients.

Usage:
    python -m src.collect              # Run full pipeline
    python -m src.collect --stage steam       # Steam crawl only
    python -m src.collect --stage steamspy    # SteamSpy details only
    python -m src.collect --stage rawg        # RAWG metadata only
    python -m src.collect --stage clean       # Clean & merge only
"""

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from src.collectors.rawg_api import RAWGClient
from src.collectors.steam_api import SteamAPIClient
from src.collectors.steamspy_api import SteamSpyClient
from src.processing.clean import clean_rawg, clean_steamspy, clean_user_games
from src.processing.merge import build_games_table, build_user_games_table, generate_data_quality_report
from src.utils import load_config, setup_logging

logger = logging.getLogger(__name__)


def stage_steam(cfg: dict) -> None:
    """Stage 1: Crawl Steam friend graph and collect user-game data."""
    steam_cfg = cfg["steam"]
    client = SteamAPIClient(
        api_key=steam_cfg["api_key"],
        requests_per_second=steam_cfg.get("requests_per_second", 1.5),
    )

    seed_ids = steam_cfg["seed_ids"]

    # Auto-discover seeds from Steam community groups if configured seeds
    # are insufficient (e.g. new account with no friends)
    if steam_cfg.get("auto_discover_seeds", True):
        logger.info("Discovering seed profiles from Steam community groups...")
        discovered = client.discover_seeds()
        seed_ids = list(dict.fromkeys(seed_ids + discovered))  # dedup, preserve order
        logger.info("Total seed IDs: %d (%d discovered)", len(seed_ids), len(discovered))

    out_path = Path(cfg["data"]["raw_dir"]) / "user_games.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def on_checkpoint(user_games: list[dict]) -> None:
        """Write CSV at each checkpoint so data is available incrementally."""
        df = pd.DataFrame(user_games)
        df.to_csv(out_path, index=False)
        logger.info("CSV updated: %d rows → %s", len(df), out_path)

    result = client.crawl(
        seed_ids=seed_ids,
        max_users=steam_cfg.get("max_users", 10_000),
        on_checkpoint=on_checkpoint,
    )

    # Final CSV write
    df = pd.DataFrame(result["user_games"])
    df.to_csv(out_path, index=False)
    logger.info("Saved %d user-game rows to %s", len(df), out_path)


def stage_steamspy(cfg: dict) -> None:
    """Stage 2: Collect SteamSpy details for all games in the user dataset."""
    spy_cfg = cfg["steamspy"]
    client = SteamSpyClient(
        requests_per_second=spy_cfg.get("requests_per_second", 4),
        retry_attempts=spy_cfg.get("retry_attempts", 3),
        retry_backoff=spy_cfg.get("retry_backoff", 2.0),
    )

    # Get unique app IDs from the user-games data
    user_games_path = Path(cfg["data"]["raw_dir"]) / "user_games.csv"
    if not user_games_path.exists():
        logger.error("Run steam stage first: %s not found", user_games_path)
        return

    ug_df = pd.read_csv(user_games_path)
    app_ids = sorted(ug_df["app_id"].unique().tolist())
    logger.info("Collecting SteamSpy data for %d unique games", len(app_ids))

    details = client.collect_details_for_apps(app_ids)

    out_path = Path(cfg["data"]["raw_dir"]) / "steamspy_details.json"
    with open(out_path, "w") as f:
        json.dump(details, f)
    logger.info("Saved SteamSpy details for %d games to %s", len(details), out_path)


def stage_rawg(cfg: dict) -> None:
    """Stage 3: Collect RAWG metadata for all games."""
    rawg_cfg = cfg["rawg"]
    client = RAWGClient(
        api_key=rawg_cfg["api_key"],
        requests_per_second=rawg_cfg.get("requests_per_second", 1),
        retry_attempts=rawg_cfg.get("retry_attempts", 3),
        retry_backoff=rawg_cfg.get("retry_backoff", 2.0),
    )

    # Build a unique game list from user-games
    user_games_path = Path(cfg["data"]["raw_dir"]) / "user_games.csv"
    if not user_games_path.exists():
        logger.error("Run steam stage first: %s not found", user_games_path)
        return

    ug_df = pd.read_csv(user_games_path)
    games = (
        ug_df[["app_id", "name"]]
        .drop_duplicates(subset=["app_id"])
        .to_dict("records")
    )
    logger.info("Collecting RAWG metadata for %d unique games", len(games))

    matched, unmatched = client.collect_metadata_for_games(games)

    out_path = Path(cfg["data"]["raw_dir"]) / "rawg_metadata.json"
    with open(out_path, "w") as f:
        json.dump({"matched": matched, "unmatched": unmatched}, f)
    logger.info(
        "RAWG: %d matched, %d unmatched → %s", len(matched), len(unmatched), out_path
    )


def stage_clean(cfg: dict) -> None:
    """Stage 4: Clean and merge all sources into processed tables."""
    raw_dir = Path(cfg["data"]["raw_dir"])
    processed_dir = Path(cfg["data"]["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)

    # --- User games ---
    ug_df = pd.read_csv(raw_dir / "user_games.csv")
    ug_df = clean_user_games(ug_df)

    # --- SteamSpy ---
    with open(raw_dir / "steamspy_details.json") as f:
        spy_data = json.load(f)
    spy_df = pd.DataFrame(spy_data)
    spy_df = clean_steamspy(spy_df)

    # --- RAWG ---
    with open(raw_dir / "rawg_metadata.json") as f:
        rawg_data = json.load(f)
    rawg_df = pd.DataFrame(rawg_data["matched"])
    rawg_df = clean_rawg(rawg_df)

    # --- Merge ---
    games_df = build_games_table(spy_df, rawg_df)
    enriched_ug = build_user_games_table(ug_df, games_df)

    # --- Save ---
    games_df.to_csv(processed_dir / "games.csv", index=False)
    enriched_ug.to_csv(processed_dir / "user_games.csv", index=False)

    # Save list columns as JSON (CSV can't handle lists)
    games_df.to_json(processed_dir / "games.json", orient="records", lines=True)

    # --- Data quality report ---
    report = generate_data_quality_report(enriched_ug, games_df)
    with open(processed_dir / "data_quality_report.json", "w") as f:
        json.dump(report, f, indent=2)

    logger.info("Processed data saved to %s", processed_dir)


STAGES = {
    "steam": stage_steam,
    "steamspy": stage_steamspy,
    "rawg": stage_rawg,
    "clean": stage_clean,
}


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Steam Market Intelligence — Data Collection Pipeline"
    )
    parser.add_argument(
        "--stage",
        choices=list(STAGES.keys()),
        default=None,
        help="Run a single stage. Omit to run all stages sequentially.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to config YAML (default: config.yaml in project root).",
    )
    args = parser.parse_args()

    setup_logging()
    cfg = load_config(args.config)

    if args.stage:
        logger.info("Running stage: %s", args.stage)
        STAGES[args.stage](cfg)
    else:
        for name, func in STAGES.items():
            logger.info("=== Stage: %s ===", name)
            func(cfg)

    logger.info("Done.")


if __name__ == "__main__":
    main()
