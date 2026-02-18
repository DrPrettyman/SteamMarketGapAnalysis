"""SteamSpy API client with rate limiting.

Collects estimated ownership, playtime, price, and review data per game.

Endpoints used:
    - /api.php?request=appdetails&appid=X  (per-game detail)
    - /api.php?request=all                 (bulk summary)
"""

import json
import logging
import re
from pathlib import Path

import requests

from src.utils import DiskCache, RateLimiter

logger = logging.getLogger(__name__)

BASE_URL = "https://steamspy.com/api.php"


def parse_owner_range(owners_str: str) -> tuple[int, int]:
    """Parse SteamSpy owner range string into (low, high) integers.

    Example: ``"200,000 .. 500,000"`` -> ``(200_000, 500_000)``.
    """
    parts = owners_str.replace(",", "").split("..")
    nums = [int(re.sub(r"[^\d]", "", p)) for p in parts if p.strip()]
    if len(nums) == 2:
        return (nums[0], nums[1])
    if len(nums) == 1:
        return (nums[0], nums[0])
    return (0, 0)


class SteamSpyClient:
    """Client for the SteamSpy API.

    Args:
        requests_per_second: Max request rate (default 4 req/s — SteamSpy limit).
        retry_attempts: Number of retries on failure.
        retry_backoff: Exponential backoff multiplier.
        cache_dir: Directory for disk cache.
    """

    def __init__(
        self,
        requests_per_second: float = 4.0,
        retry_attempts: int = 3,
        retry_backoff: float = 2.0,
        cache_dir: str | Path = "data/raw/cache/steamspy",
    ) -> None:
        self._session = requests.Session()
        self._limiter = RateLimiter(requests_per_second)
        self._cache = DiskCache(cache_dir)
        self._retry_attempts = retry_attempts
        self._retry_backoff = retry_backoff

    def _get(self, params: dict) -> dict | None:
        """Rate-limited, cached GET with retries."""
        cached = self._cache.get(BASE_URL, params)
        if cached is not None:
            return cached

        import time

        for attempt in range(1, self._retry_attempts + 1):
            self._limiter.wait()
            try:
                resp = self._session.get(BASE_URL, params=params, timeout=15)
                resp.raise_for_status()
                data = resp.json()
                self._cache.set(BASE_URL, data, params)
                return data
            except requests.RequestException as exc:
                logger.warning(
                    "SteamSpy request failed (attempt %d/%d): %s",
                    attempt,
                    self._retry_attempts,
                    exc,
                )
                if attempt < self._retry_attempts:
                    time.sleep(self._retry_backoff**attempt)
        return None

    def get_all_games(self) -> dict[str, dict]:
        """Fetch the bulk summary of all games.

        Returns a dict mapping ``app_id`` (str) to a summary dict with
        limited fields (name, developer, publisher, score_rank, owners, etc.).
        """
        logger.info("Fetching SteamSpy bulk game list...")
        data = self._get({"request": "all"})
        if data is None:
            logger.error("Failed to fetch SteamSpy bulk game list")
            return {}
        logger.info("SteamSpy bulk list: %d games", len(data))
        return data

    def get_app_details(self, app_id: int) -> dict | None:
        """Fetch detailed stats for a single game.

        Returns dict with keys including: ``name``, ``developer``, ``publisher``,
        ``owners``, ``average_forever``, ``median_forever``, ``price`` (cents),
        ``positive``, ``negative``, ``ccu``, ``tags``, etc.
        """
        data = self._get({"request": "appdetails", "appid": str(app_id)})
        return data

    def collect_details_for_apps(
        self,
        app_ids: list[int],
        checkpoint_path: str | Path = "data/raw/steamspy_details_checkpoint.json",
    ) -> list[dict]:
        """Collect detailed stats for a list of app IDs with checkpointing.

        Args:
            app_ids: List of Steam app IDs to collect.
            checkpoint_path: Path for progress saves.

        Returns:
            List of enriched game detail dicts.
        """
        checkpoint_path = Path(checkpoint_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        collected: dict[int, dict] = {}
        if checkpoint_path.exists():
            with open(checkpoint_path) as f:
                saved = json.load(f)
            collected = {int(k): v for k, v in saved.items()}
            logger.info(
                "Resuming SteamSpy collection from checkpoint: %d already collected",
                len(collected),
            )

        remaining = [aid for aid in app_ids if aid not in collected]
        logger.info(
            "SteamSpy: %d remaining of %d total app IDs",
            len(remaining),
            len(app_ids),
        )

        for i, app_id in enumerate(remaining):
            details = self.get_app_details(app_id)
            if details:
                # Parse owner range into numeric columns
                owners_str = details.get("owners", "0 .. 0")
                low, high = parse_owner_range(owners_str)
                details["owners_low"] = low
                details["owners_high"] = high
                details["owners_mid"] = (low + high) // 2
                collected[app_id] = details

            if (i + 1) % 100 == 0:
                with open(checkpoint_path, "w") as f:
                    json.dump({str(k): v for k, v in collected.items()}, f)
                logger.info(
                    "SteamSpy progress: %d / %d collected", len(collected), len(app_ids)
                )

        # Final save
        with open(checkpoint_path, "w") as f:
            json.dump({str(k): v for k, v in collected.items()}, f)
        logger.info("SteamSpy collection complete: %d games", len(collected))

        return list(collected.values())
