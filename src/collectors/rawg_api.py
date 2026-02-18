"""RAWG API client with fuzzy title matching.

Collects game metadata: genres, tags, platforms, Metacritic scores, release
dates, developers, and publishers.

Endpoint used:
    - /api/games          (search by name)
    - /api/games/{id}     (detail by RAWG id)
"""

import json
import logging
import time
from pathlib import Path

import requests
from thefuzz import fuzz

from src.utils import DiskCache, RateLimiter

logger = logging.getLogger(__name__)

BASE_URL = "https://api.rawg.io/api"

# Minimum fuzzy-match score (0-100) to accept a RAWG result for a Steam game
_MATCH_THRESHOLD = 75


class RAWGClient:
    """Client for the RAWG.io API.

    Args:
        api_key: RAWG API key.
        requests_per_second: Max request rate (default 1 req/s for free tier).
        retry_attempts: Number of retries on failure.
        retry_backoff: Exponential backoff multiplier.
        cache_dir: Directory for disk cache.
    """

    def __init__(
        self,
        api_key: str,
        requests_per_second: float = 1.0,
        retry_attempts: int = 3,
        retry_backoff: float = 2.0,
        cache_dir: str | Path = "data/raw/cache/rawg",
    ) -> None:
        self._key = api_key
        self._session = requests.Session()
        self._limiter = RateLimiter(requests_per_second)
        self._cache = DiskCache(cache_dir)
        self._retry_attempts = retry_attempts
        self._retry_backoff = retry_backoff

    def _get(self, url: str, params: dict) -> dict | None:
        """Rate-limited, cached GET with retries."""
        params["key"] = self._key
        cached = self._cache.get(url, params)
        if cached is not None:
            return cached

        for attempt in range(1, self._retry_attempts + 1):
            self._limiter.wait()
            try:
                resp = self._session.get(url, params=params, timeout=15)
                resp.raise_for_status()
                data = resp.json()
                self._cache.set(url, data, params)
                return data
            except requests.RequestException as exc:
                logger.warning(
                    "RAWG request failed (attempt %d/%d): %s",
                    attempt,
                    self._retry_attempts,
                    exc,
                )
                if attempt < self._retry_attempts:
                    time.sleep(self._retry_backoff**attempt)
        return None

    def search_game(self, name: str) -> list[dict]:
        """Search RAWG for games matching *name*.

        Returns a list of result dicts (top results first).
        """
        url = f"{BASE_URL}/games"
        params = {"search": name, "page_size": 5}
        data = self._get(url, params)
        if data is None:
            return []
        return data.get("results", [])

    def get_game_detail(self, rawg_id: int) -> dict | None:
        """Fetch full detail for a RAWG game ID."""
        url = f"{BASE_URL}/games/{rawg_id}"
        return self._get(url, {})

    def match_steam_game(self, steam_name: str) -> dict | None:
        """Find the best RAWG match for a Steam game by title.

        Uses fuzzy string matching to handle differences in naming conventions
        (e.g. trademark symbols, subtitles).  Returns the full detail dict
        for the best match, or ``None`` if no match exceeds the threshold.
        """
        results = self.search_game(steam_name)
        if not results:
            return None

        best_score = 0
        best_result = None
        for r in results:
            score = fuzz.token_sort_ratio(steam_name.lower(), r["name"].lower())
            if score > best_score:
                best_score = score
                best_result = r

        if best_score < _MATCH_THRESHOLD:
            logger.debug(
                "No RAWG match for '%s' (best: '%s' at %d%%)",
                steam_name,
                best_result["name"] if best_result else "N/A",
                best_score,
            )
            return None

        # Fetch full details for the matched game
        detail = self.get_game_detail(best_result["id"])
        if detail:
            detail["_match_score"] = best_score
            detail["_steam_name"] = steam_name
        return detail

    def collect_metadata_for_games(
        self,
        games: list[dict],
        checkpoint_path: str | Path = "data/raw/rawg_metadata_checkpoint.json",
    ) -> tuple[list[dict], list[dict]]:
        """Collect RAWG metadata for a list of games.

        Args:
            games: List of dicts with at least ``app_id`` and ``name`` keys.
            checkpoint_path: Path for progress saves.

        Returns:
            Tuple of (matched_games, unmatched_games).  Each matched game dict
            contains the RAWG metadata plus ``steam_app_id`` for joining.
        """
        checkpoint_path = Path(checkpoint_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        matched: dict[int, dict] = {}
        unmatched_ids: set[int] = set()

        if checkpoint_path.exists():
            with open(checkpoint_path) as f:
                saved = json.load(f)
            matched = {int(k): v for k, v in saved.get("matched", {}).items()}
            unmatched_ids = set(saved.get("unmatched_ids", []))
            logger.info(
                "Resuming RAWG collection: %d matched, %d unmatched",
                len(matched),
                len(unmatched_ids),
            )

        already_processed = set(matched.keys()) | unmatched_ids
        remaining = [g for g in games if g["app_id"] not in already_processed]
        logger.info(
            "RAWG: %d remaining of %d total games", len(remaining), len(games)
        )

        for i, game in enumerate(remaining):
            app_id = game["app_id"]
            name = game["name"]

            detail = self.match_steam_game(name)
            if detail:
                detail["steam_app_id"] = app_id
                matched[app_id] = self._extract_metadata(detail)
            else:
                unmatched_ids.add(app_id)

            if (i + 1) % 50 == 0:
                self._save_checkpoint(checkpoint_path, matched, unmatched_ids)
                logger.info(
                    "RAWG progress: %d matched, %d unmatched of %d processed",
                    len(matched),
                    len(unmatched_ids),
                    len(matched) + len(unmatched_ids),
                )

        self._save_checkpoint(checkpoint_path, matched, unmatched_ids)

        match_rate = len(matched) / max(len(matched) + len(unmatched_ids), 1)
        logger.info(
            "RAWG collection complete: %d matched, %d unmatched (%.1f%% match rate)",
            len(matched),
            len(unmatched_ids),
            match_rate * 100,
        )

        return (
            list(matched.values()),
            [{"app_id": aid} for aid in unmatched_ids],
        )

    @staticmethod
    def _extract_metadata(detail: dict) -> dict:
        """Extract and flatten relevant fields from a RAWG detail response."""
        return {
            "steam_app_id": detail.get("steam_app_id"),
            "rawg_id": detail.get("id"),
            "rawg_name": detail.get("name"),
            "match_score": detail.get("_match_score", 0),
            "released": detail.get("released"),
            "metacritic": detail.get("metacritic"),
            "rating": detail.get("rating"),
            "ratings_count": detail.get("ratings_count"),
            "genres": [g["name"] for g in detail.get("genres", [])],
            "tags": [t["name"] for t in detail.get("tags", [])],
            "platforms": [
                p["platform"]["name"]
                for p in detail.get("platforms", [])
                if p.get("platform")
            ],
            "developers": [d["name"] for d in detail.get("developers", [])],
            "publishers": [p["name"] for p in detail.get("publishers", [])],
            "esrb_rating": (
                detail.get("esrb_rating", {}) or {}
            ).get("name"),
            "description_raw": (detail.get("description_raw") or "")[:500],
        }

    @staticmethod
    def _save_checkpoint(
        path: Path, matched: dict, unmatched_ids: set
    ) -> None:
        with open(path, "w") as f:
            json.dump(
                {
                    "matched": {str(k): v for k, v in matched.items()},
                    "unmatched_ids": list(unmatched_ids),
                },
                f,
            )
