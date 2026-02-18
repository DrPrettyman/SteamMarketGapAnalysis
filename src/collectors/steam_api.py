"""Steam Web API client with BFS friend-graph crawler.

Collects user-game ownership and playtime data by traversing the public
friend graph starting from a set of seed Steam IDs.

Endpoints used:
    - IPlayerService/GetOwnedGames
    - ISteamUser/GetFriendList
    - ISteamUser/GetPlayerSummaries
"""

import json
import logging
from collections import deque
from pathlib import Path

import requests

from src.utils import DiskCache, RateLimiter

logger = logging.getLogger(__name__)

BASE_URL = "https://api.steampowered.com"

# communityvisibilitystate == 3 means the profile is public
_PUBLIC_VISIBILITY = 3


class SteamAPIClient:
    """Client for the Steam Web API with BFS crawling capability.

    Args:
        api_key: Steam Web API key.
        requests_per_second: Max request rate (default 1.5 req/s).
        cache_dir: Directory for disk cache.  Defaults to ``data/raw/cache/steam``.
    """

    def __init__(
        self,
        api_key: str,
        requests_per_second: float = 1.5,
        cache_dir: str | Path = "data/raw/cache/steam",
    ) -> None:
        self._key = api_key
        self._session = requests.Session()
        self._limiter = RateLimiter(requests_per_second)
        self._cache = DiskCache(cache_dir)

    # ------------------------------------------------------------------
    # Low-level API helpers
    # ------------------------------------------------------------------

    def _get(self, url: str, params: dict) -> dict | None:
        """Issue a rate-limited, cached GET request.

        Returns parsed JSON on success, ``None`` on HTTP/network errors.
        """
        cached = self._cache.get(url, params)
        if cached is not None:
            return cached

        self._limiter.wait()
        try:
            resp = self._session.get(url, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            self._cache.set(url, data, params)
            return data
        except requests.RequestException as exc:
            logger.warning("Steam API error for %s: %s", url, exc)
            return None

    # ------------------------------------------------------------------
    # Public endpoint wrappers
    # ------------------------------------------------------------------

    def get_owned_games(self, steam_id: str) -> list[dict] | None:
        """Return list of games owned by *steam_id* with playtime.

        Each entry contains ``appid``, ``playtime_forever`` (minutes), and
        ``name`` (when available).  Returns ``None`` on error or private profile.
        """
        url = f"{BASE_URL}/IPlayerService/GetOwnedGames/v1/"
        params = {
            "key": self._key,
            "steamid": steam_id,
            "include_appinfo": 1,
            "include_played_free_games": 1,
            "format": "json",
        }
        data = self._get(url, params)
        if data is None:
            return None
        games = data.get("response", {}).get("games")
        return games  # May be None if profile hides game details

    def get_friend_list(self, steam_id: str) -> list[str] | None:
        """Return list of friend Steam IDs for *steam_id*.

        Returns ``None`` if the friend list is private or an error occurs.
        """
        url = f"{BASE_URL}/ISteamUser/GetFriendList/v1/"
        params = {
            "key": self._key,
            "steamid": steam_id,
            "relationship": "friend",
            "format": "json",
        }
        data = self._get(url, params)
        if data is None:
            return None
        friends = data.get("friendslist", {}).get("friends", [])
        return [f["steamid"] for f in friends]

    def get_player_summaries(self, steam_ids: list[str]) -> list[dict]:
        """Return player summaries for up to 100 Steam IDs.

        Only public profiles (``communityvisibilitystate == 3``) are useful
        for game data collection.
        """
        url = f"{BASE_URL}/ISteamUser/GetPlayerSummaries/v2/"
        # API accepts comma-separated list, max 100 per call
        results: list[dict] = []
        for i in range(0, len(steam_ids), 100):
            batch = steam_ids[i : i + 100]
            params = {
                "key": self._key,
                "steamids": ",".join(batch),
                "format": "json",
            }
            data = self._get(url, params)
            if data:
                results.extend(data.get("response", {}).get("players", []))
        return results

    def is_public(self, summary: dict) -> bool:
        """Check whether a player summary indicates a public profile."""
        return summary.get("communityvisibilitystate") == _PUBLIC_VISIBILITY

    # ------------------------------------------------------------------
    # BFS crawler
    # ------------------------------------------------------------------

    def crawl(
        self,
        seed_ids: list[str],
        max_users: int = 10_000,
        checkpoint_path: str | Path = "data/raw/steam_crawl_checkpoint.json",
    ) -> dict:
        """BFS-crawl the friend graph collecting game ownership data.

        Args:
            seed_ids: Starting Steam IDs.
            max_users: Stop after collecting this many public profiles.
            checkpoint_path: Path for periodic progress saves.

        Returns:
            Dictionary with keys ``users`` (list of user-game rows) and
            ``visited`` (set of visited Steam IDs).
        """
        checkpoint_path = Path(checkpoint_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        # Resume from checkpoint if it exists
        visited: set[str] = set()
        queue: deque[str] = deque()
        user_games: list[dict] = []

        if checkpoint_path.exists():
            logger.info("Resuming crawl from checkpoint %s", checkpoint_path)
            with open(checkpoint_path) as f:
                ckpt = json.load(f)
            visited = set(ckpt.get("visited", []))
            queue = deque(ckpt.get("queue", []))
            user_games = ckpt.get("user_games", [])
            logger.info(
                "Checkpoint loaded: %d visited, %d in queue, %d game rows",
                len(visited),
                len(queue),
                len(user_games),
            )

        # Seed the queue with any IDs we haven't visited yet
        for sid in seed_ids:
            if sid not in visited:
                queue.append(sid)

        collected_users = len({row["steam_id"] for row in user_games})
        total_api_calls = 0

        while queue and collected_users < max_users:
            steam_id = queue.popleft()
            if steam_id in visited:
                continue
            visited.add(steam_id)

            # Check if profile is public
            summaries = self.get_player_summaries([steam_id])
            total_api_calls += 1
            if not summaries or not self.is_public(summaries[0]):
                continue

            # Collect owned games
            games = self.get_owned_games(steam_id)
            total_api_calls += 1
            if games:
                for g in games:
                    user_games.append(
                        {
                            "steam_id": steam_id,
                            "app_id": g["appid"],
                            "name": g.get("name", ""),
                            "playtime_forever": g.get("playtime_forever", 0),
                            "playtime_2weeks": g.get("playtime_2weeks", 0),
                        }
                    )
                collected_users += 1

            # Expand friends into the queue
            friends = self.get_friend_list(steam_id)
            total_api_calls += 1
            if friends:
                for fid in friends:
                    if fid not in visited:
                        queue.append(fid)

            # Periodic checkpoint
            if collected_users % 100 == 0 and collected_users > 0:
                self._save_checkpoint(
                    checkpoint_path, visited, queue, user_games
                )
                logger.info(
                    "Progress: %d users collected, %d visited, %d in queue, %d API calls",
                    collected_users,
                    len(visited),
                    len(queue),
                    total_api_calls,
                )

        # Final checkpoint
        self._save_checkpoint(checkpoint_path, visited, queue, user_games)
        logger.info(
            "Crawl complete: %d users, %d game rows, %d total API calls",
            collected_users,
            len(user_games),
            total_api_calls,
        )

        return {"user_games": user_games, "visited": list(visited)}

    @staticmethod
    def _save_checkpoint(
        path: Path,
        visited: set[str],
        queue: deque[str],
        user_games: list[dict],
    ) -> None:
        with open(path, "w") as f:
            json.dump(
                {
                    "visited": list(visited),
                    "queue": list(queue),
                    "user_games": user_games,
                },
                f,
            )
