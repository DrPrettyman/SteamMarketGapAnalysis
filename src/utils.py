"""Shared utilities: configuration loading, rate limiting, caching, and logging."""

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_config(path: str | Path | None = None) -> dict:
    """Load YAML configuration from disk.

    Args:
        path: Path to config file. Defaults to ``config.yaml`` in the project root.

    Returns:
        Parsed configuration dictionary.

    Raises:
        FileNotFoundError: If neither the given path nor the default exists.
    """
    if path is None:
        path = PROJECT_ROOT / "config.yaml"
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Config file not found at {path}. "
            "Copy config.example.yaml to config.yaml and fill in your API keys."
        )
    with open(path) as f:
        return yaml.safe_load(f)


class RateLimiter:
    """Token-bucket rate limiter for API calls."""

    def __init__(self, requests_per_second: float) -> None:
        self._min_interval = 1.0 / requests_per_second
        self._last_call: float = 0.0

    def wait(self) -> None:
        """Block until the next request is allowed."""
        elapsed = time.monotonic() - self._last_call
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_call = time.monotonic()


class DiskCache:
    """Simple JSON-based disk cache for API responses.

    Stores each response as a JSON file keyed by a hash of the request URL +
    params.  Entries older than ``ttl_hours`` are treated as stale.
    """

    def __init__(self, cache_dir: str | Path, ttl_hours: int = 168) -> None:
        self._dir = Path(cache_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._ttl_seconds = ttl_hours * 3600

    def _key(self, url: str, params: dict | None = None) -> str:
        raw = url + json.dumps(params or {}, sort_keys=True)
        return hashlib.sha256(raw.encode()).hexdigest()

    def get(self, url: str, params: dict | None = None) -> Any | None:
        """Return cached value or ``None`` if missing/stale."""
        path = self._dir / f"{self._key(url, params)}.json"
        if not path.exists():
            return None
        age = time.time() - path.stat().st_mtime
        if age > self._ttl_seconds:
            path.unlink()
            return None
        with open(path) as f:
            return json.load(f)

    def set(self, url: str, data: Any, params: dict | None = None) -> None:
        """Write a value to the cache."""
        path = self._dir / f"{self._key(url, params)}.json"
        with open(path, "w") as f:
            json.dump(data, f)


def setup_logging(level: int = logging.INFO) -> None:
    """Configure project-wide logging to both console and file."""
    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(exist_ok=True)

    fmt = logging.Formatter(
        "%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console = logging.StreamHandler()
    console.setLevel(level)
    console.setFormatter(fmt)

    # File handler (append mode — survives restarts)
    file_handler = logging.FileHandler(log_dir / "collect.log")
    file_handler.setLevel(level)
    file_handler.setFormatter(fmt)

    root = logging.getLogger()
    root.setLevel(level)
    root.addHandler(console)
    root.addHandler(file_handler)
