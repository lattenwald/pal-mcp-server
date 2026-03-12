"""Fetch live model lists from provider APIs with TTL cache.

This module discovers models available from API endpoints that may not
be in the static JSON configuration files. Results are cached in memory
to avoid repeated API calls.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import httpx

from utils.env import get_env

logger = logging.getLogger(__name__)

# Provider prefixes we care about on OpenRouter
TARGET_PROVIDER_PREFIXES = (
    "openai/",
    "google/",
    "x-ai/",
    "anthropic/",
    "meta-llama/",
    "mistralai/",
)

DEFAULT_CACHE_TTL = 3600  # 1 hour
DEFAULT_TOP_N = 5
HTTP_TIMEOUT = 15  # seconds


def _format_discovered_context(ctx: int) -> str:
    """Format a context length integer for display."""
    if not ctx:
        return "?"
    if ctx >= 1_000_000:
        return f"{ctx // 1_000_000}M"
    if ctx >= 1_000:
        return f"{ctx // 1_000}K"
    return str(ctx)


class LiveModelFetcher:
    """Fetches and caches live model lists from provider APIs.

    Usage:
        fetcher = get_live_model_fetcher()
        models = await fetcher.fetch_openrouter_models(api_key="...")
    """

    _instance: LiveModelFetcher | None = None

    def __init__(
        self,
        *,
        cache_ttl_seconds: int | None = None,
        top_n_per_provider: int | None = None,
    ) -> None:
        if cache_ttl_seconds is not None:
            self._cache_ttl = cache_ttl_seconds
        else:
            ttl_env = get_env("LIVE_DISCOVERY_CACHE_TTL")
            self._cache_ttl = int(ttl_env) if ttl_env else DEFAULT_CACHE_TTL

        if top_n_per_provider is not None:
            self._top_n = top_n_per_provider
        else:
            top_n_env = get_env("LIVE_DISCOVERY_TOP_N")
            self._top_n = int(top_n_env) if top_n_env else DEFAULT_TOP_N

        self._cache: dict[str, dict[str, Any]] = {}
        self._static_model_ids: set[str] | None = None
        self._static_xai_model_ids: set[str] | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def fetch_openrouter_models(self, *, api_key: str) -> list[dict[str, Any]]:
        """Fetch models from OpenRouter, filtered and cached.

        Returns list of dicts with keys: id, name, context_length.
        Only includes models from target providers that are NOT already
        in the static openrouter_models.json registry.
        """
        cached = self._get_cached("openrouter")
        if cached is not None:
            return cached

        try:
            data = await self._http_get_json(
                "https://openrouter.ai/api/v1/models",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "HTTP-Referer": get_env(
                        "OPENROUTER_REFERER",
                        "https://github.com/BeehiveInnovations/pal-mcp-server",
                    )
                    or "https://github.com/BeehiveInnovations/pal-mcp-server",
                },
            )
        except Exception:
            logger.warning("Failed to fetch live models from OpenRouter", exc_info=True)
            stale = self._get_stale_cached("openrouter")
            return stale if stale is not None else []

        raw_models = data.get("data", [])
        filtered = self._filter_and_rank(raw_models)
        self._set_cached("openrouter", filtered)
        return filtered

    async def fetch_xai_models(self, *, api_key: str) -> list[dict[str, Any]]:
        """Fetch models from X.AI, filtered to grok-* and cached.

        Returns list of dicts with keys: id, name, context_length.
        Only includes grok-* models NOT already in the static XAI registry.
        """
        cached = self._get_cached("xai")
        if cached is not None:
            return cached

        try:
            data = await self._http_get_json(
                "https://api.x.ai/v1/models",
                headers={"Authorization": f"Bearer {api_key}"},
            )
        except Exception:
            logger.warning("Failed to fetch live models from X.AI", exc_info=True)
            stale = self._get_stale_cached("xai")
            return stale if stale is not None else []

        static_ids = self._get_static_xai_model_ids()
        result = []
        for model in data.get("data", []):
            model_id = model.get("id", "")
            if not model_id.startswith("grok-"):
                continue
            if model_id in static_ids:
                continue
            result.append(
                {
                    "id": model_id,
                    "name": model.get("name", model_id),
                    "context_length": model.get("context_length", 0),
                }
            )

        self._set_cached("xai", result)
        return result

    # ------------------------------------------------------------------
    # Filtering
    # ------------------------------------------------------------------

    def _filter_and_rank(self, raw_models: list[dict]) -> list[dict[str, Any]]:
        """Filter to target providers, exclude static, keep top N per provider."""
        static_ids = self._get_static_model_ids()

        # Group by provider prefix
        by_provider: dict[str, list[dict]] = {}
        for model in raw_models:
            model_id = model.get("id", "")

            # Only target providers
            if not any(model_id.startswith(prefix) for prefix in TARGET_PROVIDER_PREFIXES):
                continue

            # Skip models already in static config
            if model_id in static_ids:
                continue

            prefix = model_id.split("/")[0]
            by_provider.setdefault(prefix, []).append(model)

        # Sort each group by context_length desc, then id desc (newer versions)
        result = []
        for _prefix, models in sorted(by_provider.items()):
            models.sort(
                key=lambda m: (
                    m.get("top_provider", {}).get("context_length", 0) or m.get("context_length", 0),
                    m.get("id", ""),
                ),
                reverse=True,
            )
            for model in models[: self._top_n]:
                ctx = model.get("top_provider", {}).get("context_length", 0) or model.get("context_length", 0)
                result.append(
                    {
                        "id": model["id"],
                        "name": model.get("name", model["id"]),
                        "context_length": ctx,
                    }
                )

        return result

    def _get_static_model_ids(self) -> set[str]:
        """Lazily load the set of model IDs from the static OpenRouter registry."""
        if self._static_model_ids is not None:
            return self._static_model_ids

        try:
            from providers.registries.openrouter import OpenRouterModelRegistry

            registry = OpenRouterModelRegistry()
            self._static_model_ids = set(registry.list_models())
        except Exception:
            logger.debug("Could not load static OpenRouter registry for exclusion", exc_info=True)
            self._static_model_ids = set()

        return self._static_model_ids

    def _get_static_xai_model_ids(self) -> set[str]:
        """Lazily load the set of model IDs from the static XAI registry."""
        if self._static_xai_model_ids is not None:
            return self._static_xai_model_ids

        try:
            from providers.registries.xai import XAIModelRegistry

            registry = XAIModelRegistry()
            self._static_xai_model_ids = set(registry.list_models())
        except Exception:
            logger.debug("Could not load static XAI registry for exclusion", exc_info=True)
            self._static_xai_model_ids = set()

        return self._static_xai_model_ids

    # ------------------------------------------------------------------
    # Cache
    # ------------------------------------------------------------------

    def _get_cached(self, key: str) -> list[dict[str, Any]] | None:
        """Return cached data if within TTL, else None."""
        entry = self._cache.get(key)
        if entry is None:
            return None
        if time.monotonic() - entry["timestamp"] > self._cache_ttl:
            return None
        return entry["data"]

    def _get_stale_cached(self, key: str) -> list[dict[str, Any]] | None:
        """Return cached data regardless of TTL (fallback for errors)."""
        entry = self._cache.get(key)
        if entry is None:
            return None
        return entry["data"]

    def _set_cached(self, key: str, data: list[dict[str, Any]]) -> None:
        self._cache[key] = {"data": data, "timestamp": time.monotonic()}

    # ------------------------------------------------------------------
    # HTTP
    # ------------------------------------------------------------------

    async def _http_get_json(self, url: str, headers: dict | None = None) -> dict:
        """Make an async HTTP GET and return parsed JSON."""
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
            response = await client.get(url, headers=headers or {})
            response.raise_for_status()
            return response.json()


def get_live_model_fetcher() -> LiveModelFetcher:
    """Return the singleton LiveModelFetcher instance."""
    if LiveModelFetcher._instance is None:
        LiveModelFetcher._instance = LiveModelFetcher()
    return LiveModelFetcher._instance
