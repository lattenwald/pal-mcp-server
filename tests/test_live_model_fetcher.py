"""Tests for live model discovery fetcher."""

import time
from unittest.mock import AsyncMock, patch

import pytest

from providers.live_model_fetcher import LiveModelFetcher

# Sample OpenRouter API response with mix of providers
SAMPLE_OPENROUTER_RESPONSE = {
    "data": [
        {
            "id": "openai/gpt-5.4",
            "name": "OpenAI: GPT-5.4",
            "context_length": 256000,
            "top_provider": {"context_length": 256000},
        },
        {
            "id": "openai/gpt-5.2",
            "name": "OpenAI: GPT-5.2",
            "context_length": 128000,
            "top_provider": {"context_length": 128000},
        },
        {
            "id": "google/gemini-3.5-pro",
            "name": "Google: Gemini 3.5 Pro",
            "context_length": 2000000,
            "top_provider": {"context_length": 2000000},
        },
        {
            "id": "x-ai/grok-5",
            "name": "xAI: Grok 5",
            "context_length": 200000,
            "top_provider": {"context_length": 200000},
        },
        {
            "id": "nousresearch/hermes-3",
            "name": "Nous: Hermes 3",
            "context_length": 32000,
            "top_provider": {"context_length": 32000},
        },
    ]
}


class TestLiveModelFetcher:
    """Tests for LiveModelFetcher."""

    def setup_method(self):
        """Reset singleton state between tests."""
        LiveModelFetcher._instance = None

    @pytest.mark.asyncio
    async def test_fetch_openrouter_models_returns_filtered_results(self):
        """Fetcher returns only models from target provider prefixes."""
        fetcher = LiveModelFetcher(cache_ttl_seconds=3600)

        with patch.object(fetcher, "_http_get_json", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = SAMPLE_OPENROUTER_RESPONSE

            models = await fetcher.fetch_openrouter_models(api_key="test-key")

            assert any(m["id"] == "openai/gpt-5.4" for m in models)
            assert any(m["id"] == "google/gemini-3.5-pro" for m in models)
            assert any(m["id"] == "x-ai/grok-5" for m in models)
            # nousresearch is NOT in the target prefixes
            assert not any(m["id"] == "nousresearch/hermes-3" for m in models)

    @pytest.mark.asyncio
    async def test_fetch_openrouter_models_excludes_known_static_models(self):
        """Fetcher excludes models already in static registry."""
        fetcher = LiveModelFetcher(cache_ttl_seconds=3600)
        # gpt-5.2 is in openrouter_models.json as openai/gpt-5.2 (simulate)
        fetcher._static_model_ids = {"openai/gpt-5.2"}

        with patch.object(fetcher, "_http_get_json", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = SAMPLE_OPENROUTER_RESPONSE

            models = await fetcher.fetch_openrouter_models(api_key="test-key")

            assert any(m["id"] == "openai/gpt-5.4" for m in models)
            assert not any(m["id"] == "openai/gpt-5.2" for m in models)

    @pytest.mark.asyncio
    async def test_cache_returns_same_results_within_ttl(self):
        """Second call within TTL returns cached data without HTTP request."""
        fetcher = LiveModelFetcher(cache_ttl_seconds=3600)

        with patch.object(fetcher, "_http_get_json", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = SAMPLE_OPENROUTER_RESPONSE

            first = await fetcher.fetch_openrouter_models(api_key="test-key")
            second = await fetcher.fetch_openrouter_models(api_key="test-key")

            assert first == second
            assert mock_get.call_count == 1  # Only one HTTP call

    @pytest.mark.asyncio
    async def test_cache_expires_after_ttl(self):
        """After TTL expires, fetcher makes a new HTTP request."""
        fetcher = LiveModelFetcher(cache_ttl_seconds=1)

        with patch.object(fetcher, "_http_get_json", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = SAMPLE_OPENROUTER_RESPONSE

            await fetcher.fetch_openrouter_models(api_key="test-key")
            # Expire the cache
            fetcher._cache["openrouter"]["timestamp"] = time.monotonic() - 2

            await fetcher.fetch_openrouter_models(api_key="test-key")
            assert mock_get.call_count == 2

    @pytest.mark.asyncio
    async def test_fetch_returns_empty_on_http_error_no_cache(self):
        """HTTP failures return empty list when no stale cache exists."""
        fetcher = LiveModelFetcher(cache_ttl_seconds=3600)

        with patch.object(fetcher, "_http_get_json", new_callable=AsyncMock) as mock_get:
            mock_get.side_effect = Exception("connection refused")

            models = await fetcher.fetch_openrouter_models(api_key="test-key")
            assert models == []

    @pytest.mark.asyncio
    async def test_fetch_returns_stale_cache_on_http_error(self):
        """HTTP failures return stale cached data when available."""
        fetcher = LiveModelFetcher(cache_ttl_seconds=1)

        with patch.object(fetcher, "_http_get_json", new_callable=AsyncMock) as mock_get:
            # First call succeeds and populates cache
            mock_get.return_value = SAMPLE_OPENROUTER_RESPONSE
            first = await fetcher.fetch_openrouter_models(api_key="test-key")
            assert len(first) > 0

            # Expire the cache
            fetcher._cache["openrouter"]["timestamp"] = time.monotonic() - 2

            # Second call fails — should return stale data, not empty
            mock_get.side_effect = Exception("connection refused")
            models = await fetcher.fetch_openrouter_models(api_key="test-key")
            assert models == first

    @pytest.mark.asyncio
    async def test_fetch_xai_models_filters_grok_prefix(self):
        """X.AI fetcher only returns grok-* model IDs."""
        fetcher = LiveModelFetcher(cache_ttl_seconds=3600)

        with patch.object(fetcher, "_http_get_json", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = {
                "data": [
                    {"id": "grok-4", "object": "model"},
                    {"id": "grok-4-1-fast-reasoning", "object": "model"},
                    {"id": "grok-5-beta", "object": "model"},
                    {"id": "embedding-v1", "object": "model"},
                    {"id": "moderation-latest", "object": "model"},
                ]
            }

            models = await fetcher.fetch_xai_models(api_key="test-key")

            ids = {m["id"] for m in models}
            assert "grok-5-beta" in ids
            assert "embedding-v1" not in ids
            assert "moderation-latest" not in ids

    @pytest.mark.asyncio
    async def test_fetch_xai_models_excludes_static_registry(self):
        """X.AI fetcher excludes models already in static XAI registry."""
        fetcher = LiveModelFetcher(cache_ttl_seconds=3600)
        fetcher._static_xai_model_ids = {"grok-4", "grok-4-1-fast-reasoning"}

        with patch.object(fetcher, "_http_get_json", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = {
                "data": [
                    {"id": "grok-4", "object": "model"},
                    {"id": "grok-4-1-fast-reasoning", "object": "model"},
                    {"id": "grok-5-beta", "object": "model"},
                ]
            }

            models = await fetcher.fetch_xai_models(api_key="test-key")

            ids = {m["id"] for m in models}
            assert ids == {"grok-5-beta"}

    @pytest.mark.asyncio
    async def test_fetch_xai_models_returns_empty_on_error(self):
        """X.AI fetcher returns empty list on HTTP error."""
        fetcher = LiveModelFetcher(cache_ttl_seconds=3600)

        with patch.object(fetcher, "_http_get_json", new_callable=AsyncMock) as mock_get:
            mock_get.side_effect = Exception("connection refused")

            models = await fetcher.fetch_xai_models(api_key="test-key")
            assert models == []

    @pytest.mark.asyncio
    async def test_top_n_filtering(self):
        """Only top N models per provider prefix are returned."""
        fetcher = LiveModelFetcher(cache_ttl_seconds=3600, top_n_per_provider=1)

        with patch.object(fetcher, "_http_get_json", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = SAMPLE_OPENROUTER_RESPONSE

            models = await fetcher.fetch_openrouter_models(api_key="test-key")

            openai_models = [m for m in models if m["id"].startswith("openai/")]
            # Only 1 OpenAI model (the one with highest context_length)
            assert len(openai_models) == 1
            assert openai_models[0]["id"] == "openai/gpt-5.4"
