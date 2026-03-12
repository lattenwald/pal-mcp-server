"""Tests for dynamic model discovery in listmodels."""

import json
import os
from unittest.mock import AsyncMock, patch

import pytest

from tools.listmodels import ListModelsTool


class TestListModelsDiscovery:
    """Test the discovered models section in listmodels output."""

    @pytest.fixture
    def tool(self):
        return ListModelsTool()

    @pytest.mark.asyncio
    async def test_discovery_section_shown_when_openrouter_configured(self, tool):
        """When OpenRouter is configured and live models are found, show discovery section."""
        discovered = [
            {"id": "openai/gpt-5.4", "name": "OpenAI: GPT-5.4", "context_length": 256000},
            {"id": "google/gemini-3.5-pro", "name": "Google: Gemini 3.5 Pro", "context_length": 2000000},
        ]

        env_vars = {"OPENROUTER_API_KEY": "test-key", "DEFAULT_MODEL": "auto"}

        with (
            patch.dict(os.environ, env_vars, clear=True),
            patch("providers.live_model_fetcher.get_live_model_fetcher") as mock_get_fetcher,
        ):
            mock_fetcher = AsyncMock()
            mock_fetcher.fetch_openrouter_models.return_value = discovered
            mock_get_fetcher.return_value = mock_fetcher

            result = await tool.execute({})
            content = json.loads(result[0].text)["content"]

            assert "Discovered via API" in content
            assert "openai/gpt-5.4" in content
            assert "google/gemini-3.5-pro" in content
            assert "256K" in content
            assert "2M" in content

    @pytest.mark.asyncio
    async def test_discovery_section_hidden_when_no_models_found(self, tool):
        """When no new models are discovered, don't show the section."""
        env_vars = {"OPENROUTER_API_KEY": "test-key", "DEFAULT_MODEL": "auto"}

        with (
            patch.dict(os.environ, env_vars, clear=True),
            patch("providers.live_model_fetcher.get_live_model_fetcher") as mock_get_fetcher,
        ):
            mock_fetcher = AsyncMock()
            mock_fetcher.fetch_openrouter_models.return_value = []
            mock_get_fetcher.return_value = mock_fetcher

            result = await tool.execute({})
            content = json.loads(result[0].text)["content"]

            assert "Discovered via API" not in content

    @pytest.mark.asyncio
    async def test_discovery_section_hidden_when_openrouter_not_configured(self, tool):
        """When OpenRouter is not configured, no discovery section."""
        env_vars = {"DEFAULT_MODEL": "auto"}

        with patch.dict(os.environ, env_vars, clear=True):
            result = await tool.execute({})
            content = json.loads(result[0].text)["content"]

            assert "Discovered via API" not in content

    @pytest.mark.asyncio
    async def test_discovery_graceful_on_fetcher_error(self, tool):
        """If fetcher raises, listmodels still works without discovery section."""
        env_vars = {"OPENROUTER_API_KEY": "test-key", "DEFAULT_MODEL": "auto"}

        with (
            patch.dict(os.environ, env_vars, clear=True),
            patch("providers.live_model_fetcher.get_live_model_fetcher") as mock_get_fetcher,
        ):
            mock_fetcher = AsyncMock()
            mock_fetcher.fetch_openrouter_models.side_effect = Exception("boom")
            mock_get_fetcher.return_value = mock_fetcher

            result = await tool.execute({})
            content = json.loads(result[0].text)["content"]

            # Should still produce valid output
            assert "Available AI Models" in content
            assert "Discovered via API" not in content
