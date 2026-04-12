"""Shared fixtures for frontend tests."""

from __future__ import annotations

import os

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    """Add --llm-api-key option for integration tests."""
    parser.addoption(
        "--llm-api-key",
        action="store",
        default=None,
        help="LLM API key for integration tests.",
    )


@pytest.fixture(autouse=True)
def _set_api_key(request: pytest.FixtureRequest) -> None:
    """Set OPENAI_API_KEY from --llm-api-key if provided."""
    key = request.config.getoption("--llm-api-key", default=None)
    if key is not None:
        os.environ["OPENAI_API_KEY"] = key
