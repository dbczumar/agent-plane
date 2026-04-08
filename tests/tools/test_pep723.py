"""Tests for agent_plane.tools._pep723 (PEP 723 inline metadata parser)."""

from __future__ import annotations

from agent_plane.tools._pep723 import InlineMetadata, parse_inline_metadata


def test_parse_with_dependencies() -> None:
    """
    A source file with a ``# /// script`` block containing
    ``dependencies = [...]`` returns the parsed dep list.
    """
    source = """\
# /// script
# dependencies = ["requests>=2.28", "beautifulsoup4"]
# requires-python = ">=3.10"
# ///

SCHEMA = {}
async def run(args):
    return "ok"
"""
    result = parse_inline_metadata(source)
    assert result is not None
    assert result == InlineMetadata(
        dependencies=["requests>=2.28", "beautifulsoup4"],
    )


def test_parse_no_metadata_block() -> None:
    """
    A source file without a ``# /// script`` block returns None.
    """
    source = """\
SCHEMA = {}
async def run(args):
    return "ok"
"""
    assert parse_inline_metadata(source) is None


def test_parse_empty_dependencies() -> None:
    """
    A metadata block with an empty dependency list returns None
    (no deps to install).
    """
    source = """\
# /// script
# dependencies = []
# ///
"""
    assert parse_inline_metadata(source) is None


def test_parse_single_quotes() -> None:
    """
    Dependencies with single quotes are parsed correctly.
    """
    source = """\
# /// script
# dependencies = ['httpx', 'pydantic>=2.0']
# ///
"""
    result = parse_inline_metadata(source)
    assert result is not None
    assert result.dependencies == ["httpx", "pydantic>=2.0"]


def test_parse_block_without_dependencies_key() -> None:
    """
    A metadata block that exists but has no ``dependencies``
    key returns None.
    """
    source = """\
# /// script
# requires-python = ">=3.10"
# ///
"""
    assert parse_inline_metadata(source) is None
