"""PEP 723 inline script metadata parser.

Extracts dependency declarations from Python tool files that use the
`PEP 723 <https://peps.python.org/pep-0723/>`_ inline metadata format::

    # /// script
    # dependencies = ["requests>=2.28", "beautifulsoup4"]
    # requires-python = ">=3.10"
    # ///

When dependencies are found, the tool subprocess is invoked via
``uv run --with dep1 --with dep2 -- python _runner.py`` so that
deps are auto-resolved and cached by uv.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class InlineMetadata:
    """
    Parsed PEP 723 inline script metadata.

    :param dependencies: List of PEP 508 dependency specifiers,
        e.g. ``["requests>=2.28", "beautifulsoup4"]``.
    """

    dependencies: list[str]


# Matches the opening marker: ``# /// script``
_BLOCK_START_RE = re.compile(r"^#\s*///\s*script\s*$")
# Matches the closing marker: ``# ///``
_BLOCK_END_RE = re.compile(r"^#\s*///\s*$")
# Matches a dependencies line: ``# dependencies = [...]``
_DEPS_RE = re.compile(
    r"^#\s*dependencies\s*=\s*\[([^\]]*)\]",
)


def parse_inline_metadata(source: str) -> InlineMetadata | None:
    """
    Extract PEP 723 inline script metadata from Python source.

    Scans for a ``# /// script`` ... ``# ///`` block and extracts
    the ``dependencies`` list. Returns ``None`` if no metadata
    block is found or if the block contains no dependencies.

    :param source: The full source text of a Python file.
    :returns: Parsed metadata with dependencies, or ``None``.
    """
    lines = source.splitlines()
    in_block = False
    block_lines: list[str] = []

    for line in lines:
        if not in_block:
            if _BLOCK_START_RE.match(line):
                in_block = True
            continue
        if _BLOCK_END_RE.match(line):
            break
        block_lines.append(line)

    if not block_lines:
        return None

    # Join block lines and search for dependencies = [...]
    block_text = "\n".join(block_lines)
    match = _DEPS_RE.search(block_text)
    if match is None:
        return None

    raw_deps = match.group(1)
    deps = _parse_dep_list(raw_deps)
    if not deps:
        return None
    return InlineMetadata(dependencies=deps)


def _parse_dep_list(raw: str) -> list[str]:
    """
    Parse a comma-separated list of quoted dependency specifiers.

    Handles both single and double quotes, strips whitespace.
    Example input: ``'"requests>=2.28", "beautifulsoup4"'``

    :param raw: The raw string between ``[`` and ``]``.
    :returns: List of dependency specifier strings.
    """
    # Extract all quoted strings (single or double)
    return [m.group(1) or m.group(2) for m in re.finditer(r'"([^"]*?)"|\'([^\']*?)\'', raw)]
