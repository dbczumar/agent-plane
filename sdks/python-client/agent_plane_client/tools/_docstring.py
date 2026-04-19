"""
Google-style docstring parsing for ``@tool``-decorated functions.

Extracts the function description (everything before the first
section header) and per-parameter descriptions (from the
``Args:`` / ``Arguments:`` / ``Parameters:`` section).

Used by the schema-derivation logic to populate the function-calling
JSON schema's ``description`` and ``properties[name].description``
fields.

We don't depend on a third-party docstring library because the
Google-style format is simple and our needs are narrow. NumPy and
Sphinx styles are intentionally not supported — authors should
use Google style or the explicit ``Annotated[T, Field(description=...)]``
form for per-param descriptions.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass

# Recognized section headers that terminate the description and
# the args section. Case-sensitive (Google convention).
_SECTION_HEADERS = (
    "Args:",
    "Arguments:",
    "Parameters:",
    "Returns:",
    "Return:",
    "Yields:",
    "Yield:",
    "Raises:",
    "Raise:",
    "Note:",
    "Notes:",
    "Example:",
    "Examples:",
    "See Also:",
    "Warning:",
    "Warnings:",
    "Attributes:",
)

_ARGS_HEADERS = ("Args:", "Arguments:", "Parameters:")


@dataclass(frozen=True)
class ParsedDocstring:
    """
    Result of parsing a Google-style docstring.

    :param description: The function-level description, taken from the
        text preceding the first section header. Whitespace-trimmed.
    :param param_descriptions: Mapping from parameter name to its
        description, extracted from the ``Args:`` / ``Arguments:`` /
        ``Parameters:`` section. Empty if no such section exists.
    """

    description: str
    param_descriptions: dict[str, str]


def parse_google_docstring(doc: str) -> ParsedDocstring:
    """
    Parse a Google-style docstring into description and per-param docs.

    Recognizes ``Args:`` / ``Arguments:`` / ``Parameters:`` as the
    parameter-list section header. Within that section, lines like
    ``    name: description`` (or ``    name (type): description``)
    are parsed as parameter entries; subsequent more-indented lines
    are treated as continuations of the current parameter's
    description. The args section ends at the next recognized
    section header.

    :param doc: The raw docstring text (typically from
        ``fn.__doc__``). May be ``None``-equivalent (empty string).
    :returns: A :class:`ParsedDocstring` with the extracted description
        and parameter descriptions. Returns empty values rather than
        raising for malformed input.
    """
    if not doc:
        return ParsedDocstring(description="", param_descriptions={})

    cleaned = inspect.cleandoc(doc)
    if not cleaned:
        return ParsedDocstring(description="", param_descriptions={})

    lines = cleaned.split("\n")

    # Locate the first recognized section header. Everything before
    # it is the description; the args section (if any) is what
    # contains parameter docs.
    description_lines: list[str] = []
    args_section_lines: list[str] = []
    in_args_section = False
    in_other_section = False

    for line in lines:
        stripped = line.strip()
        if stripped in _SECTION_HEADERS:
            in_args_section = stripped in _ARGS_HEADERS
            in_other_section = not in_args_section
            continue
        if in_args_section:
            args_section_lines.append(line)
        elif in_other_section:
            # Skip non-args sections (Returns:, Raises:, etc.)
            continue
        else:
            description_lines.append(line)

    description = "\n".join(description_lines).strip()
    param_descriptions = _parse_args_lines(args_section_lines)

    return ParsedDocstring(
        description=description,
        param_descriptions=param_descriptions,
    )


def _parse_args_lines(lines: list[str]) -> dict[str, str]:
    """
    Parse the body of an ``Args:`` section into per-param descriptions.

    Param lines have the form ``    name: description`` or
    ``    name (type): description`` at the section's base indent.
    Lines indented further are treated as continuations of the
    current parameter's description.

    :param lines: The lines following the ``Args:`` header (not
        including the header itself), up to the next section.
    :returns: Mapping from parameter name to its (whitespace-collapsed)
        description.
    """
    # Determine the base indent — the indent of the first non-empty line.
    base_indent: int | None = None
    for line in lines:
        if line.strip():
            base_indent = len(line) - len(line.lstrip())
            break

    if base_indent is None:
        return {}

    param_descriptions: dict[str, str] = {}
    current_name: str | None = None
    current_parts: list[str] = []

    for line in lines:
        if not line.strip():
            # Blank lines within args section are continuation separators;
            # they don't terminate a param.
            continue
        line_indent = len(line) - len(line.lstrip())

        if line_indent == base_indent:
            # Start of a new param entry.
            if current_name is not None:
                param_descriptions[current_name] = " ".join(current_parts).strip()
                current_name = None
                current_parts = []

            stripped = line.strip()
            if ":" not in stripped:
                # Malformed entry; skip it.
                continue

            name_part, _, desc = stripped.partition(":")
            # Handle "name (type)" form by trimming the parenthetical.
            if "(" in name_part:
                name_only = name_part.split("(", 1)[0].strip()
            else:
                name_only = name_part.strip()

            if name_only.isidentifier():
                current_name = name_only
                current_parts = [desc.strip()]
            # Else: not a valid param entry; ignore.
        else:
            # Continuation line for the current param.
            if current_name is not None:
                current_parts.append(line.strip())

    if current_name is not None:
        param_descriptions[current_name] = " ".join(current_parts).strip()

    return param_descriptions
