"""Agent image spec: parsing, validation, and safe extraction."""

from __future__ import annotations

from pathlib import Path

from agent_plane.spec.parser import parse
from agent_plane.spec.tar_utils import ExtractionError, extract_safe
from agent_plane.spec.types import (
    AgentSpec,
    InteractionConfig,
    LLMConfig,
    LocalToolInfo,
    MCPServerConfig,
    ModalityConfig,
    SkillSpec,
    ToolsConfig,
)
from agent_plane.spec.validator import ValidationResult, validate

__all__ = [
    "AgentSpec",
    "ExtractionError",
    "InteractionConfig",
    "LLMConfig",
    "LocalToolInfo",
    "MCPServerConfig",
    "ModalityConfig",
    "SkillSpec",
    "ToolsConfig",
    "ValidationResult",
    "extract_safe",
    "load",
    "parse",
    "validate",
]


def load(source: Path | bytes, *, dest: Path | None = None) -> AgentSpec:
    """
    Load an agent spec from a directory, tarball path, or raw bytes.

    If *source* is a directory, parse and validate it directly.
    If *source* is a file path (tarball) or raw bytes, extract to
    *dest* first, then parse and validate the extracted directory.

    Args:
        source: Path to an agent image directory or .tar.gz bundle,
            or raw tarball bytes (e.g. from an HTTP upload).
        dest: Extraction destination — required when source is a
              tarball or bytes, ignored when source is a directory.

    Returns:
        A validated AgentSpec.

    Raises:
        ValueError: If the spec fails validation, or if source is a
            tarball/bytes and dest is not provided.
        FileNotFoundError: If source is a Path that does not exist,
            or if the extracted directory is missing config.yaml.
        ExtractionError: If the tarball fails safety checks.
    """
    if isinstance(source, bytes):
        if dest is None:
            raise ValueError("dest is required when loading from bytes")
        extract_safe(source, dest)
        root = dest
    elif source.is_dir():
        root = source
    elif source.is_file():
        if dest is None:
            raise ValueError("dest is required when loading from a tarball")
        extract_safe(source, dest)
        root = dest
    else:
        raise FileNotFoundError(f"source not found: {source}")

    spec = parse(root)
    result = validate(spec)
    if not result.valid:
        errors = "; ".join(f"{e.path}: {e.message}" for e in result.errors)
        raise ValueError(f"invalid agent spec: {errors}")
    return spec
