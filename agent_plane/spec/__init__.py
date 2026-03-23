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


def load(source: Path, *, dest: Path | None = None) -> AgentSpec:
    """
    Load an agent spec from a directory or tarball.

    If *source* is a directory, parse and validate it directly.
    If *source* is a file (tarball), extract it to *dest* first,
    then parse and validate the extracted directory.

    Args:
        source: Path to an agent image directory or a .tar.gz bundle.
        dest: Extraction destination — required when source is a
              tarball, ignored when source is a directory.

    Returns:
        A validated AgentSpec.

    Raises:
        ValueError: If the spec fails validation, or if source is a
            tarball and dest is not provided.
        FileNotFoundError: If source does not exist, or if the
            extracted directory is missing config.yaml.
        ExtractionError: If the tarball fails safety checks.
    """
    if source.is_dir():
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
