"""Agent image spec: parsing, validation, and safe extraction."""

from __future__ import annotations

from pathlib import Path

from agent_plane.errors import AgentPlaneError, ErrorCode
from agent_plane.spec.parser import expand_env_vars, parse
from agent_plane.spec.tar_utils import ExtractionError, extract_safe
from agent_plane.spec.types import (
    AgentSpec,
    BuiltinToolConfig,
    ExecutionConfig,
    InteractionConfig,
    LLMConfig,
    LocalToolInfo,
    MCPServerConfig,
    ModalityConfig,
    RetryConfig,
    SkillSpec,
    ToolsConfig,
)
from agent_plane.spec.validator import ValidationResult, validate

__all__ = [
    "AgentSpec",
    "BuiltinToolConfig",
    "ExecutionConfig",
    "ExtractionError",
    "InteractionConfig",
    "LLMConfig",
    "LocalToolInfo",
    "MCPServerConfig",
    "ModalityConfig",
    "RetryConfig",
    "SkillSpec",
    "ToolsConfig",
    "ValidationResult",
    "expand_env_vars",
    "extract_safe",
    "load",
    "parse",
    "validate",
]


def load(source: Path | bytes, *, dest: Path | None = None) -> AgentSpec:
    """
    Load an agent spec from a directory, tarball path, or raw
    bytes.

    If *source* is a directory, parse and validate it directly.
    If *source* is a file path (tarball) or raw bytes, extract to
    *dest* first, then parse and validate the extracted directory.

    :param source: Path to an agent image directory or ``.tar.gz``
        bundle, or raw tarball bytes (e.g. from an HTTP upload).
    :param dest: Extraction destination -- required when *source*
        is a tarball or bytes, ignored when *source* is a
        directory.
    :returns: A validated :class:`AgentSpec`.
    :raises AgentPlaneError: If the spec fails validation, or if
        *source* is a tarball/bytes and *dest* is not provided.
    :raises FileNotFoundError: If *source* is a :class:`Path` that
        does not exist, or if the extracted directory is missing
        ``config.yaml``.
    :raises ExtractionError: If the tarball fails safety checks.
    """
    if isinstance(source, bytes):
        if dest is None:
            raise AgentPlaneError(
                "dest is required when loading from bytes",
                code=ErrorCode.INVALID_INPUT,
            )
        extract_safe(source, dest)
        root = dest
    elif source.is_dir():
        root = source
    elif source.is_file():
        if dest is None:
            raise AgentPlaneError(
                "dest is required when loading from a tarball",
                code=ErrorCode.INVALID_INPUT,
            )
        extract_safe(source, dest)
        root = dest
    else:
        raise FileNotFoundError(f"source not found: {source}")

    spec = parse(root)
    result = validate(spec)
    if not result.valid:
        errors = "; ".join(f"{e.path}: {e.message}" for e in result.errors)
        raise AgentPlaneError(
            f"invalid agent spec: {errors}",
            code=ErrorCode.INVALID_INPUT,
        )
    return spec
