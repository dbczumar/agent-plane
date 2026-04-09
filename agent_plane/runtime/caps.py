"""Runtime caps — operator-configured hard ceilings."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RuntimeCaps:
    """
    Operator-configured runtime policies for agent execution.

    These are deployment/security decisions that agents cannot
    override. Agent specs are clamped to these limits.

    :param execution_timeout: Max wall-clock time for the entire
        agent loop in seconds, e.g. ``7200``.
    :param sandbox_enabled: Whether to use ``srt`` sandboxing for
        local tool execution when available on PATH. ``True`` by
        default. This is a runtime security policy — agents cannot
        opt out. The agent spec only controls ``docker_image``
        (what container to use).
    """

    execution_timeout: int = 7200
    sandbox_enabled: bool = True
