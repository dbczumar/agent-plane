"""Runtime caps — operator-configured hard ceilings."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RuntimeCaps:
    """
    Operator-configured hard ceiling for agent execution.

    Agent specs with a higher ``execution.timeout`` are clamped
    to this value via ``min(spec, cap)``.

    :param execution_timeout: Max wall-clock time for the entire
        agent loop in seconds, e.g. ``7200``.
    """

    execution_timeout: int = 7200
