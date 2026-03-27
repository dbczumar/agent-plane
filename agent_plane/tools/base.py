"""Abstract base class for agent tools."""

from __future__ import annotations

import abc
import re
from typing import Any

# OpenAI function-calling constraint: names must match this pattern.
# MCP and client-specified tools must be validated before registration.
TOOL_NAME_RE: re.Pattern[str] = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")


def is_valid_tool_name(name: str) -> bool:
    """
    Check whether a tool name satisfies the OpenAI function-calling
    constraint: 1–64 characters, alphanumeric plus ``_`` and ``-``.

    :param name: The tool name to validate, e.g. ``"get_weather"``.
    :returns: ``True`` if the name is valid, ``False`` otherwise.
    """
    return TOOL_NAME_RE.match(name) is not None


class Tool(abc.ABC):
    """
    Abstract base class for all tools available to the agent.

    Each tool has a unique name, an OpenAI-format schema for the
    LLM, and an ``invoke`` method that executes the tool and
    returns a string result.
    """

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """
        Unique tool name used for dispatch and schema registration.

        :returns: The tool name, e.g. ``"load_skill"``.
        """

    @abc.abstractmethod
    def get_schema(self) -> dict[str, Any]:
        """
        Return the OpenAI Chat Completions tool schema.

        :returns: A dict with ``"type": "function"`` and a
            ``"function"`` sub-dict describing the tool's name,
            description, and parameters.
        """

    @abc.abstractmethod
    def invoke(self, arguments: str) -> str:
        """
        Execute the tool with the given arguments.

        :param arguments: JSON-encoded arguments string from
            the LLM, e.g. ``'{"name": "summarize"}'``.
        :returns: The tool's string result.
        """
