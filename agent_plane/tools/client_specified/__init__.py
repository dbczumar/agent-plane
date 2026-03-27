"""Client-specified tools — tools whose schemas are supplied by the API caller.

These tools are defined at request time rather than baked into the agent
image. The caller provides an OpenAI-format function schema plus a
callback URL; when the agent invokes the tool, the runtime POSTs the
arguments to the callback and uses the response body as the result.

Public API:
- ``CallbackTool``: A :class:`~agent_plane.tools.base.Tool` that executes
  by making an HTTP POST to a caller-supplied URL.
- ``CallbackToolSpec``: Configuration for one callback tool (name, schema,
  callback URL and headers).
- ``parse_callback_tool_spec``: Parse one raw OpenAI tool dict (with the
  ``agent_plane`` extension key) into a :class:`CallbackToolSpec`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import httpx

from agent_plane.tools.base import Tool

_logger = logging.getLogger(__name__)

# Sync HTTP timeout for callback calls, in seconds.
# Tool-level retry (via execute_tool_with_retry) handles transient failures.
_CALLBACK_TIMEOUT = 30.0


@dataclass
class CallbackToolSpec:
    """
    Configuration for one client-specified callback tool.

    Holds the information needed to present the tool to the LLM
    and to execute it via HTTP when the LLM invokes it.

    :param name: Tool function name, e.g. ``"get_weather"``. Must match
        the ``function.name`` in the OpenAI schema.
    :param schema: OpenAI-format function tool object, e.g.
        ``{"type": "function", "function": {"name": "get_weather",
        "description": "...", "parameters": {...}}}``.
        The ``agent_plane`` extension key is stripped before storage.
    :param callback_url: URL to POST to when the tool is invoked, e.g.
        ``"https://api.example.com/tools/get_weather"``.
    :param callback_headers: HTTP headers to include with every callback
        request, e.g. ``{"Authorization": "Bearer tok_xyz"}``.
    """

    name: str
    schema: dict[str, Any]
    callback_url: str
    callback_headers: dict[str, str] = field(default_factory=dict)


class CallbackTool(Tool):
    """
    A tool that executes by POSTing to a caller-supplied HTTP endpoint.

    When the LLM invokes this tool, the runtime sends a POST request to
    :attr:`CallbackToolSpec.callback_url` with a JSON body of the form
    ``{"name": "<tool_name>", "arguments": "<json_string>"}``. The
    response body (as a plain string) is used as the tool result.

    HTTP errors and connection failures are caught and returned as error
    strings so the LLM can decide how to proceed.

    :param spec: The :class:`CallbackToolSpec` describing this tool.
    """

    def __init__(self, spec: CallbackToolSpec) -> None:
        """
        :param spec: The :class:`CallbackToolSpec` describing this tool.
        """
        self._spec = spec

    @property
    def name(self) -> str:
        """
        :returns: The tool function name, e.g. ``"get_weather"``.
        """
        return self._spec.name

    def get_schema(self) -> dict[str, Any]:
        """
        Return the OpenAI-format tool schema.

        :returns: The schema dict as supplied by the caller (the
            ``agent_plane`` extension key is already stripped).
        """
        return self._spec.schema

    def invoke(self, arguments: str) -> str:
        """
        Execute the tool by POSTing arguments to the callback URL.

        Sends ``{"name": "<name>", "arguments": "<json_string>"}``
        to :attr:`CallbackToolSpec.callback_url`. Returns the response
        body on success, or an error string on HTTP/network failure.

        :param arguments: JSON-encoded arguments from the LLM, e.g.
            ``'{"city": "San Francisco"}'``.
        :returns: The callback response body, or an error string
            beginning with ``"Error:"`` if the call failed.
        """
        try:
            response = httpx.post(
                self._spec.callback_url,
                json={"name": self._spec.name, "arguments": arguments},
                headers=self._spec.callback_headers,
                timeout=_CALLBACK_TIMEOUT,
            )
            response.raise_for_status()
            return response.text
        except httpx.HTTPStatusError as exc:
            _logger.warning(
                "Callback tool %r returned HTTP %d",
                self._spec.name,
                exc.response.status_code,
            )
            return f"Error: callback returned HTTP {exc.response.status_code}: {exc.response.text}"
        except httpx.HTTPError as exc:
            _logger.warning("Callback tool %r request failed: %s", self._spec.name, exc)
            return f"Error: callback request failed: {exc}"


def parse_callback_tool_spec(raw: dict[str, Any]) -> CallbackToolSpec:
    """
    Parse a raw OpenAI tool dict (with ``agent_plane`` extension) into
    a :class:`CallbackToolSpec`.

    The ``agent_plane`` key is expected to contain a ``callback`` sub-dict
    with a ``url`` field. Callers must validate the raw input before
    calling this function (e.g. in the route layer); this function raises
    ``ValueError`` on malformed input.

    :param raw: A dict in OpenAI function tool format extended with an
        ``agent_plane`` key, e.g.::

            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get current weather",
                    "parameters": {"type": "object", "properties": {...}}
                },
                "agent_plane": {
                    "callback": {
                        "url": "https://api.example.com/tool",
                        "headers": {"Authorization": "Bearer tok_xyz"}
                    }
                }
            }

    :returns: A :class:`CallbackToolSpec` with the callback metadata
        extracted and the ``agent_plane`` key stripped from the schema.
    :raises ValueError: If ``type`` is not ``"function"``, ``function.name``
        is missing, or ``agent_plane.callback.url`` is missing.
    """
    if raw.get("type") != "function":
        raise ValueError(
            f"client-specified tools must have type 'function', got {raw.get('type')!r}"
        )

    func = raw.get("function")
    if not isinstance(func, dict):
        raise ValueError("client-specified tool missing 'function' object")

    name = func.get("name")
    if not name:
        raise ValueError("client-specified tool missing function.name")

    ap = raw.get("agent_plane")
    if not isinstance(ap, dict):
        raise ValueError(
            f"client-specified tool {name!r} missing 'agent_plane' configuration"
        )

    callback = ap.get("callback")
    if not isinstance(callback, dict):
        raise ValueError(
            f"client-specified tool {name!r} missing 'agent_plane.callback' configuration"
        )

    callback_url = callback.get("url")
    if not callback_url:
        raise ValueError(
            f"client-specified tool {name!r} missing 'agent_plane.callback.url'"
        )

    # Strip the agent_plane key — the LLM should not see our internal metadata.
    schema = {k: v for k, v in raw.items() if k != "agent_plane"}

    # Validate headers are all strings if provided.
    raw_headers: dict[str, Any] = callback.get("headers") or {}
    if not all(isinstance(k, str) and isinstance(v, str) for k, v in raw_headers.items()):
        raise ValueError(
            f"client-specified tool {name!r}: agent_plane.callback.headers "
            "must be a mapping of string keys to string values"
        )
    callback_headers: dict[str, str] = dict(raw_headers)

    return CallbackToolSpec(
        name=name,
        schema=schema,
        callback_url=callback_url,
        callback_headers=callback_headers,
    )


def parse_callback_tool_specs(
    raw_tools: list[dict[str, Any]],
) -> list[CallbackToolSpec]:
    """
    Parse a list of raw tool dicts into :class:`CallbackToolSpec` objects.

    :param raw_tools: List of raw tool dicts from the API request, each
        in OpenAI function format extended with ``agent_plane`` metadata.
    :returns: A list of :class:`CallbackToolSpec` instances.
    :raises ValueError: If any tool in the list is malformed.
    """
    return [parse_callback_tool_spec(raw) for raw in raw_tools]


__all__ = [
    "CallbackTool",
    "CallbackToolSpec",
    "parse_callback_tool_spec",
    "parse_callback_tool_specs",
]
