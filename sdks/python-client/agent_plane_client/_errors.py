"""Typed exceptions for the agent-plane client."""

from __future__ import annotations


class AgentPlaneError(Exception):
    """Base exception for all agent-plane client errors."""

    def __init__(self, message: str, status_code: int | None = None, code: str | None = None):
        super().__init__(message)
        self.status_code = status_code
        self.code = code


class AgentNotFoundError(AgentPlaneError):
    """Agent name or ID not found (HTTP 404)."""

    pass


class ResponseNotFoundError(AgentPlaneError):
    """Response ID not found (HTTP 404)."""

    pass


class FileNotFoundError(AgentPlaneError):
    """File ID not found (HTTP 404)."""

    pass


class ConversationNotFoundError(AgentPlaneError):
    """Conversation ID not found (HTTP 404)."""

    pass


class InvalidInputError(AgentPlaneError):
    """Bad request — invalid input, missing fields, etc. (HTTP 400)."""

    pass


class ConflictError(AgentPlaneError):
    """Resource conflict — duplicate name, stale state, etc. (HTTP 409)."""

    pass


class BundleInvalidError(AgentPlaneError):
    """Agent bundle is invalid — corrupt tarball, bad config, etc. (HTTP 400)."""

    pass


class ServerError(AgentPlaneError):
    """Internal server error (HTTP 5xx)."""

    pass


class ToolCallDenied(Exception):
    """Raised by ``on_tool_call_start`` hook to deny a client-side tool call.

    The exception message is sent back to the agent as the tool's output,
    so the agent knows the call was denied and can adapt.
    """

    pass


def raise_for_status(status_code: int, body: dict[str, object] | str) -> None:
    """Raise a typed exception based on HTTP status code and error body.

    Uses the server's error ``code`` field for classification when
    available, falling back to status code only. Never relies on
    substring matching in error messages.
    """
    if status_code < 400:
        return

    if isinstance(body, dict):
        error = body.get("error", {})
        if isinstance(error, dict):
            code = str(error.get("code", ""))
            message = str(error.get("message", str(body)))
        else:
            code = ""
            message = str(body)
    else:
        code = ""
        message = str(body)

    # Use the server's error code for precise classification.
    _CODE_MAP: dict[str, type[AgentPlaneError]] = {
        "not_found": AgentPlaneError,
        "invalid_input": InvalidInputError,
        "conflict": ConflictError,
        "server_error": ServerError,
    }

    if status_code == 404:
        # 404 is always "not found" — the specific type (agent, response,
        # file, conversation) is determined by the endpoint the caller
        # hit, not the error message. Callers can catch the base
        # AgentPlaneError or the specific subclass at the call site.
        raise AgentPlaneError(message, status_code, code)

    if status_code == 409:
        raise ConflictError(message, status_code, code)

    if status_code == 400:
        raise InvalidInputError(message, status_code, code)

    if status_code >= 500:
        raise ServerError(message, status_code, code)

    raise AgentPlaneError(message, status_code, code)
