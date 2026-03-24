"""Centralized error handling for the agent-plane server.

All user-facing errors should be raised as AgentPlaneError with an
appropriate error code. The FastAPI exception handler (registered in
server/app.py) catches these and returns a JSON response with the
correct HTTP status code.

Existing HTTPException usage continues to work — FastAPI handles both.
New code should prefer AgentPlaneError for consistency.
"""

from __future__ import annotations


class ErrorCode:
    """
    Error codes and their HTTP status mappings.

    Add new codes here as needed. The string value is what appears in
    the JSON response body.
    """

    NOT_FOUND = "not_found"
    INVALID_INPUT = "invalid_input"
    ALREADY_EXISTS = "already_exists"
    INTERNAL_ERROR = "internal_error"


# Single source of truth for error code → HTTP status.
_CODE_TO_HTTP_STATUS: dict[str, int] = {
    ErrorCode.NOT_FOUND: 404,
    ErrorCode.INVALID_INPUT: 400,
    ErrorCode.ALREADY_EXISTS: 409,
    ErrorCode.INTERNAL_ERROR: 500,
}


class AgentPlaneError(Exception):
    """
    Application-level error with a machine-readable code.

    Raise this from routes, stores, or any layer. The global FastAPI
    exception handler converts it to a JSON response automatically.
    """

    def __init__(self, message: str, *, code: str = ErrorCode.INTERNAL_ERROR) -> None:
        super().__init__(message)
        self.code = code
        self.message = message

    @property
    def http_status(self) -> int:
        return _CODE_TO_HTTP_STATUS.get(self.code, 500)
