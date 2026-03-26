"""
Mock LLM server with a controllable gate for cross-server tests.

Implements the OpenAI Responses API streaming format. Blocks
LLM responses on an internal asyncio gate until released via
``POST /gate/release``, creating deterministic race windows
across server processes.

Endpoints:

- ``POST /v1/responses`` — accepts LLM request, blocks on gate,
  then returns a streaming SSE response.
- ``GET /gate/pending`` — returns ``{"pending": true}`` when a
  request is waiting on the gate.
- ``POST /gate/release`` — unblocks the pending request.
- ``GET /stats`` — returns ``{"request_count": N}``.

Usage::

    python tests/server/integration/mock_llm_server.py 9999
"""

from __future__ import annotations

import asyncio
import json
import sys
from collections.abc import AsyncIterator

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

app = FastAPI()

_gate = asyncio.Event()
_request_pending = asyncio.Event()
_request_count = 0

_RESPONSE_TEXT = "Cross-server response from mock LLM"


async def _generate_sse() -> AsyncIterator[str]:
    """
    Yield a single ``response.completed`` SSE event.

    The format matches the OpenAI Responses API streaming
    protocol parsed by ``OpenAIAdapter._stream_responses()``.
    """
    completed = {
        "response": {
            "model": "mock-model",
            "output": [
                {
                    "type": "message",
                    "content": [
                        {
                            "type": "output_text",
                            "text": _RESPONSE_TEXT,
                        },
                    ],
                },
            ],
            "usage": {
                "input_tokens": 10,
                "output_tokens": 5,
                "total_tokens": 15,
            },
        },
    }
    yield f"event: response.completed\ndata: {json.dumps(completed)}\n\n"


@app.post("/v1/responses")
async def create_response(request: Request) -> StreamingResponse:
    """
    Accept an LLM request, block on gate, then return SSE.

    :param request: The incoming FastAPI request.
    """
    global _request_count  # noqa: PLW0603
    _request_count += 1
    # Drain the request body so the client doesn't hang
    await request.body()
    _request_pending.set()
    await _gate.wait()
    return StreamingResponse(
        _generate_sse(),
        media_type="text/event-stream",
    )


@app.get("/gate/pending")
async def gate_pending() -> dict[str, bool]:
    """Check if an LLM request is waiting on the gate."""
    return {"pending": _request_pending.is_set()}


@app.post("/gate/release")
async def gate_release() -> dict[str, bool]:
    """Release the gate, unblocking the pending LLM request."""
    _gate.set()
    return {"released": True}


@app.get("/stats")
async def stats() -> dict[str, int]:
    """Return the total number of LLM requests received."""
    return {"request_count": _request_count}


if __name__ == "__main__":
    port = int(sys.argv[1])
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=port,
        log_level="warning",
    )
