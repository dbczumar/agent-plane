"""Agent execution workflow — placeholder for Layer 1.

The real implementation will contain the agent loop with LLM calls,
tool execution, and the steering inbox handshake. This placeholder
proves the DBOS plumbing works end-to-end: accepts the right params,
writes to the output stream, and returns a response dict.
"""

from __future__ import annotations

from typing import Any

from agent_plane.runtime.durability import (
    close_stream,
    get_workflow_id,
    step,
    workflow,
    write_stream,
)


@step()
def _placeholder_step(agent_id: str, conversation_id: str) -> dict[str, str]:
    """Simulates a single LLM call. Will be replaced by real inference."""
    return {
        "role": "assistant",
        "content": (
            f"Placeholder response from agent {agent_id} in conversation {conversation_id}"
        ),
    }


@workflow()
def agent_execution_workflow(
    agent_id: str,
    conversation_id: str,
    previous_response_id: str | None = None,
    instructions: str | None = None,
) -> dict[str, Any]:
    """
    Placeholder agent execution workflow.

    Accepts the same parameters the real workflow will need.
    Writes a single streaming event and returns a response dict
    matching the shape that get() expects.
    """
    task_id = get_workflow_id()

    result = _placeholder_step(agent_id, conversation_id)

    # Write a streaming event so stream() has something to read
    write_stream("output", {"type": "message", "content": result["content"]})
    close_stream("output")

    return {
        "task_id": task_id,
        "status": "completed",  # matches TaskStatus.COMPLETED value
        "output": [result],
    }
