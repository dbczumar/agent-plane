"""Prompt construction — build Responses API inputs from spec + history."""

from __future__ import annotations

from typing import Any

from agent_plane.entities import (
    ConversationItem,
    FunctionCallData,
    FunctionCallOutputData,
    MessageData,
)
from agent_plane.spec import AgentSpec


def _extract_text(content_blocks: list[dict[str, Any]]) -> str:
    """
    Extract plain text from heterogeneous content blocks.
    Handles ``input_text``, ``output_text``, and bare ``text``
    block types.

    :param content_blocks: List of content block dicts, e.g.
        ``[{"type": "input_text", "text": "Hello"}]``.
    :returns: Joined text from all recognized blocks, or
        empty string if none contain text.
    """
    parts: list[str] = []
    for block in content_blocks:
        block_type = block.get("type")
        if block_type in ("input_text", "output_text", "text"):
            text = block.get("text")
            if text:
                parts.append(text)
    return "\n".join(parts) if parts else ""


def build_instructions(
    spec: AgentSpec,
    per_request_instructions: str | None,
    tool_schemas: list[dict[str, Any]],
) -> str:
    """
    Build the system instructions string from the agent's
    instructions, per-request instructions, and skill metadata.
    Passed as the ``instructions`` parameter to
    ``client.responses.create()``.

    :param spec: The parsed AgentSpec containing the agent's
        base instructions and skill definitions.
    :param per_request_instructions: Optional additional
        instructions for this specific request, appended
        after the agent's base instructions.
    :param tool_schemas: OpenAI-format tool schemas (used
        only for future skill-awareness hinting; currently
        not included in the instructions body).
    :returns: The assembled instructions string.
    """
    parts: list[str] = []

    if spec.instructions:
        parts.append(spec.instructions)

    if per_request_instructions:
        parts.append(per_request_instructions)

    # Let the LLM know about available skills (name + description only)
    # so it can call load_skill to retrieve the full content.
    if spec.skills:
        skill_lines = ["Available skills (use the load_skill tool to load one):"]
        for skill in spec.skills:
            skill_lines.append(f"- {skill.name}: {skill.description}")
        parts.append("\n".join(skill_lines))

    return "\n\n".join(parts) if parts else "You are a helpful assistant."


def history_to_input_items(
    items: list[ConversationItem],
) -> list[dict[str, Any]]:
    """
    Convert persisted ConversationItems into Responses API input items.

    Each item type maps directly to a Responses API input item format:
    ``message`` → role/content pair, ``function_call`` → function call
    item, ``function_call_output`` → function call output item. This
    is simpler than Chat Completions format because function calls are
    kept as separate items rather than embedded in assistant messages.

    :param items: Persisted conversation items in chronological order.
    :returns: A list of Responses API input item dicts suitable for
        ``client.responses.create(input=...)``.
    """
    result: list[dict[str, Any]] = []

    for item in items:
        if item.type == "message":
            assert isinstance(item.data, MessageData)
            # Pass content blocks through as-is. After
            # resolve_content_references(), all file_id refs are
            # already resolved to inline content.
            result.append(
                {
                    "role": item.data.role,
                    "content": item.data.content,
                }
            )

        elif item.type == "function_call":
            assert isinstance(item.data, FunctionCallData)
            result.append(
                {
                    "type": "function_call",
                    "call_id": item.data.call_id,
                    "name": item.data.name,
                    "arguments": item.data.arguments,
                }
            )

        elif item.type == "function_call_output":
            assert isinstance(item.data, FunctionCallOutputData)
            result.append(
                {
                    "type": "function_call_output",
                    "call_id": item.data.call_id,
                    "output": item.data.output,
                }
            )

        # reasoning items are not included in the LLM prompt
        # (they are output-only)

    return result
