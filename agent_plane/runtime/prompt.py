"""Prompt construction — build litellm message lists from spec + history."""

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
    Handles input_text, output_text, and bare text block types.
    """
    parts: list[str] = []
    for block in content_blocks:
        block_type = block.get("type")
        if block_type in ("input_text", "output_text", "text"):
            text = block.get("text")
            if text:
                parts.append(text)
    return "\n".join(parts) if parts else ""


def build_system_message(
    spec: AgentSpec,
    per_request_instructions: str | None,
    tool_schemas: list[dict[str, Any]],
) -> dict[str, str]:
    """
    Build the system message from the agent's instructions,
    per-request instructions, and skill metadata.
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

    content = "\n\n".join(parts) if parts else "You are a helpful assistant."
    return {"role": "system", "content": content}


def history_to_messages(
    items: list[ConversationItem],
) -> list[dict[str, Any]]:
    """
    Convert persisted ConversationItems into litellm/OpenAI chat messages.

    Merges consecutive function_call items into the preceding assistant
    message's tool_calls list (OpenAI expects them in one message).
    """
    messages: list[dict[str, Any]] = []

    for item in items:
        if item.type == "message":
            assert isinstance(item.data, MessageData)
            content = _extract_text(item.data.content)
            messages.append({"role": item.data.role, "content": content})

        elif item.type == "function_call":
            assert isinstance(item.data, FunctionCallData)
            tc: dict[str, Any] = {
                "id": item.data.call_id,
                "type": "function",
                "function": {
                    "name": item.data.name,
                    "arguments": item.data.arguments,
                },
            }
            # Merge with preceding assistant message if one exists
            if messages and messages[-1]["role"] == "assistant":
                messages[-1].setdefault("tool_calls", []).append(tc)
                # OpenAI requires content to be null (not missing)
                # when tool_calls is present and there was no text
                if not messages[-1].get("content"):
                    messages[-1]["content"] = None
            else:
                messages.append({
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [tc],
                })

        elif item.type == "function_call_output":
            assert isinstance(item.data, FunctionCallOutputData)
            messages.append({
                "role": "tool",
                "tool_call_id": item.data.call_id,
                "content": item.data.output,
            })

        # reasoning items are not included in the LLM prompt
        # (they are output-only)

    return messages


def build_messages(
    spec: AgentSpec,
    history: list[ConversationItem],
    instructions: str | None,
    tool_schemas: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Build the complete message list for a litellm.completion() call.
    """
    messages: list[dict[str, Any]] = [
        build_system_message(spec, instructions, tool_schemas),
    ]
    messages.extend(history_to_messages(history))
    return messages
