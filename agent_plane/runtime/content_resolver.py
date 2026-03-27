"""Resolve file_id references in content blocks to inline content.

Scans conversation items for content blocks that reference uploaded
files via ``file_id`` and replaces them with inline base64 content.
This runs as a pre-processing step before prompt construction so the
prompt builder remains pure (no I/O).

See ``MULTIMODAL_INFERENCE.md`` for the full design.
"""

from __future__ import annotations

import base64
import copy
import logging
from typing import Any

from agent_plane.entities import ConversationItem, MessageData
from agent_plane.stores import ArtifactStore, FileStore

_logger = logging.getLogger(__name__)


def resolve_content_references(
    items: list[ConversationItem],
    file_store: FileStore,
    artifact_store: ArtifactStore,
) -> list[ConversationItem]:
    """
    Resolve ``file_id`` references in content blocks to inline content.

    Returns **copies** of items whose content was modified. Items
    without ``file_id`` references are returned as-is (no copy).
    The originals in the conversation store remain unchanged.

    Resolves ``file_id`` on **any** block type (``input_image``,
    ``input_file``, or future types like ``input_audio``). External
    URLs (``image_url``, ``file_url``) are never fetched — they pass
    through unchanged (SSRF protection).

    :param items: Persisted conversation items in chronological
        order, e.g. from ``conversation_store.fetch_all()``.
    :param file_store: Store for looking up file metadata
        (``content_type``, ``filename``).
    :param artifact_store: Store for fetching file binary content.
    :returns: A list of conversation items with all ``file_id``
        references replaced by inline base64 content.
    :raises ValueError: If a referenced ``file_id`` does not exist
        in the file store.
    :raises KeyError: If a referenced ``file_id`` exists in the
        file store but its binary content is missing from the
        artifact store.
    """
    result: list[ConversationItem] = []
    for item in items:
        if item.type == "message" and isinstance(item.data, MessageData):
            resolved_content = _resolve_message_content(
                item.data.content, file_store, artifact_store
            )
            if resolved_content is item.data.content:
                # No file_id references found — reuse original.
                result.append(item)
            else:
                # Content was modified — deep-copy and replace.
                item_copy = copy.deepcopy(item)
                assert isinstance(item_copy.data, MessageData)
                item_copy.data.content = resolved_content
                result.append(item_copy)
        else:
            result.append(item)
    return result


def _resolve_message_content(
    content: list[dict[str, Any]],
    file_store: FileStore,
    artifact_store: ArtifactStore,
) -> list[dict[str, Any]]:
    """
    Resolve ``file_id`` references in a list of content blocks.

    Returns the **original list** if no blocks contain ``file_id``
    (caller uses identity check to detect changes). Returns a
    **new list** with resolved blocks if any ``file_id`` was found.

    :param content: Content block dicts from ``MessageData.content``.
    :param file_store: Store for file metadata lookups.
    :param artifact_store: Store for binary content fetches.
    :returns: The original list (unchanged) or a new list with
        ``file_id`` references resolved to inline content.
    """
    resolved: list[dict[str, Any]] = []
    changed = False
    for block in content:
        if "file_id" in block:
            resolved.append(
                _resolve_file_id_block(block, file_store, artifact_store)
            )
            changed = True
        else:
            resolved.append(block)
    # Return original list when nothing changed so caller can use
    # identity check (``is``) to skip unnecessary deep-copies.
    return resolved if changed else content


def _resolve_file_id_block(
    block: dict[str, Any],
    file_store: FileStore,
    artifact_store: ArtifactStore,
) -> dict[str, Any]:
    """
    Resolve a single content block's ``file_id`` to inline content.

    For ``input_image`` blocks: replaces ``file_id`` with
    ``image_url`` containing a ``data:`` URI.

    For all other block types (``input_file``, future types):
    replaces ``file_id`` with ``file_data`` (raw base64 string).

    :param block: A content block dict containing ``file_id``,
        e.g. ``{"type": "input_image", "file_id": "file_abc123"}``.
    :param file_store: Store for file metadata lookups.
    :param artifact_store: Store for binary content fetches.
    :returns: A new dict with ``file_id`` replaced by inline
        content. All other fields are preserved.
    :raises ValueError: If ``file_id`` is not found in the file
        store.
    """
    file_id = block["file_id"]
    file_meta = file_store.get(file_id)
    if file_meta is None:
        raise ValueError(
            f"file_id '{file_id}' not found in file store"
        )

    content_bytes = artifact_store.get(file_id)
    encoded = base64.b64encode(content_bytes).decode("ascii")

    # Copy all fields except file_id.
    resolved: dict[str, Any] = {
        k: v for k, v in block.items() if k != "file_id"
    }

    block_type = block.get("type")
    if block_type == "input_image":
        # Image blocks use a data: URI in the image_url field.
        content_type = (
            file_meta.content_type or "application/octet-stream"
        )
        resolved["image_url"] = (
            f"data:{content_type};base64,{encoded}"
        )
    else:
        # input_file and any future type: inline as file_data.
        resolved["file_data"] = encoded
        if file_meta.content_type:
            resolved["content_type"] = file_meta.content_type

    return resolved
