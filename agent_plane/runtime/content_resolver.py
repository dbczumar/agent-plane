"""Resolve file_id references in content blocks to inline content.

Scans conversation items for content blocks that reference uploaded
files via ``file_id`` and replaces them with inline base64 content.
This runs as a pre-processing step before prompt construction so the
prompt builder remains pure (no I/O).

See ``designs/MULTIMODAL_INFERENCE.md`` for the full design.
"""

from __future__ import annotations

import base64
import copy
import logging
from typing import Any

from agent_plane.entities import ConversationItem, MessageData
from agent_plane.stores import ArtifactStore, FileStore

_logger = logging.getLogger(__name__)

# Extensions that Python's mimetypes module doesn't always know,
# depending on the platform and Python version. Used as a fallback
# when the stored content_type is missing or generic. LLM providers
# (OpenAI) reject application/octet-stream for text files, so any
# text-like format needs a proper MIME type.
_EXTRA_MIME_TYPES: dict[str, str] = {
    # Markup / config
    ".md": "text/markdown",
    ".markdown": "text/markdown",
    ".yaml": "text/yaml",
    ".yml": "text/yaml",
    ".toml": "text/plain",
    ".jsonl": "application/jsonl",
    ".ndjson": "application/x-ndjson",
    ".proto": "text/plain",
    ".graphql": "text/plain",
    ".gql": "text/plain",
    # Languages mimetypes misses
    ".rs": "text/x-rust",
    ".go": "text/x-go",
    ".ts": "text/typescript",
    ".tsx": "text/typescript",
    ".jsx": "text/javascript",
    ".swift": "text/x-swift",
    ".kt": "text/x-kotlin",
    ".scala": "text/x-scala",
    ".r": "text/x-r",
    ".jl": "text/x-julia",
    ".lua": "text/x-lua",
    ".ex": "text/x-elixir",
    ".exs": "text/x-elixir",
    ".erl": "text/x-erlang",
    ".hs": "text/x-haskell",
    ".clj": "text/x-clojure",
    ".dart": "text/x-dart",
    ".vue": "text/plain",
    ".svelte": "text/plain",
    # Infra / build
    ".tf": "text/plain",
    ".hcl": "text/plain",
    ".dockerfile": "text/plain",
    ".gradle": "text/plain",
    ".ipynb": "application/x-ipynb+json",
    # Dotfiles
    ".env": "text/plain",
    ".lock": "text/plain",
}


def resolve_content_references(
    items: list[ConversationItem],
    file_store: FileStore,
    artifact_store: ArtifactStore,
    cache: dict[str, str] | None = None,
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
    :param cache: Optional per-task cache mapping ``file_id`` to
        its base64-encoded content. Avoids re-fetching and
        re-encoding the same file across agent loop iterations.
        Pass ``None`` to disable caching.
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
                item.data.content, file_store, artifact_store, cache
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
    cache: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    """
    Resolve ``file_id`` references in a list of content blocks.

    Returns the **original list** if no blocks contain ``file_id``
    (caller uses identity check to detect changes). Returns a
    **new list** with resolved blocks if any ``file_id`` was found.

    :param content: Content block dicts from ``MessageData.content``.
    :param file_store: Store for file metadata lookups.
    :param artifact_store: Store for binary content fetches.
    :param cache: Optional per-task base64 cache (see
        :func:`resolve_content_references`).
    :returns: The original list (unchanged) or a new list with
        ``file_id`` references resolved to inline content.
    """
    resolved: list[dict[str, Any]] = []
    changed = False
    for block in content:
        if "file_id" in block:
            resolved.append(_resolve_file_id_block(block, file_store, artifact_store, cache))
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
    cache: dict[str, str] | None = None,
) -> dict[str, Any]:
    """
    Resolve a single content block's ``file_id`` to inline content.

    For ``input_image`` blocks: replaces ``file_id`` with
    ``image_url`` containing a ``data:`` URI.

    For all other block types (``input_file``, future types):
    replaces ``file_id`` with ``file_data`` containing a ``data:``
    URI (e.g. ``"data:application/pdf;base64,..."``).  Provider
    adapters parse the URI to extract the media type and payload.

    :param block: A content block dict containing ``file_id``,
        e.g. ``{"type": "input_image", "file_id": "file_abc123"}``.
    :param file_store: Store for file metadata lookups.
    :param artifact_store: Store for binary content fetches.
    :param cache: Optional per-task base64 cache (see
        :func:`resolve_content_references`).
    :returns: A new dict with ``file_id`` replaced by inline
        content. All other fields are preserved.
    :raises ValueError: If ``file_id`` is not found in the file
        store — the file was deleted between request validation
        and agent loop execution.
    """
    file_id = block["file_id"]
    file_meta = file_store.get(file_id)
    if file_meta is None:
        raise ValueError(
            f"Referenced file '{file_id}' no longer exists — "
            f"it may have been deleted after the request was accepted"
        )

    # Use cached base64 if available; otherwise fetch, encode, and cache.
    if cache is not None and file_id in cache:
        encoded = cache[file_id]
    else:
        content_bytes = artifact_store.get(file_id)
        encoded = base64.b64encode(content_bytes).decode("ascii")
        if cache is not None:
            cache[file_id] = encoded

    # Copy all fields except file_id.
    resolved: dict[str, Any] = {k: v for k, v in block.items() if k != "file_id"}

    content_type = _resolve_content_type(file_meta.content_type, file_meta.filename)

    block_type = block.get("type")
    if block_type == "input_image":
        resolved["image_url"] = f"data:{content_type};base64,{encoded}"
    else:
        # input_file and any future type: inline as file_data.
        # Uses a data: URI so providers (OpenAI, etc.) can parse
        # the media type alongside the payload.
        resolved["file_data"] = f"data:{content_type};base64,{encoded}"

    return resolved


def _resolve_content_type(
    stored_type: str | None,
    filename: str | None,
) -> str:
    """
    Determine the MIME type for a file, with fallbacks.

    Priority: stored content_type (unless it's the generic
    ``application/octet-stream``) → ``mimetypes.guess_type``
    from filename → ``_EXTRA_MIME_TYPES`` lookup → ``text/plain``
    for text-like extensions → ``application/octet-stream``.

    Some LLM providers (OpenAI) reject ``application/octet-stream``
    for text files, so we try hard to resolve a specific type.

    :param stored_type: The content_type from file metadata, or
        ``None``.
    :param filename: The original filename, e.g. ``"report.md"``.
    :returns: A MIME type string.
    """
    import mimetypes as _mt
    from pathlib import PurePath

    # Use stored type if it's specific (not the generic fallback).
    if stored_type and stored_type != "application/octet-stream":
        return stored_type

    if filename:
        suffix = PurePath(filename).suffix.lower()
        # Try stdlib first.
        guessed = _mt.guess_type(filename)[0]
        if guessed and guessed != "application/octet-stream":
            return guessed
        # Fallback for extensions mimetypes doesn't know.
        if suffix in _EXTRA_MIME_TYPES:
            return _EXTRA_MIME_TYPES[suffix]
        # Text-like extensions default to text/plain rather than
        # octet-stream, which providers are more likely to accept.
        if suffix in {".txt", ".log", ".cfg", ".ini", ".env"}:
            return "text/plain"

    return stored_type or "application/octet-stream"
