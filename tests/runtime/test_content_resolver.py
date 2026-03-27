"""Tests for agent_plane.runtime.content_resolver."""

from __future__ import annotations

import base64
from dataclasses import dataclass
from typing import Any

import pytest

from agent_plane.entities import ConversationItem, StoredFile
from agent_plane.entities.conversation import (
    FunctionCallData,
    MessageData,
)
from agent_plane.runtime.content_resolver import resolve_content_references

# ── Fake stores ──────────────────────────────────────────────────────


@dataclass
class FakeFileStore:
    """
    In-memory FileStore that returns pre-configured StoredFile
    objects by file_id.

    :param files: Mapping of file_id to StoredFile.
    """

    files: dict[str, StoredFile]

    def get(self, file_id: str) -> StoredFile | None:
        """
        Return the StoredFile for *file_id*, or ``None``.

        :param file_id: The file identifier to look up.
        :returns: The matching StoredFile or None.
        """
        return self.files.get(file_id)


@dataclass
class FakeArtifactStore:
    """
    In-memory ArtifactStore that returns pre-configured binary
    blobs by key.

    :param blobs: Mapping of artifact key to binary content.
    """

    blobs: dict[str, bytes]

    def get(self, key: str) -> bytes:
        """
        Return binary content for *key*.

        :param key: The artifact key to look up.
        :returns: The binary content.
        :raises KeyError: If no blob exists for the key.
        """
        return self.blobs[key]


# ── Helpers ──────────────────────────────────────────────────────────


def _make_conversation_item(
    content: list[dict[str, Any]],
    *,
    role: str = "user",
    item_id: str = "msg_001",
    response_id: str = "resp_001",
) -> ConversationItem:
    """
    Build a message ConversationItem with the given content blocks.

    :param content: Content block dicts, e.g.
        ``[{"type": "input_text", "text": "Hello"}]``.
    :param role: Message role, ``"user"`` or ``"assistant"``.
    :param item_id: Store-assigned item ID.
    :param response_id: The response/task ID.
    :returns: A ConversationItem with type ``"message"``.
    """
    data = MessageData(
        role=role,
        content=content,
        # assistant messages require an agent name.
        agent="test-agent" if role == "assistant" else None,
    )
    return ConversationItem(
        id=item_id,
        type="message",
        status="completed",
        response_id=response_id,
        created_at=1000,
        data=data,
    )


def _make_function_call_item(
    *,
    item_id: str = "fc_001",
    response_id: str = "resp_001",
) -> ConversationItem:
    """
    Build a function_call ConversationItem.

    :param item_id: Store-assigned item ID.
    :param response_id: The response/task ID.
    :returns: A ConversationItem with type ``"function_call"``.
    """
    return ConversationItem(
        id=item_id,
        type="function_call",
        status="completed",
        response_id=response_id,
        created_at=1000,
        data=FunctionCallData(
            agent="test-agent",
            call_id="call_001",
            name="grep",
            arguments="{}",
        ),
    )


PNG_BYTES = b"\x89PNG\r\n\x1a\n fake png content"
PDF_BYTES = b"%PDF-1.4 fake pdf content"


@pytest.fixture()
def file_store() -> FakeFileStore:
    """
    FileStore with two pre-configured files: an image and a PDF.

    :returns: A FakeFileStore with ``file_img`` and ``file_pdf``.
    """
    return FakeFileStore(
        files={
            "file_img": StoredFile(
                id="file_img",
                created_at=1000,
                filename="photo.png",
                bytes=len(PNG_BYTES),
                content_type="image/png",
            ),
            "file_pdf": StoredFile(
                id="file_pdf",
                created_at=1000,
                filename="report.pdf",
                bytes=len(PDF_BYTES),
                content_type="application/pdf",
            ),
        }
    )


@pytest.fixture()
def artifact_store() -> FakeArtifactStore:
    """
    ArtifactStore with binary content for the image and PDF files.

    :returns: A FakeArtifactStore with blobs for ``file_img`` and
        ``file_pdf``.
    """
    return FakeArtifactStore(
        blobs={
            "file_img": PNG_BYTES,
            "file_pdf": PDF_BYTES,
        }
    )


# ── Tests ────────────────────────────────────────────────────────────


def test_text_only_message_passes_through(
    file_store: FakeFileStore,
    artifact_store: FakeArtifactStore,
) -> None:
    """
    Messages with only text blocks should pass through unchanged
    (no copy, no modification).
    """
    item = _make_conversation_item([{"type": "input_text", "text": "Hello"}])
    result = resolve_content_references(
        [item],
        file_store,
        artifact_store,  # type: ignore[arg-type]
    )

    # Exactly one item returned, same object (no copy needed).
    # Failure would mean text-only messages are unnecessarily copied.
    assert len(result) == 1
    assert result[0] is item


def test_input_image_file_id_resolved_to_data_uri(
    file_store: FakeFileStore,
    artifact_store: FakeArtifactStore,
) -> None:
    """
    input_image with file_id must be resolved to a data: URI in
    image_url, with file_id removed from the block.
    """
    item = _make_conversation_item(
        [
            {"type": "input_text", "text": "What's in this image?"},
            {
                "type": "input_image",
                "file_id": "file_img",
                "detail": "auto",
            },
        ]
    )
    result = resolve_content_references(
        [item],
        file_store,
        artifact_store,  # type: ignore[arg-type]
    )

    # Result is a copy — original must not be modified.
    # Failure would mean in-place mutation corrupts conversation store.
    assert result[0] is not item
    assert isinstance(result[0].data, MessageData)
    blocks = result[0].data.content

    # Text block passes through unchanged.
    assert blocks[0] == {"type": "input_text", "text": "What's in this image?"}

    # Image block: file_id replaced with data: URI.
    img_block = blocks[1]
    expected_b64 = base64.b64encode(PNG_BYTES).decode("ascii")
    # file_id must be removed — it's a local reference the LLM can't use.
    # Failure would mean the LLM receives a meaningless file_id.
    assert "file_id" not in img_block
    # image_url must be a data: URI with the correct content type.
    # Failure would mean the LLM receives an invalid image reference.
    assert img_block["image_url"] == (f"data:image/png;base64,{expected_b64}")
    # detail field must be preserved — it controls provider image resolution.
    # Failure would mean the client's detail preference is lost.
    assert img_block["detail"] == "auto"
    # Block type must be preserved for downstream translation layers.
    assert img_block["type"] == "input_image"


def test_input_file_file_id_resolved_to_file_data(
    file_store: FakeFileStore,
    artifact_store: FakeArtifactStore,
) -> None:
    """
    input_file with file_id must be resolved to file_data (base64),
    with content_type from file store metadata.
    """
    item = _make_conversation_item(
        [
            {"type": "input_text", "text": "Summarize"},
            {
                "type": "input_file",
                "file_id": "file_pdf",
                "filename": "report.pdf",
            },
        ]
    )
    result = resolve_content_references(
        [item],
        file_store,
        artifact_store,  # type: ignore[arg-type]
    )

    assert result[0] is not item
    assert isinstance(result[0].data, MessageData)
    file_block = result[0].data.content[1]

    expected_b64 = base64.b64encode(PDF_BYTES).decode("ascii")
    # file_id must be removed.
    assert "file_id" not in file_block
    # file_data must contain the base64-encoded content.
    # Failure would mean the LLM receives no file content.
    assert file_block["file_data"] == expected_b64
    # content_type from file store metadata must be included.
    # Failure would mean the provider can't determine the file format.
    assert file_block["content_type"] == "application/pdf"
    # filename must be preserved from the original block.
    assert file_block["filename"] == "report.pdf"
    assert file_block["type"] == "input_file"


def test_image_url_passes_through_unchanged(
    file_store: FakeFileStore,
    artifact_store: FakeArtifactStore,
) -> None:
    """
    input_image with image_url (no file_id) must pass through
    unchanged — URLs are never fetched server-side (SSRF protection).
    """
    item = _make_conversation_item(
        [
            {
                "type": "input_image",
                "image_url": "https://example.com/photo.png",
                "detail": "high",
            },
        ]
    )
    result = resolve_content_references(
        [item],
        file_store,
        artifact_store,  # type: ignore[arg-type]
    )

    # No file_id in any block — original item returned as-is.
    # Failure would mean URL-only messages are unnecessarily copied.
    assert result[0] is item


def test_inline_file_data_passes_through_unchanged(
    file_store: FakeFileStore,
    artifact_store: FakeArtifactStore,
) -> None:
    """
    input_file with file_data (no file_id) must pass through
    unchanged — content is already inline.
    """
    item = _make_conversation_item(
        [
            {
                "type": "input_file",
                "file_data": "JVBERi0xLjQK",
                "filename": "report.pdf",
            },
        ]
    )
    result = resolve_content_references(
        [item],
        file_store,
        artifact_store,  # type: ignore[arg-type]
    )

    # No file_id — original item returned.
    assert result[0] is item


def test_non_message_items_pass_through(
    file_store: FakeFileStore,
    artifact_store: FakeArtifactStore,
) -> None:
    """
    Non-message items (function_call, function_call_output) must
    pass through unchanged — only message content blocks are scanned.
    """
    fc_item = _make_function_call_item()
    result = resolve_content_references(
        [fc_item],
        file_store,
        artifact_store,  # type: ignore[arg-type]
    )

    # function_call items have no content blocks to resolve.
    # Failure would mean non-message items are incorrectly processed.
    assert len(result) == 1
    assert result[0] is fc_item


def test_missing_file_id_raises_value_error(
    file_store: FakeFileStore,
    artifact_store: FakeArtifactStore,
) -> None:
    """
    Referencing a file_id that doesn't exist in the file store
    must raise ValueError — fail loud, no silent dropping.
    """
    item = _make_conversation_item(
        [
            {
                "type": "input_image",
                "file_id": "file_nonexistent",
            },
        ]
    )

    with pytest.raises(ValueError, match="file_nonexistent"):
        resolve_content_references(
            [item],
            file_store,
            artifact_store,  # type: ignore[arg-type]
        )


def test_original_item_not_mutated(
    file_store: FakeFileStore,
    artifact_store: FakeArtifactStore,
) -> None:
    """
    The original ConversationItem must not be mutated — the resolver
    returns copies for modified items.
    """
    original_content = [
        {"type": "input_image", "file_id": "file_img"},
    ]
    item = _make_conversation_item(original_content)

    resolve_content_references(
        [item],
        file_store,
        artifact_store,  # type: ignore[arg-type]
    )

    # Original content must still contain file_id — not mutated.
    # Failure would mean the resolver modifies conversation store data.
    assert isinstance(item.data, MessageData)
    assert item.data.content[0]["file_id"] == "file_img"
    assert "image_url" not in item.data.content[0]


def test_unknown_block_type_with_file_id_resolved(
    file_store: FakeFileStore,
    artifact_store: FakeArtifactStore,
) -> None:
    """
    An unrecognized block type (e.g. input_audio) with file_id must
    still have its file_id resolved — the resolver resolves file_id
    on any block type, not just known ones.
    """
    item = _make_conversation_item(
        [
            {
                "type": "input_audio",
                "file_id": "file_img",
            },
        ]
    )
    result = resolve_content_references(
        [item],
        file_store,
        artifact_store,  # type: ignore[arg-type]
    )

    assert isinstance(result[0].data, MessageData)
    block = result[0].data.content[0]
    # file_id resolved even for unknown type.
    # Failure would mean new content types can't use file_id.
    assert "file_id" not in block
    # Unknown types get file_data (not image_url, which is
    # only for input_image).
    assert "file_data" in block
    assert block["type"] == "input_audio"


def test_mixed_items_preserves_order(
    file_store: FakeFileStore,
    artifact_store: FakeArtifactStore,
) -> None:
    """
    A mixed list of message and non-message items must preserve
    chronological order after resolution.
    """
    items = [
        _make_conversation_item(
            [{"type": "input_text", "text": "first"}],
            item_id="msg_001",
        ),
        _make_conversation_item(
            [{"type": "input_image", "file_id": "file_img"}],
            item_id="msg_002",
        ),
        _make_function_call_item(item_id="fc_001"),
        _make_conversation_item(
            [{"type": "input_text", "text": "third"}],
            item_id="msg_003",
        ),
    ]
    result = resolve_content_references(
        items,
        file_store,
        artifact_store,  # type: ignore[arg-type]
    )

    # Order must be preserved: msg, msg(resolved), fc, msg.
    # Failure would mean resolution reorders conversation history.
    assert len(result) == 4
    assert result[0].id == "msg_001"
    assert result[1].id == "msg_002"
    assert result[2].id == "fc_001"
    assert result[3].id == "msg_003"

    # Only msg_002 should be a copy (it had file_id).
    assert result[0] is items[0]
    assert result[1] is not items[1]
    assert result[2] is items[2]
    assert result[3] is items[3]


@pytest.mark.parametrize(
    ("block_type", "expected_field"),
    [
        pytest.param(
            "input_image",
            "image_url",
            id="input_image_gets_data_uri",
        ),
        pytest.param(
            "input_file",
            "file_data",
            id="input_file_gets_file_data",
        ),
    ],
)
def test_resolution_field_varies_by_block_type(
    block_type: str,
    expected_field: str,
    file_store: FakeFileStore,
    artifact_store: FakeArtifactStore,
) -> None:
    """
    input_image blocks get image_url (data: URI), while input_file
    blocks get file_data (raw base64). The resolution target field
    depends on block type.
    """
    item = _make_conversation_item(
        [
            {"type": block_type, "file_id": "file_img"},
        ]
    )
    result = resolve_content_references(
        [item],
        file_store,
        artifact_store,  # type: ignore[arg-type]
    )

    assert isinstance(result[0].data, MessageData)
    block = result[0].data.content[0]
    # The expected field must be present after resolution.
    # Failure would mean the wrong inline format is used for this type.
    assert expected_field in block
    assert "file_id" not in block


# ── Cache tests ─────────────────────────────────────────────────────


def test_cache_avoids_redundant_artifact_fetch(
    file_store: FakeFileStore,
    artifact_store: FakeArtifactStore,
) -> None:
    """
    When a cache dict is provided, the second resolution of the same
    file_id must use the cached base64 instead of re-fetching from
    the artifact store.
    """
    cache: dict[str, str] = {}
    item = _make_conversation_item([{"type": "input_image", "file_id": "file_img"}])

    # First call — populates cache.
    resolve_content_references(
        [item],
        file_store,
        artifact_store,  # type: ignore[arg-type]
        cache,
    )

    # Cache should now contain the file_id.
    assert "file_img" in cache
    expected_b64 = base64.b64encode(PNG_BYTES).decode("ascii")
    # Cached value is the raw base64, not a data: URI.
    assert cache["file_img"] == expected_b64

    # Sabotage the artifact store — if the cache is working, the
    # resolver won't call artifact_store.get() again.
    artifact_store.blobs = {}

    result2 = resolve_content_references(
        [item],
        file_store,
        artifact_store,  # type: ignore[arg-type]
        cache,
    )

    # Second call should still succeed using cached value.
    assert isinstance(result2[0].data, MessageData)
    img_block = result2[0].data.content[0]
    assert img_block["image_url"] == f"data:image/png;base64,{expected_b64}"


def test_cache_none_disables_caching(
    file_store: FakeFileStore,
    artifact_store: FakeArtifactStore,
) -> None:
    """
    Passing ``cache=None`` (the default) must still resolve
    correctly — caching is optional.
    """
    item = _make_conversation_item([{"type": "input_image", "file_id": "file_img"}])

    result = resolve_content_references(
        [item],
        file_store,
        artifact_store,  # type: ignore[arg-type]
        # Explicit None — no cache.
        None,
    )

    assert isinstance(result[0].data, MessageData)
    img_block = result[0].data.content[0]
    expected_b64 = base64.b64encode(PNG_BYTES).decode("ascii")
    assert img_block["image_url"] == f"data:image/png;base64,{expected_b64}"


# ── Error handling tests ────────────────────────────────────────────


def test_deleted_file_raises_clear_error(
    file_store: FakeFileStore,
    artifact_store: FakeArtifactStore,
) -> None:
    """
    When a file is deleted between request validation and agent loop
    execution, the error message must clearly indicate the file was
    deleted — not a generic "not found".
    """
    item = _make_conversation_item([{"type": "input_image", "file_id": "file_nonexistent"}])

    with pytest.raises(ValueError, match="no longer exists"):
        resolve_content_references(
            [item],
            file_store,
            artifact_store,  # type: ignore[arg-type]
        )
