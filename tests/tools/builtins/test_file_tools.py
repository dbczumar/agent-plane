"""Unit tests for list_files and download_file builtin tools."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from agent_plane.tools.base import ToolContext
from agent_plane.tools.builtins.download_file import DownloadFileTool
from agent_plane.tools.builtins.list_files import ListFilesTool

# ── Stubs ─────────────────────────────────────────────────


@dataclass
class _FakeFile:
    """
    Minimal stub for StoredFile.

    :param id: File ID.
    :param filename: Original filename.
    :param bytes: File size.
    :param content_type: MIME type.
    :param created_at: Unix timestamp.
    """

    id: str
    filename: str
    bytes: int
    content_type: str | None
    created_at: int


@dataclass
class _FakePage:
    """
    Minimal stub for PagedList.

    :param data: List of items.
    :param has_more: Whether there are more pages.
    :param first_id: First item ID.
    :param last_id: Last item ID.
    """

    data: list[Any]
    has_more: bool = False
    first_id: str | None = None
    last_id: str | None = None


class _FakeFileStore:
    """
    Stub file store for testing.

    :param files: Pre-populated file records.
    """

    def __init__(self, files: list[_FakeFile] | None = None) -> None:
        self._files = {f.id: f for f in (files or [])}

    def list(
        self,
        limit: int = 20,
        after: str | None = None,
        before: str | None = None,
        order: str = "desc",
    ) -> _FakePage:
        """
        Return all files as a single page.

        :param limit: Max results.
        :param after: Ignored in stub.
        :param before: Ignored in stub.
        :param order: Ignored in stub.
        :returns: A page of files.
        """
        data = list(self._files.values())[:limit]
        return _FakePage(data=data)

    def get(self, file_id: str) -> _FakeFile | None:
        """
        Look up a file by ID.

        :param file_id: The file ID.
        :returns: The file record, or None.
        """
        return self._files.get(file_id)


class _FakeArtifactStore:
    """
    Stub artifact store for testing.

    :param blobs: Pre-populated key → bytes mapping.
    """

    def __init__(self, blobs: dict[str, bytes] | None = None) -> None:
        self._blobs = dict(blobs or {})

    def get(self, key: str) -> bytes:
        """
        Retrieve blob by key.

        :param key: Artifact key.
        :returns: The blob bytes.
        :raises KeyError: If not found.
        """
        if key not in self._blobs:
            raise KeyError(key)
        return self._blobs[key]


@pytest.fixture()
def tool_ctx(tmp_path: Path) -> ToolContext:
    """
    ToolContext with a temporary workspace.

    :param tmp_path: Pytest temp directory.
    :returns: A ToolContext with workspace set.
    """
    return ToolContext(
        task_id="task_test",
        agent_id="agent_test",
        workspace=tmp_path,
    )


# ── list_files tests ─────────────────────────────────────


def test_list_files_returns_metadata(
    monkeypatch: pytest.MonkeyPatch,
    tool_ctx: ToolContext,
) -> None:
    """
    list_files returns file metadata for all stored files.

    :param monkeypatch: Pytest monkeypatch fixture.
    :param tool_ctx: Tool execution context.
    """
    files = [
        _FakeFile("file_1", "report.pdf", 1024, "application/pdf", 1000),
        _FakeFile("file_2", "chart.png", 2048, "image/png", 2000),
    ]
    monkeypatch.setattr(
        "agent_plane.runtime.get_file_store",
        lambda: _FakeFileStore(files),
    )

    tool = ListFilesTool()
    result = json.loads(tool.invoke("{}", tool_ctx))

    assert len(result["files"]) == 2
    assert result["files"][0]["file_id"] == "file_1"
    assert result["files"][0]["filename"] == "report.pdf"
    assert result["files"][0]["bytes"] == 1024
    assert result["files"][0]["content_type"] == "application/pdf"
    assert result["files"][1]["file_id"] == "file_2"


def test_list_files_empty(
    monkeypatch: pytest.MonkeyPatch,
    tool_ctx: ToolContext,
) -> None:
    """
    list_files returns empty list when no files exist.

    :param monkeypatch: Pytest monkeypatch fixture.
    :param tool_ctx: Tool execution context.
    """
    monkeypatch.setattr(
        "agent_plane.runtime.get_file_store",
        lambda: _FakeFileStore([]),
    )

    tool = ListFilesTool()
    result = json.loads(tool.invoke("{}", tool_ctx))

    assert result["files"] == []


def test_list_files_respects_limit(
    monkeypatch: pytest.MonkeyPatch,
    tool_ctx: ToolContext,
) -> None:
    """
    list_files caps at the requested limit.

    :param monkeypatch: Pytest monkeypatch fixture.
    :param tool_ctx: Tool execution context.
    """
    files = [_FakeFile(f"file_{i}", f"f{i}.txt", 100, None, i) for i in range(50)]
    monkeypatch.setattr(
        "agent_plane.runtime.get_file_store",
        lambda: _FakeFileStore(files),
    )

    tool = ListFilesTool()
    result = json.loads(tool.invoke('{"limit": 5}', tool_ctx))

    assert len(result["files"]) == 5


# ── download_file tests ──────────────────────────────────


def test_download_file_saves_to_workspace(
    monkeypatch: pytest.MonkeyPatch,
    tool_ctx: ToolContext,
) -> None:
    """
    download_file retrieves content and writes it to the workspace.

    :param monkeypatch: Pytest monkeypatch fixture.
    :param tool_ctx: Tool execution context.
    """
    content = b"hello world"
    monkeypatch.setattr(
        "agent_plane.runtime.get_file_store",
        lambda: _FakeFileStore(
            [
                _FakeFile("file_abc", "hello.txt", len(content), "text/plain", 1000),
            ]
        ),
    )
    monkeypatch.setattr(
        "agent_plane.runtime.get_artifact_store",
        lambda: _FakeArtifactStore({"file_abc": content}),
    )

    tool = DownloadFileTool()
    result = json.loads(tool.invoke('{"file_id": "file_abc"}', tool_ctx))

    assert result["filename"] == "hello.txt"
    assert result["bytes"] == 11
    assert result["content_type"] == "text/plain"

    saved = Path(result["path"])
    assert saved.exists()
    assert saved.read_bytes() == content
    assert saved.name == "hello.txt"


def test_download_file_custom_destination(
    monkeypatch: pytest.MonkeyPatch,
    tool_ctx: ToolContext,
) -> None:
    """
    download_file saves to a custom path within the workspace.

    :param monkeypatch: Pytest monkeypatch fixture.
    :param tool_ctx: Tool execution context.
    """
    content = b"data"
    monkeypatch.setattr(
        "agent_plane.runtime.get_file_store",
        lambda: _FakeFileStore(
            [
                _FakeFile("file_xyz", "data.csv", len(content), "text/csv", 1000),
            ]
        ),
    )
    monkeypatch.setattr(
        "agent_plane.runtime.get_artifact_store",
        lambda: _FakeArtifactStore({"file_xyz": content}),
    )

    tool = DownloadFileTool()
    result = json.loads(
        tool.invoke('{"file_id": "file_xyz", "destination": "output/saved.csv"}', tool_ctx)
    )

    saved = Path(result["path"])
    assert saved.exists()
    assert saved.name == "saved.csv"
    assert "output" in str(saved)


def test_download_file_not_found(
    monkeypatch: pytest.MonkeyPatch,
    tool_ctx: ToolContext,
) -> None:
    """
    download_file returns error for unknown file_id.

    :param monkeypatch: Pytest monkeypatch fixture.
    :param tool_ctx: Tool execution context.
    """
    monkeypatch.setattr(
        "agent_plane.runtime.get_file_store",
        lambda: _FakeFileStore([]),
    )
    monkeypatch.setattr(
        "agent_plane.runtime.get_artifact_store",
        lambda: _FakeArtifactStore({}),
    )

    tool = DownloadFileTool()
    result = json.loads(tool.invoke('{"file_id": "file_nope"}', tool_ctx))

    assert "error" in result
    assert "not found" in result["error"].lower()


def test_download_file_missing_content(
    monkeypatch: pytest.MonkeyPatch,
    tool_ctx: ToolContext,
) -> None:
    """
    download_file returns error when metadata exists but content is missing.

    :param monkeypatch: Pytest monkeypatch fixture.
    :param tool_ctx: Tool execution context.
    """
    monkeypatch.setattr(
        "agent_plane.runtime.get_file_store",
        lambda: _FakeFileStore(
            [
                _FakeFile("file_orphan", "ghost.bin", 100, None, 1000),
            ]
        ),
    )
    monkeypatch.setattr(
        "agent_plane.runtime.get_artifact_store",
        lambda: _FakeArtifactStore({}),
    )

    tool = DownloadFileTool()
    result = json.loads(tool.invoke('{"file_id": "file_orphan"}', tool_ctx))

    assert "error" in result
    assert "content" in result["error"].lower()
