"""Integration tests for /v1/files endpoints."""

from __future__ import annotations

import httpx
import pytest

pytestmark = pytest.mark.asyncio


async def test_upload_file(client: httpx.AsyncClient) -> None:
    resp = await client.post(
        "/v1/files",
        files={"file": ("hello.txt", b"hello world", "text/plain")},
    )
    assert resp.status_code == 201
    body = resp.json()
    assert body["object"] == "file"
    assert body["filename"] == "hello.txt"
    assert body["bytes"] == 11
    assert isinstance(body["id"], str)
    assert isinstance(body["created_at"], int)


async def test_list_files_empty(client: httpx.AsyncClient) -> None:
    resp = await client.get("/v1/files")
    assert resp.status_code == 200
    body = resp.json()
    assert body["object"] == "list"
    assert body["data"] == []
    assert body["has_more"] is False


async def test_list_files(client: httpx.AsyncClient) -> None:
    await client.post(
        "/v1/files",
        files={"file": ("a.txt", b"aaa", "text/plain")},
    )
    await client.post(
        "/v1/files",
        files={"file": ("b.txt", b"bbb", "text/plain")},
    )

    resp = await client.get("/v1/files")
    body = resp.json()
    assert len(body["data"]) == 2
    filenames = {f["filename"] for f in body["data"]}
    assert filenames == {"a.txt", "b.txt"}
    # Verify PaginatedList structure
    assert isinstance(body["first_id"], str)
    assert isinstance(body["last_id"], str)
    assert body["has_more"] is False


async def test_get_file(client: httpx.AsyncClient) -> None:
    create_resp = await client.post(
        "/v1/files",
        files={"file": ("doc.pdf", b"pdf-content", "application/pdf")},
    )
    file_id = create_resp.json()["id"]

    resp = await client.get(f"/v1/files/{file_id}")
    assert resp.status_code == 200
    body = resp.json()
    assert body["id"] == file_id
    assert body["filename"] == "doc.pdf"


async def test_get_file_not_found(client: httpx.AsyncClient) -> None:
    resp = await client.get("/v1/files/nonexistent")
    assert resp.status_code == 404
    assert isinstance(resp.json()["detail"], str)


async def test_get_file_content(client: httpx.AsyncClient) -> None:
    content = b"binary content here"
    create_resp = await client.post(
        "/v1/files",
        files={"file": ("data.bin", content, "application/octet-stream")},
    )
    file_id = create_resp.json()["id"]

    resp = await client.get(f"/v1/files/{file_id}/content")
    assert resp.status_code == 200
    assert resp.content == content


async def test_delete_file(client: httpx.AsyncClient) -> None:
    create_resp = await client.post(
        "/v1/files",
        files={"file": ("del.txt", b"delete me", "text/plain")},
    )
    file_id = create_resp.json()["id"]

    del_resp = await client.delete(f"/v1/files/{file_id}")
    assert del_resp.status_code == 200
    body = del_resp.json()
    assert body["id"] == file_id
    assert body["object"] == "file"
    assert body["deleted"] is True

    # Confirm it's gone
    get_resp = await client.get(f"/v1/files/{file_id}")
    assert get_resp.status_code == 404


async def test_delete_file_not_found(client: httpx.AsyncClient) -> None:
    resp = await client.delete("/v1/files/nonexistent")
    assert resp.status_code == 404
    assert isinstance(resp.json()["detail"], str)


async def test_list_files_pagination(client: httpx.AsyncClient) -> None:
    for i in range(3):
        await client.post(
            "/v1/files",
            files={"file": (f"file{i}.txt", f"content{i}".encode(), "text/plain")},
        )

    # Fetch first page of 2
    resp = await client.get("/v1/files", params={"limit": 2})
    body = resp.json()
    assert len(body["data"]) == 2
    assert body["has_more"] is True

    # Fetch next page using last_id as cursor
    resp2 = await client.get("/v1/files", params={"limit": 2, "after": body["last_id"]})
    body2 = resp2.json()
    assert len(body2["data"]) == 1
    assert body2["has_more"] is False
