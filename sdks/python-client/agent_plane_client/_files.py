"""Files namespace — upload, list, get, delete."""

from __future__ import annotations

import mimetypes
import pathlib

import httpx

from ._errors import raise_for_status
from ._types import File, PaginatedList


class FilesNamespace:
    """Methods for ``/v1/files`` endpoints."""

    def __init__(self, http: httpx.AsyncClient, base_url: str) -> None:
        self._http = http
        self._base = base_url

    async def upload(self, path: str) -> File:
        """Upload a local file.

        :param path: Path to the file on disk.
        :returns: The uploaded file metadata.
        """
        p = pathlib.Path(path)
        content_type = mimetypes.guess_type(str(p))[0]
        with open(p, "rb") as f:
            resp = await self._http.post(
                f"{self._base}/v1/files",
                files={"file": (p.name, f, content_type)},
                timeout=30.0,
            )
        data = resp.json() if resp.status_code < 500 else resp.text
        raise_for_status(resp.status_code, data)
        return File.from_dict(resp.json())

    async def list(
        self,
        *,
        limit: int = 20,
        after: str | None = None,
        order: str = "desc",
    ) -> list[File]:
        """List uploaded files.

        :param limit: Max files to return.
        :param after: Cursor for pagination.
        :param order: Sort order.
        :returns: List of files.
        """
        params: dict[str, object] = {"limit": limit, "order": order}
        if after is not None:
            params["after"] = after
        resp = await self._http.get(f"{self._base}/v1/files", params=params)
        data = resp.json() if resp.status_code < 500 else resp.text
        raise_for_status(resp.status_code, data)
        page = PaginatedList.from_dict(resp.json())
        return [File.from_dict(d) for d in page.data]

    async def get(self, file_id: str) -> File:
        """Get file metadata by ID.

        :param file_id: The file ID.
        :returns: File metadata.
        """
        resp = await self._http.get(f"{self._base}/v1/files/{file_id}")
        data = resp.json() if resp.status_code < 500 else resp.text
        raise_for_status(resp.status_code, data)
        return File.from_dict(resp.json())

    async def get_content(self, file_id: str) -> bytes:
        """Download file content.

        :param file_id: The file ID.
        :returns: Raw file bytes.
        """
        resp = await self._http.get(
            f"{self._base}/v1/files/{file_id}/content",
            timeout=30.0,
        )
        if resp.status_code >= 400:
            raise_for_status(resp.status_code, resp.text)
        return resp.content

    async def download(self, file_id: str, to_path: str | pathlib.Path) -> pathlib.Path:
        """Download file content and write it to disk.

        Convenience wrapper over :meth:`get_content`. Creates any
        missing parent directories and returns the :class:`~pathlib.Path`
        that was written.

        :param file_id: The file ID, e.g. ``"file_abc123"``.
        :param to_path: Local path to write to, e.g. ``"./out/chart.png"``
            or ``pathlib.Path("/tmp/chart.png")``. Parent directories
            are created if they don't exist.
        :returns: The resolved path the bytes were written to.
        """
        content = await self.get_content(file_id)
        path = pathlib.Path(to_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return path

    async def delete(self, file_id: str) -> None:
        """Delete a file.

        :param file_id: The file ID.
        """
        resp = await self._http.delete(f"{self._base}/v1/files/{file_id}")
        if resp.status_code >= 400:
            data = resp.json() if resp.status_code < 500 else resp.text
            raise_for_status(resp.status_code, data)
