"""Routes for the /v1/files endpoints."""

import mimetypes
import time
import uuid

from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from starlette.responses import Response

from agent_plane.stores import ArtifactStore
from agent_plane.server.models import FileDeleted, FileObject, PaginatedList


def create_files_router(artifact_store: ArtifactStore) -> APIRouter:
    """
    Factory that builds the files router. File metadata is in-memory
    (a simple dict inside this closure) as a placeholder until real
    metadata storage is implemented. Binary content is stored via
    the injected ArtifactStore.
    """
    router = APIRouter()

    # In-memory storage: metadata keyed by file id.
    files_by_id: dict[str, FileObject] = {}

    # ── POST /files ──────────────────────────────────────────────

    @router.post("/files", status_code=201)
    async def upload_file(
        file: UploadFile = File(...),
    ) -> FileObject:
        content = await file.read()
        file_id = f"file_{uuid.uuid4().hex[:12]}"

        file_obj = FileObject(
            id=file_id,
            filename=file.filename or "unknown",
            bytes=len(content),
            created_at=int(time.time()),
        )

        files_by_id[file_id] = file_obj
        artifact_store.put(file_id, content)

        return file_obj

    # ── GET /files ───────────────────────────────────────────────

    @router.get("/files")
    async def list_files(
        limit: int = Query(default=20, ge=1, le=100),
        after: str | None = Query(default=None),
        before: str | None = Query(default=None),
        order: str = Query(default="desc", pattern="^(asc|desc)$"),
    ) -> PaginatedList:
        sorted_files = sorted(
            files_by_id.values(),
            key=lambda f: f.created_at,
            reverse=(order == "desc"),
        )

        # Apply cursor-based pagination.
        if after is not None:
            idx = next(
                (
                    i
                    for i, f in enumerate(sorted_files)
                    if f.id == after
                ),
                None,
            )
            if idx is not None:
                sorted_files = sorted_files[idx + 1 :]

        if before is not None:
            idx = next(
                (
                    i
                    for i, f in enumerate(sorted_files)
                    if f.id == before
                ),
                None,
            )
            if idx is not None:
                sorted_files = sorted_files[:idx]

        has_more = len(sorted_files) > limit
        page = sorted_files[:limit]

        return PaginatedList(
            data=page,
            first_id=page[0].id if page else None,
            last_id=page[-1].id if page else None,
            has_more=has_more,
        )

    # ── GET /files/{file_id} ─────────────────────────────────────

    @router.get("/files/{file_id}")
    async def get_file(file_id: str) -> FileObject:
        file_obj = files_by_id.get(file_id)
        if file_obj is None:
            raise HTTPException(
                status_code=404, detail="File not found"
            )
        return file_obj

    # ── DELETE /files/{file_id} ──────────────────────────────────

    @router.delete("/files/{file_id}")
    async def delete_file(file_id: str) -> FileDeleted:
        file_obj = files_by_id.pop(file_id, None)
        if file_obj is None:
            raise HTTPException(
                status_code=404, detail="File not found"
            )
        artifact_store.delete(file_id)
        return FileDeleted(id=file_id)

    # ── GET /files/{file_id}/content ─────────────────────────────

    @router.get("/files/{file_id}/content")
    async def get_file_content(file_id: str) -> Response:
        file_obj = files_by_id.get(file_id)
        if file_obj is None:
            raise HTTPException(
                status_code=404, detail="File not found"
            )

        content = artifact_store.get(file_id)
        media_type = (
            mimetypes.guess_type(file_obj.filename)[0]
            or "application/octet-stream"
        )

        return Response(content=content, media_type=media_type)

    return router
