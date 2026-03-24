"""Routes for the /v1/files endpoints."""

import mimetypes

from fastapi import APIRouter, File, Query, UploadFile
from starlette.responses import Response

from agent_plane.entities import StoredFile
from agent_plane.errors import AgentPlaneError, ErrorCode
from agent_plane.server.schemas import FileDeleted, FileObject, PaginatedList
from agent_plane.stores import ArtifactStore, FileStore


def _to_file_object(f: StoredFile) -> FileObject:
    return FileObject(
        id=f.id,
        filename=f.filename,
        bytes=f.bytes,
        created_at=f.created_at,
    )


def create_files_router(
    file_store: FileStore,
    artifact_store: ArtifactStore,
) -> APIRouter:
    router = APIRouter()

    # ── POST /files ──────────────────────────────────────────────

    @router.post("/files", status_code=201)
    async def upload_file(
        file: UploadFile = File(...),
    ) -> FileObject:
        if not file.filename:
            raise AgentPlaneError("filename is required", code=ErrorCode.INVALID_INPUT)
        content = await file.read()
        content_type = mimetypes.guess_type(file.filename)[0]
        stored = file_store.create(
            filename=file.filename,
            bytes=len(content),
            content_type=content_type,
        )
        artifact_store.put(stored.id, content)

        return _to_file_object(stored)

    # ── GET /files ───────────────────────────────────────────────

    @router.get("/files")
    async def list_files(
        limit: int = Query(default=20, ge=1, le=100),
        after: str | None = Query(default=None),
        before: str | None = Query(default=None),
        order: str = Query(default="desc", pattern="^(asc|desc)$"),
    ) -> PaginatedList:
        page = file_store.list(limit=limit, after=after, before=before, order=order)
        data = [_to_file_object(f) for f in page.data]
        return PaginatedList(
            data=data,
            first_id=page.first_id,
            last_id=page.last_id,
            has_more=page.has_more,
        )

    # ── GET /files/{file_id} ─────────────────────────────────────

    @router.get("/files/{file_id}")
    async def get_file(file_id: str) -> FileObject:
        stored = file_store.get(file_id)
        if stored is None:
            raise AgentPlaneError("File not found", code=ErrorCode.NOT_FOUND)
        return _to_file_object(stored)

    # ── DELETE /files/{file_id} ──────────────────────────────────

    @router.delete("/files/{file_id}")
    async def delete_file(file_id: str) -> FileDeleted:
        if not file_store.delete(file_id):
            raise AgentPlaneError("File not found", code=ErrorCode.NOT_FOUND)
        artifact_store.delete(file_id)
        return FileDeleted(id=file_id)

    # ── GET /files/{file_id}/content ─────────────────────────────

    @router.get("/files/{file_id}/content")
    async def get_file_content(file_id: str) -> Response:
        stored = file_store.get(file_id)
        if stored is None:
            raise AgentPlaneError("File not found", code=ErrorCode.NOT_FOUND)

        content = artifact_store.get(stored.id)
        media_type = mimetypes.guess_type(stored.filename)[0] or "application/octet-stream"

        return Response(content=content, media_type=media_type)

    return router
