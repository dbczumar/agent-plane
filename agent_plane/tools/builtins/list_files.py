"""Built-in tool: list uploaded files."""

from __future__ import annotations

import json
from typing import Any

from agent_plane.tools.base import Tool, ToolContext


class ListFilesTool(Tool):
    """
    List files stored in the file store.

    Returns metadata for uploaded files — ID, filename, size,
    and creation timestamp. Supports pagination via ``limit``
    and ``after`` cursor.
    """

    @classmethod
    def name(cls) -> str:
        """
        :returns: ``"list_files"``.
        """
        return "list_files"

    def get_schema(self) -> dict[str, Any]:
        """
        Return the OpenAI-format tool schema.

        :returns: A tool schema dict.
        """
        return {
            "type": "function",
            "function": {
                "name": "list_files",
                "description": (
                    "List files that have been uploaded or created. "
                    "Returns file metadata: ID, filename, size, and "
                    "creation time. Use the file_id to download content "
                    "with download_file."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "limit": {
                            "type": "integer",
                            "description": (
                                "Maximum number of files to return. Default 20, max 100."
                            ),
                        },
                        "after": {
                            "type": "string",
                            "description": (
                                "Cursor for pagination. Return files after this file ID."
                            ),
                        },
                    },
                    "required": [],
                },
            },
        }

    def invoke(self, arguments: str, ctx: ToolContext) -> str:
        """
        List files from the file store.

        :param arguments: JSON with optional ``"limit"`` and
            ``"after"`` keys.
        :param ctx: Server-side execution context (unused).
        :returns: JSON string with file list and pagination info.
        """
        args: dict[str, Any] = json.loads(arguments)
        limit = min(args.get("limit", 20), 100)
        after = args.get("after")

        from agent_plane.runtime import get_file_store

        file_store = get_file_store()
        if file_store is None:
            return json.dumps({"error": "File store not configured."})

        page = file_store.list(
            limit=limit,
            after=after,
            before=None,
            order="desc",
        )

        files = [
            {
                "file_id": f.id,
                "filename": f.filename,
                "bytes": f.bytes,
                "content_type": f.content_type,
                "created_at": f.created_at,
            }
            for f in page.data
        ]

        result: dict[str, Any] = {"files": files}
        if page.has_more:
            result["has_more"] = True
            result["last_id"] = page.last_id
        return json.dumps(result)
