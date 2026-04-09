"""E2E test: archer agent generates a chart and uploads it.

Verifies the full output attachment pipeline: agent uses
``code_sandbox`` to generate a matplotlib chart, calls
``upload_file`` to store it, the response includes a
``file_citation`` annotation, and the file is downloadable
via ``GET /v1/files/{file_id}/content``.

Usage::

    pytest tests/e2e/test_archer_output_files.py \
        --llm-api-key $(cat /tmp/mykey) -v
"""

from __future__ import annotations

from typing import Any

import httpx

from tests.e2e.conftest import poll_until_terminal


def _extract_file_annotations(body: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Extract all ``file_citation`` annotations from assistant messages.

    :param body: The terminal response body.
    :returns: List of file_citation annotation dicts.
    """
    annotations: list[dict[str, Any]] = []
    for item in body.get("output", []):
        if item.get("type") != "message" or item.get("role") != "assistant":
            continue
        for block in item.get("content", []):
            for ann in block.get("annotations", []):
                if ann.get("type") == "file_citation":
                    annotations.append(ann)
    return annotations


def test_archer_generates_chart_and_uploads(
    http_client: httpx.Client,
    archer_agent: str,
) -> None:
    """
    The archer agent generates a matplotlib chart via
    ``code_sandbox``, uploads it via ``upload_file``, and the
    response includes a ``file_citation`` annotation. The file
    is downloadable via the files API.

    :param http_client: HTTP client pointed at the live e2e server.
    :param archer_agent: The uploaded archer agent name.
    """
    resp = http_client.post(
        "/v1/responses",
        json={
            "model": archer_agent,
            "input": (
                "Do these two steps:\n"
                '1. code_sandbox: python -c "'
                "import matplotlib;matplotlib.use('Agg');"
                "import matplotlib.pyplot as plt;"
                "plt.plot([1,2,3],[1,4,9]);"
                "plt.savefig('chart.png');"
                "print('done')\"\n"
                "2. upload_file: path=chart.png\n"
                "Reply with the file_id."
            ),
            "background": True,
        },
    )
    resp.raise_for_status()
    response_id = resp.json()["id"]

    body = poll_until_terminal(http_client, response_id, timeout=300)

    assert body["status"] == "completed", (
        f"Expected completed, got {body['status']}. Error: {body.get('error')}"
    )

    # Check for file_citation annotation on the assistant message.
    annotations = _extract_file_annotations(body)
    assert len(annotations) >= 1, (
        f"Expected at least 1 file_citation annotation. "
        f"Got {len(annotations)}. The agent may not have called "
        f"upload_file or the annotation pipeline is broken. "
        f"Output types: "
        f"{[it.get('type') for it in body.get('output', [])]}"
    )

    ann = annotations[0]
    file_id = ann.get("file_id")
    assert file_id, f"file_citation annotation missing file_id: {ann}"
    assert ann.get("filename"), f"file_citation annotation missing filename: {ann}"

    # Download the file via the files API.
    download = http_client.get(f"/v1/files/{file_id}/content")
    assert download.status_code == 200, (
        f"File download failed: {download.status_code}. "
        f"file_id={file_id} may not exist in the file store."
    )

    # The file should be a PNG image (matplotlib default).
    content = download.content
    assert len(content) > 100, (
        f"Downloaded file is too small ({len(content)} bytes) to be a valid PNG image."
    )
    # PNG magic bytes: \x89PNG\r\n\x1a\n
    assert content[:4] == b"\x89PNG", (
        f"Expected PNG file (magic bytes \\x89PNG), got: {content[:8]!r}"
    )
