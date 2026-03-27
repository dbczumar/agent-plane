"""Shared helpers for translating multimodal content across adapters.

Provides data-URI parsing used by Anthropic, Gemini, and Bedrock
adapters when converting Chat Completions ``image_url`` parts to
their provider-native image formats.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DataUriParts:
    """
    Parsed components of a ``data:`` URI.

    :param media_type: MIME type, e.g. ``"image/png"``.
    :param data: Base64-encoded payload.
    """

    media_type: str
    data: str


def parse_data_uri(uri: str) -> DataUriParts | None:
    """
    Parse a ``data:`` URI into its media type and base64 payload.

    Returns ``None`` for non-data URIs (e.g. ``https://...``)
    so callers can distinguish between inline content and external
    URLs that must be passed through to the provider.

    :param uri: A URI string, e.g.
        ``"data:image/png;base64,iVBOR..."`` or
        ``"https://example.com/img.png"``.
    :returns: Parsed :class:`DataUriParts`, or ``None`` if the
        URI is not a ``data:`` URI.
    """
    if not uri.startswith("data:"):
        return None

    # Format: data:<media_type>;base64,<data>
    rest = uri[len("data:") :]
    separator = ";base64,"
    sep_idx = rest.find(separator)
    if sep_idx == -1:
        return None

    media_type = rest[:sep_idx]
    data = rest[sep_idx + len(separator) :]
    return DataUriParts(media_type=media_type, data=data)
