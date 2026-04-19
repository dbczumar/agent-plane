"""
Strict JSON-schema normalization for ``@tool``-derived schemas.

Strict-mode schemas have two extra constraints beyond ordinary
JSON Schema:

- Every object schema sets ``additionalProperties: false``.
- Every property in an object is listed in ``required`` (Python
  defaults are still applied on the executor side after the LLM
  emits a value).

This is the form most major LLM providers accept for
function-calling tool schemas without further coercion. Authors
who hit a real schema that strict mode breaks can opt out via
``@tool(strict=False)``.
"""

from __future__ import annotations

from typing import Any


def ensure_strict_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """
    Recursively normalize a JSON schema to strict-mode rules.

    Returns a new dict (does not mutate the input). Recurses into
    ``properties``, ``items``, ``$defs``, and union variants
    (``anyOf`` / ``oneOf`` / ``allOf``).

    :param schema: A JSON schema dict (typically produced by
        :func:`pydantic.BaseModel.model_json_schema`).
    :returns: A new dict with strict-mode constraints applied.
    """
    if not isinstance(schema, dict):
        return schema

    out: dict[str, Any] = dict(schema)

    # Recurse into nested $defs FIRST so the recursion sees the
    # normalized definitions when it walks references.
    if "$defs" in out:
        out["$defs"] = {k: ensure_strict_schema(v) for k, v in out["$defs"].items()}

    # Recurse into union variants.
    for union_key in ("anyOf", "oneOf", "allOf"):
        if union_key in out and isinstance(out[union_key], list):
            out[union_key] = [ensure_strict_schema(variant) for variant in out[union_key]]

    # Recurse into array items.
    if "items" in out:
        out["items"] = ensure_strict_schema(out["items"])

    if out.get("type") == "object":
        properties = out.get("properties", {}) or {}
        out["additionalProperties"] = False
        out["properties"] = {k: ensure_strict_schema(v) for k, v in properties.items()}
        # Strict mode requires every property to be listed as required.
        # The default value (if any) still applies on the executor side
        # via Pydantic — strict mode only governs what the LLM emits.
        if properties:
            out["required"] = list(properties.keys())

    return out
