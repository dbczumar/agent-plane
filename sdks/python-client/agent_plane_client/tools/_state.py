"""Per-agent ToolState for stateful ``@tool`` functions.

See ``designs/TOOL_STATE.md`` for the full design. The primitive is
a simple key-value store, JSON-serialized, scoped to one
(conversation, agent) pair via the storage directory provided by
the framework. The ``@tool`` decorator hides ``ToolState``-typed
parameters from the LLM-facing schema; the subprocess runner
reconstructs a ``ToolState`` from the directory path and injects it
when the tool function is called.

Tool authors see::

    from agent_plane_client import tool, ToolState

    @tool
    def add_task(desc: str, state: ToolState) -> str:
        with state.transaction("queue") as q:
            q = q or []
            q.append({"desc": desc})
            return f"#{len(q) - 1}"

and nothing else.
"""

from __future__ import annotations

import fcntl
import json
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

# Subdirectory segments reserved by the framework — JSON key files
# live at ``{root}/{key}.json``. Keys must not contain path
# separators; we sanitize eagerly rather than allowing the bug to
# surface as directory traversal.
_KEY_SUFFIX = ".json"


class ToolState:
    """Per-agent, per-conversation key-value state for ``@tool`` functions.

    Values are JSON-serialized. The keyspace is shared across every
    tool invoked for the same registered agent within the same
    conversation. Use :meth:`transaction` for atomic read-modify-write;
    plain :meth:`get` and :meth:`set` do not serialize concurrent
    writers on the same key.

    Instances are constructed by the framework. Tool authors receive
    a ``ToolState`` by declaring a parameter of this type on their
    ``@tool``-decorated function; the decorator strips the parameter
    from the LLM-facing schema and the subprocess runner injects the
    live ``ToolState`` at call time.

    :param root: The directory this namespace lives in, e.g.
        ``{workspace}/.tool_state/{agent_id}``. The directory does
        not need to exist yet; it is created lazily on first write.
    """

    def __init__(self, root: Path) -> None:
        self._root = root

    # ── Primary API ──────────────────────────────────────────

    def get(self, key: str, *, default: Any = None) -> Any:
        """Return the stored value at ``key``, or ``default`` if absent.

        :param key: The state key, e.g. ``"queue"``.
        :param default: Value to return when the key has never been
            written. ``None`` by default.
        :returns: The deserialized JSON value, or ``default``.
        """
        path = self._path_for(key)
        if not path.exists():
            return default
        with path.open("r") as f:
            # Shared lock: allow parallel reads, block concurrent writers
            # briefly so we see a complete JSON payload.
            fcntl.flock(f, fcntl.LOCK_SH)
            try:
                return json.loads(f.read() or "null")
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)

    def set(self, key: str, value: Any) -> None:
        """Replace (or create) the value at ``key``. JSON-serialized.

        Non-atomic relative to concurrent writers on the same key —
        use :meth:`transaction` for read-modify-write sequences.

        :param key: The state key.
        :param value: Any JSON-serializable value.
        """
        path = self._path_for(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Write through a temp file + rename so a reader never sees
        # a half-written JSON payload even without a lock.
        tmp = path.with_suffix(_KEY_SUFFIX + ".tmp")
        with tmp.open("w") as f:
            json.dump(value, f)
        tmp.replace(path)

    def delete(self, key: str) -> None:
        """Remove ``key``. No-op if absent.

        :param key: The state key to remove.
        """
        path = self._path_for(key)
        try:
            path.unlink()
        except FileNotFoundError:
            # Idempotent delete — tools commonly don't know whether
            # the key was ever set.
            pass

    def keys(self) -> list[str]:
        """List all keys currently stored in this namespace.

        :returns: Sorted list of keys, e.g. ``["counter", "queue"]``.
            Empty list if nothing has been written yet.
        """
        if not self._root.exists():
            return []
        return sorted(p.stem for p in self._root.iterdir() if p.suffix == _KEY_SUFFIX)

    @contextmanager
    def transaction(self, key: str, *, default: Any = None) -> Iterator[Any]:
        """Atomic read-modify-write for one key.

        Typical usage — supply a ``default`` so first-time callers
        get a usable container without a ``None`` check::

            with state.transaction("queue", default=[]) as queue:
                queue.append(item)
            # queue is written back on normal exit.

        The yielded value is the current contents, or a fresh
        ``default`` if the key was never set. Mutating the yielded
        object in place is the expected pattern — the same object
        is serialized back on exit. Rebinding the local name inside
        the ``with`` block does NOT propagate (Python closures), so
        for "replace the value" semantics use :meth:`set` explicitly.

        On a normal exit the yielded object is JSON-serialized and
        written back. On exception no write happens — the prior
        value is preserved.

        :param key: The state key to lock + read + write.
        :param default: Value yielded when the key has never been
            written. Defaults to ``None``. Pass ``[]`` or ``{}``
            (or any JSON-serializable value) to skip the absent-key
            branch in caller code.
        :yields: The current value at ``key``, or ``default`` if
            the key has no stored value yet. Mutate in place.
        """
        path = self._path_for(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        # ``a+`` creates the file if missing and positions at end;
        # we seek to 0 before read. Opening with ``r+`` would fail
        # when the file doesn't exist yet, which is a common first-
        # call case for a tool.
        with path.open("a+") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            try:
                value = _read_transaction_value(f, default)
                yield value
                _write_transaction_value(f, value)
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)

    # ── Internals ────────────────────────────────────────────

    def _path_for(self, key: str) -> Path:
        """Resolve ``key`` to the on-disk path, rejecting traversal.

        :param key: Caller-supplied key.
        :returns: The ``{root}/{key}.json`` path.
        :raises ValueError: If ``key`` is empty, contains a path
            separator, or starts with a dot (no hidden or
            escaped paths).
        """
        if not key:
            raise ValueError("ToolState key must be a non-empty string")
        if "/" in key or "\\" in key or key.startswith("."):
            # Rejects traversal and hidden-file sigils. Authors who
            # really need slashes can encode them (e.g. "a__b") —
            # we'd rather break loudly than accept quiet bugs.
            raise ValueError(
                f"ToolState key {key!r} contains an illegal character. "
                f"Keys must be plain names (no '/', '\\', or leading '.')."
            )
        return self._root / f"{key}{_KEY_SUFFIX}"


def _read_transaction_value(f: Any, default: Any) -> Any:
    """Seek to 0 and decode the JSON value under the open file handle.

    Returns ``default`` when the file is empty (first-time use of
    the key). Factored out of :meth:`ToolState.transaction` so the
    context manager stays short.

    :param f: Open file handle positioned anywhere; will be seek(0)ed.
    :param default: Value to return on empty/whitespace content.
    :returns: Decoded JSON value or ``default``.
    """
    f.seek(0)
    raw = f.read()
    if raw.strip():
        return json.loads(raw)
    return default


def _write_transaction_value(f: Any, value: Any) -> None:
    """Truncate and re-serialize ``value`` as JSON under the file handle.

    Caller must hold the exclusive flock before calling. Factored
    out of :meth:`ToolState.transaction` so the context manager
    stays short.

    :param f: Open file handle (must support ``r+``-style truncate).
    :param value: Any JSON-serializable value to persist.
    """
    f.seek(0)
    f.truncate()
    json.dump(value, f)
