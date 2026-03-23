"""Local filesystem implementation of ArtifactStore."""

from __future__ import annotations

from pathlib import Path, PurePosixPath, PureWindowsPath

from agent_plane.stores.artifact_store import ArtifactStore


class LocalArtifactStore(ArtifactStore):
    """
    Stores binary blobs as flat files under a local directory.

    The ``storage_location`` is a filesystem path used as the root
    directory.  Layout::

        storage_location/
            <key1>
            nested/key2
            ...

    Keys use forward slashes as separators and are mapped to the
    native OS path on disk.  Traversal sequences (``..``) and
    backslashes are rejected; a post-resolution containment check
    ensures the resolved path stays within the root even if symlinks
    are involved.
    """

    def __init__(self, storage_location: str) -> None:
        super().__init__(storage_location)
        self._root = Path(storage_location)
        self._root.mkdir(parents=True, exist_ok=True)

    def _resolve(self, key: str) -> Path:
        """
        Map *key* (forward-slash separated) to an absolute filesystem
        path.  Raises ``ValueError`` if the key is empty, contains
        traversal sequences, or resolves outside the root.
        """
        parts = PurePosixPath(key).parts
        if (
            not parts
            or ".." in parts
            or "\\" in key
            or PurePosixPath(key).is_absolute()
            or PureWindowsPath(key).is_absolute()
        ):
            raise ValueError(f"invalid artifact key: {key!r}")

        # Join validated parts with OS-native separator
        resolved = (self._root / Path(*parts)).resolve()
        if not resolved.is_relative_to(self._root.resolve()):
            raise ValueError(f"artifact key escapes root directory: {key!r}")
        return resolved

    # ── ArtifactStore interface ──────────────────────────────

    def put(self, key: str, data: bytes) -> None:
        path = self._resolve(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    def get(self, key: str) -> bytes:
        path = self._resolve(key)
        if not path.exists():
            raise KeyError(key)
        return path.read_bytes()

    def delete(self, key: str) -> None:
        path = self._resolve(key)
        if path.exists():
            path.unlink()

    def exists(self, key: str) -> bool:
        return self._resolve(key).exists()
