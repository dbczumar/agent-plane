"""Safe tarball extraction for agent image bundles."""

from __future__ import annotations

import io
import tarfile
from pathlib import Path, PurePosixPath

# 500 MB default limit to guard against decompression bombs
DEFAULT_MAX_BYTES = 500 * 1024 * 1024

# Maximum number of entries to prevent zip-bomb style attacks
DEFAULT_MAX_ENTRIES = 10_000


class ExtractionError(Exception):
    """
    Raised when a tarball fails safety checks during extraction.

    Safety violations include path traversal, symlinks/hardlinks,
    decompression bombs, and entry count bombs.
    """


def extract_safe(
    source: Path | bytes,
    dest: Path,
    *,
    max_bytes: int = DEFAULT_MAX_BYTES,
    max_entries: int = DEFAULT_MAX_ENTRIES,
) -> Path:
    """
    Extract a tarball to *dest* with safety checks.

    Rejects:

    - Path traversal (members outside *dest* via ``..`` or
      absolute paths)
    - Symlinks and hardlinks (could escape *dest*)
    - Decompression bombs (total extracted bytes > *max_bytes*)
    - Entry count bombs (more than *max_entries* files)

    :param source: Path to a ``.tar.gz`` / ``.tar`` file, or raw
        tarball bytes (e.g. from an HTTP upload).
    :param dest: Directory to extract into (created if needed).
    :param max_bytes: Maximum total extracted size in bytes.
        Defaults to 500 MB.
    :param max_entries: Maximum number of tar entries. Defaults
        to 10,000.
    :returns: The *dest* path for convenience.
    :raises ExtractionError: If any safety check fails.
    :raises FileNotFoundError: If *source* is a :class:`Path` and
        does not exist.
    """
    if isinstance(source, Path):
        if not source.exists():
            raise FileNotFoundError(f"tarball not found: {source}")

    dest.mkdir(parents=True, exist_ok=True)
    resolved_dest = dest.resolve()

    total_bytes = 0
    entry_count = 0

    with _open_tar(source) as tf:
        for member in tf:
            entry_count += 1
            if entry_count > max_entries:
                raise ExtractionError(f"tarball exceeds max entry count ({max_entries})")

            _check_member_safety(member, resolved_dest)

            if member.isfile():
                total_bytes += member.size
                if total_bytes > max_bytes:
                    raise ExtractionError(
                        f"tarball exceeds max extracted size ({max_bytes} bytes)"
                    )

            tf.extract(member, dest, set_attrs=False)

    return dest


def _open_tar(source: Path | bytes) -> tarfile.TarFile:
    """
    Open a tarball from a file path or raw bytes.

    :param source: Path to a ``.tar.gz`` / ``.tar`` file, or raw
        tarball bytes.
    :returns: An opened :class:`tarfile.TarFile` ready for
        iteration.
    :raises ExtractionError: If the data is not a valid tarball.
    """
    try:
        if isinstance(source, bytes):
            return tarfile.open(fileobj=io.BytesIO(source), mode="r:*")
        return tarfile.open(source, "r:*")
    except (tarfile.ReadError, tarfile.CompressionError) as exc:
        raise ExtractionError(f"invalid tarball: {exc}") from exc


def _check_member_safety(member: tarfile.TarInfo, resolved_dest: Path) -> None:
    """
    Validate a single tar member against safety rules.

    :param member: The :class:`tarfile.TarInfo` entry to validate.
    :param resolved_dest: The resolved (absolute) extraction
        destination directory, used for containment checks.
    :raises ExtractionError: If the member is a symlink/hardlink,
        uses an absolute path, contains ``..`` traversal, or
        resolves outside *resolved_dest*.
    """
    # Reject symlinks and hardlinks
    if member.issym() or member.islnk():
        raise ExtractionError(f"tarball contains a link (not allowed): {member.name!r}")

    # Reject absolute paths
    if PurePosixPath(member.name).is_absolute():
        raise ExtractionError(f"tarball contains an absolute path: {member.name!r}")

    # Reject path traversal via .. segments
    parts = PurePosixPath(member.name).parts
    if ".." in parts:
        raise ExtractionError(f"tarball contains path traversal: {member.name!r}")

    # Post-resolution containment check
    resolved_target = (resolved_dest / member.name).resolve()
    if not resolved_target.is_relative_to(resolved_dest):
        raise ExtractionError(f"tarball member escapes destination: {member.name!r}")
