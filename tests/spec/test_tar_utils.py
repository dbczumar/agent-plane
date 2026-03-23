"""Tests for agent_plane.spec.tar_utils."""

from __future__ import annotations

import io
import tarfile
from pathlib import Path

import pytest
import yaml

from agent_plane.spec.tar_utils import ExtractionError, extract_safe


def _create_tar(tmp_path: Path, members: dict[str, bytes | str]) -> Path:
    """Create a tar.gz at tmp_path/bundle.tar.gz with given members."""
    tar_path = tmp_path / "bundle.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tf:
        for name, content in members.items():
            data = content.encode() if isinstance(content, str) else content
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    return tar_path


@pytest.fixture()
def dest(tmp_path: Path) -> Path:
    return tmp_path / "extracted"


def test_extract_valid_tarball(tmp_path: Path, dest: Path) -> None:
    config = yaml.dump({"spec_version": 1, "name": "test"})
    tar_path = _create_tar(tmp_path, {"config.yaml": config})
    result = extract_safe(tar_path, dest)
    assert result == dest
    assert (dest / "config.yaml").exists()
    assert yaml.safe_load((dest / "config.yaml").read_text())["name"] == "test"


def test_extract_nested_files(tmp_path: Path, dest: Path) -> None:
    tar_path = _create_tar(
        tmp_path,
        {
            "config.yaml": "spec_version: 1",
            "skills/search/SKILL.md": "---\nname: search\n---\ncontent",
        },
    )
    extract_safe(tar_path, dest)
    assert (dest / "skills" / "search" / "SKILL.md").exists()


def test_extract_rejects_path_traversal(tmp_path: Path, dest: Path) -> None:
    tar_path = _create_tar(tmp_path, {"../escape.txt": "evil"})
    with pytest.raises(ExtractionError, match="path traversal"):
        extract_safe(tar_path, dest)


def test_extract_rejects_absolute_path(tmp_path: Path, dest: Path) -> None:
    tar_path = _create_tar(tmp_path, {"/etc/passwd": "evil"})
    with pytest.raises(ExtractionError, match="absolute path"):
        extract_safe(tar_path, dest)


def test_extract_rejects_symlink(tmp_path: Path, dest: Path) -> None:
    tar_path = tmp_path / "symlink.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tf:
        info = tarfile.TarInfo(name="evil-link")
        info.type = tarfile.SYMTYPE
        info.linkname = "/etc/passwd"
        tf.addfile(info)
    with pytest.raises(ExtractionError, match="link"):
        extract_safe(tar_path, dest)


def test_extract_rejects_hardlink(tmp_path: Path, dest: Path) -> None:
    tar_path = tmp_path / "hardlink.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tf:
        info = tarfile.TarInfo(name="evil-link")
        info.type = tarfile.LNKTYPE
        info.linkname = "/etc/passwd"
        tf.addfile(info)
    with pytest.raises(ExtractionError, match="link"):
        extract_safe(tar_path, dest)


def test_extract_rejects_size_bomb(tmp_path: Path, dest: Path) -> None:
    tar_path = tmp_path / "bomb.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tf:
        data = b"x" * 1024
        info = tarfile.TarInfo(name="big.bin")
        info.size = len(data)
        tf.addfile(info, io.BytesIO(data))
    with pytest.raises(ExtractionError, match="max extracted size"):
        extract_safe(tar_path, dest, max_bytes=512)


def test_extract_rejects_entry_bomb(tmp_path: Path, dest: Path) -> None:
    tar_path = tmp_path / "entries.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tf:
        for i in range(10):
            info = tarfile.TarInfo(name=f"file_{i}.txt")
            info.size = 1
            tf.addfile(info, io.BytesIO(b"x"))
    with pytest.raises(ExtractionError, match="max entry count"):
        extract_safe(tar_path, dest, max_entries=5)


def test_extract_missing_tarball(tmp_path: Path, dest: Path) -> None:
    with pytest.raises(FileNotFoundError, match="tarball not found"):
        extract_safe(tmp_path / "nonexistent.tar.gz", dest)


def test_extract_creates_dest_directory(tmp_path: Path) -> None:
    dest = tmp_path / "deep" / "nested" / "dir"
    tar_path = _create_tar(tmp_path, {"config.yaml": "spec_version: 1"})
    extract_safe(tar_path, dest)
    assert dest.is_dir()
    assert (dest / "config.yaml").exists()
