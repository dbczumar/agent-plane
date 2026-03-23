"""Tests for agent_plane.spec.load()."""

from __future__ import annotations

import io
import tarfile
from pathlib import Path

import pytest
import yaml

from agent_plane.spec import load


@pytest.fixture()
def agent_dir(tmp_path: Path) -> Path:
    """Create a minimal valid agent image directory."""
    config = {"spec_version": 1, "name": "test-agent"}
    (tmp_path / "config.yaml").write_text(yaml.dump(config))
    return tmp_path


def _make_tarball(tmp_path: Path, files: dict[str, str]) -> Path:
    """Build a tar.gz at tmp_path/bundle.tar.gz."""
    tar_path = tmp_path / "bundle.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tf:
        for name, content in files.items():
            data = content.encode()
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    return tar_path


def test_load_from_directory(agent_dir: Path) -> None:
    spec = load(agent_dir)
    assert spec.name == "test-agent"
    assert spec.spec_version == 1


def test_load_from_tarball(tmp_path: Path) -> None:
    config = yaml.dump({"spec_version": 1, "name": "tarball-agent"})
    tar_path = _make_tarball(tmp_path, {"config.yaml": config})
    dest = tmp_path / "extracted"

    spec = load(tar_path, dest=dest)

    assert spec.name == "tarball-agent"
    assert dest.is_dir()
    assert (dest / "config.yaml").exists()


def test_load_tarball_without_dest_raises(tmp_path: Path) -> None:
    config = yaml.dump({"spec_version": 1, "name": "x"})
    tar_path = _make_tarball(tmp_path, {"config.yaml": config})

    with pytest.raises(ValueError, match="dest is required"):
        load(tar_path)


def test_load_invalid_spec_raises(tmp_path: Path) -> None:
    config = {"spec_version": 99, "name": "bad"}
    (tmp_path / "config.yaml").write_text(yaml.dump(config))

    with pytest.raises(ValueError, match="invalid agent spec"):
        load(tmp_path)


def test_load_missing_source(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="source not found"):
        load(tmp_path / "nonexistent")
