"""Tests for agent_plane.runtime.agent_cache."""

from __future__ import annotations

import io
import tarfile
from pathlib import Path

import pytest
import yaml

from agent_plane.errors import AgentPlaneError
from agent_plane.runtime.agent_cache import AgentCache
from agent_plane.stores.artifact_store.local import LocalArtifactStore

# Minimal valid config.yaml for a spec_version=1 agent
_MINIMAL_CONFIG = yaml.dump({"spec_version": 1, "name": "test-agent"})


def _make_bundle_bytes(files: dict[str, str]) -> bytes:
    """
    Build a tar.gz in memory from a dict of {path: content}.
    Returns the raw bytes of the tarball.
    """
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for name, content in files.items():
            data = content.encode()
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    return buf.getvalue()


@pytest.fixture()
def artifact_store(tmp_path: Path) -> LocalArtifactStore:
    return LocalArtifactStore(str(tmp_path / "artifacts"))


@pytest.fixture()
def cache_dir(tmp_path: Path) -> Path:
    return tmp_path / "cache"


@pytest.fixture()
def agent_cache(artifact_store: LocalArtifactStore, cache_dir: Path) -> AgentCache:
    return AgentCache(artifact_store=artifact_store, cache_dir=cache_dir)


def _store_bundle(
    artifact_store: LocalArtifactStore,
    bundle_location: str,
    files: dict[str, str] | None = None,
) -> bytes:
    """
    Store a tarball bundle in the artifact store under the given
    bundle_location. Uses minimal valid config.yaml if no files
    provided. Returns the bundle bytes.
    """
    if files is None:
        files = {"config.yaml": _MINIMAL_CONFIG}
    data = _make_bundle_bytes(files)
    artifact_store.put(bundle_location, data)
    return data


def test_load_cache_miss_downloads_and_extracts(
    agent_cache: AgentCache,
    artifact_store: LocalArtifactStore,
    cache_dir: Path,
) -> None:
    """
    On a full cache miss, load() downloads from artifact store,
    extracts to disk, parses spec, and returns LoadedAgent.
    """
    loc = "agent-1/abc123"
    _store_bundle(artifact_store, loc)

    loaded = agent_cache.load("agent-1", loc)

    assert loaded.spec.name == "test-agent"
    assert loaded.spec.spec_version == 1
    assert loaded.workdir == cache_dir / "agent-1"
    assert loaded.workdir.is_dir()
    assert (loaded.workdir / "config.yaml").exists()


def test_load_memory_cache_hit(
    agent_cache: AgentCache,
    artifact_store: LocalArtifactStore,
) -> None:
    """
    Second call to load() returns from in-memory cache without
    re-parsing from disk.
    """
    loc = "agent-2/abc123"
    _store_bundle(artifact_store, loc)

    first = agent_cache.load("agent-2", loc)
    second = agent_cache.load("agent-2", loc)

    # Same spec object (identity check — memory cache returns same ref)
    assert first.spec is second.spec
    assert first.workdir == second.workdir


def test_load_disk_cache_hit(
    artifact_store: LocalArtifactStore,
    cache_dir: Path,
) -> None:
    """
    When the disk directory exists but memory cache is empty (e.g.
    after server restart), load() re-parses from disk without
    downloading.
    """
    loc = "agent-3/abc123"
    _store_bundle(artifact_store, loc)

    # First cache instance populates disk
    cache_1 = AgentCache(artifact_store=artifact_store, cache_dir=cache_dir)
    first = cache_1.load("agent-3", loc)

    # New cache instance simulates server restart — empty memory cache
    cache_2 = AgentCache(artifact_store=artifact_store, cache_dir=cache_dir)

    # Remove from artifact store to prove we don't re-download
    artifact_store.delete(loc)

    second = cache_2.load("agent-3", loc)
    assert second.spec.name == first.spec.name
    assert second.workdir == first.workdir


def test_load_missing_agent_raises_key_error(
    agent_cache: AgentCache,
) -> None:
    """load() raises KeyError when the bundle doesn't exist."""
    with pytest.raises(KeyError):
        agent_cache.load("nonexistent", "nonexistent/abc123")


def test_load_invalid_spec_raises_agent_plane_error(
    agent_cache: AgentCache,
    artifact_store: LocalArtifactStore,
) -> None:
    """
    ``load()`` raises ``AgentPlaneError`` when the extracted spec
    is invalid.

    :param agent_cache: The cache under test.
    :param artifact_store: Store for uploading test bundles.
    """
    # spec_version=99 is invalid (must be 1)
    bad_config = yaml.dump({"spec_version": 99, "name": "bad"})
    loc = "bad-agent/abc123"
    _store_bundle(artifact_store, loc, {"config.yaml": bad_config})

    with pytest.raises(AgentPlaneError, match="invalid agent spec"):
        agent_cache.load("bad-agent", loc)


def test_evict_clears_both_tiers(
    agent_cache: AgentCache,
    artifact_store: LocalArtifactStore,
    cache_dir: Path,
) -> None:
    """evict() removes from memory and disk."""
    loc = "agent-4/abc123"
    _store_bundle(artifact_store, loc)

    agent_cache.load("agent-4", loc)
    assert (cache_dir / "agent-4").is_dir()

    agent_cache.evict("agent-4")

    # Disk cache cleared
    assert not (cache_dir / "agent-4").exists()

    # Memory cache cleared — remove from artifact store to prove
    # load() can't fall back to a cached spec in memory
    artifact_store.delete(loc)
    with pytest.raises(KeyError):
        agent_cache.load("agent-4", loc)


def test_evict_noop_for_uncached_agent(
    agent_cache: AgentCache,
) -> None:
    """evict() on a non-existent agent is a silent no-op."""
    agent_cache.evict("never-loaded")


def test_replace_swaps_spec(
    agent_cache: AgentCache,
    artifact_store: LocalArtifactStore,
    cache_dir: Path,
) -> None:
    """
    replace() extracts new bundle, swaps the in-memory spec,
    and replaces the disk directory.
    """
    # Load original bundle
    loc_v1 = "agent-5/v1hash"
    _store_bundle(artifact_store, loc_v1)
    loaded_v1 = agent_cache.load("agent-5", loc_v1)
    assert loaded_v1.spec.name == "test-agent"

    # Build a new bundle with a different description
    new_config = yaml.dump(
        {
            "spec_version": 1,
            "name": "test-agent",
            "description": "updated agent",
        }
    )
    new_bytes = _make_bundle_bytes({"config.yaml": new_config})

    # Warm-swap
    loc_v2 = "agent-5/v2hash"
    loaded_v2 = agent_cache.replace("agent-5", loc_v2, new_bytes)

    # New spec is returned and cached
    assert loaded_v2.spec.description == "updated agent"
    assert loaded_v2.workdir == cache_dir / "agent-5"
    assert loaded_v2.workdir.is_dir()

    # Subsequent load() returns the new spec from memory cache
    loaded_again = agent_cache.load("agent-5", loc_v2)
    assert loaded_again.spec is loaded_v2.spec
