"""Two-tier agent cache — disk + in-memory — backed by ArtifactStore."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from agent_plane.entities.agent import LoadedAgent
from agent_plane.spec.parser import parse
from agent_plane.spec.tar_utils import extract_safe
from agent_plane.spec.types import AgentSpec
from agent_plane.spec.validator import validate
from agent_plane.stores.artifact_store import ArtifactStore


class AgentCache:
    """
    Two-tier cache for loaded agents.

    Tier 1 (in-memory): parsed AgentSpec objects keyed by agent_id.
    Tier 2 (disk): extracted agent directories under cache_dir/<agent_id>/.
    Source of truth: ArtifactStore (tarball bytes).

    On cache miss the bundle is downloaded from the ArtifactStore,
    extracted to disk, parsed, validated, and stored in both tiers.
    """

    def __init__(
        self, artifact_store: ArtifactStore, cache_dir: Path
    ) -> None:
        self._artifact_store = artifact_store
        self._cache_dir = cache_dir
        self._specs: dict[str, AgentSpec] = {}

    def load(self, agent_id: str) -> LoadedAgent:
        """
        Load an agent, populating caches on miss.

        Raises KeyError if the agent bundle does not exist in the
        ArtifactStore. Raises ValueError if the spec is invalid.
        """
        workdir = self._cache_dir / agent_id

        # Tier 1: in-memory spec
        if agent_id in self._specs:
            return LoadedAgent(spec=self._specs[agent_id], workdir=workdir)

        # Tier 2: disk cache (directory already extracted)
        if workdir.is_dir():
            spec = _parse_and_validate(workdir)
            self._specs[agent_id] = spec
            return LoadedAgent(spec=spec, workdir=workdir)

        # Cache miss — download from artifact store
        bundle_bytes = self._artifact_store.get(agent_id)
        _extract_bundle(bundle_bytes, workdir)

        spec = _parse_and_validate(workdir)
        self._specs[agent_id] = spec
        return LoadedAgent(spec=spec, workdir=workdir)

    def evict(self, agent_id: str) -> None:
        """
        Remove an agent from both cache tiers. Called when an agent
        is deleted. No-op if the agent is not cached.
        """
        self._specs.pop(agent_id, None)
        workdir = self._cache_dir / agent_id
        if workdir.is_dir():
            shutil.rmtree(workdir)


def _parse_and_validate(workdir: Path) -> AgentSpec:
    """
    Parse and validate the agent spec from an extracted directory.

    Raises ValueError if validation fails.
    """
    spec = parse(workdir)
    result = validate(spec)
    if not result.valid:
        errors = "; ".join(
            f"{e.path}: {e.message}" for e in result.errors
        )
        raise ValueError(f"invalid agent spec: {errors}")
    return spec


def _extract_bundle(bundle_bytes: bytes, dest: Path) -> None:
    """
    Write bundle bytes to a temp file and extract safely to dest.

    extract_safe() requires a file path (not bytes), so we write to
    a temporary file first.
    """
    tmp_fd, tmp_name = tempfile.mkstemp(suffix=".tar.gz")
    tmp_path = Path(tmp_name)
    try:
        tmp_path.write_bytes(bundle_bytes)
        extract_safe(tmp_path, dest)
    finally:
        tmp_path.unlink()
