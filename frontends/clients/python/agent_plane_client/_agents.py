"""Agents namespace — CRUD for agent bundles."""

from __future__ import annotations

import io
import pathlib
import tarfile

import httpx

from ._errors import raise_for_status
from ._types import Agent, PaginatedList


class AgentsNamespace:
    """Methods for ``/api/agents`` endpoints."""

    def __init__(self, http: httpx.AsyncClient, base_url: str) -> None:
        self._http = http
        self._base = base_url

    async def create(
        self,
        *,
        bundle_path: str,
        replace: bool = False,
    ) -> Agent:
        """Upload an agent bundle.

        :param bundle_path: Path to an agent directory (containing
            ``config.yaml``) or a ``.tar.gz`` file.
        :param replace: If True and an agent with the same name already
            exists, delete it first and re-upload.
        :returns: The created agent.
        """
        bundle_bytes = _load_bundle(bundle_path)
        resp = await self._http.post(
            f"{self._base}/api/agents",
            files={"bundle": ("agent.tar.gz", bundle_bytes, "application/gzip")},
        )

        if resp.status_code == 409 and replace:
            # Agent with same name exists — delete and retry.
            name = _extract_name_from_bundle(bundle_bytes)
            if name is not None:
                await self._delete_by_name(name)
                resp = await self._http.post(
                    f"{self._base}/api/agents",
                    files={"bundle": ("agent.tar.gz", bundle_bytes, "application/gzip")},
                )

        data = resp.json() if resp.status_code < 500 else resp.text
        raise_for_status(resp.status_code, data)
        return Agent.from_dict(resp.json())

    async def list(
        self,
        *,
        limit: int = 20,
        after: str | None = None,
        order: str = "desc",
    ) -> list[Agent]:
        """List agents.

        :param limit: Max agents to return.
        :param after: Cursor for pagination.
        :param order: Sort order (``"asc"`` or ``"desc"``).
        :returns: List of agents.
        """
        params: dict[str, object] = {"limit": limit, "order": order}
        if after is not None:
            params["after"] = after
        resp = await self._http.get(f"{self._base}/api/agents", params=params)
        data = resp.json() if resp.status_code < 500 else resp.text
        raise_for_status(resp.status_code, data)
        page = PaginatedList.from_dict(resp.json())
        return [Agent.from_dict(d) for d in page.data]

    async def get(self, agent_id: str) -> Agent:
        """Get an agent by ID.

        :param agent_id: The agent ID.
        :returns: The agent.
        """
        resp = await self._http.get(f"{self._base}/api/agents/{agent_id}")
        data = resp.json() if resp.status_code < 500 else resp.text
        raise_for_status(resp.status_code, data)
        return Agent.from_dict(resp.json())

    async def get_by_name(self, name: str) -> Agent | None:
        """Find an agent by name.

        :param name: The agent name.
        :returns: The agent, or None if not found.
        """
        agents = await self.list(limit=100)
        for agent in agents:
            if agent.name == name:
                return agent
        return None

    async def delete(self, agent_id: str) -> None:
        """Delete an agent.

        :param agent_id: The agent ID.
        """
        resp = await self._http.delete(f"{self._base}/api/agents/{agent_id}")
        if resp.status_code >= 400:
            data = resp.json() if resp.status_code < 500 else resp.text
            raise_for_status(resp.status_code, data)

    async def _delete_by_name(self, name: str) -> None:
        """Delete an agent by name (for replace semantics)."""
        agent = await self.get_by_name(name)
        if agent is not None:
            await self.delete(agent.id)


def _load_bundle(path: str) -> bytes:
    """Load an agent bundle from a directory or tarball path."""
    p = pathlib.Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Bundle path not found: {path}")

    if p.is_file():
        return p.read_bytes()

    # Directory — tar it up.
    if not (p / "config.yaml").exists():
        raise FileNotFoundError(f"No config.yaml in {path}")

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for file_path in sorted(p.rglob("*")):
            if file_path.is_file():
                arcname = str(file_path.relative_to(p))
                tf.add(str(file_path), arcname=arcname)
    return buf.getvalue()


def _extract_name_from_bundle(bundle: bytes) -> str | None:
    """Extract the agent name from config.yaml inside a tarball."""
    try:
        import yaml  # type: ignore[import-untyped]
    except ImportError:
        return None

    try:
        with tarfile.open(fileobj=io.BytesIO(bundle), mode="r:gz") as tf:
            for member in tf.getmembers():
                if member.name == "config.yaml":
                    f = tf.extractfile(member)
                    if f is None:
                        continue
                    config = yaml.safe_load(f.read())
                    if isinstance(config, dict):
                        name = config.get("name")
                        if isinstance(name, str):
                            return name
    except Exception:
        pass
    return None
