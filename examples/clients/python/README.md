# Python client examples

Examples for the [`agent_plane_client`](../../../sdks/python-client/) package.

## `quickstart.py`

A tour of the patterns you'll actually use when building apps on top
of agent-plane:

1. `client.query(...)` — ask, get a `QueryResult(text, files)` back.
2. `client.query(..., stream=True)` — stream `str` chunks; `.files`
   after exhaustion.
3. `session.query(...)` — multi-turn conversation, IDs threaded for you.
4. Client-side tools — register an `@tool`-decorated function; agent calls it.
5. Text file attachment — pass a text file via `files=[...]`.
6. Image attachment — pass a PNG via `files=[...]`; archer describes it.
7. Agent-produced files — agent writes a file, we download via
   `client.files.download(id, path)`.
8. Pointer to `BlockStream` for cases where plain text isn't enough.

### Run

From the agent-plane repo root:

```bash
pip install -e sdks/python-client
OPENAI_API_KEY=sk-... python examples/clients/python/quickstart.py
```

The script spins up a temporary server via `LocalServer`, deploys the
`archer` agent, runs all six sections, and shuts the server down on
exit.

### Pointing at an existing server

Replace the `LocalServer` block with:

```python
async with AgentPlaneClient(base_url="http://localhost:8080") as client:
    text = await client.query(model="archer", input="hello")
```
