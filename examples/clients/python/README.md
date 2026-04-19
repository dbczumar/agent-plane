# Python client examples

Examples for the `agent_plane_client` package (see `sdks/python-client/`).

## `quickstart.py`

A five-demo tour covering the SDK surface you're most likely to need:

1. One-shot invocation (send → collect → print)
2. Raw event stream (typed SSE events)
3. Semantic blocks via `BlockStream`
4. Multi-turn sessions (automatic `previous_response_id` threading)
5. Client-side tools (register a local function, let the agent call it)

### Run

From the agent-plane repo root:

```bash
pip install -e sdks/python-client
OPENAI_API_KEY=sk-... python examples/clients/python/quickstart.py
```

The script spins up a temporary `agent-plane` server via `LocalServer`,
deploys the `archer` research agent, runs all five demos sequentially,
and shuts the server down on exit.

### Customizing

Each demo is a standalone `async` function. To try a demo in isolation,
copy the function into your own script alongside the top-level imports
and the `LocalServer` block from `main()`.

To point at an already-running agent-plane server instead of spinning
one up, replace:

```python
async with LocalServer(agent_path=AGENT_PATH) as server:
    await demo_one_shot(server.client)
```

with:

```python
async with AgentPlaneClient(base_url="http://localhost:8080") as client:
    await demo_one_shot(client)
```
