# client/ — How to TALK to the server

Depends on: spec/ (for types only)
Does NOT depend on: server/, runtime/

HTTP client and CLI for interacting with the agent-plane server.

## Planned files

### client.py
`AgentPlaneClient` — thin HTTP wrapper:
- `client = AgentPlaneClient(base_url="http://localhost:8080")`
- `client.create_agent(name, description, bundle_path) → CreateAgentResponse`
  - Reads tarball from local path, sends multipart POST
- `client.get_agent(agent_id) → GetAgentResponse`
- `client.delete_agent(agent_id) → None`
- Uses `httpx` for async support
- Raises typed exceptions for 4xx/5xx responses

### cli.py
CLI entrypoint (click or argparse):
```
agent-plane create-agent --name my-agent --bundle ./my-agent.tar.gz
agent-plane create-agent --name my-agent --bundle-dir ./my-agent-repo/
    (tars the directory automatically before uploading)
agent-plane get-agent <agent_id>
agent-plane delete-agent <agent_id>
```
- `--bundle-dir` convenience: tar a directory on the fly before uploading
- Output: JSON by default, `--format table` for human-readable
- `--server` flag or `AGENT_PLANE_SERVER_URL` env var for server address

## Key design decisions
- Client is a standalone module — can be used as a library or via CLI
- Depends on spec/ types so responses are properly typed, not raw dicts
- Does NOT import anything from server/ or runtime/
- The CLI is a thin wrapper around the client library

## Not yet (future)
- `agent-plane invoke <agent_id> --input "..."` — invoke agent
- `agent-plane list-agents` — list all agents
- `agent-plane logs <agent_id>` — stream execution logs
- Auth (API keys, OAuth tokens)
