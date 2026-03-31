#!/usr/bin/env bash
# Launch Toad connected to an agent-plane server.
#
# Usage:
#   ./scripts/launch-toad.sh [AGENT_NAME] [PORT]
#
# Arguments:
#   AGENT_NAME  Name of deployed agent (default: coder)
#   PORT        Server port (default: 8923)
#
# Prerequisites:
#   - toad installed: uv tool install -U batrachian-toad --exclude-newer 2026-03-09
#   - agent-plane installed in mlflow env: pip install -e ~/agent-plane
#   - OpenAI API key at /tmp/mykey
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

AGENT_NAME="${1:-coder}"
PORT="${2:-8923}"
AGENT_PLANE_URL="http://127.0.0.1:$PORT"
CONDA_BIN="${HOME}/miniconda3/envs/mlflow/bin"
TMPDIR="$(mktemp -d)"

# --- Load API key ---

KEY_FILE="/tmp/mykey"
if [[ ! -f "$KEY_FILE" ]]; then
    echo "Error: OpenAI API key not found at $KEY_FILE"
    exit 1
fi
export OPENAI_API_KEY="$(cat "$KEY_FILE")"

# --- Preflight checks ---

if ! command -v toad &>/dev/null; then
    echo "Error: toad not found."
    echo "Install: uv tool install -U batrachian-toad --exclude-newer 2026-03-09 --no-build-package watchdog"
    exit 1
fi

if [[ ! -f "$CONDA_BIN/python" ]]; then
    echo "Error: mlflow conda env not found at $CONDA_BIN"
    exit 1
fi

AGENT_DIR="$ROOT_DIR/examples/agents/$AGENT_NAME"
if [[ ! -d "$AGENT_DIR" ]]; then
    echo "Error: agent directory not found: $AGENT_DIR"
    exit 1
fi

# --- Cleanup on exit ---

cleanup() {
    echo ""
    echo "Shutting down..."
    if [[ -n "${SERVER_PID:-}" ]]; then
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
    rm -rf "$TMPDIR"
}
trap cleanup EXIT

# --- Start agent-plane server ---

echo "Starting agent-plane server on port $PORT..."
"$CONDA_BIN/python" -m agent_plane.cli server \
    --host 127.0.0.1 \
    --port "$PORT" \
    --database-uri "sqlite:///$TMPDIR/ap.db" \
    --artifact-location "$TMPDIR/artifacts" \
    > "$TMPDIR/server.log" 2>&1 &
SERVER_PID=$!

# Wait for server (up to 15s) — no /health endpoint, so check
# /health which returns 200 on a running server.
for i in $(seq 1 30); do
    if curl -sf "$AGENT_PLANE_URL/health" >/dev/null 2>&1; then
        break
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "Error: server exited unexpectedly. Logs:"
        cat "$TMPDIR/server.log"
        exit 1
    fi
    sleep 0.5
done

if ! curl -sf "$AGENT_PLANE_URL/health" >/dev/null 2>&1; then
    echo "Error: server did not start within 15s. Logs:"
    tail -20 "$TMPDIR/server.log"
    exit 1
fi
echo "Server ready at $AGENT_PLANE_URL"

# --- Deploy the agent ---

echo "Deploying $AGENT_NAME..."
"$CONDA_BIN/python" -m agent_plane.cli deploy "$AGENT_DIR" \
    --server "$AGENT_PLANE_URL"
echo "Agent deployed."

# --- Launch Toad ---

echo ""
echo "Launching Toad with agent '$AGENT_NAME'..."
echo "Press Ctrl+C to exit."
echo ""

export PATH="$CONDA_BIN:$PATH"
export AGENT_PLANE_URL
export AGENT_PLANE_AGENT="$AGENT_NAME"

toad acp "agent-plane-acp" .
