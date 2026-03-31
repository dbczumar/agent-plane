#!/usr/bin/env bash
# Launch Toad connected to an agent-plane server.
# Starts the server automatically if it's not already running.
#
# Usage:
#   ./scripts/launch-toad.sh [AGENT_NAME] [PORT]
#
# Arguments:
#   AGENT_NAME   Name of deployed agent (default: coder)
#   PORT         Server port (default: 8923)
#
# Prerequisites:
#   - toad installed: uv tool install -U batrachian-toad --python 3.14 --exclude-newer 2026-03-09
#   - agent-plane installed in mlflow env: pip install -e ~/agent-plane
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
AGENT_NAME="${1:-coder}"
PORT="${2:-8923}"
AGENT_PLANE_URL="http://127.0.0.1:$PORT"
CONDA_BIN="${HOME}/miniconda3/envs/mlflow/bin"
DATA_DIR="${HOME}/.agent-plane"
STARTED_SERVER=false

# --- Preflight checks ---

if ! command -v toad &>/dev/null; then
    echo "Error: toad not found."
    echo "Install: uv tool install -U batrachian-toad --python 3.14 --exclude-newer 2026-03-09 --no-build-package watchdog"
    exit 1
fi

if [[ ! -f "$CONDA_BIN/agent-plane-acp" ]]; then
    echo "Error: agent-plane-acp not found at $CONDA_BIN"
    echo "Install: conda run -n mlflow pip install -e ~/agent-plane"
    exit 1
fi

export PATH="$CONDA_BIN:$PATH"
export OPENAI_API_KEY="${OPENAI_API_KEY:-$(cat /tmp/mykey 2>/dev/null || echo '')}"

# --- Start server if not running ---

if ! curl -sf "$AGENT_PLANE_URL/health" >/dev/null 2>&1; then
    echo "No server at $AGENT_PLANE_URL — starting one..."
    mkdir -p "$DATA_DIR"
    "$CONDA_BIN/python" -m agent_plane.cli server \
        --host 127.0.0.1 \
        --port "$PORT" \
        --database-uri "sqlite:///$DATA_DIR/agent_plane.db" \
        --artifact-location "$DATA_DIR/artifacts" \
        > "$DATA_DIR/server.log" 2>&1 &
    SERVER_PID=$!
    STARTED_SERVER=true
    for _ in $(seq 1 30); do
        if curl -sf "$AGENT_PLANE_URL/health" >/dev/null 2>&1; then break; fi
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            echo "Error: server crashed. See $DATA_DIR/server.log"
            exit 1
        fi
        sleep 0.5
    done
    if ! curl -sf "$AGENT_PLANE_URL/health" >/dev/null 2>&1; then
        echo "Error: server didn't start. See $DATA_DIR/server.log"
        exit 1
    fi
    echo "Server started (pid $SERVER_PID, db: $DATA_DIR/agent_plane.db)"
fi

# --- Deploy agent if not already deployed ---

AGENT_DIR="$ROOT_DIR/examples/agents/$AGENT_NAME"
if [[ -d "$AGENT_DIR" ]]; then
    AGENTS_JSON=$(curl -sf "$AGENT_PLANE_URL/api/agents" || echo '{"data":[]}')
    if ! echo "$AGENTS_JSON" | python3 -c "
import sys, json
data = json.load(sys.stdin).get('data', [])
sys.exit(0 if any(a.get('name') == '$AGENT_NAME' for a in data) else 1)
" 2>/dev/null; then
        echo "Deploying $AGENT_NAME..."
        "$CONDA_BIN/python" -m agent_plane.cli deploy "$AGENT_DIR" \
            --server "$AGENT_PLANE_URL" 2>&1
    fi
fi

# --- Cleanup server on exit (only if we started it) ---

cleanup() {
    if [[ "$STARTED_SERVER" == true && -n "${SERVER_PID:-}" ]]; then
        echo ""
        echo "Stopping server (pid $SERVER_PID)..."
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

# --- Launch Toad ---

echo "Connecting to $AGENT_PLANE_URL (agent: $AGENT_NAME)"

export AGENT_PLANE_URL
export AGENT_PLANE_AGENT="$AGENT_NAME"

exec toad acp "agent-plane-acp" .
