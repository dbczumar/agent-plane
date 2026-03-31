#!/usr/bin/env bash
# Launch Toad connected to a running agent-plane server.
#
# Usage:
#   ./scripts/launch-toad.sh [AGENT_NAME] [SERVER_URL]
#
# Arguments:
#   AGENT_NAME   Name of deployed agent (default: coder)
#   SERVER_URL   Agent-plane server URL (default: http://127.0.0.1:8923)
#
# Prerequisites:
#   - agent-plane server already running
#   - toad installed: uv tool install -U batrachian-toad --python 3.14 --exclude-newer 2026-03-09
#   - agent-plane installed in mlflow env: pip install -e ~/agent-plane
#
set -euo pipefail

AGENT_NAME="${1:-coder}"
AGENT_PLANE_URL="${2:-http://127.0.0.1:8923}"
CONDA_BIN="${HOME}/miniconda3/envs/mlflow/bin"

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

if ! curl -sf "$AGENT_PLANE_URL/health" >/dev/null 2>&1; then
    echo "Error: no agent-plane server at $AGENT_PLANE_URL"
    echo "Start one first."
    exit 1
fi

# --- Launch Toad ---

echo "Connecting to $AGENT_PLANE_URL (agent: $AGENT_NAME)"

export PATH="$CONDA_BIN:$PATH"
export OPENAI_API_KEY="${OPENAI_API_KEY:-$(cat /tmp/mykey 2>/dev/null || echo '')}"
export AGENT_PLANE_URL
export AGENT_PLANE_AGENT="$AGENT_NAME"

exec toad acp "agent-plane-acp" .
