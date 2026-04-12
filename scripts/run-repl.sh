#!/usr/bin/env bash
# Run the agent-plane REPL with the coder agent (full features).
#
# Usage:
#   ./run.sh                          # coder with client tools
#   ./run.sh ../../examples/agents/archer/   # archer (no client tools)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# API key
if [ -z "${OPENAI_API_KEY:-}" ]; then
    if [ -f /tmp/mykey ]; then
        export OPENAI_API_KEY="$(cat /tmp/mykey)"
    else
        echo "Error: OPENAI_API_KEY not set and /tmp/mykey not found"
        exit 1
    fi
fi

AGENT="${1:-$REPO_ROOT/examples/agents/coder/}"
CLIENT_TOOLS_FLAG=""

# Auto-detect: if using coder agent, add --client-tools coder
if [[ "$AGENT" == *"/coder"* ]] && [[ "$AGENT" != *"claude-coder"* ]] && [[ "$AGENT" != *"openai-coder"* ]]; then
    CLIENT_TOOLS_FLAG="--client-tools coder"
fi

cd "$REPO_ROOT/frontends/repl"
PYTHONPATH="../clients/python:.:${PYTHONPATH:-}" exec python repl.py "$AGENT" $CLIENT_TOOLS_FLAG
