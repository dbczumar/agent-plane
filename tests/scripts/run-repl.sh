#!/usr/bin/env bash
# Run the agent-plane REPL with the coder agent (full features).
#
# Usage:
#   OPENAI_API_KEY=... ./run-repl.sh
#   OPENAI_API_KEY=... ./run-repl.sh ../../examples/agents/archer/

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# API key
if [ -z "${OPENAI_API_KEY:-}" ]; then
    echo "Error: OPENAI_API_KEY not set"
    exit 1
fi

AGENT="${1:-$REPO_ROOT/examples/agents/coder/}"
CLIENT_TOOLS_FLAG=""

# Auto-detect: if using coder agent, add --tools coding
if [[ "$AGENT" == *"/coder"* ]] && [[ "$AGENT" != *"claude-coder"* ]] && [[ "$AGENT" != *"openai-coder"* ]]; then
    CLIENT_TOOLS_FLAG="--tools coding"
fi

cd "$REPO_ROOT"
exec ap chat "$AGENT" $CLIENT_TOOLS_FLAG
