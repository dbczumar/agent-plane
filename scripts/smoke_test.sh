#!/usr/bin/env bash
# Smoke test for the agent execution loop.
#
# Usage:
#   ./scripts/smoke_test.sh <OPENAI_API_KEY>
#   ./scripts/smoke_test.sh $(cat /tmp/mykey)
set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: $0 <OPENAI_API_KEY>"
    echo "       $0 \$(cat /tmp/mykey)"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
exec python "$SCRIPT_DIR/smoke_test.py" "$1"
