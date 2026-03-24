#!/usr/bin/env bash
# Interactive chat shell for agent-plane.
#
# Usage:
#   ./scripts/chat.sh <OPENAI_API_KEY>
#   ./scripts/chat.sh $(cat /tmp/mykey)
set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: $0 <OPENAI_API_KEY>"
    echo "       $0 \$(cat /tmp/mykey)"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
exec python "$SCRIPT_DIR/chat.py" "$1"
