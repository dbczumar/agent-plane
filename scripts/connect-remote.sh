#!/usr/bin/env bash
# Connect the TUI to a remote agent-plane server on Databricks Apps.
#
# Usage:
#   ./scripts/connect-remote.sh <app-url> <agent-name>
#
# Example:
#   ./scripts/connect-remote.sh \
#     https://agent-plane-6051921418418893.staging.aws.databricksapps.com \
#     archer
#
# Requires:
#   - DATABRICKS_HOST set to the workspace URL
#   - databricks-sdk installed (pip install databricks-sdk)
#   - A browser for OAuth consent

set -euo pipefail

if [ $# -lt 2 ]; then
    echo "Usage: $0 <app-url> <agent-name>"
    echo ""
    echo "Example:"
    echo "  DATABRICKS_HOST=https://e2-dogfood.staging.cloud.databricks.com \\"
    echo "  $0 https://agent-plane-6051921418418893.staging.aws.databricksapps.com archer"
    exit 1
fi

APP_URL="$1"
AGENT_NAME="$2"

if [ -z "${DATABRICKS_HOST:-}" ]; then
    echo "Error: DATABRICKS_HOST must be set to your workspace URL"
    echo "  export DATABRICKS_HOST=https://your-workspace.cloud.databricks.com"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

exec python "$REPO_DIR/examples/frontends/terminal.py" \
    --server "$APP_URL" \
    "$AGENT_NAME"
