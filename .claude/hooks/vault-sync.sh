#!/usr/bin/env bash
# vault-sync.sh — Auto-commit and push research-vault changes
# Called by Claude Code hooks on PostToolUse (Edit|Write) and Stop events.
# Reads JSON from stdin, checks if file_path is in research-vault.

set -euo pipefail

VAULT_DIR="$(cd "$(dirname "$0")/../../.." && pwd)/research-vault"

# Read JSON event from stdin
INPUT=$(cat)

# Extract file_path from the JSON (handles both tool_input.file_path and direct file_path)
FILE_PATH=$(echo "$INPUT" | python3 -c "
import json, sys
data = json.load(sys.stdin)
# PostToolUse events have tool_input.file_path
ti = data.get('tool_input', {})
fp = ti.get('file_path', '') if isinstance(ti, dict) else ''
if not fp:
    fp = data.get('file_path', '')
print(fp)
" 2>/dev/null || echo "")

# Exit silently if not a vault file
if [[ -z "$FILE_PATH" ]] || [[ "$FILE_PATH" != *"research-vault"* ]]; then
    exit 0
fi

# Check vault exists
if [[ ! -d "$VAULT_DIR/.git" ]]; then
    exit 0
fi

cd "$VAULT_DIR"

# Stage, commit, and push
git add -A
if git diff --cached --quiet; then
    exit 0
fi

FILENAME=$(basename "$FILE_PATH")
git commit -m "vault: update $FILENAME" --no-gpg-sign 2>/dev/null || true
git push 2>/dev/null || true
