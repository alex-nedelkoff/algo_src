# Linear MCP Setup

One-time setup to connect Claude Code to the Corvidx Drone Grand Prix Linear workspace.

## Steps

1. Add the MCP server:
   ```bash
   claude mcp add --transport http -s user linear-grandprix https://mcp.linear.app/sse
   ```
   This opens OAuth in the browser. Authenticate with the Linear account that has access to the **corvidx-drone-grand-prix** workspace.

2. Restart Claude Code (MCP servers load at startup).

3. Verify by listing teams — confirm "Corvidx-drone-grand-prix" appears:
   ```
   mcp__linear-grandprix__list_teams
   ```

4. Optionally add tool permissions to `.claude/settings.local.json` so common operations don't require approval each time:
   ```json
   {
     "permissions": {
       "allow": [
         "mcp__linear-grandprix__list_teams",
         "mcp__linear-grandprix__list_projects",
         "mcp__linear-grandprix__list_issues",
         "mcp__linear-grandprix__get_issue",
         "mcp__linear-grandprix__list_comments",
         "mcp__linear-grandprix__list_issue_statuses"
       ]
     }
   }
   ```

## Troubleshooting

- **OAuth fails**: Ensure your Linear account has access to the workspace. Ask an admin to send an invite.
- **Tools not available after adding**: Restart the Claude Code session.
- **Permission denied on tool calls**: Add the tool name to `.claude/settings.local.json` under `permissions.allow`.
