# Debugger Companion Agent

## Problem

When an agent is deployed to agent-plane and serving real users,
problems surface gradually: the agent hallucinates, misses edge
cases, gives stale information, ignores instructions, fails on
certain tool calls, or frustrates users with tone issues. Today
there's no systematic way to detect these problems, track them,
or fix them. The operator has to manually read conversation logs,
guess what went wrong, edit the agent's files, and redeploy — then
hope the fix worked.

### What operators do today

1. Notice a user complaint or read a conversation log
2. Try to reproduce the issue
3. Guess the root cause (bad instructions? missing tool? wrong model?)
4. Edit AGENTS.md or config.yaml manually
5. Redeploy the agent
6. Check future conversations to see if it's fixed
7. Repeat

This is slow, manual, and doesn't scale. Issues accumulate faster
than they're fixed. The same problems recur because there's no
tracking. Fixes aren't validated against regressions.

---

## Design: A Companion Agent

A **debugger agent** that runs alongside every deployed agent. It
watches conversations, detects issues, tracks them, and — when
asked — fixes them by modifying the main agent's spec and
redeploying.

Three jobs:

1. **Watch** — analyze conversations for problems
2. **Track** — maintain a lightweight issue backlog
3. **Fix** — propose and apply changes, monitor for regression

The debugger is always watching but only speaks when spoken to
(or when something is critical). Like a QA engineer sitting next
to the operator.

---

## What the Debugger Detects

### Signal types

| Signal | Source | Example |
|---|---|---|
| Explicit negative feedback | User message | "That's wrong", "No, I meant...", "This isn't helpful" |
| User frustration | Conversation pattern | Repeated rephrasing, short terse replies after long agent output, conversation abandonment |
| Tool failures | Agent output | Tool calls that error, tools called with wrong arguments, tools that return empty results |
| Incomplete responses | Response status | `status: "incomplete"`, `status: "failed"`, context overflow |
| Instruction violations | Agent output vs spec | Agent does something its AGENTS.md explicitly says not to do |
| Hallucination | Agent output | Agent invents facts, APIs, file paths that don't exist |
| Repeated questions | Cross-conversation | Multiple users ask the same question the agent can't handle |
| Stale information | Agent output vs reality | Agent references outdated docs, deprecated APIs, old processes |
| Tone mismatch | Agent output vs spec | Too formal, too casual, too verbose for the stated persona |

### Severity levels

- **Critical** — agent gives dangerous/harmful advice, tool call
  modifies wrong files, data loss risk
- **High** — agent is wrong and user corrected it, repeated across
  multiple conversations
- **Medium** — agent is suboptimal but not wrong, user expressed
  mild frustration
- **Low** — style/tone issue, minor suboptimal phrasing, could be
  better but not broken

---

## Issue Tracking

Lightweight — not Jira. A list of issues with links to evidence.

### Issue structure

```json
{
  "id": 1,
  "title": "Hallucinates v2 API endpoints",
  "severity": "high",
  "status": "open",
  "source_conversations": ["conv_abc123", "conv_def456"],
  "source_messages": [
    {"conversation_id": "conv_abc123", "item_id": "msg_789"}
  ],
  "description": "Agent invents /v2/responses and /v2/conversations routes that don't exist. Seen in 3 conversations when users ask about the API.",
  "root_cause": null,
  "proposed_fix": null,
  "fix_applied": null,
  "created_at": "2026-04-13T10:00:00Z",
  "updated_at": "2026-04-13T10:00:00Z",
  "resolved_at": null
}
```

### Statuses

```
open → investigating → fix_proposed → fixing → fixed → verified
                                              → regressed → open
open → dismissed
```

### Storage

**Option A: JSON file** — a `debugger-issues.json` in the agent's
workspace. Simplest, no schema changes, debugger reads/writes it
via file tools.

**Option B: Agent-plane store** — a new `IssueStore` with a
`sql_issues` table. Proper persistence, queryable, survives
agent redeployments.

**Option C: Conversation-based** — issues stored as items in a
dedicated "debugger" conversation. The debugger reads its own
conversation history to reconstruct the issue list. No new storage
needed, but querying is awkward.

**Recommendation:** Option A for v1 (JSON file). Upgrade to
Option B when the issue count justifies it.

---

## UX: Terminal REPL

### Commands

| Command | Action |
|---|---|
| `/debug` | Open a chat session with the debugger |
| `/debug analyze` | Analyze recent conversations for issues |
| `/issues` | List all open issues with severity and count |
| `/issues 3` | Show details of issue #3 |
| `/fix 3` | Ask debugger to fix issue #3 |
| `/fix` | Fix the most critical open issue |
| `/dismiss 3` | Dismiss issue #3 (not a real problem) |

### Toolbar integration

The REPL toolbar shows the issue count:

```
── archer · ready · 3 issues  esc cancel · ctrl+c exit ──
```

The count updates when the debugger runs analysis.

### Chat with the debugger

`/debug` switches the REPL to talking to the debugger agent.
The debugger has full context on all conversations and issues:

```
/debug

 ◆ debugger
   I've found 3 issues across 12 conversations:

   #1 [high] Hallucinates v2 API endpoints (3 conversations)
   #2 [medium] Doesn't know about the new auth flow (2 corrections)
   #3 [low] Tone too formal for casual questions

   Want me to fix any of these?

 ❯ tell me more about #1

 ◆ debugger
   In conversations conv_abc, conv_def, and conv_ghi, users asked
   about v2 API routes. The agent responded with invented endpoints:
   - /v2/responses (doesn't exist)
   - /v2/conversations/stream (doesn't exist)

   The agent's AGENTS.md doesn't mention API versioning at all.
   The real API only has /v1/ routes.

   Root cause: missing documentation in AGENTS.md.

   Proposed fix: Add a "API Versioning" section to AGENTS.md:
   "There is only one API version: /v1/. There are no /v2/ routes."

   Apply this fix?

 ❯ yes

 ◆ debugger
   ⏵ Edit(AGENTS.md)
   ⏵ validate_agent()
   ⏵ redeploy(archer)
   ✓ Fixed. Issue #1 → status: fixed.
   I'll check the next 5 conversations for regression.

 ❯ /debug exit
```

`/debug exit` or `/back` returns to the main agent chat.

### Fix workflow

```
1. Debugger identifies issue → creates entry (status: open)
2. User reviews → /fix 3 or "fix #1" in chat
3. Debugger analyzes root cause:
   - Reads AGENTS.md, config.yaml, tools, skills
   - Reads the source conversations
   - Identifies what's wrong and why
   (status: investigating)
4. Debugger proposes fix:
   - Shows a diff or description of what to change
   - Explains why this should fix it
   (status: fix_proposed)
5. User approves → debugger applies:
   - Edits files
   - Validates spec
   - Redeploys agent
   (status: fixing → fixed)
6. Debugger monitors future conversations:
   - If the issue recurs → status: regressed → re-opens
   - If 5+ conversations pass without issue → status: verified
```

---

## UX: Web UI

### Layout

Split-pane or sidebar:

```
┌─────────────────────────────────┬──────────────────────┐
│                                 │ Debugger         (3) │
│   Main Agent Chat               │                      │
│   or Conversation List          │ #1 HIGH              │
│                                 │ Hallucinates v2 API  │
│                                 │ 3 conversations      │
│                                 │ [Fix] [Dismiss]      │
│                                 │                      │
│                                 │ #2 MEDIUM            │
│                                 │ Missing auth flow    │
│                                 │ 2 corrections        │
│                                 │ [Fix] [Dismiss]      │
│                                 │                      │
│   ❯ hello                       │ ─────────────────── │
│                                 │ Chat with debugger:  │
│   ◆ archer                      │                      │
│     Hi! How can I help?         │ ❯ fix #1             │
│                                 │                      │
└─────────────────────────────────┴──────────────────────┘
```

### Issue cards

Each issue is a card with:
- Severity badge (color-coded)
- Title (auto-generated)
- Source count ("3 conversations")
- Quick actions: [View] [Fix] [Dismiss]

**[View]** opens the source conversation with the problematic
messages highlighted.

**[Fix]** opens a chat with the debugger in the sidebar, focused
on that issue.

**[Dismiss]** marks the issue as not a real problem.

### Conversation annotations

When viewing a conversation, the debugger can annotate messages
that contributed to an issue:

```
 user: What's the v2 API endpoint for streaming?

 assistant: The v2 streaming endpoint is /v2/responses/stream...
            ⚠️ Issue #1: This endpoint doesn't exist

 user: That doesn't work. There's no /v2.

 assistant: I apologize...
            ⚠️ User correction detected
```

---

## Architecture

### The debugger is a separate agent

Deployed alongside the main agent on the same server. Independent
lifecycle — if the main agent breaks, the debugger still works.

```
agent-plane server
├── archer (main agent)
│   ├── config.yaml
│   ├── AGENTS.md
│   └── tools/
└── archer-debugger (companion)
    ├── config.yaml
    ├── AGENTS.md
    └── tools/
        ├── analyze_conversations.py
        ├── manage_issues.py
        ├── edit_agent.py
        └── redeploy_agent.py
```

The debugger agent's name follows a convention:
`{main_agent_name}-debugger`.

### Debugger tools

| Tool | Description |
|---|---|
| `list_conversations` | Fetch recent conversations via API |
| `get_conversation_items` | Read a specific conversation's messages |
| `search_conversations` | Find conversations matching keywords or patterns |
| `analyze_conversation` | Run sentiment/failure analysis on one conversation |
| `create_issue` | Create a new issue in the tracker |
| `update_issue` | Update issue status, add root cause, proposed fix |
| `list_issues` | List issues filtered by status/severity |
| `get_issue` | Get full details of one issue |
| `dismiss_issue` | Mark an issue as not a problem |
| `read_agent_file` | Read the main agent's AGENTS.md, config.yaml, etc. |
| `edit_agent_file` | Modify the main agent's files |
| `validate_agent` | Run spec validation on the main agent |
| `redeploy_agent` | Tar + re-upload the main agent bundle |

### Debugger instructions (AGENTS.md)

```markdown
You are a debugging companion for the "{agent_name}" agent.

Your job:
1. Analyze conversations for problems (hallucinations, errors,
   user frustration, instruction violations)
2. Track issues with severity and evidence
3. When asked, propose and apply fixes

Rules:
- Never modify the agent without explicit user approval
- Always explain your reasoning before proposing a fix
- Propose minimal changes — don't rewrite everything
- After fixing, monitor for regression
- Link every issue to specific conversations and messages
- When analyzing, check the agent's AGENTS.md to understand
  what it's supposed to do vs what it actually does
```

---

## How the Debugger Watches

### Option 1: On-demand analysis

The debugger only analyzes when the user asks:
```
/debug analyze
```
or
```
/debug analyze --since 24h
```

The debugger fetches recent conversations via the API, reads each
one, and runs analysis. Creates issues for any problems found.

**Pros:** Simplest. No background infrastructure. User controls when
analysis runs.

**Cons:** Issues aren't found until the user asks. No real-time
alerting.

### Option 2: Periodic polling

A background task (cron job or agent-plane scheduled trigger) runs
the debugger analysis every N minutes:

```yaml
# Scheduled trigger
schedule: "*/30 * * * *"  # Every 30 minutes
agent: archer-debugger
input: "Analyze conversations from the last 30 minutes. Create issues for any problems found."
```

The debugger reads new conversations since its last run and creates
issues. The REPL toolbar shows the count.

**Pros:** Issues found automatically. Near-real-time detection.

**Cons:** Requires scheduled triggers (agent-plane already has this).
LLM cost for each analysis run.

### Option 3: Post-response hook

Agent-plane fires a hook after each conversation completes. The
debugger receives the conversation ID and analyzes it:

```json
{
  "hooks": {
    "PostResponse": [{
      "agent": "archer-debugger",
      "input": "Analyze conversation {conversation_id}"
    }]
  }
}
```

**Pros:** Real-time analysis of every conversation. No polling delay.

**Cons:** Requires a new hook type in agent-plane (PostResponse
doesn't exist yet). LLM cost per conversation.

### Option 4: Streaming tap

The debugger subscribes to the main agent's SSE stream in real-time.
It sees every token as it's generated and can flag issues mid-
conversation.

**Pros:** True real-time. Could even intervene mid-conversation
("the agent is about to hallucinate").

**Cons:** Requires new infrastructure (stream tapping). High
complexity. Unclear if mid-conversation intervention is desirable.

### Recommendation

**Start with Option 1** (on-demand). Add Option 2 (periodic polling)
when the user validates the concept. Option 3 is the long-term
goal. Option 4 is future/research.

---

## The Redeploy Tool

The debugger needs to redeploy the main agent after applying fixes.
This is a tool that:

1. Reads the main agent's directory path
2. Tars it into a bundle
3. Deletes the old agent registration
4. Uploads the new bundle
5. Returns success/failure

```python
def redeploy_agent(agent_name: str, agent_dir: str) -> str:
    """Tar the agent directory and re-upload to the server."""
    bundle = tar_directory(agent_dir)
    # Delete old registration
    client.delete(f"/api/agents/{agent_id}")
    # Upload new bundle
    resp = client.post("/api/agents", files={"bundle": bundle})
    return f"Redeployed {agent_name}: {resp.json()['id']}"
```

**Validation before redeploy:** The debugger should ALWAYS run
`validate_agent` before redeploying. If validation fails, the fix
is rejected and the issue stays open.

**Rollback:** Keep the previous bundle (or git commit) so the
operator can revert if the fix makes things worse.

---

## Regression Monitoring

After applying a fix, the debugger monitors future conversations
for the same issue:

1. Fix applied → issue status: `fixed`
2. Debugger checks next N conversations for the same pattern
3. If the issue recurs → status: `regressed`, severity bumps up
4. If N conversations pass clean → status: `verified`

The monitoring happens during the normal analysis cycle (on-demand
or polling). The debugger remembers which issues are in the
"monitoring" phase and specifically checks for them.

---

## Multi-Agent Considerations

In a multi-agent setup (parent + sub-agents), the debugger should:

- Analyze sub-agent conversations too (they have their own
  conversation IDs with `kind: "sub_agent"`)
- Attribute issues to the correct agent (parent vs sub-agent)
- Fix the right agent's files (not the parent when the sub-agent
  is broken)
- Understand the delegation pattern (issue might be in the parent's
  delegation instructions, not the sub-agent)

---

## REPL Integration

### Switching between agents

The REPL needs to support switching between the main agent and
the debugger mid-session:

```
 ❯ hello                     ← talking to archer

 ◆ archer
   Hi!

 ❯ /debug                    ← switch to debugger

 ◆ debugger
   3 open issues. Want a summary?

 ❯ yes
   ...

 ❯ /back                     ← switch back to archer

 ❯ hello again

 ◆ archer
   Welcome back!
```

Implementation: the REPL maintains two sessions (one per agent).
`/debug` switches the active session. `/back` restores the
previous one. The `on_input` handler routes to the active session.

### Issue count in toolbar

The toolbar shows the issue count from the JSON file:

```
── archer · ready · 3 issues  esc cancel · ctrl+c exit ──
```

The count is read from the issue tracker file. Updated when the
debugger runs analysis or when issues are fixed/dismissed.

---

## Implementation Order

### Phase 1: Manual debugging (v1)

1. **Debugger agent spec** — config.yaml + AGENTS.md with analysis
   instructions
2. **Issue tracking tools** — create/list/update/dismiss backed by
   JSON file
3. **Conversation analysis tools** — list_conversations,
   get_conversation_items, analyze_conversation
4. **`/debug` command** in the REPL — switch to debugger chat
5. **`/issues` command** — list issues from the JSON file
6. **`/fix` command** — tell debugger to fix an issue

### Phase 2: Automated detection

7. **`/debug analyze` command** — on-demand batch analysis
8. **Periodic polling** via scheduled triggers
9. **Regression monitoring** after fixes
10. **Issue count in toolbar**

### Phase 3: Self-healing

11. **`redeploy` tool** — debugger can apply and deploy fixes
12. **Validation before redeploy** — spec validator gate
13. **Rollback support** — keep previous bundles
14. **Auto-fix mode** — debugger fixes critical issues without
    asking (opt-in, with notification)

### Phase 4: Web UI

15. **Issue sidebar** in the web chat UI
16. **Conversation annotations** — highlight problematic messages
17. **Fix approval workflow** — review diffs before applying
18. **Regression dashboard** — track fix effectiveness over time

---

## Not Yet (Future)

- Real-time stream tapping (Option 4)
- A/B testing fixes (deploy fix to subset of conversations)
- Cross-agent issue correlation (same issue across multiple agents)
- Automatic severity escalation (Slack/email alerts for critical)
- Issue deduplication (merge similar issues)
- Fix templates (common patterns: "add to instructions", "add tool",
  "change model")
- User feedback integration (thumbs up/down buttons in chat UI
  that directly create issues)
