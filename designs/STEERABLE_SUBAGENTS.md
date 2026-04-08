# Steerable Sub-Agent Collection

## Problem

The parent agent cannot process steering messages while waiting for
sub-agents. Both `collect_sub_agents` (explicit LLM call) and
auto-collect (loop-level guard before completion) block inside
`wait_sync`, a `time.sleep(0.5)` polling loop in the task store.
Steering messages are only checked at the agent loop boundary — after
tool execution completes — so a steering message sent during collect
sits in the conversation store unprocessed until all sub-agents finish.

This means a user cannot redirect, cancel, or provide new instructions
to the parent while it's waiting for slow sub-agents. The delay is
unbounded — if a sub-agent runs for minutes, the steering message is
invisible for minutes.

### What exists

| Component | File | Status |
|-----------|------|--------|
| `SpawnTool` | `tools/builtins/spawn.py:57` | Non-blocking — returns IDs immediately |
| `CollectTool` | `tools/builtins/spawn.py:131` | Blocks in `wait_sync` until all sub-agents terminal |
| `_collect_all()` | `tools/builtins/spawn.py:461` | Iterates `wait_sync` per task, splitting timeout budget |
| `wait_sync()` | `stores/task_store/sqlalchemy_store.py:440` | `time.sleep(0.5)` poll loop inside `@step()` |
| `_auto_collect_sub_agents()` | `runtime/workflow.py:1790` | Calls `_collect_all(timeout=None)` — blocks indefinitely |
| `_sync_history()` | `runtime/workflow.py:2067` | Checks for steering — only runs at loop top |
| `_sync_steered_after_tools()` | `runtime/workflow.py:2016` | Checks for steering — only runs after tool execution returns |
| `spawned_ids` / `collected_ids` | `runtime/workflow.py:2611` | Track which sub-agents need auto-collect |

### What's wrong

1. **`CollectTool` blocks opaquely.** `_collect_all` → `wait_sync` →
   `time.sleep(0.5)` inside a `@step()`. The agent loop is frozen.
   Steering messages delivered via `try_deliver` go into the
   conversation store but the loop never checks because it's stuck
   inside a tool call.

2. **Auto-collect blocks opaquely.** When the LLM produces a final
   response with uncollected sub-agents, `_auto_collect_sub_agents`
   calls `_collect_all(timeout=None)`. Same blocking problem — the
   loop is frozen at the completion gate.

3. **No way to check sub-agent progress.** The LLM can only
   block-wait via `collect_sub_agents`. It cannot peek at sub-agent
   status, cancel a sub-agent, or get partial results without
   blocking.

---

## Survey: How Other Frameworks Handle This

### OpenAI Agents SDK

No collect tool. Sub-agents are invoked as regular tool calls via
`agent.as_tool()`. The parent's agent loop blocks on the tool call
like any other tool. No mechanism for interrupting while waiting.
Cancellation is coarse-grained (cancel the entire run).

### LangGraph

No explicit collect. Parallel work uses the `Send` API — the runtime
implicitly waits for all branches in a "superstep" before advancing
to the next node. Interruptions use explicit `interrupt()` calls
placed *before* tool execution (checkpoint-based). Cannot interrupt
mid-tool — the node re-executes from the beginning on resume.

### OmniAgents (internal experimental framework)

Non-blocking pattern with explicit control:

1. **`sys_session_send`** — launches sub-agent in background, returns
   a handle immediately.
2. **`sys_read_inbox`** — parent reads completed results from a queue
   (pull-based).
3. **`sys_session_peek`** — inspect sub-agent progress without
   blocking.
4. **`sys_session_cancel_turn`** — interrupt a running sub-agent.
5. **`_wake_event`** — asyncio Event wakes the parent when a
   sub-agent completes. A framework notice is injected:
   *"[SYSTEM] There are N unread inbox items available."*

The parent never blocks on sub-agents. It sleeps in
`_wait_for_turn_trigger()` which awaits **either** a user message
**or** the wake event.

### Takeaway

The pattern that handles steering correctly (OmniAgents) never blocks
inside a tool. Spawn is non-blocking, results are delivered via
notification, and the parent loop waits on **multiple signals**
(sub-agent completion, user message, timeout) at the same level.

---

## Design

### Principle

**No tool ever blocks waiting for sub-agents.** All waiting happens at
the agent loop level, where steering is already checked every
iteration. This means:

- Steering messages are processed within one poll interval (~0.5s)
  regardless of sub-agent state.
- The LLM can check, cancel, or ignore sub-agents at any time.
- Auto-collect at the completion gate is interruptible by steering.

### Overview

Replace the blocking `collect_sub_agents` tool with three non-blocking
tools and move all sub-agent waiting to the agent loop:

| Before | After |
|--------|-------|
| `collect_sub_agents` (blocks in `wait_sync`) | **Removed** |
| N/A | `check_sub_agents` — non-blocking status read |
| N/A | `cancel_sub_agent` — stop a running sub-agent |
| `_auto_collect_sub_agents` (blocks in `_collect_all`) | Loop-level poll with steering checks |
| N/A | Completion notifications at loop top |

---

## New Tools

### `check_sub_agents` (replaces `collect_sub_agents`)

Non-blocking. Checks the specified sub-agents (the caller chooses
which to check — it need not check all spawned sub-agents at once).
Reads current state via `get_sync` and the sub-agent's conversation
via `conv_store.list_items`. Returns immediately.

- **Completed sub-agents:** `output` has the extracted final text,
  `recent_activity` is null.
- **In-progress sub-agents:** `output` is null, `recent_activity`
  has the last N conversation items in a compact format (content
  truncated to avoid blowing up the parent's context).
- **Failed/cancelled:** `output` has an error message,
  `recent_activity` has the tail so the LLM can see what went wrong.

```python
_ACTIVITY_TAIL = 5
_ACTIVITY_MAX_CHARS = 300


class CheckSubAgentsTool(Tool):
    """
    Non-blocking status check for specified sub-agents.

    Checks only the sub-agents whose response IDs are passed
    in — the caller need not check all spawned sub-agents at
    once. Returns immediately with each sub-agent's current
    status. Completed sub-agents include their extracted
    output text. Non-terminal and non-completed terminal
    sub-agents include recent conversation activity so the
    parent can see what the sub-agent is doing.
    """

    @classmethod
    def name(cls) -> str:
        return "check_sub_agents"

    def invoke(self, arguments: str, ctx: ToolContext) -> str:
        args = _parse_check_args(arguments)
        if isinstance(args, str):
            return args

        response_ids: list[str] = args["response_ids"]
        task_store = get_task_store()
        conv_store = get_conversation_store()

        results: list[dict[str, Any]] = []
        for tid in response_ids:
            task = task_store.get_sync(tid)
            if task is None:
                results.append({
                    "response_id": tid,
                    "status": "not_found",
                    "output": None,
                    "recent_activity": None,
                })
                continue
            results.append(
                _task_to_check_result(task, conv_store),
            )
        return json.dumps({"results": results})
```

**`_task_to_check_result`** builds the result with activity:

```python
def _task_to_check_result(
    task: Task,
    conv_store: ConversationStore,
) -> dict[str, Any]:
    """
    Build a check result for a single sub-agent.

    Completed sub-agents get extracted output text and no
    activity. All others get recent conversation activity
    so the parent LLM can see what the sub-agent is doing
    (or what it was doing when it failed).

    :param task: The sub-agent's task.
    :param conv_store: For fetching recent conversation items.
    :returns: Result dict with response_id, agent_name,
        status, output, and recent_activity fields.
    """
    if task.status == "completed" and task.output:
        return {
            "response_id": task.id,
            "agent_name": task.agent_name,
            "status": task.status,
            "output": _extract_output_text(task.output),
            "recent_activity": None,
        }

    # Non-completed: include recent activity
    activity = _get_recent_activity(
        task.conversation_id, conv_store,
    )
    output = None
    if task.status in TERMINAL_STATUSES:
        output = (
            f"Sub-agent {task.agent_name!r} finished "
            f"with status: {task.status}."
        )

    return {
        "response_id": task.id,
        "agent_name": task.agent_name,
        "status": task.status,
        "output": output,
        "recent_activity": activity,
    }
```

**`_get_recent_activity`** fetches the tail of the conversation in a
compact format:

```python
def _get_recent_activity(
    conversation_id: str,
    conv_store: ConversationStore,
) -> list[dict[str, str]]:
    """
    Fetch the last few conversation items and project them
    into a compact format for the parent LLM.

    :param conversation_id: The sub-agent's conversation ID.
    :param conv_store: For fetching items.
    :returns: List of compact activity dicts.
    """
    page = conv_store.list_items(
        conversation_id,
        limit=_ACTIVITY_TAIL,
        order="desc",
    )
    items = list(reversed(page.data))

    activity: list[dict[str, str]] = []
    for item in items:
        activity.append(_project_activity_item(item))
    return activity


def _project_activity_item(
    item: ConversationItem,
) -> dict[str, str]:
    """
    Project a conversation item into a compact dict.

    - Messages: ``{"role": "assistant", "type": "text",
      "content": "truncated..."}``
    - Tool calls: ``{"role": "assistant", "type": "tool_call",
      "name": "web_search", "args": "truncated..."}``
    - Tool results: ``{"role": "tool", "type": "tool_result",
      "name": "web_search", "content": "truncated..."}``

    All content fields are truncated to ``_ACTIVITY_MAX_CHARS``.
    """
    # Implementation varies by item.type and item.data
    # structure — omitted here for brevity. The key
    # invariant: every content string is capped at
    # _ACTIVITY_MAX_CHARS with a " [truncated]" suffix.
    ...
```

**Schema:**

```json
{
  "type": "function",
  "function": {
    "name": "check_sub_agents",
    "description": "Check the current status of one or more spawned sub-agent tasks. Pass only the response IDs you want to check — you do not need to check all spawned sub-agents at once. Returns immediately with each specified sub-agent's current status, output (if completed), and recent conversation activity (if still running). Does not wait.",
    "parameters": {
      "type": "object",
      "properties": {
        "response_ids": {
          "type": "array",
          "items": { "type": "string" },
          "description": "One or more response IDs returned by spawn_sub_agents to check."
        }
      },
      "required": ["response_ids"]
    }
  }
}
```

No `timeout` parameter — there is nothing to wait for.

**Return format:**

```json
{
  "results": [
    {
      "response_id": "resp_child1",
      "agent_name": "researcher",
      "status": "completed",
      "output": "The findings are...",
      "recent_activity": null
    },
    {
      "response_id": "resp_child2",
      "agent_name": "analyzer",
      "status": "in_progress",
      "output": null,
      "recent_activity": [
        {"role": "assistant", "type": "tool_call", "name": "web_search", "args": "{\"query\": \"RLHF 2025\"}"},
        {"role": "tool", "type": "tool_result", "name": "web_search", "content": "Found 12 results: 1. \"Direct Prefere... [truncated]"},
        {"role": "assistant", "type": "text", "content": "I found several relevant papers. Let me now... [truncated]"}
      ]
    }
  ]
}
```

Completed sub-agents have `output` with extracted text and no
activity (the journey doesn't matter — only the answer). In-progress
and non-completed terminal sub-agents have `recent_activity` with
the last 5 conversation items, each content-truncated to 300 chars.
This gives the parent LLM enough to see what the sub-agent is doing
without overwhelming the context.

### `cancel_sub_agent`

Cancels a running sub-agent. Delegates to `task_store.cancel()`,
which is made sync in this design (see [Store changes](#store-changes)).
The underlying DBOS primitive is a single DB `UPDATE` — non-blocking.
Does not wait for the workflow's `finally` block to run; the
sub-agent will observe the cancellation on its next DBOS checkpoint
and wind down on its own.

```python
class CancelSubAgentTool(Tool):
    """
    Cancel a running sub-agent task.

    Delegates to ``task_store.cancel`` — non-blocking.
    The sub-agent workflow observes the cancellation on its
    next DBOS checkpoint and winds down.
    """

    @classmethod
    def name(cls) -> str:
        return "cancel_sub_agent"

    def invoke(self, arguments: str, ctx: ToolContext) -> str:
        args = json.loads(arguments)
        response_id = args["response_id"]
        task_store = get_task_store()
        task_store.cancel(response_id)
        return json.dumps({
            "status": "cancelled",
            "response_id": response_id,
        })
```

**Schema:**

```json
{
  "type": "function",
  "function": {
    "name": "cancel_sub_agent",
    "description": "Cancel a running sub-agent task. The sub-agent will stop execution and its status will become 'cancelled'.",
    "parameters": {
      "type": "object",
      "properties": {
        "response_id": {
          "type": "string",
          "description": "The response ID of the sub-agent to cancel."
        }
      },
      "required": ["response_id"]
    }
  }
}
```

### Registration

In `ToolManager._register_sub_agent_tools`:

```python
def _register_sub_agent_tools(self) -> None:
    if not self._spec.tools.agents:
        return
    sub_specs = {
        sa.name: sa
        for sa in self._spec.sub_agents
        if sa.name is not None
    }
    self._tools[SpawnTool.name()] = SpawnTool(sub_specs=sub_specs)
    self._tools[CheckSubAgentsTool.name()] = CheckSubAgentsTool()
    self._tools[CancelSubAgentTool.name()] = CancelSubAgentTool()
```

`CollectTool` is no longer registered.

### Schema update for `spawn_sub_agents`

The `spawn_sub_agents` description currently says *"Use
collect_sub_agents() to gather results."* Update to reference
`check_sub_agents`:

```
"Launch one or more sub-agents as independent parallel tasks.
Returns response IDs immediately. You will be notified when
sub-agents complete. Use check_sub_agents() to retrieve results."
```

---

## Loop Changes

### 1. Sub-agent completion notifications

New function, called at the top of each loop iteration after
`_sync_history`:

```python
def _notify_subagent_completions(
    task_id: str,
    conversation_id: str,
    spawned_ids: set[str],
    notified_ids: set[str],
    task_store: TaskStore,
    conv_store: ConversationStore,
    history: list[ConversationItem],
    output_items: list[dict[str, Any]],
) -> str | None:
    """
    Check if any spawned sub-agents have newly reached a
    terminal state and inject a system notice for each.

    :param spawned_ids: All spawned sub-agent response IDs.
    :param notified_ids: Mutable set — updated in place with
        newly-completed IDs to avoid duplicate notifications.
    :returns: The last_seen cursor after persisting notices,
        or None if no new completions detected.
    """
    newly_done: list[Task] = []
    for tid in spawned_ids - notified_ids:
        task = task_store.get_sync(tid)
        if task is not None and task.status in TERMINAL_STATUSES:
            newly_done.append(task)
            notified_ids.add(tid)

    if not newly_done:
        return None

    lines = [
        f"Sub-agent {t.agent_name!r} ({t.id}) finished "
        f"with status: {t.status}"
        for t in newly_done
    ]
    notice_text = (
        "[System: sub-agent completion notice]\n"
        + "\n".join(lines)
        + "\nCall check_sub_agents to retrieve their results."
    )

    new_items = [
        NewConversationItem(
            type="message",
            response_id=task_id,
            data=MessageData(
                role="user",
                content=[
                    {"type": "input_text", "text": notice_text},
                ],
            ),
        ),
    ]
    persisted = _persist_and_stream(
        task_id, conv_store, conversation_id,
        new_items, output_items,
    )
    history.extend(persisted)
    return persisted[-1].id
```

This is the equivalent of OmniAgents' `_wake_event` + framework
notice, implemented as a poll. The loop already runs at ~0.5s
granularity (bounded by LLM call latency), so the notification
latency is acceptable.

### 2. Steerable auto-collect

Replace the blocking `_auto_collect_sub_agents` with a loop-level
poll that checks both sub-agent status and steering:

```python
@dataclass
class _SubagentWaitResult:
    """
    Result of waiting for sub-agents at the auto-collect gate.

    :param steering: True if a steering message interrupted
        the wait, False if all sub-agents completed.
    :param last_seen: Updated conversation cursor.
    """

    steering: bool
    last_seen: str | None


def _wait_for_subagents_or_steering(
    task_id: str,
    conversation_id: str,
    response_ids: list[str],
    task_store: TaskStore,
    conv_store: ConversationStore,
    last_seen: str | None,
    history: list[ConversationItem],
    output_items: list[dict[str, Any]],
) -> _SubagentWaitResult:
    """
    Poll until all sub-agents finish OR a steering message
    arrives.

    Called from the auto-collect gate when the LLM wants to
    produce a final response but sub-agents are still running.
    Checks both sub-agent terminal status and conversation
    store for new items every poll interval.

    :param response_ids: Sub-agent response IDs to wait for.
    :param last_seen: Current conversation cursor.
    :returns: A result indicating whether all sub-agents
        completed or steering interrupted the wait.
    """
    pending = set(response_ids)

    while True:
        # 1. Check sub-agent status
        for tid in list(pending):
            task = task_store.get_sync(tid)
            if (
                task is not None
                and task.status in TERMINAL_STATUSES
            ):
                pending.discard(tid)

        if not pending:
            # All done — collect results and inject
            results = _collect_results_sync(
                response_ids, task_store,
            )
            new_last_seen = _inject_auto_collect_message(
                task_id, conversation_id, results,
                conv_store, history, output_items,
            )
            return _SubagentWaitResult(
                steering=False, last_seen=new_last_seen,
            )

        # 2. Check for steering messages
        new_last_seen = _sync_history(
            conv_store, conversation_id,
            last_seen, history,
        )
        if new_last_seen != last_seen:
            return _SubagentWaitResult(
                steering=True, last_seen=new_last_seen,
            )

        # 3. Sleep and retry
        time.sleep(_SUBAGENT_POLL_S)


def _collect_results_sync(
    response_ids: list[str],
    task_store: TaskStore,
) -> list[dict[str, str]]:
    """
    Read final results for sub-agents that are already in
    terminal state. No waiting — caller must ensure all
    tasks are terminal before calling.
    """
    results: list[dict[str, str]] = []
    for tid in response_ids:
        task = task_store.get_sync(tid)
        results.append(_task_to_result(task))
    return results
```

`_inject_auto_collect_message` is the same as today's
`_auto_collect_sub_agents` but without the `_collect_all` call — it
takes pre-built results and persists the system message.

### 3. Updated agent loop

New tracking set and updated paths (showing only changed sections):

```python
    spawned_ids: set[str] = set()
    collected_ids: set[str] = set()
    notified_ids: set[str] = set()  # NEW

    for iteration in range(max_iterations):
        elapsed = time.monotonic() - start_time
        if elapsed >= execution_timeout:
            return _handle_execution_timeout(...)

        # ── Sync steering ──────────────────────────
        last_seen = _sync_history(
            conv_store, conversation_id,
            last_seen, history,
        )

        # ── NEW: Notify sub-agent completions ──────
        if spawned_ids - notified_ids:
            notice_cursor = _notify_subagent_completions(
                task_id, conversation_id,
                spawned_ids, notified_ids,
                task_store, conv_store,
                history, output_items,
            )
            if notice_cursor is not None:
                last_seen = notice_cursor

        # ── LLM call ──────────────────────────────
        llm_resp = _executor_turn_with_compaction(...)

        _emit_native_tool_items(...)

        if not _has_tool_calls(llm_resp):
            # ── Final response path ───────────────
            uncollected = spawned_ids - collected_ids
            if uncollected:
                # CHANGED: Non-blocking wait with
                # steering check
                wait_result = \
                    _wait_for_subagents_or_steering(
                        task_id, conversation_id,
                        list(uncollected),
                        task_store, conv_store,
                        last_seen, history,
                        output_items,
                    )
                if wait_result.steering:
                    # Steering arrived — abort
                    # completion, re-run LLM
                    last_seen = wait_result.last_seen
                    continue
                # All sub-agents done, results injected
                collected_ids.update(uncollected)
                notified_ids.update(uncollected)
                last_seen = wait_result.last_seen
                continue

            result = _handle_final_response(...)
            # ... rest unchanged ...

        # ── Tool execution path ───────────────────
        # (unchanged — tools are now non-blocking)
        pre_tool_last_seen = last_seen
        handle_result = _handle_tool_calls(...)
        # ... rest unchanged ...

        _track_spawn_collect(
            output_items, spawned_ids, collected_ids,
        )

        last_seen = _sync_steered_after_tools(...)
```

### 4. Updated `_track_spawn_collect`

Replace `CollectTool.name()` with `CheckSubAgentsTool.name()`. Only
mark sub-agents as collected when `check_sub_agents` returns them in a
terminal status:

```python
elif name == CheckSubAgentsTool.name():
    for r in parsed.get("results", []):
        rid = r.get("response_id", "")
        status = r.get("status", "")
        if rid and status in TERMINAL_STATUSES:
            collected_ids.add(rid)
```

This ensures that calling `check_sub_agents` on an in-progress
sub-agent does not mark it as collected — auto-collect will still
trigger at the completion gate if the LLM tries to finish without
all sub-agents done.

---

## Design Decisions

### Store changes

`TaskStore.cancel` becomes sync (was async for no good reason). The
underlying DBOS primitive (`cancel_workflow`) is sync — a single DB
`UPDATE`. The method calls `cancel_workflow()` + `get_sync()`, both
sync. The three async call sites in `responses.py` drop their
`await`. No new methods, no new DB columns, no changes to
`ToolContext`.

### Notifications are status-only

Completion notifications include the sub-agent name and status but not
the full output text. The LLM calls `check_sub_agents` to retrieve
output. This keeps notifications lightweight and avoids bloating the
context window with potentially large sub-agent outputs that the LLM
may not need immediately.

### Auto-collect remains mandatory

The LLM cannot produce a final response with uncollected sub-agents.
The auto-collect gate ensures all sub-agent results are incorporated
before the parent completes. The change is that the gate is now
interruptible by steering — the LLM can be redirected mid-wait.

### `collect_sub_agents` is deleted, not aliased

Clean break — all collect-related code is deleted from the codebase:

**Delete from `tools/builtins/spawn.py`:**
- `CollectTool` class (entire class)
- `_collect_all()` function
- `_build_collect_schema()` function
- `_parse_collect_args()` function
- `_remaining_timeout()` helper (only used by `_collect_all`)
- `_WAIT_SYNC_POLL_S` constant comment referencing CollectTool
- Update `SpawnTool` schema description: replace
  `"Use collect_sub_agents() to gather results."` with
  `"Use check_sub_agents() to retrieve results."`

**Delete from `runtime/workflow.py`:**
- `_auto_collect_sub_agents()` function (replaced by
  `_wait_for_subagents_or_steering`)
- Remove `CollectTool` from import at line 100
- Update `_track_spawn_collect` to reference
  `CheckSubAgentsTool` instead of `CollectTool`

**Update in `tools/manager.py`:**
- Remove `CollectTool` import (line 20)
- Remove `CollectTool` registration in
  `_register_sub_agent_tools` (line 148)

**Update in `tools/builtins/__init__.py`:**
- Remove `CollectTool` from imports (line 28) and
  `__all__` (line 42)

**Update in `stores/task_store/sqlalchemy_store.py`:**
- Update `wait_sync` docstring: remove reference to
  `collect_sub_agents` (line 449)

Existing agent specs and prompts that reference
`collect_sub_agents` must update to `check_sub_agents`. Since
sub-agent support is not yet GA and there are no external consumers,
backward compatibility is not a concern.

### Polling frequency

The notification check iterates `spawned_ids - notified_ids` and
calls `get_sync` per task. For a handful of sub-agents (typical case),
this is negligible — a few DB reads per loop iteration. If we ever
support 100+ concurrent sub-agents, a batch `get_many_sync` method
on the task store would be needed. Not necessary now.

The auto-collect wait loop uses `_SUBAGENT_POLL_S` (same 0.5s
interval as `wait_sync`). This bounds the worst-case steering
detection latency to 0.5s.

---

## End-to-End Flow

### Happy path: sub-agents complete before LLM finishes

```
1. LLM calls spawn_sub_agents({agents: [{name: "researcher", input: "..."}]})
   → returns {response_ids: ["resp_1"]}

2. LLM makes other tool calls (its own work)

3. [Loop iteration top] _notify_subagent_completions detects resp_1
   completed → injects system message:
   "[System: sub-agent completion notice]
    Sub-agent 'researcher' (resp_1) finished with status: completed
    Call check_sub_agents to retrieve their results."

4. LLM sees notification → calls check_sub_agents({response_ids: ["resp_1"]})
   → gets {results: [{status: "completed", output: "..."}]}

5. LLM produces final response. No uncollected sub-agents.
   → _handle_final_response proceeds normally.
```

### Steering during auto-collect

```
1. LLM spawns sub-agents, does some work, tries to finish.

2. Auto-collect gate: uncollected sub-agents exist.
   → _wait_for_subagents_or_steering starts polling.

3. User sends steering message: "Actually, cancel everything."
   → try_deliver appends to conversation store.

4. _wait_for_subagents_or_steering detects new item via _sync_history.
   → returns _SubagentWaitResult(steering=True, ...)

5. Loop continues. LLM sees steering message on next iteration.
   → LLM calls cancel_sub_agent for each, then responds to user.
```

### Steering during mid-execution wait

```
1. LLM spawns sub-agents. Sub-agents are slow.

2. LLM makes more tool calls. All tools return quickly
   (check_sub_agents is non-blocking). Loop keeps iterating.

3. Each iteration: _sync_history checks steering,
   _notify_subagent_completions checks sub-agents.

4. User sends "Focus on the first sub-agent only."
   → _sync_history picks it up on next iteration.
   → LLM sees it, calls cancel_sub_agent on the others.
```

---

## What Changes, What Doesn't

| Component | Before | After |
|-----------|--------|-------|
| `spawn_sub_agents` tool | Non-blocking | **No change** (description updated) |
| `collect_sub_agents` tool | Blocks in `wait_sync` | **Removed** |
| `check_sub_agents` tool | N/A | **New** — non-blocking `get_sync` per task |
| `cancel_sub_agent` tool | N/A | **New** — `task_store.cancel` |
| `_collect_all()` | Blocking poll loop | **Removed** |
| `_auto_collect_sub_agents()` | Calls `_collect_all` | **Replaced** by `_wait_for_subagents_or_steering` |
| `wait_sync()` | Used by collect | **No longer called from spawn.py** (remains for other uses) |
| Loop iteration top | `_sync_history` only | **+ `_notify_subagent_completions`** |
| `_track_spawn_collect()` | Tracks spawn + collect | **Updated** to track spawn + check (terminal only) |
| `spawned_ids` / `collected_ids` | Existing | **No change** |
| N/A | N/A | **New**: `notified_ids` tracking set |
| `TaskStore.cancel` | Async | **Made sync** (underlying DBOS primitive is sync) |
| `ToolContext` | — | **No changes** |
| DB schema | — | **No changes** |

---

## Open Questions

1. **Should notifications include output text for short outputs?**
   Current design: status-only, LLM calls `check_sub_agents` for
   output. Alternative: include output inline when it's under a
   threshold (e.g. 500 chars). Reduces tool call round-trips at the
   cost of notification size.

2. **Should `check_sub_agents` support a `wait` flag?** A hybrid
   where `check_sub_agents({response_ids: [...], wait: true})`
   blocks briefly (e.g. 5s max) then returns. This would let the
   LLM opt into short waits without a full loop-level mechanism.
   Current design says no — keep tools non-blocking, keep waiting
   at the loop level.

3. **Execution timeout interaction.** The auto-collect wait loop
   (`_wait_for_subagents_or_steering`) should also respect the
   parent's execution timeout. If the timeout expires while waiting
   for sub-agents, treat it the same as the existing timeout path.

4. **Notification deduplication across turns.** `notified_ids` is
   in-memory for the current execution. If the parent is a
   multi-turn conversation, a sub-agent spawned in turn 1 and
   completed in turn 2 would be re-notified in turn 2 (since
   `notified_ids` resets). This is acceptable — the notification
   is idempotent and the LLM can handle seeing it again.

---

## Test Plan

### Unit tests — `tests/tools/builtins/test_spawn.py`

Extend the existing test file. These test the new tool helpers and
result formatting in isolation — no stores, no workflow.

**`_task_to_check_result`:**

- `test_check_result_completed_has_output_no_activity` — Completed
  task with output returns extracted text in `output`, `null`
  `recent_activity`.
- `test_check_result_completed_empty_output` — Completed task with
  no output_text items returns empty string output, `null` activity.
- `test_check_result_in_progress_has_activity_no_output` — In-progress
  task returns `null` output and a `recent_activity` list from the
  conversation store.
- `test_check_result_failed_has_both` — Failed task returns error
  message in `output` and `recent_activity` showing what happened.
- `test_check_result_cancelled_has_both` — Same for cancelled.

**`_get_recent_activity`:**

- `test_recent_activity_returns_last_n_items` — Conversation with 20
  items returns only the last `_ACTIVITY_TAIL` items, in
  chronological order.
- `test_recent_activity_truncates_long_content` — Message text
  exceeding `_ACTIVITY_MAX_CHARS` is truncated with `[truncated]`
  suffix.
- `test_recent_activity_empty_conversation` — Returns empty list.

**`_project_activity_item`:**

- `test_project_text_message` — Assistant text message projected as
  `{"role": "assistant", "type": "text", "content": "..."}`.
- `test_project_tool_call` — Function call projected with `name` and
  truncated `args`.
- `test_project_tool_result` — Function call output projected with
  `name` and truncated `content`.
- `test_project_user_message` — User message projected correctly.

**`_track_spawn_collect` update:**

- `test_track_check_sub_agents_terminal_marks_collected` — Terminal
  status in `check_sub_agents` output adds to `collected_ids`.
- `test_track_check_sub_agents_in_progress_not_collected` —
  In-progress status does NOT add to `collected_ids`.

### Unit tests — `tests/stores/test_task_store.py`

Extend the existing test file for the `cancel` signature change.

- `test_cancel_is_sync_and_returns_task` — `cancel()` is callable
  without `await`, returns a `Task` with status `"cancelled"`.
- `test_cancel_already_completed_is_noop` — Cancelling a completed
  task leaves status as `"completed"` (DBOS skips already-terminal
  workflows).

### Integration tests — `tests/runtime/test_workflow.py`

These test the new loop-level functions with real stores but a mock
LLM. Extend the existing test file.

**`_notify_subagent_completions`:**

- `test_notify_detects_newly_completed_subagent` — Spawn a sub-agent
  task, set it to completed in the store. Call
  `_notify_subagent_completions`. Assert a system message was
  persisted to the conversation and the sub-agent ID was added to
  `notified_ids`.
- `test_notify_skips_already_notified` — Call twice with the same
  completed sub-agent. Assert only one notification message
  persisted (deduplication via `notified_ids`).
- `test_notify_skips_in_progress` — Sub-agent still running. Assert
  no notification, `notified_ids` unchanged.
- `test_notify_multiple_completions_single_message` — Two sub-agents
  complete between iterations. Assert both appear in a single
  notification message.

**`_wait_for_subagents_or_steering`:**

- `test_wait_returns_when_all_subagents_terminal` — Create two
  sub-agent tasks, set both to completed. Call
  `_wait_for_subagents_or_steering`. Assert
  `steering=False`, results injected into history.
- `test_wait_returns_on_steering_message` — Create a sub-agent task
  (still running). Deliver a steering message to the conversation
  via `conv_store.append` (simulating `try_deliver`). Call
  `_wait_for_subagents_or_steering`. Assert `steering=True`,
  steering message appears in history.
- `test_wait_steering_beats_slow_subagent` — Sub-agent never
  completes, but steering message arrives. Assert the function
  returns promptly with `steering=True` (does not block until
  sub-agent finishes).
- `test_wait_injects_auto_collect_message` — All sub-agents
  complete. Assert the injected system message contains the
  sub-agent results JSON and is persisted to the conversation.

**Full loop integration (mock LLM):**

- `test_loop_notification_triggers_check` — Mock LLM: first call
  returns spawn tool call, second call (after seeing notification)
  returns check_sub_agents tool call, third call returns final
  response. Assert the loop completes, sub-agent results are in
  output, and the notification message is in the conversation.
- `test_loop_auto_collect_with_steering` — Mock LLM: first call
  returns spawn, second call returns final response (triggering
  auto-collect). While auto-collect polls, inject a steering message
  into the conversation. Assert the loop re-runs the LLM with the
  steering message in history instead of completing.
- `test_loop_auto_collect_without_steering` — Same setup but no
  steering arrives. Sub-agent completes during the auto-collect
  wait. Assert auto-collect injects results and the LLM produces
  the final response incorporating them.
- `test_loop_cancel_sub_agent_then_finish` — Mock LLM: spawn,
  then cancel_sub_agent tool call, then final response. Assert
  sub-agent is cancelled, auto-collect does not block on a
  cancelled sub-agent.

### Integration tests — `tests/server/integration/test_routes_responses.py`

Test the `cancel` change at the HTTP layer.

- `test_cancel_response_returns_cancelled_task` — `POST
  /v1/responses/{id}/cancel` returns a response object with
  `status: "cancelled"`. Verify `cancel()` works as sync from
  the async route handler.

### E2E tests — `tests/e2e/test_coder_subagent.py`

Extend the existing e2e test file. These use a real LLM and real
server — they validate the full system works end-to-end.

- `test_subagent_completion_notification_and_check` — Spawn a
  sub-agent via a real LLM. Wait for the parent to receive a
  completion notification and call `check_sub_agents`. Assert the
  parent's final response incorporates the sub-agent's output.
  Verifies the full notification → check → respond flow with a
  real LLM.
- `test_steering_during_subagent_execution` — Spawn a sub-agent
  via a real LLM. While the sub-agent is running, send a steering
  message to the parent (via `PATCH /v1/responses`). Assert the
  parent processes the steering message and responds to it —
  proving steering is not blocked by sub-agent execution. **This
  is the core bug fix validation.**
- `test_cancel_subagent_via_steering` — Spawn a sub-agent. Send a
  steering message telling the parent to cancel it. Assert the
  parent calls `cancel_sub_agent` and produces a response
  acknowledging the cancellation.

---

## Not Yet

- **Batch status check on store.** `get_many_sync(task_ids)` for
  efficient polling with many sub-agents. Add when needed.
- **Sub-agent progress streaming.** Forwarding sub-agent SSE events
  to the parent's stream for real-time progress. Orthogonal to
  steering — can be added independently.
- **Cancellation propagation.** When the parent is cancelled, cancel
  all spawned sub-agents automatically. Currently sub-agents run
  independently with their own execution timeouts.
