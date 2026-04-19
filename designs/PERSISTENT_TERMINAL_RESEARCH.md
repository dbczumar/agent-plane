# Persistent Terminal Environments for AI Agents — Landscape + Plan

**Status**: Decision recorded. All §6 sub-decisions firm.
Implementation deferred.
**Date**: 2026-04-19.
**Scope**: Answers the question "what does it mean to give an agent a
persistent terminal, how are other projects doing it, and how would
this differ from agent-plane's existing `code_sandbox` tool?" Based on
live web research (April 2026) and inspection of `agent_plane/tools/builtins/code_sandbox.py`.

**Decision (§6)**: Replace `code_sandbox` entirely with a PTY-backed
persistent-terminal builtin. No parallel tool, no backwards-compatible
wrapper, no "simple mode" — one path only (Design Principle #2).
- Tool surface (§6.1): three terminal-specific tools — `terminal_run`
  / `terminal_list` / `terminal_close` — addressed by agent-named
  **shells**, default `"default"`, with explicit `synchronous` flag
  and optional `timeout_ms` task lifetime bound. Async invocations
  return task handles routed through the unified task lifecycle
  (`check_task` / `cancel_task` / `list_tasks` from
  `session_model_notes.md`). (Option D in the comparison.)
- Backend (§6.2): `pexpect` + `pyte`.
- Completion detection (§6.3): OSC 633 markers from an injected shell-
  integration snippet.
- Durability tier (§6.4): conversation-scoped (shells outlive individual
  tasks, live with their conversation).
- Sub-agent terminals (§6.6): isolated per conversation.
- Threading & registry integration (§6.9): one module-level accessor;
  all locks, state, and coordination encapsulated inside the registry
  hierarchy. New "conversation-scoped server state" pattern — does
  NOT follow the ToolManager ContextVar idiom.

---

## 1. What "persistent terminal" actually means

The term bundles together several orthogonal capabilities that the
research conflates. Disentangling them is the first step, because the
engineering cost of each is very different.

| Capability | Meaning | Does `code_sandbox` have it? |
|---|---|---|
| **Persistent filesystem** | A workspace dir that survives across tool calls | ✅ Yes — workspace dir is per-conversation |
| **Persistent shell state** | `cd`, `export`, aliases, functions carry across calls | ❌ No — each call is a fresh `bash -c` |
| **Persistent processes** | A `npm run dev` started in one call is still running in the next call | ❌ No — subprocess dies when `communicate()` returns |
| **PTY allocation** | Programs see a real terminal; prompts, ANSI, readline, curses work | ❌ No — `stdin=DEVNULL`, pipe for stdout |
| **Interactive input** | Agent can send keystrokes mid-execution (answer `y/n`, type a password) | ❌ No — `stdin=DEVNULL` |
| **Screen emulation** | In-place updates (progress bars, spinners, vim, htop) render correctly for the agent | ❌ No — raw pipe |
| **Multiple concurrent sessions** | Agent can run `npm run dev` + `pytest --watch` in parallel, poll each | ❌ No — one call at a time |

Different projects pick different subsets. A design that says "add a
persistent terminal" needs to say which of these rows it is buying.
They compose roughly but not exactly: PTY without interactivity is
possible (capture-only); interactivity without screen emulation is
possible (raw stdin); persistent processes without a PTY is possible
(background subprocess). The expensive row is screen emulation.

---

## 2. How `code_sandbox` works today

`agent_plane/tools/builtins/code_sandbox.py`:

```python
self._proc = subprocess.Popen(
    cmd,
    cwd=str(ctx.workspace),
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    stdin=subprocess.DEVNULL,
    env=env,
)
stdout, _ = self._proc.communicate()
```

One `Popen` per `invoke()` call. Command is `["bash", "-c", command]`
or `["node", _SRT_WRAP_PATH, config_json, wrapped]` when srt sandbox
is on. No PTY, no stdin, no persistent shell, no concurrent processes.

The *only* thing that persists is:

1. The **workspace directory** itself (`ctx.workspace`) — files the
   agent writes are still there next call.
2. `PIP_TARGET`, `npm_config_prefix` point into the workspace so
   installed packages survive.

Everything else — cwd, env vars, running processes, shell aliases,
virtualenv activation — is thrown away at the end of each call.

This is the **AWS Bedrock AgentCore pattern**: filesystem persists,
shell does not. It's a deliberate simplification, and it covers most
coding-agent workflows because agents rarely need to `cd` — they pass
absolute paths. But it fails on three things:

- **Interactive commands**: `gcloud auth login`, `docker login`,
  installers that prompt. These hang or fail.
- **Long-running processes**: can't start `npm run dev` and monitor it
  from a later tool call.
- **Shell state**: `source venv/bin/activate` in one call has no
  effect on the next.

---

## 3. Survey of how other projects solve this

Synthesized from documentation, issue trackers, and the Claude Code
source leak (March 2026). Roughly ordered by architectural strategy.

### 3.1 Stateless subprocess (same as `code_sandbox`)

| Project | Implementation | Notes |
|---|---|---|
| **agent-plane `code_sandbox`** | `subprocess.Popen` per call, `bash -c` | Current baseline |
| **SWE-agent** | `subprocess.run` / `docker exec` per call | **Explicit design choice**: detecting command completion in persistent shells is flaky, bad commands can kill the session, interrupting a persistent shell corrupts subsequent output |
| **AWS Bedrock AgentCore** | Per-call bash process + persistent `/mnt/workspace` filesystem | Managed session storage handles the persistence concern at the filesystem layer |
| **Smolagents** (Hugging Face) | Custom AST Python interpreter — blocks subprocess entirely | Not really a terminal — it's a code executor |

**Takeaway**: The stateless pattern is a legitimate, defensible choice
for containerized/microVM environments. SWE-agent's position is the
most clearly articulated: persistent shells are a reliability hazard
because you can't tell when a command is done, and a crashed session
poisons the rest of the task. Worth taking seriously before jumping
to a more complex model.

### 3.2 Persistent bash subprocess (no PTY)

| Project | Implementation | Notes |
|---|---|---|
| **Claude Code** | Long-lived bash subprocess spawned via Node `child_process`; uses file descriptor 3 as an out-of-band "done" signal to detect command completion | The leaked source (2026-03-31) confirms a persistent child process with ~2,593 lines of shell-injection defense in `tools/BashTool/bashSecurity.ts`. Skips login shell by default when a shell snapshot is available. Tool results >50KB are persisted to disk. Issue #28407 documents a subtle bug where `claude -p` as a child inherits fd 3 and breaks output capture — evidence that the fd-based completion detection is load-bearing. |
| **Deep Agents CLI** (LangChain) | `LocalShellBackend` (unrestricted) and `HarborSandbox` (containerized) | Shell execution disabled by default in non-interactive mode; `--shell-allow-list` gates specific commands |

**Takeaway**: Persistent bash without a PTY is the Claude Code choice.
It gives you `cd` / `export` / aliases persistence and can background
processes with `&`, but interactive programs that probe `isatty()` or
want raw terminal I/O don't work. The hard part is command-completion
detection — Claude Code solved it by having the shell write to fd 3
when a command finishes, which is robust but requires cooperation from
the shell (a wrapper script).

### 3.3 PTY + headless terminal emulator

| Project | Implementation | Notes |
|---|---|---|
| **Gemini CLI** (Google) | `node-pty` for PTY allocation + `xterm-headless` for screen emulation; falls back to `child_process` when node-pty unavailable | Defaults to off in headless mode; opt-in for interactive shell support |
| **OpenAI Codex CLI** | Rust native PTY: `exec_command` launches a long-lived PTY for streaming/REPLs, `write_stdin` feeds keystrokes to an existing session | Rewritten from Node to Rust in late 2025; PTY is a first-class primitive tracked by session ID |
| **OmniAgents** | tmux with per-instance server sockets | Research/experimental; the inspiration for Phase 4 of `PORTING_FROM_OMNIAGENTS.md` |
| **Pilotty** (msmps) | Separate daemon managing multiple PTY sessions; agents communicate via Unix socket + JSON-line protocol. Full VT100 emulation, exposes structured screen state (cursor pos, detected UI elements) as JSON | Designed specifically for AI-agent consumption — "agent-browser for terminals." Auto-shuts down after 5 min idle. |

**Takeaway**: PTY + headless terminal emulator is the dominant pattern
for standalone agents that need interactivity. The JavaScript canonical
choice is `node-pty` + `xterm-headless`; the Python equivalent is
`pexpect` + `pyte`. This buys you interactive programs, TUIs
(vim, htop, lazygit), progress bars rendering correctly. Pilotty
additionally exposes *semantic* screen state (not just text) so the
agent can navigate TUIs reliably.

### 3.4 IDE-integrated terminal (OSC 633)

| Project | Implementation | Notes |
|---|---|---|
| **Cline** | VS Code integrated terminal + OSC 633 shell integration markers (`]633;C` command start, `]633;D` command end) | Only works inside VS Code. Breaks when shell integration confidence is low (the terminal silently emits no markers) |
| **Cursor** | VS Code-fork integrated terminal, sandboxed. Supports legacy allow-list mode | Persistence is buggy — Cursor 1.6+ doesn't apply the user's shell profile, so virtualenv/rc config isn't available. Community workarounds via `.cursor/agent-*.zsh` files |
| **Roo Code** | Dual: VS Code OSC 633 shell integration OR inline `execa` subprocess | |

**Takeaway**: OSC 633 solves the command-boundary problem by having the
shell emit escape codes at prompt-start / prompt-end / pre-exec /
post-exec. This is **strictly better** than Claude Code's fd 3 hack if
you can mandate a specific shell config. Cline/Cursor rely on VS Code
injecting the shell-integration snippet on startup. For a standalone
CLI (like agent-plane), you'd have to ship the snippet yourself and
source it when launching the shell — doable, but it means the shell is
no longer "the user's shell." OSC 133 is the vendor-neutral variant
(same four markers, different IDs).

### 3.5 tmux-backed sessions

| Project | Implementation | Notes |
|---|---|---|
| **OpenHands** | `libtmux`-backed `TmuxBashSession` (primary) + `SubprocessBashSession` fallback. Abstract `BashSession` base class | **Actively migrating away from tmux-only** because tmux isn't available on fresh Docker images. Previously used `pexpect`, switched to libtmux for reliability (PR #4881). Known gaps: no good way to sleep for long periods from the agent's POV (issue #7723) |
| **OmniAgents** | Raw `tmux` with per-instance server sockets for isolation | Each `TerminalInstance` is its own tmux server — not the system tmux |

**Takeaway**: tmux was the default choice in 2024-2025 but has fallen
out of favor. Reasons: (1) not pre-installed on many containers/macOS;
(2) every send/read forks a `tmux` CLI process (slow); (3) tmux server
lifetime is independent of the controlling agent, which complicates
durability semantics; (4) multi-pane / detach-reattach features are
useless for agents. OpenHands adding a non-tmux fallback is the clearest
signal.

### 3.6 Infrastructure-as-sandbox (delegate everything)

| Project | Implementation | Notes |
|---|---|---|
| **E2B** | Firecracker microVMs; `sandbox.pty.create()` as a first-class primitive. Per-session PTY, `sendInput`/`resize`/`connect`/`kill`/`wait`, disconnect/reconnect supported, `onData` callback for streaming | Hardware-level isolation per sandbox |
| **Modal** | gVisor containers; `Sandbox.exec(pty=True)` | System-call interception instead of hypervisor isolation |
| **Daytona, Sprites** | Similar pattern — PTY as a provider-level capability | |

**Takeaway**: If you're willing to move execution out of the
agent-server process, the PTY problem becomes someone else's problem.
The provider gives you an HTTP/gRPC API that accepts commands and
returns streaming output, and handles isolation + durability. This is
the most operationally clean option and is what serious production
coding-agent products use. It's also the option that furthest from
`code_sandbox`'s current architecture — it would be a different tool,
not a variant of this one.

### 3.7 "Human does it" (Aider)

**Aider** proposes commands to the user and lets them run it themselves.
Not a terminal implementation — a punt on the problem. Worth mentioning
because it's a legitimate UX choice for a certain kind of agent, and
it dodges every complexity in this document.

---

## 4. Competitive snapshot

### 4.1 Capability axes

A quick visual of where each project lands on the capability axes:

```
                      Persistent    PTY /      Interactive   Multi-
                      shell state   screen     stdin         session
                      ────────────  ─────────  ─────────────  ────────
agent-plane (today)   no            no         no            no
SWE-agent             no            no         no            no
AWS Bedrock           no (fs only)  no         no            no
Claude Code           YES           partial    partial       YES*
OpenHands (tmux)      YES           YES        YES           YES
Gemini CLI            YES           YES        YES           optional
OpenAI Codex CLI      YES           YES        YES           YES
E2B / Modal           YES           YES        YES           YES
Pilotty               YES           YES        YES           YES
Cline / Cursor        YES (IDE)     via IDE    via IDE       IDE
OmniAgents            YES           YES        YES           YES

* Claude Code's default Bash session is a single persistent shell, but
  it can spawn additional concurrent shells via the
  `run_in_background: true` parameter, each addressable by a
  `shell_id` and pollable via `BashOutput` / killable via `KillShell`.
```

Performance evidence: Terminal-Bench 2.0 leaderboard (April 2026) —
top performers (Claude Mythos Preview 82%, GPT-5.3 Codex 77.3%) all
use full-PTY-with-persistence harnesses. Stateless-subprocess
harnesses are absent from the top of the leaderboard. That doesn't
prove causation but it's consistent with the benchmark explicitly
testing "stateful, interactive nature of terminal environments."

### 4.2 Agent-facing tool surface

Orthogonal to backend choice is the **tool surface** — how many
distinct tools the LLM actually sees in its function-calling schema.
Projects converge on a narrow set of shapes:

| Project | Agent-facing tools | Shape |
|---|---|---|
| **Claude Code CLI** | `Bash` (with `run_in_background: true`), `BashOutput`, `KillShell` | One-shot by default. Background is opt-in per call and returns a `shell_id`; `BashOutput` polls new output, `KillShell` kills. |
| **OpenAI Codex CLI** | `exec_command`, `write_stdin` | Split by verb: `exec_command` launches a long-lived PTY and streams output back; `write_stdin` feeds keystrokes into a running session. |
| **OpenHands** | `terminal` (single) | One tool, one `command` parameter. Persistent tmux session hidden behind it. |
| **Anthropic Bash API** (`bash_20250124`) | `bash` (+ `restart`) | Schema-less tool with `command` and `restart`. Persistent session maintained client-side. One tool. |
| **Gemini CLI** | `shell` (single) | One tool. |
| **Pilotty** | Multi-tool API (send, snapshot, …) | Full split, but specialist TUI-navigation daemon — not a general shell. |
| **OmniAgents** | `sys_terminal_launch`, `_send`, `_read`, `_list`, `_close` | Full five-tool split. The outlier — nobody mainstream does this. |

**Dominant pattern**: **one "run" tool for the 80% case, plus narrow
supplementary tools for cases that genuinely don't fit one-shot** —
background execution (Claude Code's `BashOutput`/`KillShell`) or
interactive input into a running session (Codex CLI's `write_stdin`).
Fully splitting launch/send/read/wait/close into separate tools, as
OmniAgents does, is outside the industry pattern. The intuition is
that most agent shell use is one-shot (`ls`, `pytest`, `git status`) —
making the 80% case require multiple tool calls is an ergonomic tax
that doesn't pay for itself.

But there is a second use case — §4.3 — that flips this intuition:
using the terminal to host *another agent's CLI* (Claude Code, Codex,
aider, pi) as a long-lived worker. When the use case is "launch a
coding agent and drive it for 20 minutes," the heavy 5-tool shape
stops looking absurd.

### 4.3 Terminal as a host for another agent's CLI: an adjacent pattern

Three projects use terminals not just for shell commands but to run
**other agent CLIs inside them** — Claude Code, Codex, aider, pi.
This is a different use case from "run `pytest`," and the industry
split on tool surface tracks it.

**Ductor** (~200 files) wraps `claude -p`, `codex`, and `gemini` as
subprocess providers with `--output-format json` — one-shot, no PTY.
Its `NamedSessionRegistry` persists CLI session IDs to disk so
`claude --resume <session_id>` can continue a conversation across
Ductor restarts. The user cannot watch workers in real time; subprocess
pipes are opaque.

**Tmux-Orchestrator** runs a hierarchy of Claude Code instances in
tmux windows (Orchestrator → PMs → Engineers). Communication is raw
`tmux send-keys` and `tmux capture-pane`. The helper script
`send-claude-message.sh` sends text, sleeps 500ms, then sends Enter —
the sleep matters because tmux needs time between text and the Enter
keystroke. No sandbox; workers share the real filesystem. Agents
self-schedule check-ins via `at` / `cron`.

**OmniAgents** (at `~/omniagents`) treats this as a first-class
capability in `DESIGN_TMUX_AGENTS.md`. A terminal spec declares
`command: claude`, and `sys_terminal_launch(terminal="worker",
session="auth")` spawns the CLI inside a per-instance tmux server with
a private Unix socket (so sandboxed workers can't reach each other's
tmux), wrapped in Landlock+seccomp, with an optional forked filesystem
(hardlinks broken upfront so shell redirects don't corrupt the parent).
The parent drives it via `sys_terminal_send` (literal keystrokes, with
a configurable delay before the terminating key) and inspects progress
via either `sys_terminal_read` (ANSI-stripped `capture-pane`) or by
mounting the worker's filesystem and reading the CLI's structured
activity log (`.claude/projects/default/activity.jsonl`) — sidestepping
screen-scraping entirely. The user can attach with
`tmux -S <socket> attach -t main -r` to watch in real time. Shipped
`examples/terminal_workers.yaml` uses `pi` and `bash` for the demo,
but the design explicitly targets `claude` and `codex`.

Key design stance from the OmniAgents doc (§3): "The tools are
deliberately minimal. There is no prompt detection, no structured
output parsing, and no turn protocol. The agent reads the screen when
it wants and decides for itself whether the process is done, stuck, or
needs correction." What runs inside the terminal is the agent's
problem — the terminal abstraction is agnostic between `bash`,
`claude`, `codex`, `python`, `aider`, `pi`.

#### Why this shifts the tool-surface argument

Named sessions (`worker:auth`, `worker:tests`, `explorer:v1`,
`explorer:v2`) are the natural identity for long-lived CLI agents the
parent drives over many turns. The parent is tracking logical
workstreams — "the auth worker," "the tests worker" — not processes.
Server-generated opaque IDs (Option B) lose this: `bash_1` vs.
`bash_2` doesn't tell the agent which is which on turn 15. Flag-based
backgrounding (Option B) doesn't map to "this CLI is always long-lived
by design." Timeout-based backgrounding (Option D) accidentally works
because the CLI never emits OSC 633 `D` on its own, but that's a
coincidence, not a semantic fit.

#### A distinct technical concern: keystroke granularity

Claude Code accepts Ctrl-J as a newline within a message and Enter as
"submit." Other CLIs use Enter for newline and Ctrl-D for submit.
Codex has its own conventions. A tool interface that auto-appends
`\n` to every `command` (Options A, B, D as currently sketched) has
no way to express "type this without pressing Enter" or "press
Ctrl-J now." OmniAgents' `sys_terminal_send(text, keys)` decouples
these: `text` is sent verbatim via `tmux send-keys -l`, and `keys`
(defaulting to `"Enter"`) is sent after. If agent-plane wants to
support driving CLI agents inside terminals someday, the tool surface
has to preserve keystroke granularity — which points toward C's
explicit `send` tool, or toward D with a mandatory `keys` parameter,
and away from A/B as currently sketched.

#### Relation to agent-plane's existing harnesses

Agent-plane already reaches Claude Code and OpenAI Agents one way —
**programmatically**, via `ClaudeAgentsExecutor` and `AgentsSdkExecutor`.
Those are *harnesses*: the executor drives the LLM via SDK calls.

A terminal-based Claude Code worker would be the *other* way —
**drive the CLI as a human would**, via keystrokes, watching the
screen. OmniAgents offers both paths (`executor.harness: claude-sdk`
for programmatic, `terminals: {worker: {command: claude}}` for
CLI-driven) and lets agent authors choose per-agent. The two paths
have different trade-offs: SDK gives structured events and clean
error handling; CLI gives user-observable tmux sessions and access to
whatever features the CLI has that the SDK hasn't exposed yet.

**The two paths are completely disjoint.** In OmniAgents,
`claude_sdk_executor.py` (947 lines) contains zero references to
`TerminalEnv`, `TerminalInstance`, `tmux`, or `sys_terminal_*`. When
an agent uses `executor.harness: claude-sdk`:

- Claude Code's own built-in tools (`Bash`, `Read`, `Edit`, `Write`,
  `Glob`, `Grep`) are exposed to Claude via `allowed_tools` — these
  are Claude Code's own implementations, not OmniAgents'.
- OmniAgents tools are re-exposed as MCP tools under
  `mcp__omniagents__<name>`. But the `sys_terminal_*` tools are
  explicitly filtered out of the MCP surface — they are "direct
  tools" that OmniAgents dispatches from its own LLM loop, not from
  Claude's.
- What OmniAgents *does* do: wraps the Claude CLI subprocess itself
  in Landlock+seccomp (`_prepare_cli_with_sandbox`). When Claude
  Code internally shells out via its own `Bash` tool, the child
  process inherits those Landlock restrictions. But that's a
  separate sandbox path from the `TerminalInstance` system — they
  share the Landlock library and nothing else.

The flow:

```
OmniAgents session (claude-sdk harness)
  └── Claude CLI subprocess  (Landlock-wrapped by OmniAgents)
        └── Claude's loop → Claude's own Bash / Read / Edit / ...
              └── pytest etc. (inherits sandbox)
```

The `TerminalInstance` system isn't in the picture.

#### Implication for agent-plane's scope

Agent-plane's persistent terminal tool (whatever §6.1 shape is
chosen) is used **only by agents that use agent-plane's default
executor** — the path where agent-plane drives the LLM directly.
Agents using `ClaudeAgentsExecutor` or `AgentsSdkExecutor` reach
their shell via the harness's own tools (Claude's `Bash`, or whatever
the Agents SDK provides) and never touch the terminal tool.

This mirrors today's situation with `code_sandbox`: Claude-harness
agents have never used it — they use Claude's built-in `Bash`. The
migration from `code_sandbox` to the terminal tool therefore affects
**only default-executor agents**. Harness-based agents are unaffected
by the removal.

This is **not a blocker for the §6 decision** — we can replace
`code_sandbox` with a basic terminal tool now and defer the "drive
another CLI" use case. But §6.1's choice should be made with this
adjacent pattern in mind: whichever tool shape we pick now is the
foundation we'd extend to support CLI-in-terminal later.

---

## 5. How a persistent terminal would differ from `code_sandbox`

Holding the rest of agent-plane fixed, here's what changes:

### 5.1 Lifecycle

- **Today**: Tool is stateless. Each `invoke()` spawns and reaps a
  subprocess. `cancel()` kills the one in-flight process. Nothing
  survives the call.
- **Persistent**: Tool owns conversation-scoped shells managed by a
  server-resident registry (see §6.4). Shells persist across task
  workflows within a conversation, so `cd`, `export`, `source`, and
  running background processes carry across turns. Teardown happens
  on conversation close, idle timeout (24h), or server restart —
  not on individual task completion.

### 5.2 API surface

Tool names below are **illustrative** — the final shape depends on
the §6.1 decision (four options under consideration). The columns
show the capability gap against `code_sandbox`, not a committed API.

| Operation | `code_sandbox` | Persistent terminal |
|---|---|---|
| Execute a command | `code_sandbox({command})` → stdout | A single "run" call returning stdout + exit code |
| Get exit code | In the return string `[exit code N]` | Returned as a dedicated field; detected via OSC 633 `D` marker |
| Kill a foreground command | `cancel()` | Tool-call timeout / cancellation (standard agent-plane path) |
| Start a long-running process | Can't — blocks the tool call | Supported, but the way the agent requests it (flag, timeout, or explicit session launch) depends on §6.1 |
| Monitor a background process | n/a | `check_task` on the returned task handle |
| Cancel a background process | n/a | `cancel_task` (Ctrl-C, shell stays alive) or `terminal_close` (kills shell) |
| Persist `cd` / `export` | No | Yes — shells are persistent per conversation, surviving across tasks (§6.4) |
| Interactive input | Not possible (`stdin=DEVNULL`) | Pre-answered via pipes in `command`; mid-stream input not supported in v1 |

For the 80% case (run a command, get output), the API surface is
the same shape as `code_sandbox` — one tool call in, stdout + exit
code out. Background / read / kill are opt-in.

### 5.3 Durability under DBOS

This is the agent-plane-specific concern and the one `code_sandbox`
dodges cleanly.

`code_sandbox` is **naturally durable** because it's stateless. If
the workflow replays from a checkpoint, the `@step` result is cached
in DBOS — the command doesn't re-run. The subprocess is gone either
way.

A persistent terminal owns a long-lived OS process that is **not** a
DBOS step. If the workflow replays, the subprocess is either (a) still
running from before the crash (parent process survived), (b) gone
(parent died), or (c) in an unknown state. Three options for handling
this:

1. **Workflow-scoped sessions**: terminal manager lives in the
   workflow-local runtime container (the same `ContextVar`-scoped
   object proposed in `PORTING_FROM_OMNIAGENTS.md` Rift 1). Sessions
   die when the workflow process dies. Crash = all sessions lost, agent
   gets an error on next `terminal_read`. Acceptable but means the
   "persistent" claim has a caveat.
2. **Daemon-managed sessions** (Pilotty pattern): a separate process
   (systemd unit, sidecar container) owns the PTYs. The agent-plane
   server talks to it over Unix socket. Sessions survive server
   restarts. Much more operational complexity.
3. **Provider-delegated sessions** (E2B/Modal pattern): `launch`
   returns a provider session ID; persistence is the provider's
   problem. Clean but couples agent-plane to an external provider.

Option 1 is the smallest step from where we are today and matches what
OmniAgents' `TerminalInstance` does. Option 3 is the cleanest long-term.
Option 2 is an intermediate that isn't obviously better than either
endpoint.

**Decision** (see §6.4): conversation-scoped shells, owned by a
server-resident `TerminalManagerRegistry` keyed by `conversation_id`.
Shells outlive individual task workflows so that `cd` / `export` /
`source` persist across turns. Server crash = all shells lost; clear
error on replay. Daemon and provider options remain on the table as
future upgrades.

### 5.4 Command-completion detection

The single hardest problem. `code_sandbox` sidesteps it because the
subprocess exits and `communicate()` returns — you know it's done.

Persistent shells don't have that signal. Options seen in the wild:

- **Prompt-pattern matching** (pexpect classic): wait for `$ ` or a
  known regex. Fragile; breaks on multi-line prompts, custom `PS1`,
  programs that write `$ ` in their output.
- **fd-3 sentinel** (Claude Code): shell wrapper writes a completion
  marker to a dedicated file descriptor. Robust but requires a shell
  wrapper and has subtle fd-inheritance bugs (Claude Code issue
  #28407 — `claude -p` inherits fd 3 and breaks everything).
- **OSC 633 / OSC 133 markers** (Cline, Cursor, VS Code): shell writes
  escape codes at command boundaries. Needs a shell-integration
  snippet sourced in the shell's rc file.
- **tmux pipe-pane + heuristics** (OpenHands old impl): watch tmux
  output, apply timeout + quiescence heuristics. Error-prone — see
  issue #7723 (agent tries to sleep 90s, gets stuck).

All four are load-bearing for their respective implementations and all
four have known failure modes. Any persistent-terminal design for
agent-plane has to pick one and document the failure modes.

### 5.5 Sandboxing interaction

`code_sandbox` has srt-based filesystem sandboxing today (deny-read
on home/tmp, allow-write to workspace only, network unrestricted).
This is implemented by wrapping each call in the srt launcher.

For a persistent terminal, the sandbox has to be applied **once at
launch** and inherited by every command the agent types into the
session. That's doable because both srt backends are one-way and
namespace-inherited — on Linux, bubblewrap creates a user namespace
the child and all its forks live inside; on macOS, sandbox-exec's
seatbelt profile is applied via `sandbox_init` and inherited by
subprocesses. Once set, you can't escape without the process tree
dying. But it means the sandbox config is baked into the session,
not per-command. Changing policy requires closing and re-launching
the session.

### 5.6 Output volume and truncation

`code_sandbox` returns all output as a single string. A persistent
terminal with a long-running process produces unbounded output. Every
implementation solves this with a bounded ring buffer. Ring-buffer
capacity, truncation policy, and how "scrollback" is exposed to the
agent are all design decisions that don't exist today.

### 5.7 Summary table

| Dimension | `code_sandbox` | Persistent terminal |
|---|---|---|
| **Lines of code (rough)** | ~330 (current file) | 500–1500 depending on approach |
| **New dependencies** | none beyond srt | `pexpect` + `pyte`, or libtmux, or provider SDK |
| **Crash durability** | trivially durable | requires design decision (§5.3) |
| **Command-boundary detection** | free (process exit) | hard problem (§5.4) |
| **Interactive programs** | no | yes |
| **Long-running processes** | no | yes |
| **Multiple concurrent sessions** | no | yes |
| **Sandboxing** | per-call srt wrap | once-at-launch srt wrap on the shell itself |
| **Output handling** | whole stdout as string | bounded ring buffer + scrollback API |
| **Workspace persistence** | yes (dir-based) | yes (same mechanism) |
| **DBOS step fit** | natural (stateless) | awkward (stateful resource) |

---

## 6. Decision: replace `code_sandbox` with a persistent-terminal builtin

`code_sandbox` is deleted. A new PTY-backed terminal builtin takes its
place. There is no parallel tool, no "simple command" convenience
wrapper around the terminal backend, and no backwards-compat shim.
Principle #2 (one correct path) and Principle #33 (no backwards-compat
shims, no external consumers yet) both apply — every agent and test
that references `code_sandbox` is migrated to the new tool in the same
change that removes the old one.

This is the most aggressive of the three options surveyed above
(previously labeled "Option C, with removal"). It trades short-term
delivery risk for long-term architectural clarity. Rationale:

- **Principle #2, literally.** Keeping both tools because they "target
  different use cases" is exactly the dual-mode antipattern the
  principles call out.
- **The project has no external consumers.** Principle #33 explicitly
  rules out deprecation wrappers; the time to remove is when the
  replacement lands, not later.
- **`code_sandbox` is a strict subset** of what a persistent terminal
  can do (§1 capability matrix). Every agent that works today with
  `code_sandbox` can work with a terminal tool that exposes
  "run this, give me stdout and exit code" as its basic operation.
- **Terminal-Bench 2.0 evidence** (§4): every top performer uses a
  full-PTY harness. `code_sandbox`-style stateless harnesses are
  absent from the top of the leaderboard. This isn't proof, but it
  says the capability ceiling of the stateless model is lower than
  where the frontier is going.

### 6.1 Tool surface shape — **Option D selected**

Three terminal-specific tools, plus the unified task-lifecycle tools
(`check_task` / `cancel_task` / `list_tasks`) from
`session_model_notes.md`. Agent-named shells with an implicit default
shell for the common case. Explicit control over synchrony and
per-task time bounds via `synchronous` and `timeout_ms` parameters.

**Two axes**:

- **Shell axis**: a persistent bash process that lives across turns
  (§6.4). Managed via `terminal_run` / `terminal_list` / `terminal_close`.
- **Task axis**: a single command invocation on a shell. Async
  command invocations create tasks that share the unified lifecycle
  with sub-agent tasks and background tool tasks. Managed via
  `check_task` / `cancel_task` / `list_tasks` (defined in
  `session_model_notes.md`).

#### Tools

```
terminal_run(command, shell="default", synchronous=True, timeout_ms=None)
  Runs `command` in the named shell (created lazily on first use).

  synchronous=True (default):
    Block until the command completes (OSC 633 `D` marker) or
    timeout_ms fires, whichever is first.
    Returns {stdout, exit_code, status, shell}.
    status is "completed" if D was seen, "killed" if timeout_ms
    fired. No task is created — it's just a blocking tool call.

  synchronous=False:
    Return immediately after starting the command.
    Returns {task_id, kind: "terminal", shell}.
    Agent uses check_task(task_id) / cancel_task(task_id) from the
    unified task lifecycle to poll and control the command.
    Completion auto-delivers as a system message between LLM
    iterations via the same `async_work_complete` path sub-agents
    use.

  timeout_ms=None (default):
    No time bound — command may run indefinitely.

  timeout_ms=N:
    Command is killed if it runs longer than N milliseconds.
    Applies regardless of synchrony — enforced by the TerminalManager
    via a scheduled deadline, independent of the tool call's lifetime.
    When it fires, the TerminalManager sends Ctrl-C to the PTY (then
    SIGKILL if it doesn't respond); the shell stays alive.

terminal_list()
  Returns [{shell, status: "idle" | "busy", last_command,
            running_command?, running_since_ms?, ...}, ...]
  Shell-axis enumeration. For task-axis (what commands are in
  flight), use list_tasks().

terminal_close(shell="default")
  Kills any command currently running in the shell (as if
  cancel_task'd), then terminates the bash process itself and
  removes the shell from the registry.
```

#### Interaction with the unified task lifecycle

Async `terminal_run` creates a task with `kind: "terminal"`. The
task lifecycle tools defined in `session_model_notes.md` operate on
it:

```
check_task(task_id) for kind="terminal":
  {
    task_id,
    kind: "terminal",
    shell: "dev",
    status: "running" | "completed" | "failed" | "cancelled",
    result: {stdout, exit_code} | null,   # present when terminal (see note)
    recent_activity: "...stdout delta since last check_task call...",
    started_at, updated_at
  }

Semantics of recent_activity for kind="terminal":
  - Delta semantics (tail -f style). Contains output produced since
    the previous check_task call on this task. Cleared after each
    read so consecutive checks don't return the same bytes twice.
  - Populated while status is "running" OR when transitioning to a
    terminal status (captures any final output the caller hasn't
    seen yet).
  - Empty once the task is terminal AND the final delta has been
    delivered — the result field then carries the authoritative
    complete stdout.
  - Subject to the same 30 KB inline cap and head+tail truncation
    as any other LLM-surfaced output (§6.7).

Semantics of result for kind="terminal":
  - Present only when status is "completed", "failed", or
    "cancelled" — null while "running".
  - result.stdout is the complete captured stdout (subject to the
    30 KB inline cap; overflow goes to disk per §6.7).
  - result.exit_code is the process exit code (negative for signals).

cancel_task(task_id) for kind="terminal":
  Sends Ctrl-C to the running command via the shell's PTY.
  Shell stays alive. Agent can continue using the shell for new
  commands. This is the "I changed my mind" path. Task status
  transitions to "cancelled".

list_tasks(filter) includes terminal tasks alongside sub-agent and
  background-tool tasks — unified cross-subsystem view.
```

Contrast with `terminal_close(shell)`, which kills the command AND
the shell itself. The two operations live on different axes:
`cancel_task` is command-level; `terminal_close` is shell-level.

Usage by case:

| Case | Call |
|---|---|
| Simple command | `terminal_run("ls")` — blocks briefly, returns output + exit code |
| Bounded synchronous | `terminal_run("pytest", timeout_ms=300_000)` — blocks up to 5 min; if exceeded, pytest is killed and response status is `"killed"` |
| Known long-running | `terminal_run("npm run dev", synchronous=False)` — returns task handle; agent uses `check_task` to poll, `cancel_task` to stop |
| Fire-and-forget with max lifetime | `terminal_run("stuck-thing", synchronous=False, timeout_ms=60_000)` — returns task handle immediately; command self-terminates after 1 min (task status becomes `"cancelled"`), shell stays alive for reuse |
| Parallelism | `terminal_run("pytest", shell="test")` + `terminal_run("npm run dev", shell="dev", synchronous=False)` — two stateful shells concurrently |
| Stateful reuse | `terminal_run("cd /proj && source venv/bin/activate")` then later `terminal_run("python script.py")` — venv is still active |

**Naming**: `"default"` is a reserved name used implicitly when the
agent doesn't supply one. Any other string matching the character-set
constraint in §7.1 sub-decision 3 is a valid user-supplied name.
Shells are keyed per conversation (§6.4, §6.6); naming collisions
between parent and sub-agent are impossible because they have
separate conversations.

We use **"shell"** rather than "session" deliberately — "session" is
already overloaded in agent-plane (conversations, tasks, HTTP
sessions), and every shell instance is literally a persistent bash
process with OSC 633 wrapping. This matches Claude Code's
`shell_id` terminology.

**Synchrony is the primary control**, not `timeout_ms`. The 80% case
is `synchronous=True` with no timeout — the tool behaves like
`code_sandbox` did. `synchronous=False` is opt-in for "I know this is
long-running, don't block." `timeout_ms` is orthogonal and optional;
it caps how long the *task* may run (not how long the tool *call*
waits — those are the same thing only when synchronous is True).

**Concurrent calls on the same shell fail fast.** If a command is
already running in a shell (including the default), a second
`terminal_run` targeting that shell returns immediately with
`{status: "shell_busy", running_command, running_since_ms, shell}`
instead of queueing or interleaving. The LLM can recover by passing
a different `shell="..."` name, calling `cancel_task(task_id)` on
the running task, or calling `terminal_close(shell)` to kill the
shell entirely. If the running command is not its own (a
background task it started earlier), it can wait for the completion
to auto-deliver or check via `check_task`.

This fail-fast policy is forced by two constraints:

- `pexpect` is not thread-safe on a single spawn
  ([pexpect#322](https://github.com/pexpect/pexpect/issues/322),
  [#369](https://github.com/pexpect/pexpect/issues/369)) —
  concurrent interaction from different threads hangs or misbehaves.
- OSC 633 command-boundary markers assume one command per shell;
  interleaved writers corrupt the `C`/`D` sequence and break
  completion detection.

Silently serializing (the alternative) has a worse failure mode:
a fast second call queued behind a slow first would wait until its
tool-step framework timeout fires, producing an opaque timeout error
with no signal that the cause was same-shell contention. Fail-fast
surfaces the collision to the LLM immediately with enough context
(the currently-running command and its runtime) for a smart recovery
decision.

**This shape was selected over alternatives below** after considering
ergonomics, parallelism expressiveness, and mental-model clarity. A
and B were rejected because they can't express multiple persistent
stateful shells (see §6.1 "Rejected: Option B" note — a real limit
shared with Claude Code). C was rejected because the send+read split
forces 2 tool calls for the 80% case without buying anything over a
blocking `run`. D unifies the send+read into a single call with
explicit synchrony control, preserves named shells for parallelism,
and decouples the "I want to wait" intent from the "task may run for
up to N ms" bound.

#### Status enumeration

Two separate enums live on different axes.

**Terminal-tool response `status`** (returned inline by `terminal_run`,
`terminal_list`, `terminal_close`). Shell-axis conditions:

| Value | Meaning |
|---|---|
| `"completed"` | Synchronous `terminal_run` finished cleanly (OSC 633 `D` marker seen). `exit_code` present. |
| `"killed"` | Synchronous `terminal_run` killed by `timeout_ms`. `exit_code` reflects signal (negative). |
| `"shell_busy"` | The target shell already has a command running. No queueing — the call fails fast so the agent can decide how to react. |
| `"shell_crashed"` | The bash process itself died (segfault, OOM-kill, or unrecoverable timeout SIGKILL). Shell removed from registry; the next call auto-spawns a fresh shell. |
| `"shell_name_invalid"` | The supplied `shell` arg fails the `^[a-zA-Z][a-zA-Z0-9_-]{0,63}$` regex, or starts with an underscore (reserved). Error message names the rule violated. |
| `"shell_cap_exceeded"` | The per-conversation 10-shell cap (§6.4) would be exceeded by creating this shell. Agent must `terminal_close` another shell first. |
| `"error"` | Validation failure unrelated to the shell itself — e.g. missing `conversation_id` or `workspace` on the tool context (setup bug, not an agent-recoverable condition). |
| `"idle"` / `"busy"` | Only used in `terminal_list` per-shell entries, not as a top-level response status. |

Note: there is **no** `"shell_not_found"` status. If an agent calls
`terminal_run(shell="dev")` and "dev" doesn't exist, we transparently
spawn it — agents address shells by name, and the first call to a
name creates it. This is why §6.4 idle-reaping is safe: agents can't
tell a reaped shell from a never-created one, and they don't need to
(their persistent state was in the reaped shell and is gone, but
that's acceptable given the 24h idle threshold).

**Task `status`** (returned by `check_task` for tasks of
`kind: "terminal"`). Reuses the unified enum from
`session_model_notes.md` — no new values:

| Value | Meaning |
|---|---|
| `"running"` | Background command still executing in the shell. |
| `"completed"` | Command finished cleanly. `result.{stdout, exit_code}` populated. |
| `"failed"` | Command exited non-zero OR the shell crashed underneath. `result.{stdout, exit_code}` populated (exit_code captures the failure). |
| `"cancelled"` | `cancel_task` was called (Ctrl-C sent to the running command); or `terminal_close` killed the shell while this task was running. |

Any other status is a bug. Both enums are closed sets.

---

Alternatives considered (for posterity):

#### Option A: Implicit singleton

Model: one session per workflow, no identification anywhere.

| Tool | Shape |
|---|---|
| `terminal_run(command, timeout_ms?)` | Runs in the (implicit, workflow-scoped) shell. Blocks until OSC 633 `D`. Returns `{stdout, exit_code}`. |

Nothing else. No read, no kill (kill is tool-call cancellation).

| | |
|---|---|
| **Pros** | Simplest possible surface. Zero naming state for the agent to track. Cheapest migration from today's `code_sandbox` (same call shape). |
| **Cons** | No parallelism. `npm run dev` + `pytest --watch` concurrently is impossible. If a command hangs, the whole shell is blocked — no way to kill it without cancelling the entire tool call. Foreground-only. |
| **Production template** | OpenHands `terminal`, Gemini CLI `shell`, Anthropic `bash_20250124`. |

#### Option B: Flag-switched — server-ID'd background (previously selected)

Model: foreground is singleton; background gets a server-generated
ID. Single tool decides which path via a boolean.

| Tool | Shape |
|---|---|
| `terminal_run(command, background=false, timeout_ms?)` | `background=false`: runs in singleton shell, returns `{stdout, exit_code}`. `background=true`: spawns fresh shell, returns `{session_id}` immediately. |
| `terminal_read(session_id)` | Poll a background session. New output + status. |
| `terminal_kill(session_id)` | Kill a background session. |

| | |
|---|---|
| **Pros** | 80% case (simple command) has zero naming overhead. Server-generated IDs eliminate collision risk. Production-proven shape (Claude Code). |
| **Cons** | Boolean `background` flag mode-switches the tool's return type and semantics — the same tool call means two different things depending on a flag. Asymmetric code paths (foreground singleton vs. background spawned-fresh). Agents can't name sessions for their own bookkeeping. |
| **Production template** | Claude Code CLI (`Bash` + `BashOutput` + `KillShell`). |

#### Option C: Caller-named sessions (OmniAgents / Daytona)

Model: every session has an agent-chosen name. Sessions are
first-class. No hidden modes.

| Tool | Shape |
|---|---|
| `terminal_launch(name, command?)` | Create a session with that name. Optionally run an initial command. |
| `terminal_send(name, text)` | Send a command (or keystrokes) to the session. Does not wait. |
| `terminal_read(name, wait?)` | Read new output + status. Optionally block for the next OSC 633 `D`. |
| `terminal_kill(name)` | Kill the session. |
| `terminal_list()` | Enumerate open sessions. |

| | |
|---|---|
| **Pros** | Explicit, no hidden modes. Parallelism is natural (`launch("dev")`, `launch("test")`). Most mental-model-clean: sessions are first-class entities the agent reasons about. |
| **Cons** | 80% case (`ls`) requires 2+ tool calls (send + read, or launch + send + read). Naming overhead for simple commands. Agent-chosen names introduce collision risk — two code paths in the same conversation picking `"default"` for different purposes (mitigated by §6.6 sub-agent isolation, but intra-conversation collisions are still possible). The send+read split interacts awkwardly with OSC 633: `read` has to block for the `D` marker to know completion, so the split doesn't buy anything over a blocking `run`. Largest tool surface (5 tools). |
| **Production template** | OmniAgents (agent framework), Daytona (sandbox infra provider). Two independent implementers picked this. |

#### Option D: Hybrid — named sessions with explicit synchrony (SELECTED, see top of §6.1)

Earlier sketches of D folded background-vs-foreground into "didn't
finish in the timeout." The final shape at the top of §6.1 separates
that into an explicit `synchronous` flag and uses `timeout_ms` only
as a task-lifetime bound (kill-after), not as a "move on" signal. The
pros/cons below refer to the selected shape.

| | |
|---|---|
| **Pros** | No mode flag between foreground and "background." Every `terminal_run` call has the same shape. `synchronous` and `timeout_ms` are orthogonal and independently optional. Named shells give clean parallelism (`shell="dev"`, `shell="test"`). Default shell makes the 80% case as simple as A (`terminal_run("ls")`). Stateful reuse across calls is native. |
| **Cons** | Two parameters (`synchronous`, `timeout_ms`) to explain vs. A's zero. Intra-workflow name collisions possible if two agent code paths independently pick the same custom session name (less likely than C because the default covers most usage). No direct production template — this is a synthesis. |
| **Production template** | None exactly. Closest analog is Claude Code's `Bash` (foreground-default with `run_in_background` flag and a `timeout`), but Claude Code's `timeout` means **kill + return error** rather than **kill + return status in synchronous call**, and there's no way to reuse a background shell statefully. |

#### Tradeoff summary

| Axis | A | B | C | D |
|---|---|---|---|---|
| Simple-case ergonomics | best | great | worst | great |
| Parallelism expressiveness | none | server-ID | named | named |
| Absence of mode flags | ✓ | boolean | ✓ | ✓ |
| Mental-model clarity | trivial | asymmetric | uniform | uniform |
| Intra-conversation collision risk | none | server IDs safe | possible | default + optional |
| Production precedent | strong | strong | strong | synthesis |
| Tool count | 1 | 3 | 5 | 3 |

#### How each option interacts with earlier §6 decisions

- **Backend (`pexpect` + `pyte`, §6.2)**: identical in all four. One
  `TerminalManager` per conversation owns shells; only the lookup key
  (none / server-ID / agent-name / default-or-agent-name) differs.
- **OSC 633 completion detection (§6.3)**: identical. The snippet
  emits markers; the tool reads them. A, B, D use the marker
  synchronously (block until `D`); C's `read(wait=true)` also blocks
  for `D` — the difference is which tool issues the blocking call.
- **Durability tier (§6.4)**: identical. Conversation-scoped shells
  in all four; server-resident registry keyed by `conversation_id`.
- **Sub-agent isolation (§6.6)**: identical. Shells are keyed on the
  conversation, not globally, so parent and sub-agent (which run in
  separate conversations) have separate namespaces regardless of
  naming pattern.

#### Mid-stream interactive input

In all four options, writing keystrokes into a shell that's currently
running a command (answering a password prompt, driving `passwd`,
feeding `y` to an installer that didn't take `-y`) is **not supported
in v1**. Pre-answered input stays possible via shell pipes in
`command` itself (`echo y | apt install`). If mid-stream input becomes
necessary later, a fourth tool (`terminal_input(shell, text)`) can
be added; no option precludes it.

Matches Claude Code's v1 limitation exactly — they ship without
mid-stream input and it's not been the Terminal-Bench blocker.

#### Why D over A/B/C

A/B cannot express multiple *persistent stateful* shells. They can
spawn concurrent one-shot background tasks (Claude Code's
`run_in_background` model) but the only reusable stateful shell is
the implicit foreground singleton. Agents wanting two persistent
workstreams — e.g., one shell with a `/project-a` virtualenv
activated and another with `/project-b` — cannot have that in A/B.

C has the right capability model but forces 2 tool calls for the
80% case (send + read), since without a blocking-on-D `run` the
agent has to poll. The send/read split doesn't buy anything over
just blocking, because `read` has to block for the marker anyway.

D gives the same capability ceiling as C (named persistent shells,
parallelism) with A's simple-case ergonomics (one call returns
output + exit code). Decoupling synchrony (`synchronous`) from
task lifetime (`timeout_ms`) lets the agent express "fire and forget"
and "bounded wait" independently.

The §4.3 adjacent use case (hosting other agent CLIs inside
terminals) is deferred for v1, but D is the right shape if it ever
becomes in scope — named shells are how long-lived CLI workers are
naturally addressed.

### 6.2 Backend choice: `pexpect` + `pyte`

Reasoning consolidated from §3:

- Pure-Python, no system binaries (pip-installable). Works on fresh
  Docker images, stock macOS, any Linux.
- Matches the industry canonical pattern — it's the Python equivalent
  of `node-pty` + `xterm-headless` (Gemini CLI) and of Codex CLI's
  native Rust PTY. This is the pattern modern standalone agents
  converge on.
- Pyte optional: default read path is raw stdout with ANSI stripping;
  pyte rendering is available for tools that explicitly need in-place
  updates (progress bars, TUIs). That's what Claude Code does at
  stdout level, and what `terminal_read(scrollback=...)` vs. a
  separate `terminal_screen` call can expose.

Rejected: `libtmux` (requires tmux system binary — OpenHands is
actively moving away from it for this reason, §3.5), provider
delegation to E2B/Modal (correct long-term but orthogonal
architectural move — a separate decision).

### 6.3 Command-completion detection: OSC 633 + shell-integration snippet

The hardest sub-problem (§5.4). Commitment:

- Inject a shell-integration snippet at shell launch that emits OSC
  633 markers (`A` prompt-start, `B` prompt-end, `C` pre-exec, `D`
  post-exec-with-exit-code, plus `E` for the interpreted command
  line). Source the snippet before handing the shell to the agent.
- OSC 633 is VS Code's superset of the vendor-neutral OSC 133 — the
  `A`/`B`/`C`/`D` markers have identical semantics, so parsers built
  for OSC 133 work on OSC 633 output. OSC 633's extras (`E` for the
  exact command line, `P` for properties like cwd) are useful enough
  to be worth adopting over plain OSC 133, and it's the most
  battle-tested of the two — VS Code, Cursor, Cline, and Roo Code
  all ship it.
- `terminal_run` (when `synchronous=True`) scans for
  `OSC 633;D;<code>` to determine completion + exit status.
  Agent-plane is not the user's interactive shell; we control the
  shell's rc file at launch (§6.8), so the snippet is always present.
- Because the shell is hardcoded to bash (§7.1 ratified) and we ship
  the snippet ourselves, there is no "missing markers" fallback path.
  If bash itself isn't available on the system, `terminal_run` fails
  loud with `shell_not_found` at first invocation (Design Principle
  #3). No prompt-pattern scraping, no fd-3 hack. No attempt to
  "degrade gracefully" to a shell we can't reliably parse.

This is better than Claude Code's fd-3 approach (which had subtle
inheritance bugs, see issue #28407) and better than prompt-pattern
matching (brittle). It's the same mechanism VS Code / Cline / Cursor
use, adapted to a standalone CLI by shipping the snippet ourselves.

### 6.4 Durability tier and shell lifetime

Conversation-scoped shells, owned by a server-resident
`TerminalManagerRegistry` keyed by `conversation_id`. A shell outlives
individual task workflows so that `cd /tmp`, `export FOO=bar`, and
`source venv/bin/activate` persist across turns — exactly the
"persistent terminal" behavior that distinguishes this tool from
today's `code_sandbox`. This matches Claude Code's model, where the
bash subprocess lives with the Claude CLI process and persists across
every LLM call in that session.

**Architecture:**

```
AgentPlaneServer (process)
└── TerminalManagerRegistry  (server-resident singleton)
    ├── conversation_abc123
    │   └── TerminalManager
    │       ├── shell "default"  (bash subprocess, persistent)
    │       └── shell "dev"      (bash subprocess running `npm run dev`)
    └── conversation_def456
        └── TerminalManager
            └── shell "default"
```

Each task workflow looks up
`registry.for_conversation(self.conversation_id)` at startup and uses
that manager throughout the task. When the task ends, the manager
and its shells stay alive, waiting for the next turn.

**Lifetime rules:**

- **Task ends normally**: shells stay alive. Server is idle between
  turns.
- **Next task in same conversation**: reuses the conversation's
  existing shells. State persists (cwd, env, running processes,
  history).
- **Conversation closed or deleted**: all shells for that conversation
  are killed; manager removed from registry. Wired via a synchronous
  call from `conv_store`'s delete path:
  `registry.cleanup_conversation(conversation_id)` is invoked as part
  of the deletion transaction, not via pub/sub or lazy reconciliation.
  This guarantees no orphaned shells survive a conversation delete.
- **Empty manager**: when the last shell in a conversation is closed
  via `terminal_close` (or killed by idle timeout or crash), the
  `TerminalManager` itself is removed from the registry. Next
  `terminal_run` re-creates it. Prevents empty managers from
  accumulating indefinitely for conversations that briefly touched
  the terminal and never will again.
- **Idle timeout**: conversations with no task activity and no
  running background tasks for **24 hours** have their shells killed
  and manager removed. A conversation counts as "active" if any task
  workflow has started within the window OR any background task is
  still executing. Abandoned conversations release resources; active
  ones with long-running background work don't get reaped under them.
  Next task on a reaped conversation spawns fresh shells.
- **Per-conversation cap**: max **10 shells** per conversation.
  `terminal_run` with a new `shell=...` beyond the cap returns an
  error; agent must close one first. Caps the blast radius of agent
  bugs.
- **Server crash**: `TerminalManagerRegistry` is in-memory only;
  crash = all shells lost. On restart, replaying workflows that
  reference old shells get a clear "shell not found" error. Agent
  handles this the same way it would handle any lost-compute
  scenario — by recreating shells as needed. Conversation history
  (the durable record of what commands ran and what output came back)
  is unaffected; only the live bash processes are lost.

**Shells are ephemeral server compute, not durable state.** The
conversation history is in the DB. The shell processes are not. A
"persistent terminal" in this design means "persistent across turns
of a live conversation on a live server" — not "survives crashes or
restarts."

**Sub-agents**: sub-agents have their own conversations, so they get
a separate `TerminalManager` naturally. Parent and sub-agent never
share shells. See §6.6.

**Alternatives considered** (§5.3): daemon-managed shells (separate
long-lived process, survives server restarts) and provider-delegated
shells (E2B/Modal) are possible upgrades but add operational
complexity disproportionate to the current need.

**Threading model, locking, delete-hook wiring, reaper implementation,
and shutdown path** are specified in §6.9. The registry is
encapsulated: one module-level accessor, all coordination primitives
internal to the class hierarchy.

### 6.5 Sandboxing

srt-based filesystem sandbox is applied **once at shell launch**, via
the same launcher mechanism `code_sandbox` uses today but wrapping the
shell instead of each command. srt uses bubblewrap on Linux (user
namespace + mount overlays) and sandbox-exec on macOS (seatbelt
profile). Both are inherited by subprocesses and cannot be escaped
from a child — every process the agent spawns inside the shell
inherits the restrictions.

Changing policy requires closing and re-launching the shell. This is
a behavior change from today (`code_sandbox` re-applies policy per
call) but is the standard pattern for persistent-terminal sandboxes
and matches how these primitives are designed to be used.

### 6.6 Sub-agent isolation

Sub-agents get their own `TerminalManager` instance, not a view into
the parent's. Shells are keyed on the sub-agent's conversation — and
in agent-plane, sub-agents run in their own conversations distinct
from the parent's — so a sub-agent cannot see, write to, or close a
shell owned by its parent (or vice-versa). The `conversation_id`
keying in the server-resident registry (§6.4) provides this
isolation by construction.

This is the safe default:

- **Avoids concurrent modification.** Two agents writing to the same
  PTY simultaneously produces interleaved input that neither can
  interpret. The OSC 633 command-boundary mechanism assumes a single
  writer per shell; two writers corrupt the markers.
- **Matches Principle #32** (spec self-containment). A sub-agent
  whose behavior depended on the parent's terminal state would be
  non-reproducible — different parents would produce different
  sub-agent behavior for the same sub-agent spec.
- **Naming collisions are fine.** Parent and sub-agent can both have
  a shell called `"default"`; they refer to different PTYs because
  they're in different conversation scopes.

If cross-agent filesystem state needs to flow between a parent and a
sub-agent, it goes through the workspace (files on disk), not through
shared terminal state. That's the mechanism we already use for
everything else agents share.

### 6.7 Output buffering and size limits

Output from a shell moves through three layers with explicit bounds
at each:

**Layer 1: Ring buffer (per-shell, in-memory)**
- **1 MB per shell**, FIFO byte-level eviction when full.
- Implemented as a dedicated byte-bounded buffer (Python `collections.deque`
  of bytes or a bytearray ring), **separate from any pyte screen state**.
  Pyte's `HistoryScreen` is line-based (default 100 lines, configurable)
  and is only relevant if we ever expose rendered screen output
  (deferred per §6.2 "pyte optional"). The ring buffer is the
  primary output record; pyte rendering is a secondary read path.
- When bytes are evicted, the next output surfaced to the agent
  (either via `terminal_run`'s synchronous return or via
  `check_task`'s `recent_activity`) is prepended with
  `[... N bytes evicted ...]` so the agent knows data was dropped.

**Layer 2: Inline return cap (tool result)**
- **30 KB**, matching Claude Code's default. Anything larger is
  truncated before being returned to the LLM.
- **Truncation policy: head + tail**. 10 KB head + a truncation
  marker with a disk path + 10 KB tail. The marker line is
  something like `[... 47KB truncated, full output at
  .agent_plane/terminal/dev-42.log ...]`. Head/tail sizes sum to
  less than the inline cap so the marker + path fit without being
  themselves truncated.
- **Why 30 KB not 50 KB**: Claude Code settled on 30 KB after
  production experience; Codex CLI's 10 KB is widely complained
  about ([codex#7906](https://github.com/openai/codex/issues/7906));
  30 KB is the industry default at the LLM-friendly end. Not
  configurable in v1.

**Layer 3: Disk persistence (overflow only)**
- When the ring buffer would overflow OR the tool result would
  truncate, flush the full accumulated output to
  `<workspace>/.agent_plane/terminal/<shell>-<run_index>.log`.
- **Lazy, not always**. Small commands (`ls`, `echo hi`, `git status`)
  never touch disk. Only long/large outputs do.
- `run_index` is a monotonic counter per shell, starting at 1 and
  incrementing with each overflow. Lives on the `TerminalManager`
  in memory — not persisted. Rationale: simpler than timestamp
  naming, readable in LLM context (`dev-42.log` beats
  `dev-1713542345123.log`), and the counter naturally resets when
  the shell does (because a server restart kills the shell, so the
  new shell is a new logical entity).
- **Cross-restart collision handling**: if the workspace survives
  a server restart (it does — workspaces are durable), old log
  files stay on disk. The new TerminalManager's counter starts at 1
  again. To avoid overwriting stale logs from before the restart,
  the write path checks for file existence and bumps `run_index`
  until a free name is found. Inefficient in principle but trivially
  fast in practice (O(1) for typical patterns) and avoids a
  persistence layer for the counter.
- Retrieval: agent reads via any path-aware tool (e.g. the filesystem
  read tool, or another `terminal_run("cat ...")`). No special
  retrieval tool.

**Units: bytes (characters), not lines.** Lines are fragile — an
ANSI-heavy progress-bar "line" can be 500+ chars. Byte bounds are
predictable; line bounds are not.

**ANSI stripping: always, at read time.** Everything surfaced to the
LLM — both `terminal_run`'s synchronous `stdout` and `check_task`'s
`recent_activity` / `result.stdout` — is plain text with ANSI escape
sequences removed. The ring buffer stores raw bytes (preserves
fidelity for disk persistence); stripping happens on the read path
between buffer and tool result. Order of operations on each read:
(1) locate and extract OSC 633 markers (`A`/`B`/`C`/`D`/`E`) so we
know command boundaries and exit codes; (2) strip all remaining ANSI
CSI / OSC / SGR sequences; (3) return the remainder. OSC 633 markers
are never surfaced to the agent — they're our internal control
plane.

No "raw mode" escape hatch in v1. Matches OmniAgents' and Claude
Code's behavior.

**Rationale grounded in industry data**: Claude Code 30 KB default
(often bumped to 100 KB), Codex CLI 10 KB (user complaints), tmux
2000 lines default, pyte 100 lines default, OmniAgents 10k lines
scrollback. Our choice (30 KB inline / 1 MB ring / lazy disk) threads
the middle: generous enough for typical pytest runs to fit inline
without truncation, bounded enough to keep context costs manageable
and avoid OOMs with many concurrent shells, with an escape hatch for
truly large output.

### 6.8 Shell launch contract

Everything that happens when a new shell is spawned. This complements
§6.2 (backend choice), §6.3 (completion detection), and §6.5
(sandboxing) by consolidating the per-shell launch sequence.

**Binary**: `bash` (hardcoded, not `$SHELL`). See §7.1. Rationale:
OSC 633 snippet is bash-specific; we own the shell config; reproducible
starting environment regardless of developer's local shell choice.

**Working directory**: `<workspace>`, the same per-conversation
workspace dir `code_sandbox` uses today. The agent can `cd` elsewhere
at any time and it persists across commands in that shell (that's
persistence's whole point). Every *new* shell in the same conversation
launches fresh at the workspace root — `cd` state is per-shell, not
per-workspace.

**Environment variables at launch**: the same workspace-relative set
`code_sandbox` applies per-call, but applied once at shell spawn via
pexpect's `env=`:

```
PIP_TARGET        = <workspace>/.pip
PIP_CACHE_DIR     = <workspace>/.cache/pip
PYTHONPATH        = <workspace>/.pip
NODE_PATH         = <workspace>/node_modules
npm_config_prefix = <workspace>
npm_config_cache  = <workspace>/.cache/npm
```

Plus whatever the OSC 633 shell-integration snippet needs (e.g.
`PROMPT_COMMAND` wrapping and `PS1` adjustments for marker emission).
Agent-`export`ed variables persist within that shell; the launch-time
vars above can be overridden by the agent if it has reason to, but
the happy-path correctness is baked in.

**OSC 633 snippet sourcing**: the integration snippet is shipped as a
static file (inside the `agent_plane` package, e.g.
`agent_plane/tools/builtins/_terminal_integration.sh`) and loaded via
`bash --rcfile <snippet-path>` on spawn. `--rcfile` is the bash-native
way to make a startup file mandatory without the user needing to
`source` it explicitly after the shell is up (which would create a
race with the agent's first command). The snippet wraps the user's
prompt with OSC 633 markers and exports a minimal set of shell
functions used for completion reporting.

**srt sandbox wrap**: the full `bash --rcfile …` command is wrapped
in `_srt_wrap.mjs` (§6.5). The sandbox policy is applied once at the
bash launch and inherited by every child process. Changing policy
requires closing and re-launching the shell.

**Launch sequence** (end to end):

1. Agent calls `terminal_run(command, shell="name")` for an unknown
   shell name (or first call on default).
2. TerminalManager resolves the conversation's workspace dir.
3. Constructs the bash invocation:
   `_srt_wrap.mjs <srt-config> bash --rcfile <snippet-path>`.
4. Environment map built from workspace-relative vars + OS baseline.
5. `pexpect.spawn(...)` with `env=` set; this creates the PTY and
   starts bash.
6. Snippet sourcing happens automatically via `--rcfile`; snippet
   emits the initial OSC 633 `A` marker.
7. TerminalManager registers the shell in its map, keyed by name.
8. The agent's `command` is sent via the PTY; OSC 633 `C` (pre-exec)
   is emitted by the snippet, then the command runs, then `D` (post-exec)
   with exit code.
9. `terminal_run` reads until `D` (synchronous) or returns a task
   handle (async).

**Launch failures**: if step 5 or 6 fails (bash not installed, srt
launcher broken, snippet bug that prevents the initial prompt),
`Shell.spawn` raises `RuntimeError` synchronously (fail-loud per
Principle #3). The `TerminalManager._get_or_create_shell` lets the
exception propagate; the tool sees it as an unhandled exception and
the request surfaces as a task failure. Launch-failure bugs never
produce a silently-broken shell — they always surface as loud
server errors.

### 6.9 Threading model and registry integration

This section captures how `TerminalManagerRegistry` integrates with
agent-plane's DBOS-based concurrency model, and **how encapsulation
is enforced** so the registry doesn't leak locks or state into the
rest of the codebase.

**Encapsulation principle: one module-level accessor, zero exported
synchronization primitives.** The registry is the sole server-resident
entity visible to the rest of the codebase. Locks, per-conversation
managers, per-shell mutexes, reaper state — all live as instance
members inside the registry hierarchy. Callers get a registry handle
and invoke methods on it; they never touch a Lock, a dict, or any
coordination primitive directly.

#### Agent-plane execution model (what we're integrating with)

From the codebase survey:

- Each `@workflow` runs on a **dedicated DBOS worker thread**;
  `@step` functions within it run on that same thread.
- No global event loop. Async-from-sync bridging uses per-ToolManager
  `EventLoopThread` instances (`agent_plane/tools/mcp.py`), **not**
  a shared process-wide loop.
- Existing server-level state (stores in `agent_plane/runtime/_globals.py`,
  `AgentCache`) is **initialized once at startup** and relies on the
  GIL. It's read-mostly after init.
- Existing per-workflow state (canonical example: `ToolManager`) is
  scoped via **`ContextVar`** and injected at workflow start, read
  from `@step` functions.

#### Why this design introduces a new pattern

Agent-plane today has two state scopes: per-workflow (ContextVar) and
server-level-initialized-once (module globals). It has no
**conversation-scoped, actively-mutated, cross-thread** state.
`TerminalManagerRegistry` is the first of its kind.

Implementer consequence: do **not** reflexively follow the ToolManager
ContextVar pattern. Per-workflow scoping would kill shells every turn
and defeat the cross-turn persistence that is the entire point of
§6.4. This design intentionally deviates, and the registry's
docstring should say so explicitly for future contributors.

#### The single global

Exactly **one** module-level symbol is added to `_globals.py`:

```python
# agent_plane/runtime/_globals.py (new addition alongside existing stores)
_terminal_registry: TerminalManagerRegistry | None = None

def init_terminal_registry() -> None:
    """Called once from init_runtime() at server startup."""
    global _terminal_registry
    _terminal_registry = TerminalManagerRegistry()

def get_terminal_registry() -> TerminalManagerRegistry:
    if _terminal_registry is None:
        raise RuntimeError("Terminal registry not initialized")
    return _terminal_registry
```

That's the entire public module surface. No exported locks, no
exported dicts, no `_terminal_managers_by_conv: dict[str, ...]` that
other code could accidentally reach into.

#### Internal structure (all encapsulated)

```python
class TerminalManagerRegistry:
    """Server-resident, conversation-scoped terminal state.

    Callers interact only via for_conversation / cleanup_conversation /
    shutdown. All locking is internal.
    """
    def __init__(self) -> None:
        self._managers: dict[str, TerminalManager] = {}
        self._lock = threading.Lock()         # registry-scope — instance member
        self._reaper_task: asyncio.Task | None = None

    def for_conversation(self, conv_id: str) -> TerminalManager:
        with self._lock:
            mgr = self._managers.get(conv_id)
            if mgr is None:
                mgr = TerminalManager(conv_id)
                self._managers[conv_id] = mgr
            return mgr

    def cleanup_conversation(self, conv_id: str) -> None:
        with self._lock:
            mgr = self._managers.pop(conv_id, None)
        if mgr is not None:
            mgr.close_all()   # outside the registry lock: close_all may block on I/O

    def shutdown(self) -> None:
        """Walk all managers and kill their shells. Called on server shutdown."""
        ...

    # Reaper is a method on the registry, not a module-level function.
    # Starts an asyncio task from init_runtime(), ticks every 10min,
    # calls self._reap_idle() which acquires self._lock internally.


class TerminalManager:
    """Per-conversation shell owner. All locking internal."""
    def __init__(self, conversation_id: str) -> None:
        self._conv_id = conversation_id
        self._shells: dict[str, Shell] = {}
        self._lock = threading.Lock()         # manager-scope — instance member
        self._last_activity = monotonic()

    def run_sync(self, shell_name: str, command: str, timeout_ms: int | None) -> RunResult:
        shell = self._get_or_create_shell(shell_name)
        return shell.run_sync(command, timeout_ms)   # Shell owns its own mutex

    def _get_or_create_shell(self, name: str) -> Shell:
        with self._lock:
            shell = self._shells.get(name)
            if shell is None:
                if len(self._shells) >= 10:
                    raise ShellCapExceeded(...)
                shell = Shell.spawn(...)
                self._shells[name] = shell
            return shell

    def close(self, name: str) -> None: ...
    def close_all(self) -> None: ...


class Shell:
    """One bash subprocess + ring buffer. Owns its command mutex."""
    def __init__(self, ...) -> None:
        self._cmd_lock = threading.Lock()     # shell-scope — instance member
        self._proc: pexpect.spawn = ...
        self._ring = RingBuffer(1_000_000)
        ...

    def run_sync(self, command: str, timeout_ms: int | None) -> RunResult:
        if not self._cmd_lock.acquire(blocking=False):
            return RunResult(status="shell_busy", ...)
        try:
            ...
        finally:
            self._cmd_lock.release()
```

Three locks, three scopes, all instance members. Callers never see
them, never pass them, never need to acquire them externally. Each
layer owns its own coordination.

#### Call flow (sync path)

1. Workflow `@step` calls `terminal_run(command, shell, ...)`.
2. Tool does `registry = get_terminal_registry()` — single global
   lookup.
3. `mgr = registry.for_conversation(self.conversation_id)` — registry
   lock acquired and released internally.
4. `result = mgr.run_sync(shell, command, timeout_ms)` — manager lock
   acquired internally for shell create-if-absent.
5. Inside: `Shell.run_sync` attempts its per-shell mutex; on success
   runs command, on failure returns `shell_busy`.

At no point does the caller (the tool, the @step, the workflow) hold
or need to know about a lock.

#### pexpect runs on the workflow thread (Phase 1, sync)

`pexpect.spawn(...).expect(D_marker)` blocks the workflow's DBOS
worker thread until completion. This is the same pattern
`code_sandbox` uses today with `subprocess.Popen.communicate()`. No
separate thread pool is needed for Phase 1's synchronous-only scope.

Phase 2 (async, when `synchronous=False` lands) needs a long-lived
thread per background command — the workflow returns while the
command keeps running. That's where `EventLoopThread`-style
infrastructure becomes relevant. Out of scope here.

#### Conversation-delete integration: one line

Agent-plane has no event/hook system. Conversation deletion is a
direct call chain in
`agent_plane/server/routes/conversations.py`:

```python
async def delete_conversation(conversation_id: str):
    conversation_store.get_conversation(conversation_id)
    await task_store.delete_all(conversation_id=conversation_id)
    get_terminal_registry().cleanup_conversation(conversation_id)  # NEW
    await conversation_store.delete_conversation(conversation_id)
```

One line. Direct synchronous call. No listener registration, no
out-of-order delivery concerns.

Placement between task deletion and conversation deletion is
deliberate: after task delete, no running task can reference a shell
we're about to kill; before conversation delete, the registry entry
is torn down while its conversation row still exists (avoids a
window where the conversation row is gone but shells still exist).

#### Initialization and shutdown

- `init_runtime()` in `agent_plane/cli.py` calls
  `init_terminal_registry()` alongside the existing store init. Also
  starts the idle-reaper task (which is a method on the registry).
- Server shutdown calls `get_terminal_registry().shutdown()`. Hooks
  into the same teardown path the stores use. Agent-plane's shutdown
  path may need to grow if it's currently minimal.

#### Locking summary

Three locks. Zero exports. Every one lives as an instance member of
its owning class:

| Lock | Where | Scope | Purpose |
|---|---|---|---|
| `TerminalManagerRegistry._lock` | registry instance | registry ops | atomic get-or-create / cleanup on the `conv_id → manager` dict |
| `TerminalManager._lock` | manager instance | per-conversation | atomic get-or-create / close on the `shell_name → shell` dict; cap enforcement |
| `Shell._cmd_lock` | shell instance | per-shell | backs the `shell_busy` fail-fast semantic (§6.1); held for the duration of a command |

No module-level `threading.Lock` anywhere. No functions that take a
lock as an argument. No "lock the registry before calling this" notes
in docstrings — the methods handle their own coordination.

### 6.10 Removal plan — *completed*

Completed as part of Slice 5:

1. `agent_plane/tools/builtins/code_sandbox.py` — deleted.
2. `agent_plane/tools/builtins/_srt_wrap.mjs` — deleted (the
   terminal's launch path uses its own PTY-compatible
   `_srt_shell.mjs` that uses `spawn(..., stdio: 'inherit')`
   rather than `execSync`).
3. `agent_plane/tools/builtins/__init__.py` — `code_sandbox`
   unregistered from `BUILTIN_NAMES` and `_BUILTIN_REGISTRY`;
   `_create_code_sandbox` factory deleted.
4. `agent_plane/tools/manager.py` — `_create_code_sandbox` method
   deleted; `_register_builtin_tools` no longer dispatches to it.
5. `agent_plane/tools/base.py`, `agent_plane/tools/builtins/web_fetch.py`,
   `agent_plane/tools/builtins/export_agent.py` — docstring references
   updated to point at `terminal_run`.
6. `agent_plane/onboarding/*` — onboarding assistant now installs
   `terminal_run` + `export_agent` (not `code_sandbox` + `export_agent`)
   when shell access is disabled.
7. `tests/tools/builtins/test_code_sandbox.py` — deleted.
8. E2E tests rewritten to use `terminal_run`:
   `test_archer_terminal.py` (renamed from `test_archer_code_sandbox.py`),
   `test_sandbox_dependencies.py`, `test_archer_output_files.py`,
   `test_file_tools.py`, `test_steering.py`, `test_web_fetch_e2e.py`,
   `test_onboarding_e2e.py`, `test_archer_introspect.py`.
9. `examples/agents/archer/config.yaml` — `code_sandbox` replaced with
   `terminal_run` + `terminal_list` + `terminal_close`.
10. README.md, AGENTSPEC.md, and onboarding SKILL.md updated to
    document `terminal_run` as the shell tool.

Principle #33 honored: no `# removed` comments, no aliases, no
`warnings.warn` shim, no dual-register period surviving past Slice 5.
The old tool ceases to exist.

**Harness-based agents are unaffected.** Agents whose executor is
`ClaudeAgentsExecutor` or `AgentsSdkExecutor` don't use `code_sandbox`
today — they use the harness's own shell tool (Claude's built-in
`Bash`, or whatever the Agents SDK provides). See §4.3 for the full
explanation. The migration only touches default-executor agents,
which is the same population that has `code_sandbox` in their tool
lists today.

---

## 7. Open questions

All design decisions are settled. Captured here for quick
reference; see §6 for rationale.

**Core design (§6):**

- Tool surface → **Option D** (§6.1): three terminal-specific tools
  (`terminal_run`, `terminal_list`, `terminal_close`), addressed by
  agent-named shells with implicit `"default"`. Async invocations
  return task handles routed through the unified task lifecycle
  (`check_task` / `cancel_task` / `list_tasks` from
  `session_model_notes.md`) with `kind: "terminal"`. Completion
  auto-delivers between LLM iterations via the same
  `async_work_complete` path sub-agents use.
- Backend → **`pexpect` + `pyte`** (§6.2) — pyte optional, only for
  rendered-screen read paths.
- Command-completion detection → **OSC 633 shell-integration snippet
  injected at launch via `bash --rcfile`** (§6.3, §6.8)
- Durability tier → **conversation-scoped shells** owned by a
  server-resident `TerminalManagerRegistry` keyed by
  `conversation_id`. 24h idle timeout (extends while any task is
  running), 10 shells max per conversation, lost on server crash
  (§6.4)
- Sandboxing → **srt applied once at shell launch**, wrapping the
  bash process so all child processes inherit (§6.5)
- Sub-agent terminals → **isolated per conversation** — sub-agents
  run in their own conversations, so they naturally get a separate
  `TerminalManager` (§6.6)
- Output buffering → **30 KB inline / 1 MB ring buffer / lazy disk
  overflow / head+tail truncation / byte-based / ANSI-stripped on
  read** (§6.7)
- Shell launch contract → **bash + workspace cwd + workspace-relative
  env (PIP_TARGET etc.) + OSC 633 snippet via `--rcfile` + srt wrap**
  (§6.8)
- Scope → **default-executor agents only**; harness-based agents
  (Claude Agents, Agents SDK) continue to use the harness's own
  shell tool and are not affected (§4.3, §6.9)

**Sub-decisions:**

- Shell selection → **hardcoded `bash`** (not `$SHELL`). We own the
  shell config, the OSC 633 snippet is bash-specific, reproducible.
- Shell name constraints → **`^[a-zA-Z][a-zA-Z0-9_-]{0,63}$`**,
  reject names starting with `_` (reserved for future framework use).
  `"default"` is the implicit fallback when the agent omits `shell`.
- Concurrent tool calls on same shell → **fail fast with `shell_busy`**
  (see §6.1 for payload). Forced by pexpect thread-safety and OSC 633
  single-writer assumption.

**Deferred / out of scope for v1:**

- Mid-stream interactive input into a running command (can be added
  later as a fourth tool `terminal_input`)
- Drive-external-CLI use case (§4.3) — running `claude`/`codex`/etc.
  inside a shell as a sub-worker. D supports this shape but the
  glue work is out of scope.
- Client-side variant (follow-on doc after server-side is stable)

### 7.2 Implementation work items (not decisions)

1. **`_srt_wrap.mjs` reuse.** Confirmed this can wrap the shell
   launch instead of individual commands. Needs verification that
   srt's restrictions survive through bash's own forks — i.e., that
   bubblewrap's user namespace (Linux) and sandbox-exec's seatbelt
   profile (macOS) stay active when the agent types `python script.py`
   inside the sandboxed bash. Both should work by design (namespaces
   and profiles are process-tree-inherited), but this hasn't been
   empirically verified with srt specifically wrapping a long-lived
   shell rather than one-shot commands.

2. **Migration of e2e tests.** The existing `code_sandbox` E2E tests
   encode real bugs we care about (native tool persistence, sandbox
   dependencies, archer workflows). They need to be rewritten, not
   deleted — the terminal tool must preserve the behaviors they
   assert. This is the largest single chunk of the removal work.

3. **JSON Schema formalization.** The tool surfaces in §6.1 are
   specified in prose. The actual JSON Schemas presented to the LLM
   (for `terminal_run`, `terminal_list`, `terminal_close`) need to
   be written out. The prose is precise enough to derive them
   mechanically; doing so prevents drift between doc and code.
   Include param descriptions, enum values, required fields, and
   the return-shape documentation.

4. **Shell-integration snippet contents.** §6.8 references
   `agent_plane/tools/builtins/_terminal_integration.sh` as the file
   sourced via `bash --rcfile`. The actual bash code has to be
   written: OSC 633 marker emission in a `DEBUG` trap (for `C`
   pre-exec) and `PROMPT_COMMAND` (for `D` post-exec with
   `$?`-captured exit code), plus `A`/`B` wrapping the prompt. VS
   Code's shell integration script is a reasonable reference
   (pattern is well-established) but we don't need its full feature
   set — a minimal subset covering A/B/C/D plus optionally E (command
   line) is enough.

5. **Thread-safety, initialization, reaper, shutdown, delete-hook
   wiring** — all covered in §6.9 with encapsulation principles and
   illustrative code. Nothing more to decide; implementer follows
   §6.9 directly.

---

## 8. References

- [Claude Code — Bash tool API docs](https://platform.claude.com/docs/en/agents-and-tools/tool-use/bash-tool)
- [Claude Code source leak analysis — wavespeed.ai](https://wavespeed.ai/blog/posts/claude-code-architecture-leaked-source-deep-dive/)
- [Claude Code issue #28407 — fd 3 inheritance bug](https://github.com/anthropics/claude-code/issues/28407)
- [Claude Code issue #26235 — OSC 133 semantic zones](https://github.com/anthropics/claude-code/issues/26235)
- [OpenHands PR #4881 — pexpect → libtmux migration](https://github.com/All-Hands-AI/OpenHands/pull/4881)
- [OpenHands issue #9971 — CLI tmux fallback proposal](https://github.com/OpenHands/OpenHands/issues/9971)
- [OpenHands issue #7723 — long-sleep detection gap](https://github.com/All-Hands-AI/OpenHands/issues/7723)
- [Gemini CLI — Google](https://github.com/google-gemini/gemini-cli)
- [node-pty — npm](https://www.npmjs.com/package/node-pty)
- [xterm-headless — npm](https://www.npmjs.com/package/xterm-headless)
- [OpenAI Codex CLI — GitHub](https://github.com/openai/codex)
- [OpenAI Codex CLI — PTY architecture](https://mintlify.wiki/openai/codex/architecture/tui)
- [Deep Agents CLI — LangChain](https://github.com/langchain-ai/deepagents)
- [Deep Agents CLI — Terminal-Bench 2.0 evaluation](https://blog.langchain.com/evaluating-deepagents-cli-on-terminal-bench-2-0/)
- [SWE-agent — cloud-native execution framework](https://createaiagent.net/tools/swe-agent/)
- [Mini SWE-agent FAQ — on statelessness](https://mini-swe-agent.com/latest/faq/)
- [Pilotty — daemon-managed PTY sessions for agents](https://github.com/msmps/pilotty)
- [E2B — Interactive PTY docs](https://e2b.dev/docs/sandbox/pty)
- [Modal — Sandbox.exec with PTY](https://www.morphllm.com/modal-sandbox)
- [E2B vs Modal — Northflank comparison (2026)](https://northflank.com/blog/e2b-vs-modal)
- [Terminal-Bench 2.0 leaderboard](https://www.vals.ai/benchmarks/terminal-bench-2)
- [Terminal-Bench 2.0 announcement](https://www.tbench.ai/news/announcement-2-0)
- [Cline — Shell Integration Troubleshooting](https://github.com/cline/cline/wiki/Troubleshooting-%E2%80%90-Shell-Integration-Unavailable)
- [Cursor — Terminal docs](https://cursor.com/docs/agent/tools/terminal)
- [Roo Code — Shell Integration](https://docs.roocode.com/features/shell-integration)
- [VS Code — Terminal Shell Integration (OSC 633)](https://code.visualstudio.com/docs/terminal/shell-integration)
- [AWS Bedrock AgentCore — shell command execution](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/runtime-execute-command.html)
- [pexpect — GitHub](https://github.com/pexpect/pexpect)
- [pexpect + pyte example](https://github.com/pexpect/pexpect/blob/master/examples/terminal_emulation.py)
- [The Agentic Shift — terminal control overview (2026)](https://www.epsilla.com/blogs/ai-agents-update-2026)
- [Choosing an AI sandbox provider in 2026 — cto.new](https://cto.new/blog/choosing-an-ai-sandbox-provider-in-2026)
