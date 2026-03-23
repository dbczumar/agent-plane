# Runtime Design

## Overview

The runtime executes agents. Given an `agent_id` and user input, it runs the agent loop (prompt → LLM → tool calls → repeat) and produces output. It is DBOS-aware for durable execution — all LLM calls and tool calls are checkpointed, so execution survives crashes without re-running completed steps.

## Public API (WIP)

User-facing API — details TBD, sketched here for context. Not the current focus. Subject to change.

```python
import agent_plane
from agent_plane.client import Agent

agent = Agent.from_location("./my-agent")
response = agent_plane.run(agent, "What's the weather?")
```

---

## Runtime Initialization

Stores are module-level globals, set once at process startup. Workflow functions reference them directly — no dependency injection needed.

```python
# runtime/_globals.py
conversation_store: ConversationStore | None = None
task_store: TaskStore | None = None
memory_store: MemoryStore | None = None  # Not yet implemented — TBD

# runtime/__init__.py
def init(
    conversation_store: ConversationStore,
    task_store: TaskStore,
    memory_store: MemoryStore,  # Not yet implemented — TBD
) -> None:
    """
    Initialize the runtime. Must be called once before any task execution.
    The server calls this at startup. For local use, agent_plane.init()
    calls it.
    """
    _globals.conversation_store = conversation_store
    _globals.task_store = task_store
    _globals.memory_store = memory_store
```

The server then calls stores directly — no executor indirection:

```python
# server handles a request (see Request Lifecycles for full flows)
task = task_store.create(conversation_id=conversation_id, agent_id="agent_123",
    agent_name="my-agent", previous_response_id="resp_abc123")
conversation_store.append(conversation_id, [NewConversationItem(type="message",
    response_id=task.task_id, data={"role": "user", ...})])
# instructions and reasoning are passed directly to DBOS, not stored in the task row.
task_store.start(task.task_id, instructions="Be concise", reasoning={"effort": "high"})

# server resolves conversation for a follow-up request
conversation_id = conversation_store.get_conversation_id("resp_abc123")
```

### Stores

```python
class TaskStore(ABC):
    def __init__(self, uri: str) -> None:
        """
        Initialize the task store and the underlying DBOS durable execution
        engine. Calls ensure_dbos(uri) to initialize the DBOS singleton if
        it hasn't been already.
        """
        ...

    @abstractmethod
    def create(
        self,
        conversation_id: str,
        agent_id: str,
        agent_name: str,
        previous_response_id: str | None = None,
        background: bool = False,
    ) -> Task:
        """
        Create a new task for executing an agent in the given conversation.
        Generates a unique task_id (which doubles as the response_id),
        stores the task record with status="queued", and returns the Task.
        Does not start execution — call start() to begin.

        instructions and reasoning are NOT stored in the task row —
        they are pure workflow inputs passed to start().
        """
        ...

    @abstractmethod
    def start(
        self,
        task_id: str,
        instructions: str | None = None,
        reasoning: dict | None = None,
    ) -> None:
        """
        Begin execution of a previously created task. Launches the DBOS
        workflow asynchronously and returns immediately — the task
        remains "queued" until the workflow actually begins running,
        at which point it transitions to "in_progress".

        instructions and reasoning are passed directly to the DBOS
        workflow as inputs (stored by DBOS, not in the tasks table).

        Enforces the task/workflow invariant: if the DBOS workflow fails
        to start, the task row is deleted via compensating transaction.
        """
        ...

    @abstractmethod
    async def stream(self, task_id: str) -> AsyncIterator[dict]:
        """
        Yield streaming events as they are produced by the runtime. Awaits
        until the next event is available. The iterator ends when the task
        completes or is cancelled. Each event is a dict with a "type" field
        (e.g. "text_delta", "tool_call"). Backed by DBOS.read_stream().
        Async because it long-polls for events — would block the event loop
        if synchronous.
        """
        ...

    @abstractmethod
    def get(self, task_id: str) -> Task | None:
        """
        Return a snapshot of the task's current state. Output is populated
        only when status is "completed". For all other terminal states
        (failed, incomplete, cancelled), output is empty — intermediate
        work is captured in the DBOS stream, not in the task output.
        Returns the task regardless of status. Returns None if the task
        does not exist (deleted by user or cleaned up by system).
        """
        ...

    @abstractmethod
    async def wait(self, task_id: str) -> Task:
        """
        Await until the task reaches a terminal state (completed, failed,
        incomplete, or cancelled) and return the final Task. Used by the
        server for blocking mode (background=false). Internally calls
        DBOS.retrieve_workflow(task_id).get_result(). Async because it
        blocks until completion — would freeze the event loop if synchronous.
        """
        ...

    @abstractmethod
    def try_deliver(
        self,
        task_id: str,
        conversation_id: str,
        item: NewConversationItem,
    ) -> bool:
        """
        Atomically deliver a steering item to a running task, or
        report that the inbox is already closed.

        Single transaction: if the agent's inbox is still open, appends
        the item to the conversation and returns True. If the agent has
        already closed its inbox (finishing up), returns False — the
        caller should create a new response instead.

        Server-side half of the steering handshake.
        """
        ...

    @abstractmethod
    def close_inbox(
        self,
        task_id: str,
        conversation_id: str,
        last_seen_item_id: str | None,
    ) -> list[ConversationItem]:
        """
        Atomically attempt to close the inbox for a finishing task.
        Within a single transaction: queries for items in the conversation
        newer than last_seen_item_id (or all items if None). If found,
        returns them (inbox stays open — agent must continue). If none
        found, sets inbox_closed=True and returns empty list (agent
        may complete). This is the agent-side half of the steering
        handshake.
        """
        ...

    @abstractmethod
    async def cancel(self, task_id: str) -> Task:
        """
        Stop execution and mark the task as cancelled. If in progress,
        stops the DBOS workflow and waits for the finally block to
        complete (close_stream, inbox drain). Sets status to "cancelled".
        The task record is preserved — get() still works, and the response
        can be referenced as previous_response_id to continue or redirect
        the conversation. The user's input message remains in the
        conversation; the assistant's incomplete output is not saved. Async
        because stopping an in-progress workflow may block while the
        finally block runs.
        """
        ...

    @abstractmethod
    async def delete(self, task_id: str) -> None:
        """
        Remove a task record entirely. If in progress, stops the DBOS
        workflow first and waits for the finally block to complete. Then
        deletes the record — subsequent get() returns None. The user's
        input message remains in the conversation (it was persisted before
        task start), but the assistant's incomplete output is not saved.
        Works on any task regardless of status. Async because stopping
        an in-progress workflow may block while the finally block runs.
        """
        ...

    @abstractmethod
    def list_tasks(
        self,
        conversation_id: str | None = None,
        agent_id: str | None = None,
    ) -> list[Task]:
        """
        Return tasks matching the given filters. All filters are
        optional and combined with AND. Used by the route layer to
        find in-flight tasks for cancellation (e.g. before deleting
        an agent or conversation).
        """
        ...

@dataclass
class Task:
    """A task representing a single response execution."""
    task_id: str                                    # doubles as response_id
    conversation_id: str                            # the conversation this task belongs to
    status: str                                     # "queued", "in_progress", "completed", "failed", "incomplete", "cancelled"
    agent_id: str                                   # ID of the agent executing this task
    agent_name: str                                 # denormalized — stable model name even if agent is renamed
    created_at: int                                 # epoch timestamp
    completed_at: int | None = None                 # epoch timestamp, set on terminal status
    output: list = field(default_factory=list)      # empty until status is "completed"
    inbox_closed: bool = False                      # True once the agent's final inbox check found no messages
    instructions: str | None = None                 # from DBOS workflow inputs (not DB row)
    reasoning: dict | None = None                   # from DBOS workflow inputs (not DB row)
    background: bool = False                        # whether this task runs in background mode
    previous_response_id: str | None = None         # response this task continues from
    usage: dict | None = None                       # token usage stats
    error: dict | None = None                       # error details for status="failed"
    incomplete_details: dict | None = None          # details for status="incomplete"

class NewConversationItem(BaseModel):
    """An item that has not yet been persisted. No ID or timestamp. (Pydantic model)"""
    type: str          # "message", "function_call", "function_call_output", "reasoning"
    response_id: str   # the response (task) this item belongs to
    data: ItemData     # type-specific Pydantic model (MessageData, FunctionCallData, etc.)

class ConversationItem(BaseModel):
    """A persisted item with a store-assigned ID. (Pydantic model)"""
    id: str
    type: str          # "message", "function_call", "function_call_output", "reasoning"
    status: str        # item status
    response_id: str   # the response (task) this item belongs to
    created_at: int
    data: ItemData     # type-specific Pydantic model (MessageData, FunctionCallData, etc.)

@dataclass
class Conversation:
    """A conversation grouping related turns."""
    id: str                                         # unique conversation identifier
    metadata: dict = field(default_factory=dict)    # caller-attached key-value pairs
    created_at: int = 0                             # epoch timestamp
    title: str | None = None                        # optional conversation title

@dataclass
class PagedList(Generic[T]):
    """A page of results with an optional cursor for the next page."""
    data: list[T]               # the items in this page
    next_page_token: str | None # opaque cursor; None means no more pages

class ConversationStore(ABC):
    @abstractmethod
    def create_conversation(self, metadata: dict | None = None) -> Conversation:
        """
        Create a new conversation. Generates a unique conversation_id.
        Metadata is optional caller-attached key-value pairs (e.g. user_id,
        title). Returns the Conversation.
        """
        ...

    @abstractmethod
    def get_conversation_id(self, response_id: str) -> str:
        """
        Resolve a response_id to the conversation it belongs to. Queries
        messages by response_id (every message carries the response_id
        that produced it). This is the durable resolution path — it
        works even after the task record has been cleaned up. Raises
        if no message with the given response_id exists.
        """
        ...

    @abstractmethod
    def get_latest_response_id(self, conversation_id: str) -> str | None:
        """
        Return the response_id of the most recent message in the conversation,
        or None if the conversation has no messages. Used by the server to
        detect forks: if previous_response_id != get_latest_response_id(),
        the caller is branching from a non-latest point.
        """
        ...

    @abstractmethod
    def search_items(
        self,
        conversation_id: str,
        limit: int = 100,
        after: str | None = None,
        before: str | None = None,
    ) -> PagedList[ConversationItem]:
        """
        Return items in a conversation with cursor-based pagination. Used by
        the runtime to load conversation history, and by the UI to display
        conversations. Items are ordered chronologically by default.

        If `after` is set, only returns items with IDs newer than the
        given item ID. Used by the agent loop to poll for steering
        items that arrived since its last check.
        """
        ...

    @abstractmethod
    def append(self, conversation_id: str, items: list[NewConversationItem]) -> list[ConversationItem]:
        """
        Append items to a conversation. Assigns a globally unique ID and
        timestamp to each item. Returns the persisted ConversationItems
        with their assigned IDs. Called twice per request: first by the
        server to persist the user's input (after task creation, before
        task start), then by the runtime to persist agent output
        (after close_inbox confirms no late items).
        """
        ...

    @abstractmethod
    def get_conversation(self, conversation_id: str) -> Conversation | None:
        """Return the conversation, or None if it does not exist."""
        ...

    @abstractmethod
    def list_conversations(
        self,
        limit: int = 20,
        after: str | None = None,
        before: str | None = None,
    ) -> PagedList[Conversation]:
        """List conversations with cursor-based pagination, newest first."""
        ...

    @abstractmethod
    def update_conversation(
        self, conversation_id: str, **kwargs
    ) -> Conversation | None:
        """
        Update mutable fields on a conversation. Currently only
        `title` is updatable. Returns the updated Conversation, or
        None if the conversation does not exist.
        """
        ...

    @abstractmethod
    async def delete_conversation(
        self, conversation_id: str
    ) -> bool:
        """
        Delete a conversation and all its items. Returns True if the
        conversation existed, False otherwise. Async because it may
        need to cancel in-flight responses in the conversation first.
        """
        ...

# MemoryStore: Not yet implemented — structure TBD.
# The interface below is illustrative and subject to change.
class MemoryStore(ABC):
    @abstractmethod
    def search(self, query: str, limit: int) -> list[Memory]:
        """
        Search for memories relevant to the given query. Used by the runtime
        to inject long-term context into the agent's prompt.
        """
        ...

    @abstractmethod
    def write(self, memories: list[Memory]) -> None:
        """
        Store new memories. Used by the runtime when the agent produces
        facts or observations worth retaining across conversations.
        """
        ...
```

## Initialization

DBOS is a global singleton initialized once per process. `TaskStore.__init__` owns this — it calls `ensure_dbos(uri)` on construction, so DBOS is live as soon as the store is built (at server startup or on first local use).

```python
# durability.py
_dbos_initialized = False

def ensure_dbos(uri: str) -> None:
    global _dbos_initialized
    if _dbos_initialized:
        return
    DBOS(config=DBOSConfig(name="agent-plane", system_database_url=uri))
    DBOS.launch()
    _dbos_initialized = True
```

## Execution model

The execution workflow and all its internal operations are DBOS-decorated. All DBOS-specific imports are isolated in `durability.py`.

```python
# durability.py — the ONLY file that imports dbos
from dbos import DBOS, SetWorkflowID
workflow = DBOS.workflow
step = DBOS.step
transaction = DBOS.transaction
start_workflow = DBOS.start_workflow
write_stream = DBOS.write_stream
read_stream = DBOS.read_stream
# ... etc
```

The agent loop (pseudocode):

```python
@workflow()
def agent_execution_workflow(
    agent_id: str,                     # agent identifier
    conversation_id: str,              # conversation
    previous_response_id: str | None,  # conversation chain
    instructions: str | None,          # per-request steering
) -> dict:                             # full response object
    task_id = DBOS.workflow_id
    history = load_history(conversation_id)         # @step — checkpointed
    last_seen = history[-1].id if history else None

    try:
        for _ in range(max_iterations):
            # Check for steering messages appended by try_deliver
            new_messages = conversation_store.search_items(     # @step
                conversation_id, after=last_seen).data
            if new_messages:
                history = history + new_messages
                last_seen = new_messages[-1].id

            response = call_llm(history, instructions)  # @step — checkpointed
            write_stream("output", response.content)    # persisted to DBOS streams table

            if response.is_final:
                # Atomic inbox close: check for late-arriving messages.
                # IMPORTANT: we must close the inbox BEFORE persisting the
                # assistant message. If we persisted first and close_inbox
                # returned late messages, we'd have a stale assistant message
                # in the conversation — the agent would continue the loop, produce
                # a different response, and append a second assistant message.
                # By closing first, we only persist when we're certain the
                # agent is done.
                late = task_store.close_inbox(task_id, conversation_id, last_seen)
                if late:
                    # Messages arrived — keep going
                    history = history + [response] + late
                    last_seen = late[-1].id
                    continue

                # Inbox closed, no late messages — safe to persist and complete
                conversation_store.append(conversation_id, [
                    NewConversationItem(type="message", response_id=task_id,
                        data={"role": "assistant", "agent": agent.name  # agent = agent_store.get(agent_id), "content": response.content})])
                return build_response(...)

            result = call_tool(response.tool_name, ...)  # @step — checkpointed
            write_stream("output", result)
            history = history + [response, result]

        # max_iterations exhausted — agent never produced a final response.
        return build_response(status="incomplete", ...)

    finally:
        # Runs on ANY exit — normal completion, max_iterations, or
        # unhandled exception. Two cleanup responsibilities:
        #
        # 1. Close the stream so server-side stream iterators terminate.
        close_stream("output")

        # 2. Close the inbox so try_deliver stops accepting messages for
        #    this task. Drain loop handles late messages (inbox only closes
        #    when none are found). If the task was deleted (get returns
        #    None), there's no inbox to close — skip.
        task = task_store.get(task_id)
        if task and not task.inbox_closed:
            while True:
                late = task_store.close_inbox(task_id, conversation_id, last_seen)
                if not late:
                    break
                last_seen = late[-1].id
```

### What DBOS provides

- **Checkpointing**: Each `@step` output is saved to Postgres/SQLite. On crash recovery, completed steps return their cached output — no re-execution. No duplicate LLM calls, no duplicate tool side effects.
- **Streams**: `write_stream("output", chunk)` persists output incrementally. Readable via `read_stream(workflow_id, "output")` — used by the server for SSE streaming and GET snapshot responses.
- **Workflow identity**: The server pins `task_id = DBOS workflow_uuid` via `SetWorkflowID(task_id)` before starting the workflow. DBOS stores workflow inputs (all request metadata) and output (full response object). No separate responses table needed — DBOS is the store.
- **Recovery**: On server restart, DBOS detects pending workflows and resumes them from the last checkpoint.
- **Cancellation**: `DBOS.cancel_workflow(task_id)` stops execution.

### Background × Stream behavior

The runtime always runs the same DBOS workflow. How the server exposes it changes:

| `background` | `stream` | Server behavior |
|---|---|---|
| `false` | `false` | Await workflow completion, return result as JSON |
| `false` | `true` | SSE reading from `read_stream()`, cancel workflow on disconnect |
| `true` | `false` | `start_workflow()`, return `{status: "queued"}` immediately |
| `true` | `true` | SSE reading from `read_stream()`, workflow continues on disconnect |

`GET /v1/responses/{id}` calls `task_store.get(task_id)` — the server never reads from DBOS directly.

### Crash recovery

DBOS recovers the **workflow** — on server restart, pending workflows resume from the last checkpoint and run to completion. But DBOS cannot recover **HTTP connections**. This has different implications depending on mode:

**Background mode (background=true):** Full recovery. The client received the task_id immediately (before any crash), so it can always poll `GET /v1/responses/{task_id}` later. DBOS resumes the workflow, it completes, messages are saved to the conversation. The client gets the result whenever it polls. This is the recommended mode for production use.

**Blocking mode (background=false):** Best-effort. If the server crashes while the client is waiting:

1. The HTTP connection dies — client gets a connection error
2. Server restarts — DBOS resumes the workflow, it completes, assistant messages are saved to the conversation
3. But the client never received a task_id (the response hadn't been sent yet)
4. The work is preserved (the user message was appended to the conversation before the task started, and the assistant message is saved on completion), but the delivery is lost

**Client recovery for blocking mode:** The client can check the conversation for new messages. The user's message was persisted before the task started, and if the workflow completed, the assistant's response is in the conversation too. The client retrieves its conversation (via `previous_response_id` from the last successful turn) and checks if an assistant response appeared after its message.

**Warning:** If the client naively retries the POST, it will append a duplicate user message and start a duplicate task. Idempotency keys (not yet implemented) would prevent this.

## Steering

Steering lets a user send additional input while an agent is running a long-horizon task. The agent incorporates the input at its next loop iteration — no need to cancel and restart.

### How it works

The client sends `POST /v1/responses` with `previous_response_id` pointing to an in-progress response. The server detects the response is still running and delivers the message as steering input rather than creating a new response.

**Server POST handler:**

```python
def handle_post(previous_response_id, input, conversation, ...):
    # ── conversation field validation (early) ──
    if conversation and not previous_response_id:
        raise 400("conversation provided without previous_response_id")

    if previous_response_id:
        # Resolve conversation via get_conversation_id — durable path that queries
        # messages by response_id. Works even if task records are later GC'd.
        # Raises 400 if no message with this response_id exists.
        conversation_id = conversation_store.get_conversation_id(previous_response_id)

        if conversation:
            # Response must belong to the specified conversation
            if conversation_id != conversation.id:
                raise 400("previous_response_id does not belong to "
                           "the specified conversation")
            # Fork + explicit conversation is not allowed. Forks always
            # auto-create a new conversation (not yet implemented — for
            # now this is just a 400).
            latest = conversation_store.get_latest_response_id(conversation_id)
            if latest != previous_response_id:
                raise 400("conversation provided with a fork — "
                           "previous_response_id is not the latest response")

        # get() for steering check. None means deleted → 400.
        prev_task = task_store.get(previous_response_id)
        if not prev_task:
            raise 400("previous_response_id not found")

        if prev_task.status in ("in_progress", "queued"):
            # Steering: try to deliver to the running agent
            item = NewConversationItem(type="message",
                response_id=previous_response_id,
                data={"role": "user", "content": input})
            delivered = task_store.try_deliver(
                previous_response_id, conversation_id, item)
            if delivered:
                return prev_task  # still in progress, message delivered
            # Inbox closed — agent is finishing. Wait for it to complete so
            # its assistant output is in the conversation before we start a new
            # response that will load history.
            task_store.wait(previous_response_id)
    else:
        # New conversation — create a fresh conversation
        conversation = conversation_store.create_conversation()
        conversation_id = conversation.id

    # Normal: create a new response
    task = task_store.create(conversation_id=conversation_id, agent_id=agent_id,
        agent_name=agent.name, previous_response_id=previous_response_id)
    conversation_store.append(conversation_id, [
        NewConversationItem(type="message", response_id=task.task_id, data={"role": "user", "content": input})])
    # instructions and reasoning are pure workflow inputs — passed to DBOS, not stored in task row.
    task_store.start(task.task_id, instructions=instructions, reasoning=reasoning)
    return task
```

### The inbox handshake (race elimination)

The agent loop and the server coordinate via a lock to ensure no message falls into a gap between the agent finishing and the server accepting a steering message.

**Two atomic operations, both run as database transactions:**

1. **`try_deliver(task_id, conversation_id, message)`** — server side. Checks `inbox_closed` for the task. If `False`, appends the message to the conversation and returns `True`. If `True`, returns `False` (caller creates a new response instead).

2. **`close_inbox(task_id, conversation_id, last_seen_item_id)`** — agent side. Checks for messages newer than `last_seen_item_id`. If found, returns them (inbox stays open). If none, sets `inbox_closed=True` and returns empty list.

Because both operations run as transactions against the same database, serialization is guaranteed. Either:
- **Server writes first:** `try_deliver` appends the message, agent's `close_inbox` finds it → agent continues
- **Agent closes first:** `close_inbox` sets `inbox_closed=True`, server's `try_deliver` sees the flag → server creates new response

No message can fall into the void. The database provides the serialization.

**Requirement:** `inbox_closed` flag and the conversation messages must share the same database so both operations can run in a single transaction.

## Conversation state

History is loaded via `conversation_store.search_items(conversation_id)` at the start of each execution. The `conversation_id` groups turns belonging to the same conversation. The server resolves `previous_response_id` → `conversation_id` via `conversation_store.get_conversation_id(previous_response_id)`, which queries messages by their `response_id` field. This is the durable resolution path — it works even after task records have been cleaned up. Each message carries a `response_id` linking it to the response that produced it — this is exposed in the conversation items API.

## Request Lifecycles

### 1. Blocking (background=false, stream=false)

Simplest case. Server holds the connection until done.

1. Server receives `POST /v1/responses` with `background: false, stream: false`
2. If `previous_response_id` is set: `conversation_id = conversation_store.get_conversation_id(previous_response_id)`, then `prev_task = task_store.get(previous_response_id)` — if None (deleted), return 400. Otherwise: `conversation = conversation_store.create_conversation()` → `conversation_id = conversation.id`
3. `task = task_store.create(conversation_id=conversation_id, agent_id=agent_id, agent_name=agent.name, previous_response_id=previous_response_id)`
4. `conversation_store.append(conversation_id, [NewConversationItem(type="message", response_id=task.task_id, data={"role": "user", "content": [...]})])` — persist user input (durable before execution begins)
5. `task_store.start(task.task_id, instructions=instructions, reasoning=reasoning)` — starts the DBOS workflow asynchronously; compensating delete on failure
6. `task = task_store.wait(task.task_id)` — server blocks here
7. Runtime loads history via `conversation_store.search_items(conversation_id)` — includes the user message from step 4
8. Runtime runs the agent loop (LLM calls, tool calls, steering inbox checks between iterations, `write_stream()` for deltas)
9. Runtime calls `task_store.close_inbox(task.task_id, ...)` — if late messages, agent continues loop; if none, sets `inbox_closed=True`
10. Runtime appends assistant output via `conversation_store.append(conversation_id, [NewConversationItem(type="message", response_id=task.task_id, data={"role": "assistant", "agent": agent.name, "content": [...]})])` — only after inbox is confirmed closed
11. `wait()` returns the finished Task with output
12. Server returns 200 JSON response

- **On client disconnect**: server calls `task_store.cancel(task_id)` — execution stops, response is preserved with status "cancelled"

---

### 2. Foreground streaming (background=false, stream=true)

Server reads deltas in real time and forwards as SSE.

1. Server receives `POST /v1/responses` with `background: false, stream: true`
2. If `previous_response_id` is set: `conversation_id = conversation_store.get_conversation_id(previous_response_id)`, then `prev_task = task_store.get(previous_response_id)` — if None (deleted), return 400. Otherwise: `conversation = conversation_store.create_conversation()` → `conversation_id = conversation.id`
3. `task = task_store.create(conversation_id=conversation_id, agent_id=agent_id, agent_name=agent.name, previous_response_id=previous_response_id)`
4. `conversation_store.append(conversation_id, [NewConversationItem(type="message", response_id=task.task_id, data={"role": "user", "content": [...]})])` — persist user input
5. `task_store.start(task.task_id, instructions=instructions, reasoning=reasoning)` — starts the DBOS workflow asynchronously; compensating delete on failure
6. Server opens SSE connection and iterates `task_store.stream(task.task_id)`
7. Runtime loads history via `conversation_store.search_items(conversation_id)` — includes the user message from step 4
8. Runtime runs the agent loop — each `write_stream()` delta is yielded by `stream()`
9. Server converts each delta to an SSE event and writes it to the client
10. Runtime calls `task_store.close_inbox(...)` — confirms no late messages, sets `inbox_closed=True`
11. Runtime appends assistant output via `conversation_store.append(conversation_id, [NewConversationItem(type="message", response_id=task.task_id, data={"role": "assistant", "agent": agent.name, "content": [...]})])` — only after inbox is confirmed closed
12. Workflow exits — `finally` block runs `close_stream()`, ending the `stream()` iterator
13. Server calls `task_store.wait(task.task_id)` — the stream ending does not guarantee the workflow has fully exited (the `finally` block may still be running), so `wait()` is needed instead of `get()` to avoid a race where `get()` sees "in_progress" with empty output
14. Server builds and sends `response.completed` SSE event (with full response object from step 13), then `[DONE]`

- **On client disconnect**: server calls `task_store.cancel(task_id)` — execution stops, response is preserved with status "cancelled"

---

### 3. Background, no streaming (background=true, stream=false)

Fire and forget. Client polls GET for result.

1. Server receives `POST /v1/responses` with `background: true, stream: false`
2. If `previous_response_id` is set: `conversation_id = conversation_store.get_conversation_id(previous_response_id)`, then `prev_task = task_store.get(previous_response_id)` — if None (deleted), return 400. Otherwise: `conversation = conversation_store.create_conversation()` → `conversation_id = conversation.id`
3. `task = task_store.create(conversation_id=conversation_id, agent_id=agent_id, agent_name=agent.name, previous_response_id=previous_response_id)`
4. `conversation_store.append(conversation_id, [NewConversationItem(type="message", response_id=task.task_id, data={"role": "user", "content": [...]})])` — persist user input
5. `task_store.start(task.task_id, instructions=instructions, reasoning=reasoning)` — starts the DBOS workflow asynchronously; compensating delete on failure
6. Server returns 200 immediately with `{id: task.task_id, status: "queued", output: []}`
7. Runtime loads history via `conversation_store.search_items(conversation_id)` — includes the user message from step 4
8. Runtime runs the agent loop in the background
9. Runtime calls `task_store.close_inbox(...)` — confirms no late messages, sets `inbox_closed=True`
10. Runtime appends assistant output via `conversation_store.append(conversation_id, [NewConversationItem(type="message", response_id=task.task_id, data={"role": "assistant", "agent": agent.name, "content": [...]})])` — only after inbox is confirmed closed
11. Workflow completes — task status becomes "completed" with output
12. Client polls `GET /v1/responses/{task.task_id}` → `task_store.get()` → returns Task with output

- **Client disconnect has no effect** — execution continues regardless

---

### 4. Background streaming (background=true, stream=true)

The laptop-closing scenario. Durable execution + live streaming while connected.

1. Server receives `POST /v1/responses` with `background: true, stream: true`
2. If `previous_response_id` is set: `conversation_id = conversation_store.get_conversation_id(previous_response_id)`, then `prev_task = task_store.get(previous_response_id)` — if None (deleted), return 400. Otherwise: `conversation = conversation_store.create_conversation()` → `conversation_id = conversation.id`
3. `task = task_store.create(conversation_id=conversation_id, agent_id=agent_id, agent_name=agent.name, previous_response_id=previous_response_id)`
4. `conversation_store.append(conversation_id, [NewConversationItem(type="message", response_id=task.task_id, data={"role": "user", "content": [...]})])` — persist user input
5. `task_store.start(task.task_id, instructions=instructions, reasoning=reasoning)` — starts the DBOS workflow asynchronously; compensating delete on failure
6. Server opens SSE connection and iterates `task_store.stream(task.task_id)`
7. Runtime loads history via `conversation_store.search_items(conversation_id)` — includes the user message from step 4
8. Runtime runs the agent loop — deltas streamed to client via SSE (same as flow 2)
9. If client stays connected: stream ends, server calls `task_store.wait()`, sends `response.completed` + `[DONE]` (same as flow 2 steps 13-14)
10. If client disconnects mid-stream: server stops reading stream but does NOT cancel. Runtime continues running, closes inbox, appends assistant output on completion
11. Client reopens laptop — `GET /v1/responses/{task.task_id}` → `task_store.get()` → completed Task with output (or empty output if still running)

- **Disconnect does NOT cancel** — execution continues regardless
- Deltas pile up unread in the DBOS stream; only the final assembled output matters for GET

---

### 5. Multi-turn conversation (any background/stream combo)

Shows how `previous_response_id` chains turns into a conversation.

**Turn 1** — no `previous_response_id`, new conversation:

1. Server receives `POST /v1/responses` with `input: "hi"`, no `previous_response_id`
2. `conversation = conversation_store.create_conversation()` — new conversation
3. `task = task_store.create(conversation_id=conversation.id, agent_id=agent_id, )` — no previous_response_id
4. `conversation_store.append(conversation.id, [NewConversationItem(type="message", response_id=task.task_id, data={"role": "user", "content": [{"type": "input_text", "text": "hi"}]})])` — persist user input
5. `task_store.start(task.task_id)`
6. Runtime calls `conversation_store.search_items(conversation.id)` → returns `[ConversationItem(type="message", response_id=task.task_id, data={"role": "user", "content": [...]})]`
7. Runtime runs agent, produces "Hello! How can I help?"
8. Runtime calls `task_store.close_inbox(...)` — confirms no late messages, sets `inbox_closed=True`
9. Runtime appends assistant output: `conversation_store.append(conversation.id, [NewConversationItem(type="message", response_id=task.task_id, data={"role": "assistant", "agent": agent.name, "content": [{"type": "output_text", "text": "Hello!..."}]})])`
10. Server returns `{id: task.task_id, output: "Hello!..."}`

**Turn 2** — continuing the conversation:

1. Server receives `POST /v1/responses` with `input: "weather?"`, `previous_response_id: task.task_id` (from turn 1)
2. `conversation_id = conversation_store.get_conversation_id(previous_response_id)` — resolves via message response_id. `prev_task = task_store.get(previous_response_id)` — if None (deleted), return 400
3. `task2 = task_store.create(conversation_id=conversation_id, agent_id=agent_id, previous_response_id=previous_response_id)`
4. `conversation_store.append(conversation_id, [NewConversationItem(type="message", response_id=task2.task_id, data={"role": "user", "content": [...]})])` — persist user input
5. `task_store.start(task2.task_id)`
6. Runtime calls `conversation_store.search_items(conversation_id)` → returns turn 1's user + assistant messages, plus turn 2's user message
7. Runtime runs agent with history, calls tools, produces answer
8. Runtime calls `task_store.close_inbox(...)` — confirms no late messages, sets `inbox_closed=True`
9. Runtime appends assistant output with `response_id=task2.task_id`
10. Server returns `{id: task2.task_id, output: "It's..."}`

- `conversation_store` builds the full conversation; `task_store` only knows about individual executions
- The server resolves conversation via `conversation_store.get_conversation_id(previous_response_id)` — durable, survives task cleanup

---

### 6. Steering (message sent during in-progress response)

User sends a new message while the agent is still running.

**Case A: Message delivered (agent still running, inbox open):**

1. Client sends `POST /v1/responses` with `previous_response_id: "resp_001"` while resp_001 is in progress
2. Server resolves `conversation_id = conversation_store.get_conversation_id("resp_001")`. Server calls `task_store.get("resp_001")` — if None (deleted), return 400
3. Status is "in_progress"
4. Server calls `task_store.try_deliver("resp_001", conversation_id, NewConversationItem(type="message", response_id="resp_001", data={"role": "user", ...}))` — atomically checks `inbox_closed=False`, appends message to conversation, returns `True`
5. Server returns the existing in-progress response: `{id: "resp_001", status: "in_progress", ...}`
6. Runtime's next `search_items(conversation_id, after=last_seen)` call finds the new message
7. Runtime adds it to history and continues the agent loop — the LLM sees the steering input on the next iteration

**Case B: Inbox closed (agent finishing, message arrives too late):**

1. Client sends `POST /v1/responses` with `previous_response_id: "resp_001"` — resp_001 is about to complete
2. Server resolves `conversation_id = conversation_store.get_conversation_id("resp_001")`. Server calls `task_store.get("resp_001")` — if None (deleted), return 400
3. Status is "in_progress"
4. Server calls `task_store.try_deliver(...)` — `inbox_closed=True`, returns `False`
5. Server calls `task_store.wait("resp_001")` — waits for resp_001 to finish so its assistant output is persisted to the conversation before resp_002 loads history
6. Server creates `resp_002 = task_store.create(conversation_id=conversation_id, agent_id=agent_id, previous_response_id="resp_001")`
7. Server appends user message with `response_id=resp_002`
8. Server starts resp_002 — runtime loads full history (including resp_001's output from step 5) and processes the message as a new turn

**Case C: Response already completed:**

1. Client sends `POST /v1/responses` with `previous_response_id: "resp_001"` — resp_001 already completed
2. Server resolves `conversation_id = conversation_store.get_conversation_id("resp_001")`. Server calls `task_store.get("resp_001")` — if None (deleted), return 400
3. Status is "completed"
4. Server skips steering — goes directly to normal flow (create resp_002, append, start)

In all three cases, the client's message gets processed. No message falls into the void.

---

### 7. Cancel mid-execution

Stops execution but preserves the response. The client can continue the conversation from the cancelled response.

1. Client sends `POST /v1/responses` with `stream: true` — streaming begins
2. Server resolves conversation, creates task, appends user input to conversation, starts workflow, streams deltas via SSE
3. Client sends `POST /v1/responses/{task_id}/cancel`
4. Server calls `task_store.cancel(task_id)` → stops the DBOS workflow, sets status to "cancelled"
5. The user's input message remains in the conversation (appended before task start), but the assistant's incomplete output is not saved
6. Server returns the response with `{id: task_id, status: "cancelled", ...}`
7. `GET /v1/responses/{task_id}` still works — returns the preserved response with status "cancelled"
8. Client can continue the conversation: `POST /v1/responses` with `previous_response_id: task_id` creates a new response in the same conversation

---

### 8. Delete response

Removes a response entirely. Works on any stored response regardless of status — completed, cancelled, failed, etc.

1. Client sends `DELETE /v1/responses/{task_id}`
2. Server calls `task_store.delete(task_id)` → stops execution if in progress, then removes the task record
3. Conversation messages are unaffected by delete. The user's input message remains (appended before task start). If the response had already completed, the assistant's output also remains. If the response was in progress, the assistant's incomplete output was never persisted (the runtime only appends after `close_inbox`)
4. Server returns `{id: task_id, object: "response.deleted", deleted: true}`
5. Subsequent `GET /v1/responses/{task_id}` returns 404
6. The conversation is unaffected — the user's input message (and any prior turns) remain. `conversation_store.get_conversation_id(task_id)` still resolves via the persisted user message's `response_id`

### 9. Retrieve response (GET)

Returns the current state of a response. Always a JSON snapshot, never a stream.

1. Client sends `GET /v1/responses/{task_id}`
2. Server calls `task_store.get(task_id)`
3. If `None` (deleted or unknown): return 404
4. Server builds the response object from the Task and returns 200:
   - **queued**: `{id: task_id, status: "queued", output: [], completed_at: null}`
   - **in_progress**: `{id: task_id, status: "in_progress", output: [], completed_at: null}` — partial output is not available via GET; intermediate work is in the DBOS stream only
   - **completed**: `{id: task_id, status: "completed", output: [...], completed_at: ...}` — full output populated
   - **failed**: `{id: task_id, status: "failed", output: [], error: {...}}`
   - **incomplete**: `{id: task_id, status: "incomplete", output: [], incomplete_details: {...}}`
   - **cancelled**: `{id: task_id, status: "cancelled", output: []}`

- No side effects — purely a read operation
- Used by clients to poll for results in background mode (flows 3 and 4)
- Used by clients to check status after reconnecting (laptop-closing scenario)

---

## File structure

```
runtime/
├── __init__.py         # init() function, public entry point
├── models.py           # Data models: Task, ConversationItem, Conversation, etc.
├── durability.py       # ALL DBOS imports isolated here
├── steps.py            # @step functions: call_llm, call_tool, load_history
├── tool_manager.py     # MCP connections, local tool loading, routing
├── skill_manager.py    # Progressive skill disclosure
stores/
├── task_store/         # TaskStore interface
├── conversation_store/ # ConversationStore interface
```

## Not yet

### DBOS data garbage collection

Task records are never GC'd — they're lightweight (`task_id`, `conversation_id`, `status`, `inbox_closed`) and needed for `previous_response_id` resolution. But DBOS stores heavyweight data per workflow: step outputs (cached LLM responses, tool results) in `dbos.operation_outputs` and workflow metadata in `dbos.workflow_status`. This data is only needed for crash recovery while a workflow is in progress. Once a workflow completes and its assistant output is persisted to the conversation, the DBOS checkpoint data is dead weight.

When this matters, a background job can delete rows from DBOS's internal tables for completed workflows older than some threshold. Our task records stay forever. DBOS doesn't expose a "delete workflow data" API, so this would require direct deletes against its tables.

Out of scope for now — revisit when storage becomes a concern.

### Context management for long-horizon tasks

The agent loop's `history` list grows with every iteration — each LLM call and tool result appends to it. For long-horizon tasks (hundreds of iterations, complex tool chains), history will exceed the model's context window. The conversation store also accumulates unboundedly across turns.

**Two distinct problems:**

1. **Within a single execution:** The in-memory `history` list grows as the agent loops. At some point the LLM call will fail or degrade because the prompt exceeds the context window. The agent loop needs a strategy — truncation, summarization, sliding window — to keep history within the model's limits while preserving enough context for coherent behavior.

2. **Across turns:** Even with context management within a single execution, `conversation_store.search_items(conversation_id)` returns the full conversation history on each new turn. A 500-turn conversation will have thousands of messages. The `load_history` step needs a strategy for compressing or selecting from long conversations.

**What already works without changes:**

- The `status: "incomplete"` + `previous_response_id` pattern gives clients a natural "checkpoint and resume" mechanism. An agent that hits `max_iterations` returns incomplete, and the client follows up — a fresh execution loads history from the conversation and continues. This is pagination of execution without new machinery.
- `search_items` supports pagination (`max_results`, `page_token`) and ordering, so selective history loading is possible at the API level.

**What needs design:**

- A context management strategy for the agent loop (summarize, truncate, or window)
- Whether `max_iterations` should be configurable per-agent or per-request (via agent config or request params)
- Whether to add a time-based limit alongside the iteration-based one
- How summarization interacts with steering (summarized-away messages that contained steering input)

Out of scope for now — next area of focus after core flows are implemented.

### Fork detection and handling

When `previous_response_id` points to a non-latest response in a conversation (i.e., `conversation_store.get_latest_response_id(conversation_id) != previous_response_id`), this is a fork. The conversation field validation already rejects forks when an explicit `conversation` is provided (returns 400). But the implicit case — forking without specifying a conversation — needs handling:

1. Server detects the fork via `get_latest_response_id()`
2. Server creates a new conversation (new conversation)
3. Items up to and including the fork point are copied into the new conversation with new response IDs
4. The new response is added to the new conversation
5. The original conversation is unchanged — each conversation is always a linear thread

This requires a new ConversationStore method (e.g., `copy_up_to(source_conversation_id, up_to_response_id) -> Conversation`) and updates to the handler pseudocode to branch on the fork condition.

Out of scope for now — the handler currently only validates forks (400 when combined with explicit conversation). Implicit fork handling comes later.

### Conversation deletion cascade

Deleting a conversation (DELETE /v1/conversations/{id}) must:

1. Find all in-flight responses in the conversation and cancel them
2. Delete all response records (task records) in the conversation — subsequent GET /v1/responses/{id} returns 404 for every response that belonged to this conversation
3. Delete the conversation itself and all conversation messages

**Resolved.** `task_store.list_tasks(conversation_id=...)` provides the ability to find all tasks belonging to a conversation. The route layer uses this to cancel in-flight tasks before deleting. `conversation_store.delete_conversation()` handles the cascade (it is async because it may need to cancel in-flight responses first).

---

## Dependencies

- `spec/` — defines the agent contract (e.g. what's in a bundle), but is not directly referenced by the runtime stores. Agent identity is passed as an `agent_id` string; the runtime looks up the agent name via `agent_store.get(agent_id)` when needed for conversation item data.
- `dbos` — for durable execution (isolated in `durability.py`)
- `litellm` — LLM provider
