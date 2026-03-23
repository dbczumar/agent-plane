# Database Schema Design

Five tables in the default schema. DBOS manages its own tables (workflow_status,
operation_outputs, streams, etc.) in a separate `dbos` schema within the same database.

Tasks and items MUST share the same database — the steering handshake
(try_deliver + close_inbox) requires single-transaction atomicity.

Initial setup uses `Base.metadata.create_all(engine)`. No Alembic until the
schema stabilizes.

---

## agents

| Column | Type | Notes |
|---|---|---|
| id | String(64) PK | "ag_" + uuid4().hex |
| name | String(256) UNIQUE NOT NULL | Used as `model` in inference requests |
| description | Text | nullable |
| bundle_location | Text NOT NULL | Path to stored tarball; spec read from bundle at runtime |
| created_at | Integer NOT NULL | Unix epoch seconds |

**Indexes:** `uq_agents_name` (unique on name), `ix_agents_created_at`

---

## files

| Column | Type | Notes |
|---|---|---|
| id | String(64) PK | "file_" + uuid4().hex |
| filename | String(512) NOT NULL | Original filename |
| bytes | Integer NOT NULL | File size |
| content_location | Text NOT NULL | Path to binary on disk / artifact store |
| content_type | String(256) | MIME type, nullable |
| created_at | Integer NOT NULL | |

**Indexes:** `ix_files_created_at`

---

## sessions

Conversations.

| Column | Type | Notes |
|---|---|---|
| id | String(64) PK | "conv_" + uuid4().hex |
| title | Text | nullable, user-settable conversation title |
| created_at | Integer NOT NULL | |

**Indexes:** `ix_sessions_created_at`

---

## tasks

Responses. `task_id` = `response_id` = DBOS `workflow_uuid`.

| Column | Type | Notes |
|---|---|---|
| id | String(64) PK | "resp_" + uuid4().hex |
| session_id | String(64) NOT NULL | No FK — app-managed integrity |
| model | String(256) NOT NULL | Agent name |
| status | String(32) NOT NULL | Default "queued" |
| output | Text NOT NULL | Default "[]" (JSON array) |
| inbox_closed | Integer NOT NULL | Default 0 (0=open, 1=closed) |
| previous_response_id | String(64) | No FK — allows dangling after mid-chain delete |
| instructions | Text | nullable |
| metadata | Text NOT NULL | Default "{}" (JSON object, max 16 keys) |
| background | Integer NOT NULL | Default 0 (0=false, 1=true) |
| error | Text | JSON {code, message}, nullable |
| incomplete_details | Text | JSON {reason}, nullable |
| usage | Text | JSON {input_tokens, output_tokens, ...}, nullable |
| context_management | Text | JSON array, nullable |
| created_at | Integer NOT NULL | |
| completed_at | Integer | nullable |

**Indexes:** `ix_tasks_session_id`, `ix_tasks_model` (for agent deletion cascade),
`ix_tasks_created_at`

---

## items

Conversation items — messages, function calls, function call outputs, reasoning, etc.
Single table with a `type` discriminator and a JSON `data` blob for type-specific fields.

| Column | Type | Notes |
|---|---|---|
| id | String(64) PK | Prefixed by type: msg_, fc_, fco_, rs_ |
| session_id | String(64) NOT NULL | No FK |
| response_id | String(64) NOT NULL | No FK — must survive task deletion |
| model | String(256) NOT NULL | Denormalized from task |
| type | String(32) NOT NULL | message, function_call, function_call_output, reasoning |
| status | String(32) NOT NULL | Default "completed" |
| position | Integer NOT NULL | Ordering within session |
| data | Text NOT NULL | JSON blob — type-specific fields (see below) |
| created_at | Integer NOT NULL | |

**Indexes:** `ix_items_session_id_position` (composite), `ix_items_response_id`

### data column by type

**message:** `{"role": "user", "content": [{"type": "input_text", "text": "..."}]}`

**function_call:** `{"name": "get_weather", "arguments": "{...}", "call_id": "call_001"}`

**function_call_output:** `{"call_id": "call_001", "output": "{...}"}`

**reasoning:** `{"summary": [...], "content": null, "encrypted_content": null}`

---

## Design Decisions

### No foreign key constraints between tasks/sessions/items

Deletion semantics are too nuanced for FKs:
- Items must survive task deletion (rules out ON DELETE CASCADE from items to tasks)
- Mid-chain response deletion leaves dangling `previous_response_id` (explicitly allowed)
- Conversation deletion cascades in application-controlled order (cancel tasks first,
  then delete tasks, items, session)

Referential integrity is maintained at the application level.

### Single items table with JSON data column

We never filter by item-internal fields — all queries are by session_id, response_id,
or position. A discriminated union via `type` + JSON is simpler than separate tables per
item type and extends to future item types (compaction, mcp_tool_call, etc.) without
schema changes.

### position for item ordering

Guaranteed unique within a session (unlike timestamps). Makes cursor pagination clean:
`WHERE session_id = ? AND position > ?`. Assigned via `SELECT MAX(position) + 1`
within the same transaction as the INSERT.

### TEXT for JSON, Integer for booleans

Portable across SQLite and PostgreSQL. Application-level json.loads/json.dumps.
SQLite stores Boolean as INTEGER internally, so Integer(0/1) avoids ORM coercion
differences.

### model denormalized on items

Tasks can be deleted but items must retain model for the conversation items API.

---

## Store Method → DB Operation Mapping

### TaskStore

| Method | DB Operation |
|---|---|
| `create(spec, session_id, prev_id)` | INSERT INTO tasks |
| `start(task_id)` | Launch DBOS workflow (no DB write) |
| `get(task_id)` | SELECT FROM tasks WHERE id = ? |
| `wait(task_id)` | DBOS.retrieve_workflow().get_result(), then SELECT task |
| `stream(task_id)` | DBOS.read_stream(task_id, "output") |
| `try_deliver(task_id, session_id, msg)` | **Txn:** SELECT tasks.inbox_closed FOR UPDATE; if open → INSERT INTO items, return True; if closed → return False |
| `close_inbox(task_id, session_id, last_seen)` | **Txn:** SELECT items WHERE session_id = ? AND position > ?; if found → return them; if not → UPDATE tasks SET inbox_closed = 1, return [] |
| `cancel(task_id)` | DBOS.cancel_workflow(), then UPDATE tasks SET status = 'cancelled' |
| `delete(task_id)` | Cancel if in-progress, then DELETE FROM tasks WHERE id = ? (items untouched) |

### SessionStore

| Method | DB Operation |
|---|---|
| `create_session(metadata)` | INSERT INTO sessions |
| `get_session_id(response_id)` | SELECT session_id FROM items WHERE response_id = ? LIMIT 1 |
| `get_latest_response_id(session_id)` | SELECT response_id FROM items WHERE session_id = ? ORDER BY position DESC LIMIT 1 |
| `search_messages(session_id, after, ...)` | SELECT FROM items WHERE session_id = ? [AND position > ?] ORDER BY position LIMIT ? |
| `append(session_id, messages)` | **Txn:** SELECT MAX(position); INSERT items with incrementing position |

### API-Level (not in runtime stores)

| Operation | DB Operation |
|---|---|
| List conversations | SELECT FROM sessions ORDER BY created_at with cursor pagination |
| Delete conversation | Cancel in-flight tasks, DELETE tasks, DELETE items, DELETE session |
| List agents | SELECT FROM agents ORDER BY created_at with cursor pagination |
| Delete agent | Cancel in-flight tasks (by model), DELETE FROM agents |
| CRUD files | TBD — may be backed by artifact store instead of DB |

---

## Cursor-Based Pagination

All list endpoints use the same pattern. For a sort column (created_at for
agents/files/sessions, position for items):

```
after cursor:  WHERE sort_col > (SELECT sort_col FROM table WHERE id = :after_id)
before cursor: WHERE sort_col < (SELECT sort_col FROM table WHERE id = :before_id)
order "asc":   ORDER BY sort_col ASC LIMIT :limit + 1
order "desc":  ORDER BY sort_col DESC LIMIT :limit + 1
```

Fetch `limit + 1` rows. If more than `limit` returned, set `has_more = true`
and discard the extra row. `first_id` / `last_id` taken from the returned page.
