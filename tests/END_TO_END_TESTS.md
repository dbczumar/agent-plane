# End-to-End Test Scenarios

Each scenario is a complete client-side flow: multiple HTTP requests in sequence, with
exact request/response shapes and assertions at every step. No handwaving — every API
call is named, every field is checked, every state transition is verified.

**Conventions:**
- `→` marks the HTTP response. Fields shown are the ones being asserted.
- `assert` lines are explicit checks the test must make.
- Response IDs, conversation IDs, and message IDs use placeholders like `resp_1`, `conv_1`, `msg_1`.
  The test captures these from responses and uses them in subsequent requests.
- All scenarios assume a clean server (no prior state) unless noted.
- Agent behavior is controlled by the agent bundle — tests use purpose-built test agents
  (e.g., an agent that always responds with a fixed string, an agent that calls a specific
  tool, an agent that loops for N iterations before completing).

---

## Scenario 1: Single-turn blocking

Tests the simplest possible flow: one request, one response, one conversation.

**Setup:**

```
POST /api/agents
  Content-Type: multipart/form-data
  Parts: bundle=<echo-agent tarball>, name="echo-agent"

→ 201 Created
  {
    "id": "ag_1",
    "object": "agent",
    "name": "echo-agent",
    "created_at": <timestamp>
  }
```

**Step 1: Send a blocking request (no previous_response_id, no conversation)**

```
POST /v1/responses
  {
    "model": "echo-agent",
    "input": "Hello, world!",
    "stream": false,
    "background": false
  }

→ 200 OK
  {
    "id": "resp_1",
    "object": "response",
    "status": "completed",
    "model": "echo-agent",
    "created_at": <timestamp>,
    "completed_at": <timestamp>,
    "output": [
      {
        "id": "msg_out_1",
        "type": "message",
        "role": "assistant",
        "status": "completed",
        "content": [{"type": "output_text", "text": "<agent output>", "annotations": []}]
      }
    ],
    "background": false,
    "store": true,
    "previous_response_id": null,
    "conversation": {"id": "conv_1"},
    "instructions": null,
    "metadata": {},
    "error": null,
    "incomplete_details": null
  }

assert resp_1.status == "completed"
assert resp_1.previous_response_id is null
assert resp_1.conversation.id is not null  → capture as conv_1
assert resp_1.output is non-empty
assert resp_1.output[0].type == "message"
assert resp_1.output[0].role == "assistant"
assert resp_1.completed_at >= resp_1.created_at
```

**Step 2: GET the response — verify it matches**

```
GET /v1/responses/resp_1

→ 200 OK
  { same shape as step 1 response }

assert response matches step 1 exactly (id, status, output, conversation.id)
```

**Step 3: Verify the conversation was auto-created**

```
GET /v1/conversations

→ 200 OK
  {
    "object": "list",
    "data": [{"id": "conv_1", "object": "conversation", "created_at": <timestamp>}],
    "first_id": "conv_1",
    "last_id": "conv_1",
    "has_more": false
  }

assert exactly 1 conversation exists
assert conversation id matches conv_1 from step 1
```

**Step 4: Verify conversation items — should have user input + assistant output**

```
GET /v1/conversations/conv_1/items

→ 200 OK
  {
    "object": "list",
    "data": [
      {
        "id": "msg_in_1",
        "response_id": "resp_1",
        "model": "echo-agent",
        "type": "message",
        "role": "user",
        "status": "completed",
        "content": [{"type": "input_text", "text": "Hello, world!"}]
      },
      {
        "id": "msg_out_1",
        "response_id": "resp_1",
        "model": "echo-agent",
        "type": "message",
        "role": "assistant",
        "status": "completed",
        "content": [{"type": "output_text", "text": "<agent output>", "annotations": []}]
      }
    ],
    "first_id": "msg_in_1",
    "last_id": "msg_out_1",
    "has_more": false
  }

assert 2 items total
assert items[0].role == "user"
assert items[0].response_id == resp_1
assert items[0].content[0].text == "Hello, world!"
assert items[1].role == "assistant"
assert items[1].response_id == resp_1
assert items[1].content[0].text matches the output from step 1
assert items are in chronological order (msg_in_1.id < msg_out_1.id)
```

---

## Scenario 2: Multi-turn conversation (3 turns)

Tests that previous_response_id chains turns into a single conversation with
accumulated history. Verifies response_id linkage on every item.

**Setup:** Create `echo-agent` (same as scenario 1).

**Turn 1: New conversation**

```
POST /v1/responses
  {
    "model": "echo-agent",
    "input": "What is 2+2?",
    "stream": false,
    "background": false
  }

→ 200 OK
  {
    "id": "resp_1",
    "status": "completed",
    "conversation": {"id": "conv_1"},
    "previous_response_id": null,
    "output": [{"type": "message", "role": "assistant", ...}]
  }

assert resp_1.conversation.id → capture as conv_1
assert resp_1.previous_response_id is null
```

**Turn 2: Continue the conversation**

```
POST /v1/responses
  {
    "model": "echo-agent",
    "input": "Now multiply that by 3",
    "previous_response_id": "resp_1"
  }

→ 200 OK
  {
    "id": "resp_2",
    "status": "completed",
    "conversation": {"id": "conv_1"},
    "previous_response_id": "resp_1",
    "output": [{"type": "message", "role": "assistant", ...}]
  }

assert resp_2.id != resp_1  (new response)
assert resp_2.conversation.id == conv_1  (same conversation)
assert resp_2.previous_response_id == resp_1
```

**Turn 3: Continue again**

```
POST /v1/responses
  {
    "model": "echo-agent",
    "input": "Divide the result by 2",
    "previous_response_id": "resp_2"
  }

→ 200 OK
  {
    "id": "resp_3",
    "status": "completed",
    "conversation": {"id": "conv_1"},
    "previous_response_id": "resp_2",
    "output": [{"type": "message", "role": "assistant", ...}]
  }

assert resp_3.conversation.id == conv_1  (still same conversation)
assert resp_3.previous_response_id == resp_2
```

**Verify conversation items — 6 items (3 user + 3 assistant), correct response_id linkage**

```
GET /v1/conversations/conv_1/items

→ 200 OK
  {
    "data": [
      {"role": "user",      "response_id": "resp_1", "content": [{"text": "What is 2+2?"}]},
      {"role": "assistant",  "response_id": "resp_1", "content": [...]},
      {"role": "user",      "response_id": "resp_2", "content": [{"text": "Now multiply that by 3"}]},
      {"role": "assistant",  "response_id": "resp_2", "content": [...]},
      {"role": "user",      "response_id": "resp_3", "content": [{"text": "Divide the result by 2"}]},
      {"role": "assistant",  "response_id": "resp_3", "content": [...]}
    ]
  }

assert 6 items total
assert items alternate user/assistant
assert items[0].response_id == resp_1, items[1].response_id == resp_1
assert items[2].response_id == resp_2, items[3].response_id == resp_2
assert items[4].response_id == resp_3, items[5].response_id == resp_3
assert items are in chronological order
assert only 1 conversation exists (GET /v1/conversations returns 1 result)
```

**Verify each response is independently retrievable**

```
GET /v1/responses/resp_1 → 200, status=completed, conversation.id=conv_1
GET /v1/responses/resp_2 → 200, status=completed, conversation.id=conv_1
GET /v1/responses/resp_3 → 200, status=completed, conversation.id=conv_1
```

---

## Scenario 3: Background poll loop

Tests background execution: immediate return, polling, and final output retrieval.

**Setup:** Create `slow-agent` — an agent that takes several seconds to complete.

**Step 1: Start background execution**

```
POST /v1/responses
  {
    "model": "slow-agent",
    "input": "Do something slow",
    "background": true,
    "stream": false
  }

→ 200 OK
  {
    "id": "resp_1",
    "status": "queued",
    "output": [],
    "completed_at": null,
    "background": true,
    "conversation": {"id": "conv_1"}
  }

assert resp_1.status == "queued"
assert resp_1.output == []
assert resp_1.completed_at is null
assert resp_1.background == true
assert resp_1.conversation.id is not null  → capture as conv_1
```

**Step 2: Poll — expect queued or in_progress**

```
GET /v1/responses/resp_1

→ 200 OK
  {
    "id": "resp_1",
    "status": "queued" | "in_progress",
    "output": [],
    "completed_at": null
  }

assert status in ("queued", "in_progress")
assert output == []  (partial output not available via GET)
```

**Step 3: Poll until completed**

```
(loop with backoff)
GET /v1/responses/resp_1

→ 200 OK (eventually)
  {
    "id": "resp_1",
    "status": "completed",
    "output": [{"type": "message", "role": "assistant", ...}],
    "completed_at": <timestamp>
  }

assert status == "completed"
assert output is non-empty
assert completed_at is not null
assert completed_at > created_at
```

**Step 4: Verify conversation items exist**

```
GET /v1/conversations/conv_1/items

→ 200 OK
  { 2 items: user input + assistant output, both with response_id=resp_1 }

assert 2 items
assert items[0].role == "user", items[0].response_id == resp_1
assert items[1].role == "assistant", items[1].response_id == resp_1
```

---

## Scenario 4: Foreground streaming

Tests SSE event stream: event ordering, event shapes, and consistency with GET.

**Setup:** Create `echo-agent`.

**Step 1: Start streaming request**

```
POST /v1/responses
  {
    "model": "echo-agent",
    "input": "Tell me about cats",
    "stream": true,
    "background": false
  }

→ 200 OK
  Content-Type: text/event-stream

  event: response.created
  data: {"type": "response.created", "response": {"id": "resp_1", "status": "queued", ...}, "sequence_number": 0}

  event: response.in_progress
  data: {"type": "response.in_progress", "response": {"id": "resp_1", "status": "in_progress", ...}, "sequence_number": 1}

  event: response.output_item.added
  data: {"type": "response.output_item.added", "item": {"id": "msg_out_1", "type": "message", "role": "assistant", ...}, "sequence_number": 2}

  event: response.content_part.added
  data: {"type": "response.content_part.added", ..., "sequence_number": 3}

  event: response.output_text.delta
  data: {"type": "response.output_text.delta", "delta": "Cats ", "sequence_number": 4}

  event: response.output_text.delta
  data: {"type": "response.output_text.delta", "delta": "are ", "sequence_number": 5}

  ... (more text deltas)

  event: response.output_text.done
  data: {"type": "response.output_text.done", "text": "<full assembled text>", "sequence_number": N-4}

  event: response.content_part.done
  data: {"type": "response.content_part.done", ..., "sequence_number": N-3}

  event: response.output_item.done
  data: {"type": "response.output_item.done", "item": {...}, "sequence_number": N-2}

  event: response.completed
  data: {"type": "response.completed", "response": {"id": "resp_1", "status": "completed", "output": [...], ...}, "sequence_number": N-1}

  data: [DONE]
```

**Assertions on the event stream:**

```
assert first event is response.created with status "queued"
assert second event is response.in_progress
assert sequence_numbers are strictly incrementing (0, 1, 2, ...)
assert at least one response.output_text.delta event exists
assert concatenation of all delta texts equals the output_text.done text
assert response.output_item.added appears before any deltas for that item
assert response.output_item.done appears after output_text.done
assert response.completed is the last event before [DONE]
assert response.completed contains full response object with status "completed" and populated output
assert stream ends with "data: [DONE]"
```

**Step 2: GET after stream completes — verify consistency**

```
GET /v1/responses/resp_1

→ 200 OK
  {
    "id": "resp_1",
    "status": "completed",
    "output": [...]
  }

assert GET response matches the response object from the response.completed event
assert output text matches the assembled delta text from the stream
```

**Step 3: Verify conversation items**

```
GET /v1/conversations/conv_1/items

assert 2 items (user + assistant)
assert assistant content text matches the streamed output
```

---

## Scenario 4b: Foreground streaming with tool calls

Tests SSE event stream for an agent that calls a tool before producing a final
text response. Verifies function_call and function_call_output event shapes and
ordering, and that conversation items include all intermediate items.

**Setup:** Create `tool-agent` — an agent with a `get_weather` tool that always
calls it once before producing a final text response.

**Step 1: Start streaming request**

```
POST /v1/responses
  {
    "model": "tool-agent",
    "input": "What's the weather in SF?",
    "stream": true,
    "background": false
  }

→ 200 OK
  Content-Type: text/event-stream

  event: response.created
  data: {"type": "response.created", "response": {"id": "resp_1", "status": "queued", ...}, "sequence_number": 0}

  event: response.in_progress
  data: {"type": "response.in_progress", ..., "sequence_number": 1}

  — Function call output item —

  event: response.output_item.added
  data: {"type": "response.output_item.added", "item": {"id": "fc_1", "type": "function_call", "name": "get_weather", "call_id": "call_1", "status": "in_progress"}, "sequence_number": 2}

  event: response.function_call.arguments.delta
  data: {"type": "response.function_call.arguments.delta", "delta": "{\"loc", "sequence_number": 3}

  event: response.function_call.arguments.delta
  data: {"type": "response.function_call.arguments.delta", "delta": "ation\": \"SF\"}", "sequence_number": 4}

  event: response.function_call.arguments.done
  data: {"type": "response.function_call.arguments.done", "arguments": "{\"location\": \"SF\"}", "sequence_number": 5}

  event: response.output_item.done
  data: {"type": "response.output_item.done", "item": {"id": "fc_1", "type": "function_call", "name": "get_weather", "call_id": "call_1", "arguments": "{\"location\": \"SF\"}", "status": "completed"}, "sequence_number": 6}

  — Function call output result (produced by runtime after tool execution) —

  event: response.output_item.added
  data: {"type": "response.output_item.added", "item": {"id": "fco_1", "type": "function_call_output", "call_id": "call_1"}, "sequence_number": 7}

  event: response.output_item.done
  data: {"type": "response.output_item.done", "item": {"id": "fco_1", "type": "function_call_output", "call_id": "call_1", "output": "{\"temp\": 65, \"condition\": \"sunny\"}"}, "sequence_number": 8}

  — Final text message (after tool result is incorporated) —

  event: response.output_item.added
  data: {"type": "response.output_item.added", "item": {"id": "msg_1", "type": "message", "role": "assistant"}, "sequence_number": 9}

  event: response.content_part.added
  data: {"type": "response.content_part.added", ..., "sequence_number": 10}

  event: response.output_text.delta
  data: {"type": "response.output_text.delta", "delta": "It's 65°F", "sequence_number": 11}

  event: response.output_text.delta
  data: {"type": "response.output_text.delta", "delta": " and sunny in SF.", "sequence_number": 12}

  event: response.output_text.done
  data: {"type": "response.output_text.done", "text": "It's 65°F and sunny in SF.", "sequence_number": 13}

  event: response.content_part.done
  data: {"type": "response.content_part.done", ..., "sequence_number": 14}

  event: response.output_item.done
  data: {"type": "response.output_item.done", "item": {"id": "msg_1", ...}, "sequence_number": 15}

  event: response.completed
  data: {"type": "response.completed", "response": {"id": "resp_1", "status": "completed", "output": [fc_1, fco_1, msg_1], ...}, "sequence_number": 16}

  data: [DONE]
```

**Assertions on the event stream:**

```
assert sequence_numbers are strictly incrementing
assert function_call output_item.added appears before function_call.arguments.delta events
assert concatenation of arguments.delta texts equals arguments.done text
assert function_call output_item.done has the complete function_call item
assert function_call_output output_item appears after function_call output_item.done
assert function_call_output.call_id matches function_call.call_id
assert text message output_item.added appears after function_call_output (tool runs before final response)
assert response.completed output contains 3 items: [function_call, function_call_output, message]
```

**Step 2: GET after stream — verify output contains all 3 items**

```
GET /v1/responses/resp_1

→ 200 OK
  {
    "id": "resp_1",
    "status": "completed",
    "output": [
      {"id": "fc_1", "type": "function_call", "name": "get_weather", "call_id": "call_1",
       "arguments": "{\"location\": \"SF\"}", "status": "completed"},
      {"id": "fco_1", "type": "function_call_output", "call_id": "call_1",
       "output": "{\"temp\": 65, \"condition\": \"sunny\"}"},
      {"id": "msg_1", "type": "message", "role": "assistant", "status": "completed",
       "content": [{"type": "output_text", "text": "It's 65°F and sunny in SF.", "annotations": []}]}
    ]
  }

assert 3 output items: function_call, function_call_output, message
assert function_call.call_id == function_call_output.call_id
assert GET output matches the response.completed event output
```

**Step 3: Verify conversation items include all intermediate items**

```
GET /v1/conversations/conv_1/items

→ 200 OK
  {
    "data": [
      {"id": "msg_in_1", "response_id": "resp_1", "type": "message", "role": "user",
       "content": [{"type": "input_text", "text": "What's the weather in SF?"}]},
      {"id": "fc_1", "response_id": "resp_1", "type": "function_call",
       "name": "get_weather", "call_id": "call_1",
       "arguments": "{\"location\": \"SF\"}", "status": "completed"},
      {"id": "fco_1", "response_id": "resp_1", "type": "function_call_output",
       "call_id": "call_1", "output": "{\"temp\": 65, \"condition\": \"sunny\"}"},
      {"id": "msg_1", "response_id": "resp_1", "type": "message", "role": "assistant",
       "content": [{"type": "output_text", "text": "It's 65°F and sunny in SF.", "annotations": []}]}
    ]
  }

assert 4 conversation items: user message, function_call, function_call_output, assistant message
assert all items have response_id == resp_1
assert items are in chronological order
assert function_call.call_id == function_call_output.call_id
```

---

## Scenario 5: Laptop-closing (background streaming with disconnect)

Tests durable execution: client streams events, disconnects mid-stream, reconnects
via GET, and finds the completed response.

**Setup:** Create `slow-agent` (takes several seconds).

**Step 1: Start background streaming**

```
POST /v1/responses
  {
    "model": "slow-agent",
    "input": "Do a long task",
    "background": true,
    "stream": true
  }

→ 200 OK
  Content-Type: text/event-stream

  event: response.created
  data: {"type": "response.created", "response": {"id": "resp_1", "status": "queued", ...}, "sequence_number": 0}

  event: response.in_progress
  data: {"type": "response.in_progress", ..., "sequence_number": 1}

  event: response.output_text.delta
  data: {"type": "response.output_text.delta", "delta": "Starting...", "sequence_number": 2}

  ... (a few more deltas)
```

**Step 2: Client disconnects (closes SSE connection)**

```
(client closes the HTTP connection after receiving a few events)

assert: capture resp_1 from the response.created event before disconnecting
```

**Step 3: Agent continues running — no cancellation**

```
(server-side: because background=true, runtime continues executing)
(no client action during this time)
```

**Step 4: Client reconnects — poll via GET**

```
GET /v1/responses/resp_1

→ 200 OK
  {
    "id": "resp_1",
    "status": "in_progress" | "completed",
    "output": []  (if in_progress) | [...]  (if completed)
  }

If status == "in_progress":
  assert output == []
  (poll again until completed)

If status == "completed":
  assert output is non-empty
  assert completed_at is not null
```

**Step 5: Verify final state after completion**

```
GET /v1/responses/resp_1

→ 200 OK
  {
    "id": "resp_1",
    "status": "completed",
    "output": [{"type": "message", "role": "assistant", ...}]
  }

assert output is non-empty — the full response is available despite disconnect
assert conversation items have both user and assistant messages
```

**Step 6: Verify conversation is intact**

```
GET /v1/conversations/conv_1/items

assert 2 items (user + assistant), both with response_id=resp_1
assert assistant output matches GET response output
```

---

## Scenario 6: Steering — message delivered to running agent

Tests that a message sent while an agent is running is delivered via the inbox
handshake and incorporated into the agent's output. No new response is created.

**Setup:** Create `multi-iteration-agent` — an agent that loops multiple times
(e.g., calls a tool on each iteration) and takes long enough for the client to
send a steering message while it's running.

**Step 1: Start a long-running background response**

```
POST /v1/responses
  {
    "model": "multi-iteration-agent",
    "input": "Research the weather in major cities",
    "background": true,
    "stream": false
  }

→ 200 OK
  {
    "id": "resp_1",
    "status": "queued",
    "output": [],
    "conversation": {"id": "conv_1"}
  }
```

**Step 2: Wait for the agent to start running**

```
(poll GET /v1/responses/resp_1 until status == "in_progress")

GET /v1/responses/resp_1
→ 200 OK
  {"id": "resp_1", "status": "in_progress", "output": []}
```

**Step 3: Send a steering message while the agent is running**

```
POST /v1/responses
  {
    "model": "multi-iteration-agent",
    "input": "Focus only on San Francisco",
    "previous_response_id": "resp_1"
  }

→ 200 OK
  {
    "id": "resp_1",
    "status": "in_progress",
    "output": [],
    "conversation": {"id": "conv_1"}
  }

assert response.id == resp_1  (same response — NOT a new one)
assert status == "in_progress"
```

**What happened server-side:**

```
1. Server resolved session_id from resp_1
2. Server called task_store.get(resp_1) → status "in_progress"
3. Server called task_store.try_deliver(resp_1, session_id,
     NewMessage(role="user", content="Focus only on San Francisco",
                response_id=resp_1))
4. try_deliver checked inbox_closed=False → appended message to session → returned True
5. Server returned the existing in-progress response (no new task created)
6. Runtime's next search_messages(session_id, after=last_seen) found the steering message
7. Runtime added it to history — LLM sees "Focus only on San Francisco" on next iteration
```

**Step 4: Wait for the response to complete**

```
(poll GET /v1/responses/resp_1 until status == "completed")

GET /v1/responses/resp_1
→ 200 OK
  {
    "id": "resp_1",
    "status": "completed",
    "output": [{"type": "message", "role": "assistant", "content": [...]}]
  }

assert status == "completed"
assert output is non-empty
assert output text references San Francisco (the steering input was incorporated —
  the agent's output should demonstrably reflect "Focus only on San Francisco"
  rather than the original "major cities" scope. The test agent is purpose-built
  to include the city name in its output so this is deterministically verifiable.)
```

**Step 5: Verify conversation items — should include the steering message**

```
GET /v1/conversations/conv_1/items

→ 200 OK
  {
    "data": [
      {"role": "user", "response_id": "resp_1",
       "content": [{"text": "Research the weather in major cities"}]},
      {"role": "user", "response_id": "resp_1",
       "content": [{"text": "Focus only on San Francisco"}]},
      {"role": "assistant", "response_id": "resp_1",
       "content": [...]}
    ]
  }

assert 3 items total (2 user messages + 1 assistant message)
assert ALL items have response_id == resp_1 (steering message is attributed to the
  same response because it was delivered to the running agent, not a new response)
assert items[0] is the original user input
assert items[1] is the steering message
assert items[2] is the assistant output
assert only 1 conversation exists (no new conversation was created)
assert only 1 response exists (no new response was created)
```

**Step 6: Verify no extra responses exist**

```
GET /v1/conversations
→ 200 OK, data has exactly 1 conversation (conv_1)

(There is no resp_2 — steering didn't create a new response)
```

---

## Scenario 7: Steering — inbox closed, falls back to new response

Tests the case where the agent is finishing (inbox closed) when a steering message
arrives. The message becomes a new turn in the same conversation.

**Setup:** Create `fast-agent` — completes quickly (1-2 iterations).

**Step 1: Start a background response and wait for it to complete**

```
POST /v1/responses
  {
    "model": "fast-agent",
    "input": "Quick question: what is 2+2?",
    "background": true
  }

→ 200 OK
  {"id": "resp_1", "status": "queued", "conversation": {"id": "conv_1"}}

(poll until completed)
GET /v1/responses/resp_1
→ 200 OK
  {"id": "resp_1", "status": "completed", "output": [...]}
```

**Step 2: Send a follow-up with previous_response_id pointing to the completed response**

This exercises Case C from the steering flow: response already completed, so steering
is skipped and a new response is created in the same conversation.

```
POST /v1/responses
  {
    "model": "fast-agent",
    "input": "Now what is 3+3?",
    "previous_response_id": "resp_1"
  }

→ 200 OK
  {
    "id": "resp_2",
    "status": "completed",
    "conversation": {"id": "conv_1"},
    "previous_response_id": "resp_1",
    "output": [{"type": "message", "role": "assistant", ...}]
  }

assert resp_2.id != resp_1  (new response — steering was not possible)
assert resp_2.conversation.id == conv_1  (same conversation)
assert resp_2.previous_response_id == resp_1
```

**What happened server-side:**

```
1. Server resolved session_id from resp_1
2. Server called task_store.get(resp_1) → status "completed"
3. Status is NOT in ("in_progress", "queued") → skip steering
4. Server created a new task (resp_2) in the same session
5. Server appended user message with response_id=resp_2
6. Server started resp_2 → runtime loaded full history (resp_1's user + assistant + resp_2's user)
```

**Step 3: Verify conversation items — 4 items across 2 responses**

```
GET /v1/conversations/conv_1/items

→ 200 OK
  {
    "data": [
      {"role": "user",      "response_id": "resp_1", "content": [{"text": "Quick question: what is 2+2?"}]},
      {"role": "assistant",  "response_id": "resp_1", "content": [...]},
      {"role": "user",      "response_id": "resp_2", "content": [{"text": "Now what is 3+3?"}]},
      {"role": "assistant",  "response_id": "resp_2", "content": [...]}
    ]
  }

assert 4 items total
assert items[0..1] have response_id == resp_1
assert items[2..3] have response_id == resp_2
assert still only 1 conversation
```

**Timing variant (Case B — inbox closed but not yet completed):**

To specifically test Case B (try_deliver returns False because inbox_closed=True,
but the task hasn't reached terminal status yet), the test needs a controllable
agent that pauses between close_inbox and workflow completion. In this case:

```
1. Agent calls close_inbox → inbox_closed=True, no late messages
2. Agent is about to persist assistant output and return (but hasn't yet)
3. Client sends POST with previous_response_id=resp_1
4. Server: get(resp_1) → status "in_progress" (workflow still running)
5. Server: try_deliver → inbox_closed=True → returns False
6. Server: wait(resp_1) → blocks until resp_1 completes
7. Server: creates resp_2 in the same session
8. Result: same as above — resp_2 is a new response in conv_1
```

The client-visible behavior is identical to Case C: a new response is created.
The difference is purely timing (the wait in step 6). The test verifies the same
assertions — the distinction is which code path was exercised.

---

## Scenario 8: Cancel and continue

Tests cancellation mid-execution and then continuing the conversation from the
cancelled response.

**Setup:** Create `slow-agent`.

**Step 1: Start a blocking streaming response**

```
POST /v1/responses
  {
    "model": "slow-agent",
    "input": "Write a long essay about climate change",
    "stream": true,
    "background": true
  }

→ 200 OK (SSE stream)
  event: response.created
  data: {"type": "response.created", "response": {"id": "resp_1", ...}, "sequence_number": 0}

  event: response.in_progress
  data: {"type": "response.in_progress", ..., "sequence_number": 1}

  event: response.output_text.delta
  data: {"type": "response.output_text.delta", "delta": "Climate change...", "sequence_number": 2}

  ... (partial output streaming)

capture resp_1 and conv_1 from the stream
```

**Step 2: Cancel the response mid-execution**

```
POST /v1/responses/resp_1/cancel

→ 200 OK
  {
    "id": "resp_1",
    "status": "cancelled",
    "output": [],
    "conversation": {"id": "conv_1"},
    "error": null,
    "incomplete_details": null
  }

assert status == "cancelled"
assert output == []  (assistant's incomplete output was not persisted)
```

**Step 3: Verify the cancelled response is still retrievable**

```
GET /v1/responses/resp_1

→ 200 OK
  {
    "id": "resp_1",
    "status": "cancelled",
    "output": [],
    "conversation": {"id": "conv_1"}
  }

assert status == "cancelled"
assert response is NOT deleted — it's preserved
```

**Step 4: Verify conversation items — user input is preserved, no assistant output**

```
GET /v1/conversations/conv_1/items

→ 200 OK
  {
    "data": [
      {"role": "user", "response_id": "resp_1",
       "content": [{"text": "Write a long essay about climate change"}]}
    ]
  }

assert 1 item (user input only — assistant output was never persisted because
  the runtime only appends after close_inbox, which never completed)
```

**Step 5: Continue the conversation from the cancelled response**

```
POST /v1/responses
  {
    "model": "slow-agent",
    "input": "Actually, write a short summary instead",
    "previous_response_id": "resp_1"
  }

→ 200 OK
  {
    "id": "resp_2",
    "status": "completed",
    "conversation": {"id": "conv_1"},
    "previous_response_id": "resp_1",
    "output": [{"type": "message", "role": "assistant", ...}]
  }

assert resp_2.id != resp_1  (new response)
assert resp_2.conversation.id == conv_1  (same conversation)
assert resp_2.previous_response_id == resp_1
assert status == "completed"
```

**What happened server-side:**

```
1. Server resolved session_id from resp_1 (user message exists with response_id=resp_1)
2. task_store.get(resp_1) → status "cancelled" (not in_progress → skip steering)
3. Server created resp_2 in the same session
4. Runtime loaded history: [user("Write a long essay..."), user("Actually, write a short summary")]
   Note: no assistant message from resp_1 — it was cancelled before persisting output
5. Runtime completed resp_2 with a short summary
```

**Step 6: Verify final conversation state**

```
GET /v1/conversations/conv_1/items

→ 200 OK
  {
    "data": [
      {"role": "user",      "response_id": "resp_1", "content": [{"text": "Write a long essay about climate change"}]},
      {"role": "user",      "response_id": "resp_2", "content": [{"text": "Actually, write a short summary instead"}]},
      {"role": "assistant",  "response_id": "resp_2", "content": [...]}
    ]
  }

assert 3 items total
assert no assistant message from resp_1 (cancelled — output never persisted)
assert resp_2's assistant output is present
assert both responses are in the same conversation
```

---

## Scenario 9: Delete response

Tests that deleting a response removes the task record (GET returns 404) but
leaves conversation items intact.

**Setup:** Create `echo-agent`.

**Step 1: Complete a two-turn conversation**

```
POST /v1/responses
  {"model": "echo-agent", "input": "Turn 1"}
→ 200 OK  {"id": "resp_1", "status": "completed", "conversation": {"id": "conv_1"}}

POST /v1/responses
  {"model": "echo-agent", "input": "Turn 2", "previous_response_id": "resp_1"}
→ 200 OK  {"id": "resp_2", "status": "completed", "conversation": {"id": "conv_1"}}
```

**Step 2: Delete the first response**

```
DELETE /v1/responses/resp_1

→ 200 OK
  {"id": "resp_1", "object": "response.deleted", "deleted": true}
```

**Step 3: GET the deleted response — 404**

```
GET /v1/responses/resp_1
→ 404 Not Found
```

**Step 4: GET the second response — still works**

```
GET /v1/responses/resp_2

→ 200 OK
  {
    "id": "resp_2",
    "status": "completed",
    "conversation": {"id": "conv_1"},
    "previous_response_id": "resp_1"
  }

assert resp_2 is still retrievable
assert resp_2.previous_response_id still references resp_1 (the field is preserved
  even though the referenced response is deleted)
```

**Step 5: Conversation items still have all messages**

```
GET /v1/conversations/conv_1/items

→ 200 OK
  {
    "data": [
      {"role": "user",      "response_id": "resp_1", "content": [{"text": "Turn 1"}]},
      {"role": "assistant",  "response_id": "resp_1", "content": [...]},
      {"role": "user",      "response_id": "resp_2", "content": [{"text": "Turn 2"}]},
      {"role": "assistant",  "response_id": "resp_2", "content": [...]}
    ]
  }

assert 4 items — delete does NOT remove messages from the conversation
assert items with response_id=resp_1 are still present
```

**Step 6: Can still continue the conversation via resp_2**

```
POST /v1/responses
  {"model": "echo-agent", "input": "Turn 3", "previous_response_id": "resp_2"}

→ 200 OK
  {"id": "resp_3", "status": "completed", "conversation": {"id": "conv_1"}}

assert conversation continues normally
```

**Step 7: Can still resolve session from the deleted response's messages**

```
POST /v1/responses
  {"model": "echo-agent", "input": "Continue from turn 1", "previous_response_id": "resp_1"}
```

This should still work because `get_session_id(resp_1)` resolves via persisted
messages (which still exist), not via the task record (which is deleted). However,
`task_store.get(resp_1)` returns None, so the handler returns 400.

```
→ 400 Bad Request ("previous_response_id not found")
```

Wait — there's a subtlety here. `get_session_id` succeeds (messages exist), but
`task_store.get` returns None (task deleted). The handler raises 400. This is correct:
the response is deleted, so it shouldn't be referenceable as previous_response_id.

```
assert 400 — deleted responses cannot be used as previous_response_id
```

---

## Scenario 10: Delete conversation (with in-flight response)

Tests that deleting a conversation cancels any in-flight responses and removes
the conversation entirely.

**Setup:** Create `slow-agent`.

**Step 1: Start a two-turn conversation with the second turn in progress**

```
POST /v1/responses
  {"model": "slow-agent", "input": "Turn 1"}
→ 200 OK  {"id": "resp_1", "status": "completed", "conversation": {"id": "conv_1"}}

POST /v1/responses
  {"model": "slow-agent", "input": "Turn 2 — take your time", "previous_response_id": "resp_1", "background": true}
→ 200 OK  {"id": "resp_2", "status": "queued", "conversation": {"id": "conv_1"}}

(poll until resp_2 is in_progress)
GET /v1/responses/resp_2
→ 200 OK  {"status": "in_progress"}
```

**Step 2: Delete the conversation while resp_2 is running**

```
DELETE /v1/conversations/conv_1

→ 200 OK
  {"id": "conv_1", "object": "conversation.deleted", "deleted": true}
```

**What happened server-side:**

```
1. Server identified all in-flight responses in conv_1 → [resp_2]
2. Server called task_store.cancel(resp_2) → stopped execution, status="cancelled"
3. Server deleted the conversation and all associated data
```

**Step 3: Verify conversation is gone**

```
GET /v1/conversations/conv_1
→ 404 Not Found

GET /v1/conversations/conv_1/items
→ 404 Not Found
```

**Step 4: Verify responses are gone**

```
GET /v1/responses/resp_1
→ 404 Not Found

GET /v1/responses/resp_2
→ 404 Not Found
```

**Step 5: Verify conversation list is empty**

```
GET /v1/conversations

→ 200 OK
  {"data": [], "has_more": false}

assert no conversations remain
```

---

## Scenario 11: Validation error cases

Tests all documented 400/404 error paths.

**Setup:** Create `echo-agent`.

### 11a: Unknown model → 404

```
POST /v1/responses
  {"model": "nonexistent-agent", "input": "hello"}

→ 404 Not Found

assert error indicates unknown model
```

### 11b: Invalid previous_response_id (deleted) → 400

```
POST /v1/responses
  {"model": "echo-agent", "input": "Turn 1"}
→ 200 OK  {"id": "resp_1"}

DELETE /v1/responses/resp_1
→ 200 OK

POST /v1/responses
  {"model": "echo-agent", "input": "Continue", "previous_response_id": "resp_1"}

→ 400 Bad Request

assert error indicates previous_response_id not found
```

### 11c: Invalid previous_response_id (never existed) → 400

```
POST /v1/responses
  {"model": "echo-agent", "input": "Continue", "previous_response_id": "resp_does_not_exist"}

→ 400 Bad Request

assert error indicates previous_response_id not found
```

### 11d: store: false → 400

```
POST /v1/responses
  {"model": "echo-agent", "input": "hello", "store": false}

→ 400 Bad Request

assert error indicates store: false is not supported
```

### 11e: conversation without previous_response_id → 400

```
POST /v1/responses
  {"model": "echo-agent", "input": "hello", "conversation": {"id": "conv_123"}}

→ 400 Bad Request

assert error indicates conversation requires previous_response_id
```

### 11f: conversation/response mismatch → 400

```
# Create two separate conversations
POST /v1/responses
  {"model": "echo-agent", "input": "Conv A"}
→ 200 OK  {"id": "resp_A", "conversation": {"id": "conv_A"}}

POST /v1/responses
  {"model": "echo-agent", "input": "Conv B"}
→ 200 OK  {"id": "resp_B", "conversation": {"id": "conv_B"}}

# Try to continue resp_A but claim it's in conv_B
POST /v1/responses
  {
    "model": "echo-agent",
    "input": "Continue",
    "previous_response_id": "resp_A",
    "conversation": {"id": "conv_B"}
  }

→ 400 Bad Request

assert error indicates previous_response_id does not belong to the specified conversation
```

### 11g: Fork + explicit conversation → 400

```
# Create a 2-turn conversation
POST /v1/responses
  {"model": "echo-agent", "input": "Turn 1"}
→ 200 OK  {"id": "resp_1", "conversation": {"id": "conv_1"}}

POST /v1/responses
  {"model": "echo-agent", "input": "Turn 2", "previous_response_id": "resp_1"}
→ 200 OK  {"id": "resp_2", "conversation": {"id": "conv_1"}}

# Now try to fork from resp_1 (not latest) with explicit conversation
POST /v1/responses
  {
    "model": "echo-agent",
    "input": "Fork from turn 1",
    "previous_response_id": "resp_1",
    "conversation": {"id": "conv_1"}
  }

→ 400 Bad Request

assert error indicates fork + explicit conversation is not allowed
(resp_1 is not the latest response in conv_1 — resp_2 is)
```

### 11h: Cancel non-existent response → 404

```
POST /v1/responses/resp_nonexistent/cancel
→ 404 Not Found
```

### 11i: Delete non-existent response → 404

```
DELETE /v1/responses/resp_nonexistent
→ 404 Not Found
```

### 11j: GET non-existent response → 404

```
GET /v1/responses/resp_nonexistent
→ 404 Not Found
```

### 11k: Delete already-deleted response → 404

```
POST /v1/responses
  {"model": "echo-agent", "input": "hello"}
→ 200 OK  {"id": "resp_1"}

DELETE /v1/responses/resp_1
→ 200 OK

DELETE /v1/responses/resp_1
→ 404 Not Found
```

### 11l: Cancel a completed response → 400

```
POST /v1/responses
  {"model": "echo-agent", "input": "hello"}
→ 200 OK  {"id": "resp_1", "status": "completed"}

POST /v1/responses/resp_1/cancel

→ 400 Bad Request

assert error indicates response is already completed (cannot cancel a terminal response)
```

### 11m: Cancel an already-cancelled response → 400

```
POST /v1/responses
  {"model": "slow-agent", "input": "take your time", "background": true}
→ 200 OK  {"id": "resp_1", "status": "queued"}

POST /v1/responses/resp_1/cancel
→ 200 OK  {"id": "resp_1", "status": "cancelled"}

POST /v1/responses/resp_1/cancel

→ 400 Bad Request

assert error indicates response is already cancelled (cancel is not idempotent
  on terminal states)
```

### 11n: Cancel a queued response → 200

```
POST /v1/responses
  {"model": "slow-agent", "input": "take your time", "background": true}
→ 200 OK  {"id": "resp_1", "status": "queued"}

POST /v1/responses/resp_1/cancel

→ 200 OK
  {"id": "resp_1", "status": "cancelled"}

assert status == "cancelled"
assert cancellation works on queued responses (not just in_progress)

GET /v1/responses/resp_1
→ 200 OK  {"status": "cancelled"}
```

### 11o: Invalid input format → 400

```
POST /v1/responses
  {"model": "echo-agent", "input": 12345}

→ 400 Bad Request

assert error indicates invalid input format (input must be string or array)
```

```
POST /v1/responses
  {"model": "echo-agent", "input": [{"type": "unknown_type", "data": "???"}]}

→ 400 Bad Request

assert error indicates invalid input format (unrecognized item type)
```

---

## Scenario 12: Agent CRUD + cascade delete

Tests the full agent lifecycle: create, list, get, use for inference, then delete
with cascade cancellation of in-flight responses.

**Step 1: Create two agents**

```
POST /api/agents
  Parts: bundle=<echo-agent tarball>, name="agent-alpha"
→ 201 Created  {"id": "ag_1", "name": "agent-alpha"}

POST /api/agents
  Parts: bundle=<slow-agent tarball>, name="agent-beta"
→ 201 Created  {"id": "ag_2", "name": "agent-beta"}
```

**Step 2: List agents**

```
GET /api/agents

→ 200 OK
  {
    "object": "list",
    "data": [
      {"id": "ag_2", "name": "agent-beta", ...},
      {"id": "ag_1", "name": "agent-alpha", ...}
    ],
    "first_id": "ag_2",
    "last_id": "ag_1",
    "has_more": false
  }

assert 2 agents
assert ordered by created_at descending (ag_2 first — newest)
```

**Step 3: Get a specific agent**

```
GET /api/agents/ag_1

→ 200 OK
  {"id": "ag_1", "object": "agent", "name": "agent-alpha", ...}
```

**Step 4: Create a duplicate name → 409**

```
POST /api/agents
  Parts: bundle=<tarball>, name="agent-alpha"

→ 409 Conflict

assert error indicates name already exists
```

**Step 5: Use agent-beta for two responses — one completed, one in-flight**

```
POST /v1/responses
  {"model": "agent-beta", "input": "Task 1"}
→ 200 OK  {"id": "resp_1", "status": "completed", "conversation": {"id": "conv_1"}}

POST /v1/responses
  {"model": "agent-beta", "input": "Task 2 — long running", "background": true}
→ 200 OK  {"id": "resp_2", "status": "queued", "conversation": {"id": "conv_2"}}

(poll until resp_2 is in_progress)
GET /v1/responses/resp_2
→ 200 OK  {"status": "in_progress"}
```

**Step 6: Delete agent-beta — cascade cancels in-flight responses**

```
DELETE /api/agents/ag_2

→ 200 OK
  {"id": "ag_2", "object": "agent.deleted", "deleted": true}
```

**What happened server-side:**

```
1. Server found all in-flight responses for agent-beta → [resp_2]
2. Server called task_store.cancel(resp_2) → execution stopped, status="cancelled"
3. Server deleted the agent record
```

**Step 7: Verify agent-beta is gone**

```
GET /api/agents/ag_2
→ 404 Not Found

GET /api/agents
→ 200 OK  {"data": [{"id": "ag_1", "name": "agent-alpha"}], ...}

assert only agent-alpha remains
```

**Step 8: Verify in-flight response was cancelled**

```
GET /v1/responses/resp_2

→ 200 OK
  {"id": "resp_2", "status": "cancelled"}

assert status == "cancelled"  (not deleted — the response is preserved)
```

**Step 9: Verify completed response is unaffected**

```
GET /v1/responses/resp_1

→ 200 OK
  {"id": "resp_1", "status": "completed"}

assert completed response is still retrievable
```

**Step 9b: Verify conversations survive agent deletion**

```
GET /v1/conversations

→ 200 OK
  {
    "data": [
      {"id": "conv_2", ...},
      {"id": "conv_1", ...}
    ],
    "has_more": false
  }

assert both conversations still exist (agent deletion does NOT delete conversations)

GET /v1/conversations/conv_1/items
→ 200 OK  { 2 items: user + assistant for "Task 1" }

assert conversation items for conv_1 are intact

GET /v1/conversations/conv_2/items
→ 200 OK  { 1 item: user input for "Task 2 — long running" (no assistant output — was cancelled) }

assert conversation items for conv_2 are intact (user input preserved, no assistant output)
```

**Step 10: Cannot create new responses for deleted agent**

```
POST /v1/responses
  {"model": "agent-beta", "input": "hello"}

→ 404 Not Found

assert error indicates unknown model
```

**Step 11: agent-alpha still works**

```
POST /v1/responses
  {"model": "agent-alpha", "input": "Still here?"}

→ 200 OK
  {"id": "resp_3", "status": "completed", "model": "agent-alpha"}

assert agent-alpha is fully functional
```

**Step 12: Clean up — delete agent-alpha**

```
DELETE /api/agents/ag_1
→ 200 OK  {"id": "ag_1", "object": "agent.deleted", "deleted": true}

GET /api/agents
→ 200 OK  {"data": [], "has_more": false}

assert no agents remain
```
