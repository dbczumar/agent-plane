# Audit: responses.py vs API.md / RUNTIME.md

Auditor compared `/agent_plane/server/routes/responses.py` and `/agent_plane/server/models.py`
against the spec in `/agent_plane/server/API.md` (Inference section, Status Lifecycle,
Background x Stream Behavior Matrix) and the runtime design in `/agent_plane/runtime/RUNTIME.md`
(handler pseudocode lines 456-527, all 9 flows lines 518-755).

---

## 1. POST /responses — Validation

1. **store:false -> 400**
   ✅ Lines 86-89 of `responses.py`: raises HTTPException 400 when `req.store` is falsy.

2. **Unknown model -> 404**
   ✅ Lines 98-102: `get_agent_by_name(req.model)` returns None -> 404.

3. **Invalid previous_response_id -> 400**
   ✅ Lines 116-124: `conversation_store.get_conversation_id()` raises -> caught -> 400 "invalid previous_response_id". Lines 148-153: `task_store.get()` returns None -> 400 "previous_response_id not found".

4. **Conversation without previous_response_id -> 400**
   ✅ Lines 104-109: checks `req.conversation and not req.previous_response_id` -> 400.

5. **Conversation / response mismatch -> 400**
   ✅ Lines 127-135: `conversation_id != req.conversation.id` -> 400.

6. **Fork + explicit conversation -> 400**
   ✅ Lines 136-145: `latest != req.previous_response_id` -> 400.

7. **Invalid input format -> 400**
   ✅ Lines 92-95: checks `isinstance(req.input, (str, list))` -> 400. Also, Pydantic model
   `CreateResponseRequest.input` is typed `str | list[Any]`, so Pydantic will reject
   other types before the handler runs.

---

## 2. POST /responses — Handler Flow (RUNTIME.md pseudocode compliance)

8. **Conversation resolution: `conversation_store.get_conversation_id(previous_response_id)`**
   ✅ Lines 117-119: uses `conversation_store.get_conversation_id(req.previous_response_id)`.

9. **Steering check: prev_task status in ("in_progress", "queued") -> try_deliver**
   ✅ Lines 157-173: checks status, builds `NewMessage`, calls `task_store.try_deliver`.

10. **Steering delivered -> return existing in-progress response**
    ✅ Lines 166-169: if `delivered`, returns `_build_response_object(prev_task)`.

11. **Steering inbox closed -> wait for previous response before creating new**
    ✅ Lines 172-173: calls `await task_store.wait(req.previous_response_id)`.

12. **No previous_response_id -> create fresh conversation**
    ✅ Lines 176-177: `conversation_store.create_conversation()`.

13. **Normal flow: create task, append user message, start**
    ✅ Lines 180-198: `task_store.create(...)`, `conversation_store.append(...)`, `task_store.start(...)`.

14. **`context_management` field accepted on request**
    ❌ **FAIL**: API.md (lines 451-459) specifies a `context_management` field on the request
    (`array | null, optional, default: null`). The `CreateResponseRequest` model in `models.py`
    does not include this field. Pydantic by default silently drops unknown fields, so requests
    with `context_management` won't error — but the value is never passed to `task_store.create()`
    or the runtime. The field is defined in the spec but completely unimplemented.

---

## 3. POST /responses — Background x Stream Matrix

15. **background=true, stream=false: return immediately with queued status**
    ✅ Lines 201-202: returns `_build_response_object(task)` immediately. Since
    `task_store.create()` produces a task with status "queued", the response will have
    `status: "queued"` and `output: []` per spec.

16. **background=false, stream=false: blocking wait, return final task**
    ✅ Lines 272-273: `await task_store.wait(task.task_id)` then returns the finished
    response object.

17. **stream=true (foreground): SSE events created -> in_progress -> deltas -> completed -> [DONE]**
    ✅ Lines 207-264: generator yields `response.created`, `response.in_progress`, then
    iterates `task_store.stream()`, then yields terminal event, then `data: [DONE]\n\n`.

18. **stream=true (background): same streaming behavior**
    ✅ Lines 204-205: `if req.stream:` handles both background=true and background=false
    streaming identically. Per the spec (API.md line 705), background+stream streams
    events the same way; the difference is durability on disconnect (which is a runtime
    concern, not a route concern).

19. **background+stream: `response.queued` event missing**
    ❌ **FAIL**: API.md lines 720-722 states that for background+stream, the stream shows
    `response.created -> response.queued -> response.in_progress -> text deltas -> response.completed`.
    The implementation emits `response.created -> response.in_progress` (skipping `response.queued`).
    The `response.queued` event type is listed in the spec (line 588) as a supported streaming event.
    For background responses, there should be a `response.queued` event between `response.created`
    and `response.in_progress`.

---

## 4. POST /responses — Streaming Events

20. **SSE format: `event: type\ndata: json\n\n`**
    ✅ `_format_sse()` on lines 64-67 produces `event: {type}\ndata: {json}\n\n`.

21. **sequence_number included on every event**
    ✅ Lines 216-218, 230-234, 241-242, 258-260: every event gets `sequence_number`.

22. **Stream ends with `data: [DONE]`**
    ✅ Line 264: `yield "data: [DONE]\n\n"`.

23. **Terminal event uses actual status (completed/failed/incomplete/cancelled)**
    ✅ Line 253: `terminal_event = f"response.{final_task.status}"` — dynamically builds
    the correct event type from the task's actual terminal status.

24. **After stream ends, `wait()` is called before emitting terminal event**
    ✅ Lines 248: `final_task = await task_store.wait(task.task_id)` — matches RUNTIME.md
    flow 2 step 13 which explicitly calls for `wait()` after the stream iterator ends.

---

## 5. GET /responses/{id}

25. **Returns 404 for missing/deleted responses**
    ✅ Lines 279-283: `task_store.get()` returns None -> 404.

26. **Response shape matches API.md**
    ✅ `_build_response_object()` maps all Task fields to `ResponseObject` which has all
    required fields (see item 30 below for full field check).

27. **Output empty for non-completed statuses**
    ⚠️ **PARTIAL**: The spec (API.md line 638 and RUNTIME.md lines 746-750) says output is empty
    for in_progress, failed, incomplete, and cancelled statuses. The implementation returns
    `task.output` directly (line 37). This means the behavior depends entirely on the runtime
    correctly keeping `task.output` empty for non-completed statuses. The route layer does not
    enforce this invariant — if the runtime ever puts partial data in `task.output` before
    setting status to "completed", it would leak via GET. The spec says "partial output is not
    available via GET" (API.md line 638). This is a correctness dependency on the runtime rather
    than an explicit enforcement in the route.

---

## 6. POST /responses/{id}/cancel

28. **Returns 404 for missing responses**
    ✅ Lines 291-294: `task_store.get()` returns None -> 404.

29. **Returns 400 for already-terminal responses**
    ✅ Lines 295-302: checks `task.status in _TERMINAL_STATUSES` -> 400. Note: the API.md spec
    does not explicitly call out 400 for already-terminal responses (it only lists 404), but
    this is reasonable defensive behavior matching the OpenAI reference implementation. The spec
    says "Stops execution if in progress" — returning 400 for already-terminal is a sensible
    interpretation.

30. **Returns the response with status "cancelled"**
    ✅ Lines 303-304: `task_store.cancel()` returns the cancelled task, which is converted to
    a ResponseObject.

---

## 7. DELETE /responses/{id}

31. **Returns 404 for missing responses**
    ✅ Lines 311-314: `task_store.get()` returns None -> 404.

32. **Returns correct deleted shape (id, object: "response.deleted", deleted: true)**
    ✅ Lines 315-316: returns `ResponseDeleted(id=response_id)`. The `ResponseDeleted` model
    (models.py lines 146-149) has `object: str = "response.deleted"` and `deleted: bool = True`.

---

## 8. ResponseObject Shape — Field-by-field check against API.md

The spec (API.md lines 485-524) defines the response object shape. Checking each field:

33. **id** ✅ Present (models.py line 128).
34. **object** ✅ Present, defaults to `"response"` (line 129).
35. **status** ✅ Present (line 130).
36. **model** ✅ Present (line 131).
37. **created_at** ✅ Present (line 132).
38. **completed_at** ✅ Present, defaults to None (line 133).
39. **output** ✅ Present, defaults to `[]` (line 134).
40. **background** ✅ Present, defaults to False (line 135).
41. **store** ✅ Present, defaults to True (line 136).
42. **usage** ✅ Present, defaults to None (line 137).
43. **previous_response_id** ✅ Present, defaults to None (line 138).
44. **conversation** ✅ Present, defaults to None (line 139).
45. **instructions** ✅ Present, defaults to None (line 140).
46. **metadata** ✅ Present, defaults to `{}` (line 141).
47. **error** ✅ Present, defaults to None (line 142).
48. **incomplete_details** ✅ Present, defaults to None (line 143).

All 16 fields from the spec are present in `ResponseObject`.

---

## 9. Usage field population

49. **Usage populated on completed responses**
    ❌ **FAIL**: The API.md spec (lines 512-517) shows that completed responses include a
    `usage` object with `input_tokens`, `output_tokens`, `output_tokens_details`, and
    `total_tokens`. The Task dataclass (`agent_plane/entities/task.py`) has no `usage` field at all.
    `_build_response_object()` never sets `usage`, so it always defaults to `None`. The
    spec shows `"usage": null` only for the background/queued response (line 545) but
    shows a populated usage object for completed responses (line 512). Usage is always
    `null` in the current implementation.

---

## 10. Metadata default value

50. **Metadata defaults to `{}` (empty dict) vs `null`**
    ⚠️ **MINOR**: The spec shows `"metadata": {"user_id": "u_123"}` for one response and
    `"metadata": {}` for another (background queued, line 549). The `ResponseObject` defaults
    to `{}` (empty dict) which matches the background example. However, when no metadata is
    provided, `CreateResponseRequest.metadata` defaults to `None`, and `_build_response_object`
    passes `task.metadata` (which defaults to `{}` in the Task dataclass). This appears
    consistent — metadata is never null in the response.

---

## 11. Foreground disconnect cancellation

51. **background=false: client disconnect cancels execution**
    ❌ **FAIL**: API.md (lines 702-703) and RUNTIME.md (lines 571, 594) specify that for
    `background: false`, if the client disconnects, execution stops (server calls
    `task_store.cancel(task_id)`). The implementation does not have disconnect handling logic.
    The streaming path returns a `StreamingResponse` (line 266) but does not register an
    `on_disconnect` callback. The blocking path (line 272) awaits `task_store.wait()` but
    has no try/except or disconnect detection to cancel the task. If the client drops the
    connection for a non-background request, the task will continue running instead of
    being cancelled.

---

## 12. Background streaming: durable execution on disconnect

52. **background=true, stream=true: execution continues on disconnect**
    ⚠️ **NOT ENFORCED AT ROUTE LEVEL**: The spec says background execution continues on
    disconnect (API.md line 705). The route treats background and foreground streaming
    identically (both go through the same `event_generator`). Whether execution actually
    continues after disconnect depends on whether the streaming path has disconnect-cancels
    logic (see item 51). Since there is no disconnect-cancel logic at all, background
    streaming accidentally gets the right behavior (execution continues). But foreground
    streaming is broken (see item 51).

---

## 13. Fork detection and new conversation creation

53. **Fork creates a new conversation**
    ❌ **FAIL**: API.md (lines 217-220) and RUNTIME.md (line 494) state that when
    `previous_response_id` points to a non-latest response (a fork), the server creates a
    new conversation. The implementation does not implement fork detection at all — when no
    explicit `conversation` is provided and the previous_response_id is not the latest
    response in the conversation, the implementation simply reuses the existing conversation_id
    (line 117-119 resolves conversation_id from the previous response and uses it directly).
    Fork detection and conversation splitting only trigger as a 400 error when the caller
    explicitly passes a `conversation` field (lines 136-145). Without an explicit
    `conversation`, forks silently append to the existing conversation, violating the spec's
    "each conversation is always a linear thread" guarantee.

---

## 14. CreateResponseRequest model completeness

54. **`context_management` field missing from request model**
    ❌ **FAIL** (same as item 14): API.md lines 451-459 define `context_management` as an
    accepted request field. `CreateResponseRequest` in `models.py` does not include it.

---

## Summary

| # | Check | Result |
|---|-------|--------|
| 1 | store:false -> 400 | ✅ |
| 2 | Unknown model -> 404 | ✅ |
| 3 | Invalid previous_response_id -> 400 | ✅ |
| 4 | Conversation without previous_response_id -> 400 | ✅ |
| 5 | Conversation/response mismatch -> 400 | ✅ |
| 6 | Fork + explicit conversation -> 400 | ✅ |
| 7 | Invalid input format -> 400 | ✅ |
| 8-13 | Handler follows RUNTIME.md pseudocode | ✅ |
| 14 | context_management field on request | ❌ Missing from model |
| 15 | background=true, stream=false: return immediately | ✅ |
| 16 | background=false, stream=false: blocking wait | ✅ |
| 17-18 | Streaming event sequence | ✅ |
| 19 | background+stream: response.queued event | ❌ Missing queued event |
| 20 | SSE format | ✅ |
| 21 | sequence_number on every event | ✅ |
| 22 | Stream ends with [DONE] | ✅ |
| 23 | Terminal event uses actual status | ✅ |
| 24 | wait() before terminal event | ✅ |
| 25 | GET 404 for missing/deleted | ✅ |
| 26 | GET response shape | ✅ |
| 27 | GET output empty for non-completed | ⚠️ Relies on runtime invariant |
| 28 | Cancel 404 for missing | ✅ |
| 29 | Cancel 400 for terminal | ✅ (extra, not in spec) |
| 30 | Cancel returns cancelled response | ✅ |
| 31 | Delete 404 for missing | ✅ |
| 32 | Delete returns correct shape | ✅ |
| 33-48 | ResponseObject has all 16 fields | ✅ |
| 49 | Usage populated on completed | ❌ Always null |
| 50 | Metadata defaults | ✅ |
| 51 | Foreground disconnect -> cancel | ❌ Not implemented |
| 52 | Background disconnect -> continue | ⚠️ Accidentally correct |
| 53 | Fork -> new conversation | ❌ Not implemented |
| 54 | context_management on request model | ❌ Missing |

**Total: 5 failures, 2 warnings**

### Critical failures (behavioral):

1. **No disconnect-cancels for foreground requests** (item 51) — foreground tasks survive client
   disconnect, violating both the API spec and runtime design. This is a core behavioral contract.

2. **No fork detection / conversation splitting** (item 53) — forks silently append to existing
   conversations instead of creating new ones. Violates the "linear thread" invariant.

3. **Missing `response.queued` event for background streaming** (item 19) — background+stream
   skips the queued event that the spec explicitly requires.

### Non-critical failures:

4. **Usage always null** (item 49) — the Task dataclass lacks a `usage` field, so completed
   responses never report token usage. This is a data gap, not a crash.

5. **`context_management` not on request model** (item 14/54) — the field is silently dropped.
   Since the feature is described in the spec, the model should at least accept and store the
   field for future use.
