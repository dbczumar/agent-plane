# D6 direct-cancel E2E agent

Each turn, follow this exact sequence:

1. Call the `slow_compute` tool **with `synchronous: false`**
   and `seconds: 30` (always 30 — long enough that the test
   can prove the body was cancelled before it naturally
   returned).
2. You will receive a `function_call_output` with a handle
   JSON: `{"task_id": "...", "kind": "client_tool", ...}`.
   Extract the `task_id` value.
3. In your **next** tool_call, call `cancel_task` with that
   `task_id`. Do not wait for any system message; call
   cancel_task immediately after you see the handle FCO.
4. You will receive a `function_call_output` like
   `{"cancelled": true, "prior_status": "in_progress", ...}`
   and eventually a `[System: task <id> (client_tool)
   cancelled]` user message.
5. Reply with the literal string `CANCELLED_OK` so the test
   can confirm you completed the sequence.

Do NOT skip cancel_task. Do NOT wait for the slow_compute
to finish before cancelling — the whole point is to cancel
an in-flight body.
