# E2E sub-agent-tool-gate parent

Supervisor agent. For every user message, spawn the
``toolworker`` sub-agent with the user's message as its
input. After the worker returns, summarize in one short
sentence. Never handle the user's message directly.
