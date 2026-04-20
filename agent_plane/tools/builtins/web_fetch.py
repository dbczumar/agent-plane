"""Built-in tool: web_fetch — LLM-powered web research via sub-agent.

Spawns a built-in ``__web_researcher`` sub-agent with ``code_sandbox``
to search the web and fetch page content. The sub-agent uses the
parent agent's LLM model and credentials. From the calling agent's
perspective, this is a synchronous function tool — the sub-agent
mechanics are hidden.

Requires the ``llm`` executor (the default). The ``claude_sdk`` and
``agents_sdk`` executors do not support sub-agents.

Usage in config.yaml::

    tools:
      builtins:
        - web_fetch
"""

from __future__ import annotations

import json
import logging
import time

# Any: tool schemas are heterogeneous dicts, AgentSpec.params
# has heterogeneous values.
from typing import Any

from agent_plane.spec.types import (
    AgentSpec,
    BuiltinToolConfig,
    ExecutorSpec,
    InteractionConfig,
    ToolsConfig,
)
from agent_plane.tools.base import Tool, ToolContext

_logger = logging.getLogger(__name__)

# Maximum seconds to wait for the sub-agent to complete.
_POLL_TIMEOUT: int = 120

# Sleep interval between poll attempts.
_POLL_INTERVAL: float = 0.5

# Internal sub-agent name. Double-underscore prefix prevents
# collision with user-declared sub-agent names (which use
# [a-z0-9-]+ naming convention).
RESEARCHER_NAME: str = "__web_researcher"

_RESEARCHER_INSTRUCTIONS: str = """\
You are a fast web research assistant. Speed is critical — the caller
is waiting for your result synchronously.

You have a code_sandbox tool. Use it to run commands that fetch web
content. Be direct: fetch, extract the answer, return it. Do not
write elaborate scripts or over-analyze.

## Speed rules (most important)

- **One tool call when possible.** If a URL is given, fetch it in a
  single code_sandbox call. Don't plan first — just do it.
- **Minimal script.** Use curl or a short Python one-liner. Don't
  write multi-function scripts with error handling classes.
- **Answer immediately.** Once you have the data, return the answer.
  Don't fetch additional sources unless the first one failed.
- **No unnecessary reasoning.** Don't explain your approach — just
  execute and return results.

## What you receive

- A **query**: what the caller wants to know
- An optional **URL**: a starting point to fetch

## What you do

1. If a URL is provided, fetch it immediately.
2. If no URL, search the web for the query.
3. Extract the relevant answer from the content.
4. Return the answer with source URLs. Be concise.

## Quick patterns

Fetch a URL (prefer curl for speed):
```
curl -sL "https://example.com" | head -200
```

Fetch JSON API:
```
curl -s "https://api.github.com/repos/owner/repo" | \
  python3 -c "import sys,json; d=json.load(sys.stdin); print(d['stargazers_count'])"
```

Search the web:
```
curl -sL "https://html.duckduckgo.com/html/?q=your+query" | grep -oP 'href="\\K[^"]+' | head -5
```

## If the first attempt fails

Try ONE alternative approach, then return whatever you have. Don't
loop endlessly. If nothing works, say so.
"""


def build_researcher_spec(parent_spec: AgentSpec) -> AgentSpec:
    """
    Build the ``__web_researcher`` AgentSpec using the parent's LLM config.

    The researcher gets:
    - The parent's ``llm`` config (model + connection + extras)
    - ``code_sandbox`` as its only builtin tool
    - Non-conversational mode (one-shot task)
    - Inline instructions for web research

    :param parent_spec: The parent agent's parsed spec.
    :returns: A complete AgentSpec for the web researcher sub-agent.
    """
    return AgentSpec(
        spec_version=1,
        name=RESEARCHER_NAME,
        description="Internal sub-agent for web_fetch — searches and fetches web content.",
        llm=parent_spec.llm,
        interaction=InteractionConfig(conversational=False),
        tools=ToolsConfig(
            builtins=[BuiltinToolConfig(name="code_sandbox")],
        ),
        instructions=_RESEARCHER_INSTRUCTIONS,
        # Low max_iterations to keep the sub-agent fast.
        # 1 fetch + 1 retry = 2 tool calls max, plus the
        # final response = ~3 iterations.
        executor=ExecutorSpec(max_iterations=5),
    )


class WebFetchTool(Tool):
    """
    Web research tool that spawns a sub-agent with code sandbox.

    The sub-agent searches the web and/or fetches specific URLs,
    extracts text, and returns findings. The parent agent sees
    this as a synchronous function tool call.

    Only works with the ``llm`` executor. Returns an error for
    ``claude_sdk`` and ``agents_sdk`` executors (which don't
    support sub-agents).

    :param parent_spec: The parent agent's parsed AgentSpec.
        Used to copy LLM config into the researcher sub-agent.
    """

    def __init__(self, parent_spec: AgentSpec) -> None:
        """
        Build the researcher sub-agent spec and append it to the
        parent's sub_agents list.

        :param parent_spec: The parent agent's AgentSpec.
        """
        self._parent_spec = parent_spec
        self.researcher_spec = build_researcher_spec(parent_spec)
        # Append to parent's sub_agents so _resolve_agent_spec_for_task
        # can find it when the spawned task runs. This is permanent for
        # the lifetime of the ToolManager (one workflow execution).
        # Safe for parallel tool calls — all read the same spec.
        parent_spec.sub_agents.append(self.researcher_spec)

    @classmethod
    def name(cls) -> str:
        """
        :returns: ``"web_fetch"``.
        """
        return "web_fetch"

    @classmethod
    def description(cls) -> str:
        """
        :returns: Human-readable description of the tool.
        """
        return (
            "Deep web research — fetches live web pages and "
            "summarizes relevant content. Always gets the "
            "latest version of a page. Use this when you "
            "need to read what a page actually says or need "
            "the most current info. Optionally provide a URL "
            "as a starting point; if it doesn't answer the "
            "query, other sources will be searched. Slower "
            "and less comprehensive than web_search but "
            "returns actual page content."
        )

    def get_schema(self) -> dict[str, Any]:
        """
        Return the OpenAI function schema for web_fetch.

        :returns: A function tool schema with ``query`` (required)
            and ``url`` (optional) parameters.
        """
        return {
            "type": "function",
            "function": {
                "name": "web_fetch",
                "description": (
                    "Deep web research — fetches live web pages and "
                    "summarizes relevant content. Always gets the "
                    "latest version of a page. Use this when you "
                    "need to read what a page actually says or need "
                    "the most current info. Optionally provide a URL "
                    "as a starting point; if it doesn't answer the "
                    "query, other sources will be searched. Slower "
                    "and less comprehensive than web_search but "
                    "returns actual page content."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "What to look up.",
                        },
                        "url": {
                            "type": "string",
                            "description": (
                                "Optional starting URL to fetch. If the "
                                "content doesn't answer the query, other "
                                "sources will be searched."
                            ),
                        },
                    },
                    "required": ["query"],
                },
            },
        }

    def is_async(self) -> bool:
        """
        Run web_fetch in a background workflow.

        ``invoke()`` here spawns a sub-agent (the
        ``__web_researcher``) that runs its own multi-iteration
        agent loop — typically tens of seconds per fetch.
        Returning ``True`` makes the runtime dispatch each call
        as a ``kind="tool"`` background workflow and hand the
        LLM a ``{task_id, kind: "tool"}`` handle inline, so
        the LLM can fan out multiple fetches in one turn (and
        can cancel any of them via the standard cancel path).
        The eventual researcher output is auto-delivered to the
        parent on the unified ``async_work_complete`` drain.

        :returns: ``True`` — web_fetch is always long-running.
        """
        return True

    def invoke(self, arguments: str, ctx: ToolContext) -> str:
        """
        Spawn the web researcher sub-agent and wait for results.

        :param arguments: JSON with ``query`` (required) and
            ``url`` (optional).
        :param ctx: Tool execution context with task_id and agent_id.
        :returns: The researcher's findings, or an error message.
        """
        # Guard: only llm executor supports sub-agents.
        if self._parent_spec.executor.type not in ("llm", None):
            return (
                f"web_fetch is not available with the "
                f"{self._parent_spec.executor.type!r} executor. "
                f"It requires the default llm executor."
            )

        parsed: dict[str, Any] = json.loads(arguments) if arguments else {}
        query = parsed.get("query")
        if not query:
            return "Error: 'query' parameter is required."

        url = parsed.get("url")
        prompt = _build_prompt(query, url)

        return _spawn_and_wait(
            prompt=prompt,
            agent_id=ctx.agent_id,
            task_id=ctx.task_id,
        )


def _build_prompt(query: str, url: str | None) -> str:
    """
    Build the user input for the web researcher sub-agent.

    :param query: What to look up.
    :param url: Optional starting URL.
    :returns: Formatted prompt string.
    """
    if url:
        return f"Query: {query}\n\nStart with this URL: {url}"
    return f"Query: {query}"


def _spawn_and_wait(
    prompt: str,
    agent_id: str,
    task_id: str,
) -> str:
    """
    Spawn the ``__web_researcher`` sub-agent and poll until done.

    :param prompt: The user input for the sub-agent.
    :param agent_id: The parent's registered agent ID.
    :param task_id: The current task ID (for root_task_id resolution).
    :returns: The sub-agent's text output, or an error message.
    """
    from agent_plane.tools.builtins.spawn import (
        _resolve_parent_conversation_id,
        _resolve_root_task_id,
        _spawn_one,
    )

    root_task_id = _resolve_root_task_id(task_id)
    parent_conversation_id = _resolve_parent_conversation_id(task_id)

    # Phase 4 made name required. web_fetch is fire-and-forget
    # (the parent doesn't address the researcher conversation
    # later via send_to_sub_agent), but the partial unique
    # index on (parent_conversation_id, title) means a fixed
    # name would collide if the parent calls web_fetch more than
    # once. Use the calling task_id as the unique discriminator
    # so each web_fetch invocation gets a distinct child
    # conversation.
    sa_name = f"{RESEARCHER_NAME}_{task_id}"
    child_task_id = _spawn_one(
        agent_id=agent_id,
        agent_name=RESEARCHER_NAME,
        sa_name=sa_name,
        user_input=prompt,
        root_task_id=root_task_id,
        # web_fetch is invoked as a server-side tool from the
        # calling task; audit fix #1 → store the caller as
        # parent_task_id so drain signals route correctly.
        parent_task_id=task_id,
        parent_conversation_id=parent_conversation_id,
    )

    return _poll_until_terminal(child_task_id)


def _poll_until_terminal(task_id: str) -> str:
    """
    Poll a task until it reaches a terminal state.

    Runs in a thread (tools are executed via ``_to_thread`` in the
    workflow), so ``time.sleep`` is safe and does not block the
    async event loop.

    :param task_id: The sub-agent task ID to poll.
    :returns: Extracted text output, or an error message.
    """
    from agent_plane.runtime import get_task_store
    from agent_plane.tools.builtins.spawn import _extract_output_text

    task_store = get_task_store()
    deadline = time.monotonic() + _POLL_TIMEOUT

    while time.monotonic() < deadline:
        task = task_store.get_sync(task_id)
        if task is None:
            return f"Error: web_fetch sub-agent task {task_id} not found."

        if task.status == "completed":
            if task.output:
                text = _extract_output_text(task.output)
                if text:
                    return text
            return "Web research completed but returned no content."

        if task.status == "failed":
            error_msg = ""
            if task.output:
                error_msg = _extract_output_text(task.output)
            return f"Web research failed. {error_msg}".strip()

        time.sleep(_POLL_INTERVAL)

    return f"Web research timed out after {_POLL_TIMEOUT}s."
