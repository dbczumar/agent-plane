"""End-to-end test for local Python tool execution via the archer agent.

Requires ``--llm-api-key`` and a real server. Run with::

    pytest tests/e2e/test_archer_local_tool.py \
        --llm-api-key $LLM_API_KEY -v

Exercises:
- Local tool discovery (``tools/python/word_count.py`` in the archer bundle)
- Real LLM deciding to call ``word_count`` based on user input
- Subprocess tool execution (with srt if installed, plain otherwise)
- Tool result flowing back to the LLM for the final response
- LLM judge verifying the agent actually used the tool
"""

from __future__ import annotations

import json

import httpx

from tests.e2e.conftest import poll_until_terminal


def test_archer_calls_word_count_tool(
    http_client: httpx.Client,
    archer_agent: str,
    llm_api_key: str,
) -> None:
    """
    Ask the archer agent to count words. The real LLM should
    call the ``word_count`` tool (discovered from
    ``tools/python/word_count.py``), then use the result in
    its final response.

    Verification uses an LLM judge call: a second LLM request
    checks whether the agent's response references the correct
    word count AND whether the ``word_count`` tool was actually
    invoked (visible in the output items).

    **What breaks if wrong**:
    - Tool not discovered: LLM tries to count words itself
      (often wrong) or says it can't.
    - Subprocess crash: tool returns error string, LLM says
      "the tool failed."
    - srt blocks execution: same as crash.
    """
    # Step 1: Ask archer to count words in a known phrase.
    # The phrase has exactly 7 words — easy to verify.
    test_phrase = "the quick brown fox jumps over fences"
    resp = http_client.post(
        "/v1/responses",
        json={
            "model": archer_agent,
            "input": (
                f"Use the word_count tool to count the words in this exact text: '{test_phrase}'"
            ),
            "background": True,
        },
    )
    resp.raise_for_status()
    body = resp.json()
    response_id = body["id"]

    # Step 2: Wait for completion.
    final = poll_until_terminal(http_client, response_id, timeout=120)
    assert final["status"] == "completed", (
        f"Expected completed, got {final['status']}. Error: {final.get('error')}"
    )

    # Step 3: Verify the word_count tool was called.
    output = final.get("output", [])
    fc_items = [
        item
        for item in output
        if item.get("type") == "function_call" and item.get("name") == "word_count"
    ]
    assert len(fc_items) >= 1, (
        f"Expected at least 1 word_count function_call in output, "
        f"got {len(fc_items)}. The LLM did not call the tool. "
        f"Output types: {[i.get('type') + ':' + i.get('name', '') for i in output]}"
    )

    # Step 4: Verify the tool result contains the correct count.
    fco_items = [
        item
        for item in output
        if item.get("type") == "function_call_output"
        and item.get("call_id") == fc_items[0].get("call_id")
    ]
    assert len(fco_items) == 1, (
        f"Expected 1 function_call_output for word_count, got {len(fco_items)}"
    )
    tool_output = fco_items[0].get("output", "")
    # The tool output must be valid JSON with word_count — NOT an
    # error string. If uv failed to install the ftfy dependency,
    # the output would be "Error: ModuleNotFoundError: No module
    # named 'ftfy'" instead of valid JSON.
    assert not tool_output.startswith("Error"), (
        f"Tool returned an error (uv may have failed to install "
        f"the ftfy dependency): {tool_output!r}"
    )
    tool_data = json.loads(tool_output)
    assert tool_data["word_count"] == 7, (
        f"Expected word_count=7, got {tool_data}. Tool may have failed or returned wrong result."
    )

    # Step 5: LLM judge — ask a fresh LLM whether the agent's
    # response correctly reports the word count.
    text_items = [item for item in output if item.get("type") == "message"]
    agent_text = " ".join(
        c.get("text", "") for item in text_items for c in item.get("content", [])
    )

    judge_resp = httpx.post(
        "https://api.openai.com/v1/responses",
        headers={"Authorization": f"Bearer {llm_api_key}"},
        json={
            "model": "gpt-5.4",
            "input": (
                "You are a test judge. Answer PASS or FAIL only.\n\n"
                f"The user asked an agent to count words in: '{test_phrase}'\n"
                f"The agent's response was: '{agent_text}'\n\n"
                "Does the agent's response indicate that the text "
                "has 7 words (the correct count)? Answer PASS if yes, "
                "FAIL if no."
            ),
        },
        timeout=30,
    )
    judge_resp.raise_for_status()
    judge_body = judge_resp.json()
    judge_text = ""
    for item in judge_body.get("output", []):
        for c in item.get("content", []):
            judge_text += c.get("text", "")

    assert "PASS" in judge_text.upper(), (
        f"LLM judge said FAIL. Agent response: {agent_text!r}. Judge response: {judge_text!r}"
    )
