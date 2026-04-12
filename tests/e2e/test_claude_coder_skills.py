"""E2E test: Claude SDK executor discovering and loading skills.

Verifies that the Claude SDK's native Skill tool discovers skills
written to ``.claude/skills/`` by the executor's ``on_task_start``
and can load their content.

Usage::

    pytest tests/e2e/test_claude_coder_skills.py \
        --llm-api-key $LLM_API_KEY -v
"""

from __future__ import annotations

import os
from typing import Any

import httpx

from tests.e2e.conftest import poll_until_terminal


def _extract_all_text(body: dict[str, Any]) -> str:
    """
    Concatenate all output_text blocks from a response body.

    :param body: The terminal response body.
    :returns: All assistant text joined by newlines.
    """
    parts: list[str] = []
    for item in body.get("output", []):
        if item.get("type") == "message":
            for block in item.get("content", []):
                text = block.get("text")
                if text:
                    parts.append(text)
    return "\n".join(parts)


def test_claude_coder_lists_and_loads_skills(
    http_client: httpx.Client,
    claude_coder_agent: str,
    llm_api_key: str,
) -> None:
    """
    Claude discovers custom skills via the SDK's Skill tool and
    can load their content.

    The claude-coder agent has two custom skills (code-review and
    systematic-debugging) written to ``.claude/skills/`` by
    ``on_task_start``. The SDK discovers them via
    ``setting_sources=["project"]``. This test asks Claude to
    list its skills and load the code-review skill, then verifies
    the output contains the actual skill content.

    **What breaks if the feature is wrong:**

    - If ``on_task_start`` doesn't write skills, the SDK sees an
      empty ``.claude/skills/`` directory → no custom skills.
    - If ``setting_sources`` is not set, the SDK doesn't scan for
      project skills at all → only built-in skills appear.
    - If ``"Skill"`` is not in ``allowed_tools``, Claude can't
      invoke the Skill tool even if skills are discovered.
    - If ``disable-model-invocation`` defaults to true, Claude
      sees the skill listed but gets an error loading it.
    """
    resp = http_client.post(
        "/v1/responses",
        json={
            "model": claude_coder_agent,
            "input": (
                "List your available skills. Then load the "
                "code-review skill and show me its full content."
            ),
            "background": True,
        },
    )
    resp.raise_for_status()
    response_id = resp.json()["id"]

    body = poll_until_terminal(http_client, response_id, timeout=120)
    assert body["status"] == "completed", f"Task failed: {body.get('error')}"

    text = _extract_all_text(body)

    # Use LLM judge to verify Claude found and loaded the skill.
    from mlflow.genai.judges import make_judge

    os.environ["OPENAI_API_KEY"] = llm_api_key

    judge = make_judge(
        name="skill_discovery",
        instructions=(
            "You are evaluating whether an AI assistant "
            "successfully discovered and loaded a custom skill.\n\n"
            "The assistant was asked to list its skills and load "
            "the 'code-review' skill. The code-review skill "
            "contains instructions about reviewing code with a "
            "structured format (Critical Issues, Improvements, "
            "Looks Good sections) and prioritizing security > "
            "correctness > performance > style.\n\n"
            "The assistant's response is:\n"
            "{{ outputs }}\n\n"
            "Does the response:\n"
            "1. Mention 'code-review' as an available skill?\n"
            "2. Show content from the skill (e.g. the structured "
            "review format, or the priority ordering)?\n\n"
            "Return True if BOTH conditions are met. Return False "
            "if the assistant couldn't find the skill, showed "
            "generic capabilities instead, or only listed the "
            "skill name without its content."
        ),
        feedback_value_type=bool,
    )

    feedback = judge(outputs=text)
    assert feedback.value is True, (
        f"LLM judge: skills not properly discovered/loaded.\n"
        f"Rationale: {feedback.rationale}\n"
        f"Output: {text[:500]}"
    )
