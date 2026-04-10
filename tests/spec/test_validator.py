"""Tests for agent_plane.spec.validator."""

from __future__ import annotations

import pytest

from agent_plane.spec.types import (
    AgentSpec,
    CompactionConfig,
    ExecutorSpec,
    InteractionConfig,
    LLMConfig,
    LocalToolInfo,
    MCPServerConfig,
    ModalityConfig,
    SkillSpec,
    ToolsConfig,
)
from agent_plane.spec.validator import validate


def _minimal_spec(**overrides: object) -> AgentSpec:
    """Build a minimal valid AgentSpec with optional overrides."""
    defaults: dict[str, object] = {"spec_version": 1}
    defaults.update(overrides)
    return AgentSpec(**defaults)  # type: ignore[arg-type]


def test_minimal_spec_valid() -> None:
    result = validate(_minimal_spec())
    assert result.valid


def test_invalid_spec_version() -> None:
    result = validate(_minimal_spec(spec_version=2))
    assert not result.valid
    assert any("spec_version" in e.path for e in result.errors)


def test_llm_valid() -> None:
    spec = _minimal_spec(llm=LLMConfig(model="openai/gpt-5.4"))
    result = validate(spec)
    assert result.valid


def test_llm_empty_model() -> None:
    spec = _minimal_spec(llm=LLMConfig(model=""))
    result = validate(spec)
    assert not result.valid
    assert any("llm.model" in e.path for e in result.errors)


def test_llm_arbitrary_extra_passes_validation() -> None:
    """Extra keys are passed through — validator does not reject them."""
    spec = _minimal_spec(
        llm=LLMConfig(
            model="openai/gpt-5.4",
            extra={"temperature": 0.7, "reasoning_effort": "extreme"},
        )
    )
    result = validate(spec)
    assert result.valid


def test_valid_input_modalities() -> None:
    spec = _minimal_spec(
        interaction=InteractionConfig(
            modalities=ModalityConfig(input=["text", "image", "audio", "video", "file"])
        )
    )
    result = validate(spec)
    assert result.valid


def test_invalid_input_modality() -> None:
    spec = _minimal_spec(
        interaction=InteractionConfig(modalities=ModalityConfig(input=["text", "smell"]))
    )
    result = validate(spec)
    assert not result.valid
    assert any("smell" in e.message for e in result.errors)


def test_invalid_output_modality() -> None:
    spec = _minimal_spec(
        interaction=InteractionConfig(modalities=ModalityConfig(output=["text", "file"]))
    )
    result = validate(spec)
    assert not result.valid
    assert any("file" in e.message for e in result.errors)


def test_valid_output_modalities() -> None:
    spec = _minimal_spec(
        interaction=InteractionConfig(modalities=ModalityConfig(output=["text", "image", "audio"]))
    )
    result = validate(spec)
    assert result.valid


def test_skill_valid() -> None:
    spec = _minimal_spec(
        skills=[
            SkillSpec(
                name="deep-search",
                description="Search the web.",
                content="Use search.web.",
            )
        ]
    )
    result = validate(spec)
    assert result.valid


def test_skill_name_invalid_pattern() -> None:
    spec = _minimal_spec(skills=[SkillSpec(name="Bad_Name", description="Bad.", content=".")])
    result = validate(spec)
    assert not result.valid
    assert any("must match" in e.message for e in result.errors)


def test_skill_name_too_long() -> None:
    spec = _minimal_spec(skills=[SkillSpec(name="a" * 65, description="Long name.", content=".")])
    result = validate(spec)
    assert not result.valid
    assert any("at most 64" in e.message for e in result.errors)


def test_skill_description_too_long() -> None:
    spec = _minimal_spec(skills=[SkillSpec(name="ok", description="x" * 1025, content=".")])
    result = validate(spec)
    assert not result.valid
    assert any("at most 1024" in e.message for e in result.errors)


def test_duplicate_skill_names() -> None:
    spec = _minimal_spec(
        skills=[
            SkillSpec(name="dupe", description="First.", content="."),
            SkillSpec(name="dupe", description="Second.", content="."),
        ]
    )
    result = validate(spec)
    assert not result.valid
    assert any("duplicate skill name" in e.message for e in result.errors)


def test_mcp_http_valid() -> None:
    spec = _minimal_spec(mcp_servers=[MCPServerConfig(name="svc", url="http://localhost:9000")])
    result = validate(spec)
    assert result.valid


def test_duplicate_mcp_names() -> None:
    spec = _minimal_spec(
        mcp_servers=[
            MCPServerConfig(name="dupe", url="http://a"),
            MCPServerConfig(name="dupe", url="http://b"),
        ]
    )
    result = validate(spec)
    assert not result.valid
    assert any("duplicate MCP server name" in e.message for e in result.errors)


def test_duplicate_tool_names_across_mcp_and_local() -> None:
    spec = _minimal_spec(
        mcp_servers=[MCPServerConfig(name="search", url="http://localhost:9000")],
        local_tools=[
            LocalToolInfo(name="search", path="tools/python/search.py", language="python")
        ],
    )
    result = validate(spec)
    assert not result.valid
    assert any("duplicate tool name" in e.message for e in result.errors)


def test_sub_agent_reference_valid() -> None:
    sub = _minimal_spec(name="helper", llm=LLMConfig(model="openai/gpt-4o"))
    spec = _minimal_spec(
        tools=ToolsConfig(agents=["helper"]),
        sub_agents=[sub],
    )
    result = validate(spec)
    assert result.valid


def test_sub_agent_reference_missing() -> None:
    spec = _minimal_spec(
        tools=ToolsConfig(agents=["ghost"]),
    )
    result = validate(spec)
    assert not result.valid
    assert any("ghost" in e.message for e in result.errors)


@pytest.mark.parametrize(
    "invalid_name",
    [
        "has.dot",  # dot is the tunneled model field delimiter
        "has/slash",  # slash is the litellm provider/model separator
        "has space",  # whitespace confuses API clients and log pipelines
        "has\ttab",  # tab is also whitespace
        "",  # empty string has no meaningful identity
    ],
)
def test_agent_name_invalid_characters(invalid_name: str) -> None:
    """
    Agent names with dots, slashes, whitespace, or empty string are rejected.

    Each of these characters would break either the tunneled model field
    (dots), litellm routing (slashes), or client parsing (whitespace/empty).
    """
    spec = _minimal_spec(name=invalid_name)
    result = validate(spec)
    assert not result.valid
    assert any("name" in e.path for e in result.errors)


@pytest.mark.parametrize(
    "valid_name",
    [
        "researcher",
        "my-agent",
        "agent_v2",
        "Agent123",
        "CamelCase",
        "a",
    ],
)
def test_agent_name_valid(valid_name: str) -> None:
    """Agent names using alphanumeric, hyphens, and underscores are accepted."""
    spec = _minimal_spec(name=valid_name)
    result = validate(spec)
    assert result.valid


def test_agent_name_invalid_in_sub_agent() -> None:
    """Invalid name on a sub-agent (not just the root) is caught."""
    sub = _minimal_spec(name="bad.name", llm=LLMConfig(model="openai/gpt-4o"))
    spec = _minimal_spec(
        tools=ToolsConfig(agents=["bad.name"]),
        sub_agents=[sub],
    )
    result = validate(spec)
    assert not result.valid
    assert any("name" in e.path for e in result.errors)


def test_multiple_errors_reported() -> None:
    """
    Validator reports all errors, not just the first.

    Three violations: spec_version != 1, skill name not lowercase,
    and skill description exceeds 1024 chars.
    """
    spec = _minimal_spec(
        spec_version=99,
        skills=[
            SkillSpec(name="BAD", description="x" * 2000, content="."),
        ],
    )
    result = validate(spec)
    assert not result.valid
    # spec_version error + skill name pattern error + skill description length error
    assert len(result.errors) >= 3


# ── agents_sdk executor validation ────────────────────────


def test_agents_sdk_rejects_compaction() -> None:
    """
    ``agents_sdk`` executor forbids ``compaction`` — the SDK
    manages context internally.
    """
    spec = _minimal_spec(
        executor=ExecutorSpec(type="agents_sdk"),
        compaction=CompactionConfig(),
    )
    result = validate(spec)
    assert not result.valid
    assert any("compaction" in e.path for e in result.errors), (
        f"Expected compaction error, got: {result.errors}"
    )
    # Verify actual error message content, not just path.
    assert any(
        "agents_sdk" in e.message for e in result.errors
    ), (
        f"Error message should mention 'agents_sdk':"
        f" {result.errors}"
    )


def test_agents_sdk_rejects_endpoint() -> None:
    """
    ``agents_sdk`` executor forbids ``executor.endpoint`` —
    that's remote-only.
    """
    spec = _minimal_spec(
        executor=ExecutorSpec(
            type="agents_sdk",
            endpoint="http://localhost:8000",
        ),
    )
    result = validate(spec)
    assert not result.valid
    assert any("executor.endpoint" in e.path for e in result.errors), (
        f"Expected endpoint error, got: {result.errors}"
    )
    assert any(
        "agents_sdk" in e.message for e in result.errors
    ), (
        f"Error message should mention 'agents_sdk':"
        f" {result.errors}"
    )


def test_agents_sdk_accepts_connection() -> None:
    """
    ``agents_sdk`` executor allows ``llm.connection`` — unlike
    ``claude_sdk`` which forbids it. The SDK supports custom
    OpenAI clients with per-agent API keys.
    """
    spec = _minimal_spec(
        executor=ExecutorSpec(type="agents_sdk"),
        llm=LLMConfig(
            model="gpt-5.4",
            connection={"api_key": "sk-test"},
        ),
    )
    result = validate(spec)
    assert result.valid, f"Expected valid spec, got errors: {result.errors}"
