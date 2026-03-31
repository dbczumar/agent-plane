"""Tests for agent_plane.spec.validator."""

from __future__ import annotations

from agent_plane.spec.types import (
    AgentSpec,
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
