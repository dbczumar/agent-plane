"""Tests for agent_plane.spec.parser."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from agent_plane.errors import AgentPlaneError
from agent_plane.spec.parser import parse


@pytest.fixture()
def agent_dir(tmp_path: Path) -> Path:
    """Create a minimal valid agent image directory."""
    config = {"spec_version": 1, "name": "test-agent"}
    (tmp_path / "config.yaml").write_text(yaml.dump(config))
    return tmp_path


def test_parse_minimal(agent_dir: Path) -> None:
    spec = parse(agent_dir)
    assert spec.spec_version == 1
    assert spec.name == "test-agent"
    assert spec.description is None
    assert spec.llm is None
    assert spec.interaction.conversational is True
    assert spec.interaction.modalities.input == ["text"]
    assert spec.interaction.modalities.output == ["text"]
    assert spec.tools.agents == []
    assert spec.params == {}
    assert spec.instructions is None
    assert spec.skills == []
    assert spec.mcp_servers == []
    assert spec.local_tools == []
    assert spec.sub_agents == []


def test_parse_missing_config_yaml(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="config.yaml not found"):
        parse(tmp_path)


def test_parse_non_mapping_config(tmp_path: Path) -> None:
    (tmp_path / "config.yaml").write_text("- just a list")
    with pytest.raises(AgentPlaneError, match="must be a YAML mapping"):
        parse(tmp_path)


def test_parse_missing_spec_version(tmp_path: Path) -> None:
    (tmp_path / "config.yaml").write_text(yaml.dump({"name": "no-version"}))
    with pytest.raises(AgentPlaneError, match="missing required field: spec_version"):
        parse(tmp_path)


def test_parse_full_config(tmp_path: Path) -> None:
    config = {
        "spec_version": 1,
        "name": "full-agent",
        "description": "A fully configured agent.",
        "llm": {
            "model": "openai/gpt-5.4",
            "max_completion_tokens": 4096,
            "reasoning_effort": "medium",
        },
        "interaction": {
            "conversational": True,
            "modalities": {
                "input": ["text", "image", "file"],
                "output": ["text"],
            },
        },
        "tools": {"agents": ["researcher", "critic"]},
        "params": {"max_results": 10, "prefer_recent": True},
    }
    (tmp_path / "config.yaml").write_text(yaml.dump(config))

    spec = parse(tmp_path)
    assert spec.name == "full-agent"
    assert spec.description == "A fully configured agent."
    assert spec.llm is not None
    assert spec.llm.model == "openai/gpt-5.4"
    assert spec.llm.extra == {
        "max_completion_tokens": 4096,
        "reasoning_effort": "medium",
    }
    assert spec.interaction.conversational is True
    assert spec.interaction.modalities.input == ["text", "image", "file"]
    assert spec.interaction.modalities.output == ["text"]
    assert spec.tools.agents == ["researcher", "critic"]
    assert spec.params == {"max_results": 10, "prefer_recent": True}


def test_parse_llm_missing_model(tmp_path: Path) -> None:
    config = {"spec_version": 1, "llm": {"max_completion_tokens": 100}}
    (tmp_path / "config.yaml").write_text(yaml.dump(config))
    with pytest.raises(AgentPlaneError, match="missing required field: model"):
        parse(tmp_path)


def test_parse_llm_arbitrary_extra_keys(tmp_path: Path) -> None:
    """All non-model keys in the llm block are collected into extra."""
    config = {
        "spec_version": 1,
        "llm": {
            "model": "anthropic/claude-sonnet-4-20250514",
            "temperature": 0.7,
            "top_p": 0.9,
            "max_tokens": 2048,
            "stop": ["\n\n"],
        },
    }
    (tmp_path / "config.yaml").write_text(yaml.dump(config))
    spec = parse(tmp_path)
    assert spec.llm is not None
    assert spec.llm.model == "anthropic/claude-sonnet-4-20250514"
    assert spec.llm.extra == {
        "temperature": 0.7,
        "top_p": 0.9,
        "max_tokens": 2048,
        "stop": ["\n\n"],
    }


def test_parse_llm_model_only(tmp_path: Path) -> None:
    """LLM block with only model has empty extra and no connection."""
    config = {"spec_version": 1, "llm": {"model": "openai/gpt-4o"}}
    (tmp_path / "config.yaml").write_text(yaml.dump(config))
    spec = parse(tmp_path)
    assert spec.llm is not None
    assert spec.llm.model == "openai/gpt-4o"
    assert spec.llm.extra == {}
    assert spec.llm.connection is None


def test_parse_llm_connection_block(tmp_path: Path) -> None:
    """The connection sub-block is parsed into LLMConfig.connection."""
    config = {
        "spec_version": 1,
        "llm": {
            "model": "databricks/databricks-gpt-5-4",
            "temperature": 0.5,
            "connection": {
                "api_key": "dapi_test_key",
                "base_url": "https://my-workspace.databricks.com/serving-endpoints",
            },
        },
    }
    (tmp_path / "config.yaml").write_text(yaml.dump(config))
    spec = parse(tmp_path)
    assert spec.llm is not None
    assert spec.llm.model == "databricks/databricks-gpt-5-4"
    assert spec.llm.extra == {"temperature": 0.5}
    assert spec.llm.connection == {
        "api_key": "dapi_test_key",
        "base_url": "https://my-workspace.databricks.com/serving-endpoints",
    }


def test_parse_llm_connection_expands_env_vars(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``${VAR}`` references in connection values are expanded."""
    monkeypatch.setenv("MY_API_KEY", "sk-secret-123")
    config = {
        "spec_version": 1,
        "llm": {
            "model": "openai/gpt-5.4",
            "connection": {"api_key": "${MY_API_KEY}"},
        },
    }
    (tmp_path / "config.yaml").write_text(yaml.dump(config))
    spec = parse(tmp_path)
    assert spec.llm is not None
    assert spec.llm.connection == {"api_key": "sk-secret-123"}


def test_parse_llm_connection_unresolved_var_raises(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Unresolved ``${VAR}`` in LLM connection raises ValueError.

    :param tmp_path: Temporary directory for config files.
    :param monkeypatch: Pytest monkeypatch for env vars.
    """
    monkeypatch.delenv("MY_API_KEY", raising=False)
    config = {
        "spec_version": 1,
        "llm": {
            "model": "openai/gpt-4o",
            "connection": {"api_key": "${MY_API_KEY}"},
        },
    }
    (tmp_path / "config.yaml").write_text(yaml.dump(config))
    with pytest.raises(AgentPlaneError, match="Unresolved environment variable"):
        parse(tmp_path)


def test_parse_instructions_multiline_inline(tmp_path: Path) -> None:
    """Multiline inline instructions are not treated as file paths."""
    config = {
        "spec_version": 1,
        "instructions": "Line one.\nLine two.\nLine three.",
    }
    (tmp_path / "config.yaml").write_text(yaml.dump(config))
    spec = parse(tmp_path)
    assert spec.instructions == "Line one.\nLine two.\nLine three."


def test_parse_agents_md_fallback(agent_dir: Path) -> None:
    """No instructions key in config -> falls back to AGENTS.md."""
    (agent_dir / "AGENTS.md").write_text("You are a helpful research assistant.")
    spec = parse(agent_dir)
    assert spec.instructions == "You are a helpful research assistant."


def test_parse_instructions_inline(tmp_path: Path) -> None:
    """instructions key with inline text (not a file path)."""
    config = {"spec_version": 1, "instructions": "Be concise and helpful."}
    (tmp_path / "config.yaml").write_text(yaml.dump(config))
    spec = parse(tmp_path)
    assert spec.instructions == "Be concise and helpful."


def test_parse_instructions_file_reference(agent_dir: Path) -> None:
    """instructions key pointing to an existing file."""
    (agent_dir / "SYSTEM.md").write_text("Custom system prompt from file.")
    config = {"spec_version": 1, "name": "test-agent", "instructions": "SYSTEM.md"}
    (agent_dir / "config.yaml").write_text(yaml.dump(config))
    spec = parse(agent_dir)
    assert spec.instructions == "Custom system prompt from file."


def test_parse_instructions_overrides_agents_md(agent_dir: Path) -> None:
    """Explicit instructions key takes precedence over AGENTS.md."""
    (agent_dir / "AGENTS.md").write_text("Fallback instructions.")
    config = {"spec_version": 1, "name": "test-agent", "instructions": "Inline wins."}
    (agent_dir / "config.yaml").write_text(yaml.dump(config))
    spec = parse(agent_dir)
    assert spec.instructions == "Inline wins."


def test_parse_instructions_file_overrides_agents_md(agent_dir: Path) -> None:
    """instructions pointing to a file takes precedence over AGENTS.md."""
    (agent_dir / "AGENTS.md").write_text("Fallback instructions.")
    (agent_dir / "CUSTOM.md").write_text("Custom file wins.")
    config = {"spec_version": 1, "name": "test-agent", "instructions": "CUSTOM.md"}
    (agent_dir / "config.yaml").write_text(yaml.dump(config))
    spec = parse(agent_dir)
    assert spec.instructions == "Custom file wins."


def test_parse_skill(agent_dir: Path) -> None:
    skill_dir = agent_dir / "skills" / "deep-search"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\n"
        "name: deep-search\n"
        "description: Search the web for sources.\n"
        "---\n"
        "When asked to research, use search.web."
    )
    spec = parse(agent_dir)
    assert len(spec.skills) == 1
    skill = spec.skills[0]
    assert skill.name == "deep-search"
    assert skill.description == "Search the web for sources."
    assert skill.content == "When asked to research, use search.web."
    assert skill.skill_dir == skill_dir


def test_parse_skill_missing_frontmatter(agent_dir: Path) -> None:
    skill_dir = agent_dir / "skills" / "bad"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text("No frontmatter here.")
    with pytest.raises(AgentPlaneError, match="missing YAML frontmatter"):
        parse(agent_dir)


def test_parse_skill_missing_name(agent_dir: Path) -> None:
    skill_dir = agent_dir / "skills" / "no-name"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text("---\ndescription: Missing name.\n---\nContent.")
    with pytest.raises(AgentPlaneError, match="missing required field 'name'"):
        parse(agent_dir)


def test_parse_skill_missing_description(agent_dir: Path) -> None:
    skill_dir = agent_dir / "skills" / "no-desc"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text("---\nname: no-desc\n---\nContent.")
    with pytest.raises(AgentPlaneError, match="missing required field 'description'"):
        parse(agent_dir)


def test_parse_mcp_http(
    agent_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Parse an HTTP MCP server config with env var expansion.

    :param agent_dir: Temporary agent directory fixture.
    :param monkeypatch: Pytest monkeypatch for env vars.
    """
    monkeypatch.setenv("API_KEY", "sk-test-key")
    mcp_dir = agent_dir / "tools" / "mcp"
    mcp_dir.mkdir(parents=True)
    mcp_config = {
        "name": "my-service",
        "transport": "http",
        "url": "http://localhost:9000/mcp",
        "headers": {"Authorization": "Bearer ${API_KEY}"},
    }
    (mcp_dir / "service.yaml").write_text(yaml.dump(mcp_config))
    spec = parse(agent_dir)
    mcp = spec.mcp_servers[0]
    assert mcp.url == "http://localhost:9000/mcp"
    # ${API_KEY} expanded to the value set via monkeypatch.
    assert mcp.headers == {"Authorization": "Bearer sk-test-key"}


def test_parse_mcp_env_unresolved_var_raises(
    agent_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Unresolved ``${VAR}`` in MCP env raises ``AgentPlaneError``
    at parse time instead of silently passing the literal to the
    server.

    :param agent_dir: Temporary agent directory fixture.
    :param monkeypatch: Pytest monkeypatch for env vars.
    """
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    mcp_dir = agent_dir / "tools" / "mcp"
    mcp_dir.mkdir(parents=True)
    mcp_config = {
        "name": "github",
        "transport": "http",
        "url": "http://localhost:9000/mcp",
        "headers": {"Authorization": "Bearer ${GITHUB_TOKEN}"},
    }
    (mcp_dir / "github.yaml").write_text(yaml.dump(mcp_config))
    with pytest.raises(AgentPlaneError, match="Unresolved environment variable"):
        parse(agent_dir)


def test_parse_mcp_headers_unresolved_var_raises(
    agent_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Unresolved ``${VAR}`` in MCP headers raises ValueError at
    parse time.

    :param agent_dir: Temporary agent directory fixture.
    :param monkeypatch: Pytest monkeypatch for env vars.
    """
    monkeypatch.delenv("API_KEY", raising=False)
    mcp_dir = agent_dir / "tools" / "mcp"
    mcp_dir.mkdir(parents=True)
    mcp_config = {
        "name": "my-service",
        "transport": "http",
        "url": "http://localhost:9000/mcp",
        "headers": {"Authorization": "Bearer ${API_KEY}"},
    }
    (mcp_dir / "service.yaml").write_text(yaml.dump(mcp_config))
    with pytest.raises(AgentPlaneError, match="Unresolved environment variable"):
        parse(agent_dir)


def test_parse_mcp_env_dollar_without_braces_raises(
    agent_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Unresolved ``$VAR`` (without braces) also raises ValueError.

    :param agent_dir: Temporary agent directory fixture.
    :param monkeypatch: Pytest monkeypatch for env vars.
    """
    monkeypatch.delenv("MY_SECRET", raising=False)
    mcp_dir = agent_dir / "tools" / "mcp"
    mcp_dir.mkdir(parents=True)
    mcp_config = {
        "name": "test",
        "transport": "http",
        "url": "http://localhost:9000/mcp",
        "headers": {"Secret": "$MY_SECRET"},
    }
    (mcp_dir / "test.yaml").write_text(yaml.dump(mcp_config))
    with pytest.raises(AgentPlaneError, match="Unresolved environment variable"):
        parse(agent_dir)


def test_parse_mcp_missing_name(agent_dir: Path) -> None:
    mcp_dir = agent_dir / "tools" / "mcp"
    mcp_dir.mkdir(parents=True)
    (mcp_dir / "bad.yaml").write_text(yaml.dump({"transport": "http", "url": "http://x"}))
    with pytest.raises(AgentPlaneError, match="missing required field 'name'"):
        parse(agent_dir)


def test_parse_mcp_missing_transport(agent_dir: Path) -> None:
    mcp_dir = agent_dir / "tools" / "mcp"
    mcp_dir.mkdir(parents=True)
    (mcp_dir / "bad.yaml").write_text(yaml.dump({"name": "bad"}))
    with pytest.raises(AgentPlaneError, match="missing required field 'transport'"):
        parse(agent_dir)


def test_parse_local_python_tools(agent_dir: Path) -> None:
    py_dir = agent_dir / "tools" / "python"
    py_dir.mkdir(parents=True)
    (py_dir / "arxiv_search.py").write_text("def search(): pass")
    (py_dir / "web_scrape.py").write_text("def scrape(): pass")
    spec = parse(agent_dir)
    assert len(spec.local_tools) == 2
    names = {t.name for t in spec.local_tools}
    assert names == {"arxiv_search", "web_scrape"}
    assert all(t.language == "python" for t in spec.local_tools)


def test_parse_local_typescript_tools(agent_dir: Path) -> None:
    ts_dir = agent_dir / "tools" / "typescript"
    ts_dir.mkdir(parents=True)
    (ts_dir / "code_run.ts").write_text("export function run() {}")
    spec = parse(agent_dir)
    assert len(spec.local_tools) == 1
    assert spec.local_tools[0].name == "code_run"
    assert spec.local_tools[0].language == "typescript"


def test_parse_sub_agents(tmp_path: Path) -> None:
    # Parent config referencing two sub-agents
    parent_config = {
        "spec_version": 1,
        "name": "parent",
        "tools": {"agents": ["researcher", "critic"]},
    }
    (tmp_path / "config.yaml").write_text(yaml.dump(parent_config))

    # Sub-agent: researcher
    researcher_dir = tmp_path / "agents" / "researcher"
    researcher_dir.mkdir(parents=True)
    (researcher_dir / "config.yaml").write_text(
        yaml.dump({"spec_version": 1, "name": "researcher"})
    )

    # Sub-agent: critic
    critic_dir = tmp_path / "agents" / "critic"
    critic_dir.mkdir(parents=True)
    (critic_dir / "config.yaml").write_text(yaml.dump({"spec_version": 1, "name": "critic"}))

    spec = parse(tmp_path)
    assert len(spec.sub_agents) == 2
    sub_names = {sa.name for sa in spec.sub_agents}
    assert sub_names == {"researcher", "critic"}


def test_parse_interaction_defaults(agent_dir: Path) -> None:
    """Omitting interaction block entirely gives defaults."""
    spec = parse(agent_dir)
    assert spec.interaction.conversational is True
    assert spec.interaction.modalities.input == ["text"]
    assert spec.interaction.modalities.output == ["text"]


def test_parse_interaction_partial_modalities(tmp_path: Path) -> None:
    """Omitting one side of modalities defaults that side to [text]."""
    config = {
        "spec_version": 1,
        "interaction": {"modalities": {"input": ["text", "image"]}},
    }
    (tmp_path / "config.yaml").write_text(yaml.dump(config))
    spec = parse(tmp_path)
    assert spec.interaction.modalities.input == ["text", "image"]
    assert spec.interaction.modalities.output == ["text"]


def test_parse_ignores_unknown_files(agent_dir: Path) -> None:
    """Parser ignores files/directories not in the spec."""
    (agent_dir / "README.md").write_text("Ignored")
    (agent_dir / "extra_dir").mkdir()
    (agent_dir / "extra_dir" / "stuff.txt").write_text("Ignored")
    spec = parse(agent_dir)
    assert spec.name == "test-agent"


def test_parse_multiple_skills_sorted(agent_dir: Path) -> None:
    """Skills are discovered in sorted directory order."""
    for name in ["beta-skill", "alpha-skill"]:
        skill_dir = agent_dir / "skills" / name
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            f"---\nname: {name}\ndescription: Skill {name}.\n---\nContent."
        )
    spec = parse(agent_dir)
    assert [s.name for s in spec.skills] == ["alpha-skill", "beta-skill"]


# ── Env var expansion in MCP configs ───────────────────


def test_mcp_env_vars_expanded_from_environment(
    agent_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    ``${VAR}`` references in MCP env and headers are expanded
    against the process environment at parse time.
    """
    monkeypatch.setenv("MY_TOKEN", "secret-123")
    mcp_dir = agent_dir / "tools" / "mcp"
    mcp_dir.mkdir(parents=True)
    mcp_config = {
        "name": "token-server",
        "transport": "http",
        "url": "http://localhost:9000/mcp",
        "headers": {"Authorization": "Bearer ${MY_TOKEN}"},
    }
    (mcp_dir / "token.yaml").write_text(yaml.dump(mcp_config))
    spec = parse(agent_dir)
    assert spec.mcp_servers[0].headers == {"Authorization": "Bearer secret-123"}


def test_mcp_headers_expanded_from_environment(
    agent_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    ``${VAR}`` references in HTTP headers are expanded at parse
    time.
    """
    monkeypatch.setenv("MY_API_KEY", "key-abc")
    mcp_dir = agent_dir / "tools" / "mcp"
    mcp_dir.mkdir(parents=True)
    mcp_config = {
        "name": "auth-service",
        "transport": "http",
        "url": "http://localhost:9000/mcp",
        "headers": {"Authorization": "Bearer ${MY_API_KEY}"},
    }
    (mcp_dir / "auth.yaml").write_text(yaml.dump(mcp_config))
    spec = parse(agent_dir)
    assert spec.mcp_servers[0].headers == {
        "Authorization": "Bearer key-abc",
    }


def test_mcp_env_expansion_mixed_set_and_unset_raises(
    agent_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    If any env value contains an unresolved ``${VAR}``, parsing
    raises ValueError even when other vars are set.

    :param agent_dir: Temporary agent directory fixture.
    :param monkeypatch: Pytest monkeypatch for env vars.
    """
    monkeypatch.setenv("SET_VAR", "expanded")
    monkeypatch.delenv("UNSET_VAR", raising=False)
    mcp_dir = agent_dir / "tools" / "mcp"
    mcp_dir.mkdir(parents=True)
    mcp_config = {
        "name": "mixed",
        "transport": "http",
        "url": "http://localhost:9000/mcp",
        "headers": {
            "A": "${SET_VAR}",
            "B": "${UNSET_VAR}",
            "C": "plain-value",
        },
    }
    (mcp_dir / "mixed.yaml").write_text(yaml.dump(mcp_config))
    with pytest.raises(AgentPlaneError, match="Unresolved environment variable"):
        parse(agent_dir)


# ── MCP required field validation ─────────────────────


def test_mcp_missing_url_raises(agent_dir: Path) -> None:
    """
    Parser rejects an MCP config with ``transport: http`` but no
    ``url`` field.

    :param agent_dir: Temporary agent directory with minimal
        ``config.yaml``.
    """
    mcp_dir = agent_dir / "tools" / "mcp"
    mcp_dir.mkdir(parents=True)
    mcp_config = {
        "name": "no-url-server",
        "transport": "http",
        # url intentionally omitted
    }
    (mcp_dir / "no_url.yaml").write_text(yaml.dump(mcp_config))
    with pytest.raises(AgentPlaneError, match="missing required field 'url'"):
        parse(agent_dir)


# ── Timeout / retry / execution parsing ────────────────


def test_parse_llm_timeout_and_retry(tmp_path: Path) -> None:
    """LLM block with explicit request_timeout and retry overrides."""
    config = {
        "spec_version": 1,
        "llm": {
            "model": "openai/gpt-5.4",
            "request_timeout": 120,
            "retry": {
                "max_attempts": 5,
                "status_codes": [429, 502],
            },
        },
    }
    (tmp_path / "config.yaml").write_text(yaml.dump(config))
    spec = parse(tmp_path)
    assert spec.llm is not None

    # Explicit request_timeout should override the 300s default.
    # Failure means the parser ignores the request_timeout key.
    assert spec.llm.request_timeout == 120

    # Retry max_attempts should match the YAML value.
    # Failure means retry block is not parsed or defaults are used instead.
    assert spec.llm.retry.max_attempts == 5

    # Status codes should reflect the custom list, not the defaults.
    # Failure means the parser falls back to default status codes.
    assert spec.llm.retry.status_codes == [429, 502]


def test_parse_llm_timeout_defaults(tmp_path: Path) -> None:
    """LLM block with only model inherits default timeout and retry."""
    config = {
        "spec_version": 1,
        "llm": {"model": "openai/gpt-4o"},
    }
    (tmp_path / "config.yaml").write_text(yaml.dump(config))
    spec = parse(tmp_path)
    assert spec.llm is not None

    # Default LLM request_timeout is 300s per LLMConfig dataclass.
    # Failure means the parser sets a different default.
    assert spec.llm.request_timeout == 300

    # Default retry max_attempts is 3 per RetryConfig dataclass.
    # Failure means the parser produces a non-default retry config.
    assert spec.llm.retry.max_attempts == 3


def test_parse_tools_global_timeout_and_retry(tmp_path: Path) -> None:
    """Tools block with explicit timeout and retry overrides."""
    config = {
        "spec_version": 1,
        "tools": {
            "timeout": 30,
            "retry": {"max_attempts": 4},
        },
    }
    (tmp_path / "config.yaml").write_text(yaml.dump(config))
    spec = parse(tmp_path)

    # Explicit tools timeout should override the 60s default.
    # Failure means the parser ignores the tools timeout key.
    assert spec.tools.timeout == 30

    # Retry max_attempts should match the YAML value.
    # Failure means the tools retry block is not parsed.
    assert spec.tools.retry.max_attempts == 4


def test_parse_builtins_string_entries(tmp_path: Path) -> None:
    """Plain string entries in tools.builtins produce BuiltinToolConfig
    with empty config dicts."""
    config = {
        "spec_version": 1,
        "tools": {
            "builtins": ["web_search", "web_search_alt"],
        },
    }
    (tmp_path / "config.yaml").write_text(yaml.dump(config))
    spec = parse(tmp_path)

    # Two entries parsed, both with empty config.
    assert len(spec.tools.builtins) == 2
    assert spec.tools.builtins[0].name == "web_search"
    assert spec.tools.builtins[0].config == {}
    assert spec.tools.builtins[1].name == "web_search_alt"
    assert spec.tools.builtins[1].config == {}


def test_parse_builtins_dict_entries(tmp_path: Path) -> None:
    """Dict entries in tools.builtins carry tool-specific config."""
    config = {
        "spec_version": 1,
        "tools": {
            "builtins": [
                {
                    "name": "web_search_alt",
                    "api_key": "AIza-test",
                    "engine_id": "eng-123",
                },
            ],
        },
    }
    (tmp_path / "config.yaml").write_text(yaml.dump(config))
    spec = parse(tmp_path)

    assert len(spec.tools.builtins) == 1
    entry = spec.tools.builtins[0]
    assert entry.name == "web_search_alt"
    # Config contains all keys except 'name'.
    assert entry.config == {
        "api_key": "AIza-test",
        "engine_id": "eng-123",
    }


def test_parse_builtins_mixed_entries(tmp_path: Path) -> None:
    """tools.builtins supports a mix of strings and dicts."""
    config = {
        "spec_version": 1,
        "tools": {
            "builtins": [
                "web_search",
                {
                    "name": "web_search_cfg",
                    "api_key": "pplx-test",
                },
            ],
        },
    }
    (tmp_path / "config.yaml").write_text(yaml.dump(config))
    spec = parse(tmp_path)

    assert len(spec.tools.builtins) == 2
    # First entry: string → no config.
    assert spec.tools.builtins[0].name == "web_search"
    assert spec.tools.builtins[0].config == {}
    # Second entry: dict → has config.
    assert spec.tools.builtins[1].name == "web_search_cfg"
    assert spec.tools.builtins[1].config == {"api_key": "pplx-test"}


def test_parse_builtins_dict_missing_name(tmp_path: Path) -> None:
    """Dict entry without 'name' raises AgentPlaneError."""
    config = {
        "spec_version": 1,
        "tools": {
            "builtins": [
                {"api_key": "orphan-key"},
            ],
        },
    }
    (tmp_path / "config.yaml").write_text(yaml.dump(config))
    with pytest.raises(AgentPlaneError, match="name"):
        parse(tmp_path)


def test_parse_executor_config(tmp_path: Path) -> None:
    """Executor block with explicit timeout and max_iterations."""
    config = {
        "spec_version": 1,
        "executor": {
            "timeout": 7200,
            "max_iterations": 500,
        },
    }
    (tmp_path / "config.yaml").write_text(yaml.dump(config))
    spec = parse(tmp_path)

    # Explicit executor timeout should be honored.
    # Failure means executor block parsing is broken.
    assert spec.executor.timeout == 7200

    # Explicit max_iterations should override the 1000 default.
    # Failure means max_iterations is ignored by the parser.
    assert spec.executor.max_iterations == 500

    # Default type should be "llm" when not specified.
    # Failure means the parser doesn't apply the default type.
    assert spec.executor.type == "llm"


def test_parse_executor_defaults(tmp_path: Path) -> None:
    """No executor block yields ExecutorSpec defaults."""
    config = {"spec_version": 1}
    (tmp_path / "config.yaml").write_text(yaml.dump(config))
    spec = parse(tmp_path)

    # Default executor timeout is 3600s per ExecutorSpec.
    # Failure means the parser uses a different default.
    assert spec.executor.timeout == 3600

    # Default max_iterations is 1000 per ExecutorSpec.
    # Failure means the parser uses a different default.
    assert spec.executor.max_iterations == 1000

    # Default type is "llm" per ExecutorSpec.
    # Failure means the parser uses a different default.
    assert spec.executor.type == "llm"


def test_parse_executor_remote(tmp_path: Path) -> None:
    """Executor block with type: remote parses endpoint."""
    config = {
        "spec_version": 1,
        "executor": {
            "type": "remote",
            "endpoint": "http://localhost:8000/v1/turns",
            "request_timeout": 300,
        },
    }
    (tmp_path / "config.yaml").write_text(yaml.dump(config))
    spec = parse(tmp_path)

    # Remote executor type should be parsed from YAML.
    # Failure means executor type parsing is broken.
    assert spec.executor.type == "remote"

    # Endpoint should be captured from the YAML block.
    # Failure means the endpoint field is not parsed.
    assert spec.executor.endpoint == "http://localhost:8000/v1/turns"

    # Per-call request_timeout should be parsed.
    # Failure means executor.request_timeout is not parsed.
    assert spec.executor.request_timeout == 300


def test_parse_mcp_server_with_timeout_and_retry(
    agent_dir: Path,
) -> None:
    """MCP server YAML with per-server timeout and retry overrides."""
    mcp_dir = agent_dir / "tools" / "mcp"
    mcp_dir.mkdir(parents=True)
    mcp_config = {
        "name": "slow-service",
        "transport": "http",
        "url": "http://localhost:9000/mcp",
        "timeout": 120,
        "retry": {
            "max_attempts": 7,
            "backoff_base": 3.0,
        },
    }
    (mcp_dir / "slow.yaml").write_text(yaml.dump(mcp_config))
    spec = parse(agent_dir)
    assert len(spec.mcp_servers) == 1
    mcp = spec.mcp_servers[0]

    # Per-server timeout should be parsed from the YAML.
    # Failure means MCP timeout parsing is broken (returns None).
    assert mcp.timeout == 120

    # Per-server retry should be populated, not None.
    # Failure means the retry block is ignored for MCP servers.
    assert mcp.retry is not None

    # Retry max_attempts should match the YAML value.
    # Failure means MCP retry fields are not forwarded correctly.
    assert mcp.retry.max_attempts == 7
