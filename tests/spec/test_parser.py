"""Tests for agent_plane.spec.parser."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

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
    with pytest.raises(ValueError, match="must be a YAML mapping"):
        parse(tmp_path)


def test_parse_missing_spec_version(tmp_path: Path) -> None:
    (tmp_path / "config.yaml").write_text(yaml.dump({"name": "no-version"}))
    with pytest.raises(ValueError, match="missing required field: spec_version"):
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
    assert spec.llm.max_completion_tokens == 4096
    assert spec.llm.reasoning_effort == "medium"
    assert spec.interaction.conversational is True
    assert spec.interaction.modalities.input == ["text", "image", "file"]
    assert spec.interaction.modalities.output == ["text"]
    assert spec.tools.agents == ["researcher", "critic"]
    assert spec.params == {"max_results": 10, "prefer_recent": True}


def test_parse_llm_missing_model(tmp_path: Path) -> None:
    config = {"spec_version": 1, "llm": {"max_completion_tokens": 100}}
    (tmp_path / "config.yaml").write_text(yaml.dump(config))
    with pytest.raises(ValueError, match="missing required field: model"):
        parse(tmp_path)


def test_parse_agents_md(agent_dir: Path) -> None:
    (agent_dir / "AGENTS.md").write_text("You are a helpful research assistant.")
    spec = parse(agent_dir)
    assert spec.instructions == "You are a helpful research assistant."


def test_parse_skill(agent_dir: Path) -> None:
    skill_dir = agent_dir / "skills" / "deep-search"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\n"
        "name: deep-search\n"
        "description: Search the web for sources.\n"
        "allowed_tools:\n"
        "  - search.web\n"
        "---\n"
        "When asked to research, use search.web."
    )
    spec = parse(agent_dir)
    assert len(spec.skills) == 1
    skill = spec.skills[0]
    assert skill.name == "deep-search"
    assert skill.description == "Search the web for sources."
    assert skill.allowed_tools == ["search.web"]
    assert skill.content == "When asked to research, use search.web."


def test_parse_skill_no_allowed_tools(agent_dir: Path) -> None:
    skill_dir = agent_dir / "skills" / "simple"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: simple\ndescription: A simple skill.\n---\nJust do it."
    )
    spec = parse(agent_dir)
    assert spec.skills[0].allowed_tools == []


def test_parse_skill_missing_frontmatter(agent_dir: Path) -> None:
    skill_dir = agent_dir / "skills" / "bad"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text("No frontmatter here.")
    with pytest.raises(ValueError, match="missing YAML frontmatter"):
        parse(agent_dir)


def test_parse_skill_missing_name(agent_dir: Path) -> None:
    skill_dir = agent_dir / "skills" / "no-name"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text("---\ndescription: Missing name.\n---\nContent.")
    with pytest.raises(ValueError, match="missing required field 'name'"):
        parse(agent_dir)


def test_parse_skill_missing_description(agent_dir: Path) -> None:
    skill_dir = agent_dir / "skills" / "no-desc"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text("---\nname: no-desc\n---\nContent.")
    with pytest.raises(ValueError, match="missing required field 'description'"):
        parse(agent_dir)


def test_parse_mcp_stdio(agent_dir: Path) -> None:
    mcp_dir = agent_dir / "tools" / "mcp"
    mcp_dir.mkdir(parents=True)
    mcp_config = {
        "name": "github",
        "description": "Access GitHub repos.",
        "transport": "stdio",
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-github"],
        "env": {"GITHUB_TOKEN": "${GITHUB_TOKEN}"},
    }
    (mcp_dir / "github.yaml").write_text(yaml.dump(mcp_config))
    spec = parse(agent_dir)
    assert len(spec.mcp_servers) == 1
    mcp = spec.mcp_servers[0]
    assert mcp.name == "github"
    assert mcp.transport == "stdio"
    assert mcp.command == "npx"
    assert mcp.args == ["-y", "@modelcontextprotocol/server-github"]
    assert mcp.env == {"GITHUB_TOKEN": "${GITHUB_TOKEN}"}


def test_parse_mcp_http(agent_dir: Path) -> None:
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
    assert mcp.transport == "http"
    assert mcp.url == "http://localhost:9000/mcp"
    assert mcp.headers == {"Authorization": "Bearer ${API_KEY}"}


def test_parse_mcp_missing_name(agent_dir: Path) -> None:
    mcp_dir = agent_dir / "tools" / "mcp"
    mcp_dir.mkdir(parents=True)
    (mcp_dir / "bad.yaml").write_text(yaml.dump({"transport": "stdio"}))
    with pytest.raises(ValueError, match="missing required field 'name'"):
        parse(agent_dir)


def test_parse_mcp_missing_transport(agent_dir: Path) -> None:
    mcp_dir = agent_dir / "tools" / "mcp"
    mcp_dir.mkdir(parents=True)
    (mcp_dir / "bad.yaml").write_text(yaml.dump({"name": "bad"}))
    with pytest.raises(ValueError, match="missing required field 'transport'"):
        parse(agent_dir)


def test_parse_local_python_tools(agent_dir: Path) -> None:
    py_dir = agent_dir / "tools" / "python"
    py_dir.mkdir(parents=True)
    (py_dir / "arxiv_search.py").write_text("def search(): pass")
    (py_dir / "web_scrape.py").write_text("def scrape(): pass")
    spec = parse(agent_dir)
    assert len(spec.local_tools) == 2
    names = {t.name for t in spec.local_tools}
    assert names == {"arxiv.search", "web.scrape"}
    assert all(t.language == "python" for t in spec.local_tools)


def test_parse_local_typescript_tools(agent_dir: Path) -> None:
    ts_dir = agent_dir / "tools" / "typescript"
    ts_dir.mkdir(parents=True)
    (ts_dir / "code_run.ts").write_text("export function run() {}")
    spec = parse(agent_dir)
    assert len(spec.local_tools) == 1
    assert spec.local_tools[0].name == "code.run"
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
