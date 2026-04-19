"""Validate an agent directory's config.yaml.

Parses and validates the agent spec using the same parser and
validator that ``ap server`` uses. A passing validation means
the agent will load and serve correctly.
"""

from agent_plane.tools import tool


@tool
def validate_agent(path: str) -> str:
    """
    Validate an agent directory's config.yaml.

    Returns ``"Valid: <agent-name>"`` if the spec is correct, or
    a list of errors if something is wrong. Use this after
    creating an agent to verify it will work.

    Args:
        path: Path to the agent directory containing config.yaml.
    """
    import os
    from pathlib import Path

    if not path:
        return "Error: 'path' parameter is required."

    agent_path = Path(path)
    # Resolve relative paths against the conversation workspace so
    # validate_agent("my-agent") works from sandbox mode without the
    # LLM needing to know the absolute workspace path.
    if not agent_path.is_absolute():
        workspace = os.environ.get("_AP_WORKSPACE")
        if workspace:
            agent_path = Path(workspace) / agent_path

    if not agent_path.exists():
        return f"Error: directory '{agent_path}' does not exist."

    config_yaml = agent_path / "config.yaml"
    if not config_yaml.exists():
        return f"Error: no config.yaml found in '{agent_path}'."

    try:
        from agent_plane.spec.parser import parse
        from agent_plane.spec.validator import validate

        # expand_env=False: the generated agent may reference env vars
        # like ${OPENAI_API_KEY} that aren't set in the current process.
        # These are resolved at deploy/run time, not at creation time.
        spec = parse(agent_path, expand_env=False)
        result = validate(spec)

        if result.valid:
            return f"Valid: agent '{spec.name}' parsed and validated successfully."

        errors = "; ".join(f"{e.path}: {e.message}" for e in result.errors)
        return f"Validation errors: {errors}"
    except Exception as exc:
        return f"Parse error: {exc}"
