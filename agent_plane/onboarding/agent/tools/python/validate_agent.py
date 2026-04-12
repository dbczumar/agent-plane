"""Validate an agent directory's config.yaml.

Parses and validates the agent spec using the same parser and
validator that ``ap server`` uses. A passing validation means
the agent will load and serve correctly.
"""

from typing import Any

SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "validate_agent",
        "description": (
            "Validate an agent directory's config.yaml. Returns "
            "'valid' with the agent name if the spec is correct, "
            "or a list of errors if something is wrong. Use this "
            "after creating an agent to verify it will work."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": ("Path to the agent directory containing config.yaml."),
                },
            },
            "required": ["path"],
        },
    },
}


async def run(arguments: dict[str, Any]) -> str:
    """
    Parse and validate an agent spec.

    :param arguments: Must contain ``"path"`` (str) pointing to
        an agent directory with config.yaml.
    :returns: ``"Valid: <agent-name>"`` or error details.
    """
    from pathlib import Path

    path_str = arguments.get("path", "")
    if not path_str:
        return "Error: 'path' parameter is required."

    agent_path = Path(path_str)
    if not agent_path.exists():
        return f"Error: directory '{agent_path}' does not exist."

    config_yaml = agent_path / "config.yaml"
    if not config_yaml.exists():
        return f"Error: no config.yaml found in '{agent_path}'."

    try:
        from agent_plane.spec.parser import parse
        from agent_plane.spec.validator import validate

        spec = parse(agent_path)
        result = validate(spec)

        if result.valid:
            return f"Valid: agent '{spec.name}' parsed and validated successfully."

        errors = "; ".join(f"{e.path}: {e.message}" for e in result.errors)
        return f"Validation errors: {errors}"
    except Exception as exc:
        return f"Parse error: {exc}"
