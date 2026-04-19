"""Test tools exercising the breadth of @tool decorator signature handling.

Covers:

- Plain primitive ``str`` arg (``greet``).
- Pydantic ``BaseModel`` arg with optional field (``format_record``).
- Multiple primitive args with defaults and ``Annotated`` description
  (``compute``).
"""

from typing import Annotated

from agent_plane_client import tool
from pydantic import BaseModel, Field


class PersonRecord(BaseModel):
    """A person record (test fixture)."""

    name: str
    age: int
    email: str | None = None


@tool
def greet(name: str) -> str:
    """
    Return a greeting for the given name.

    Args:
        name: The name to greet.
    """
    return f"Hello, {name}!"


@tool
def format_record(record: PersonRecord) -> str:
    """
    Format a person record as a one-line string.

    Args:
        record: The person record to format.
    """
    parts = [f"name={record.name}", f"age={record.age}"]
    if record.email is not None:
        parts.append(f"email={record.email}")
    return "Person(" + ", ".join(parts) + ")"


@tool
def compute(
    value: int,
    multiplier: int = 2,
    note: Annotated[str, Field(description="Optional note to echo back.")] = "",
) -> dict[str, int | str]:
    """
    Multiply ``value`` by ``multiplier`` and echo the optional note.

    Args:
        value: Base integer value.
        multiplier: Multiplier (defaults to 2).
    """
    return {"product": value * multiplier, "note": note}
