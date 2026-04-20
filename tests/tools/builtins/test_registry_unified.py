"""
Tests for the unified builtin tool registry (POLICIES.md §15.8).

Phase 2 unification: `BUILTIN_NAMES` and the instantiable
subset both derive from a single `_BUILTIN_REGISTRY` dict,
with `None` factories marking framework-owned names
(``web_fetch``, ``introspect``, ``request_approval``).

These tests lock the name-space invariants so any future
registry change has to either deliberately touch them or
break loudly.
"""

from __future__ import annotations

from agent_plane.tools.builtins import (
    BUILTIN_NAMES,
    INSTANTIABLE_BUILTINS,
    REQUEST_APPROVAL_TOOL_NAME,
    get_builtin_tool,
)


def test_builtin_names_includes_request_approval() -> None:
    """`request_approval` is reserved at the name-space level
    so user-declared tools can't shadow it — a missing entry
    here means the validator's collision check silently would
    not fire on `request_approval`."""
    assert "request_approval" in BUILTIN_NAMES
    # Constant matches the string; a drift between the two
    # would flip downstream tests to silent false positives.
    assert REQUEST_APPROVAL_TOOL_NAME == "request_approval"


def test_builtin_names_includes_framework_owned_tools() -> None:
    """web_fetch and introspect are framework-owned (need
    runtime context, not instantiated via the registry). They
    must still occupy the name-space so user specs can't
    declare tools with these names."""
    assert "web_fetch" in BUILTIN_NAMES
    assert "introspect" in BUILTIN_NAMES


def test_instantiable_subset_excludes_framework_owned() -> None:
    """Framework-owned names are NOT in INSTANTIABLE_BUILTINS
    because they have no factory. The onboarding assistant
    uses this set to tell the agent author what they can
    declare — listing `request_approval` there would be
    confusing and wrong."""
    assert "request_approval" not in INSTANTIABLE_BUILTINS
    assert "web_fetch" not in INSTANTIABLE_BUILTINS
    assert "introspect" not in INSTANTIABLE_BUILTINS


def test_instantiable_is_subset_of_builtin_names() -> None:
    """Every instantiable name is also a reserved name. The
    two sets can't get out of sync because they derive from
    the same dict — this test guards against a refactor that
    introduces drift."""
    # subset check expressed via issubset — clearer than
    # "for-in" iteration.
    assert INSTANTIABLE_BUILTINS.issubset(BUILTIN_NAMES)


def test_get_builtin_tool_returns_none_for_framework_owned() -> None:
    """Calling get_builtin_tool on a framework-owned name
    returns None — the caller must fall back to the special
    constructor path. This is the same behavior as an
    unknown name, which is fine because BUILTIN_NAMES is
    the authoritative "is this reserved?" set."""
    assert get_builtin_tool("request_approval") is None
    assert get_builtin_tool("web_fetch") is None
    assert get_builtin_tool("introspect") is None


def test_get_builtin_tool_returns_none_for_unknown_name() -> None:
    """Unknown names also return None. Callers that want to
    distinguish "unknown" from "framework-owned" must check
    `name in BUILTIN_NAMES` first."""
    assert get_builtin_tool("definitely_not_a_tool") is None


def test_get_builtin_tool_instantiates_known_tools() -> None:
    """Instantiable tools produce a real Tool instance. Smoke
    test — if this breaks, every agent with `code_sandbox`
    declared starts failing at load time."""
    tool = get_builtin_tool("code_sandbox")
    # Not None + correct name — proves both the factory ran
    # and produced an instance with the expected identity.
    assert tool is not None
    assert tool.name() == "code_sandbox"


def test_builtin_names_size_matches_registry() -> None:
    """A sanity check that the derivation is lossless. If
    someone adds a new registry entry but BUILTIN_NAMES
    doesn't reflect it (impossible under current derivation,
    but a refactor could miss it), this test turns red."""
    # Lock the expected set so adding / removing a name is an
    # explicit test edit.
    assert BUILTIN_NAMES == frozenset(
        {
            # Instantiable
            "web_search",
            "code_sandbox",
            "upload_file",
            "list_files",
            "download_file",
            "search_conversations",
            "export_agent",
            # Framework-owned
            "web_fetch",
            "introspect",
            "request_approval",
        }
    )
