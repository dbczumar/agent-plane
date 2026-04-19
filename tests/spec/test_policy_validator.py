"""
Tests for validator-layer handling of the guardrails block.

Phase 0 scope: the parser (see ``test_policy_parser.py``) does
all the §13 spec-load rejections loudly. This file covers the
small remaining surface — that ``validate()`` accepts a
well-formed ``AgentSpec`` with guardrails attached, and
doesn't regress existing validation when the new field is
absent.

Runtime-layer cross-field checks (``function.path``
resolvability, label key cross-references) are deferred to
the phase that owns the runtime object — Phase 4 for
FunctionPolicy path resolution.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agent_plane.spec.parser import parse
from agent_plane.spec.validator import validate


@pytest.fixture()
def agent_dir(tmp_path: Path) -> Path:
    """Minimal agent dir shared across validator tests."""
    (tmp_path / "config.yaml").write_text("")
    return tmp_path


def _write_config(agent_dir: Path, config_yaml: str) -> Path:
    """Overwrite the fixture's config.yaml with test-specific content."""
    (agent_dir / "config.yaml").write_text(config_yaml)
    return agent_dir


def test_validate_passes_without_guardrails(agent_dir: Path) -> None:
    """Existing validator path still green when `guardrails:`
    is absent — regression guard for the AgentSpec extension."""
    _write_config(
        agent_dir,
        """
spec_version: 1
name: no-guardrails
""",
    )
    result = validate(parse(agent_dir))
    # Empty errors list means valid; no assertion-depth
    # concerns here because ``.errors`` IS the value under
    # test — not a proxy for it.
    assert result.errors == []
    assert result.valid is True


def test_validate_passes_with_full_guardrails(agent_dir: Path) -> None:
    """Full guardrails block with labels + all three policy
    types parses AND validates cleanly."""
    _write_config(
        agent_dir,
        """
spec_version: 1
name: full-guardrails
guardrails:
  labels:
    integrity:
      initial: "1"
      values: ["0", "1"]
      monotonic: decreasing
  policies:
    taint_web:
      type: label
      on: [tool_call:web_search]
      action: allow
      set_labels:
        integrity: "0"
    block_canada:
      type: prompt
      on: [input]
      prompt: Deny if user mentions Canada.
    rate_limit:
      type: function
      on: [tool_call]
      function:
        path: myorg.policies.rate_limit
        arguments: {limit: 10}
  ask_timeout: 30
""",
    )
    spec = parse(agent_dir)
    # Sanity: the parse produced the guardrails we expect —
    # if this assertion fails, the validator failure below
    # would be hiding a parser bug.
    assert spec.guardrails is not None
    assert len(spec.guardrails.policies) == 3

    result = validate(spec)
    # If this breaks, it means `validate()` grew a new check
    # that rejects a shape the parser accepts — investigate
    # which rule, and decide whether to reject earlier (in
    # the parser) or relax the validator.
    assert result.errors == [], (
        f"Expected validate() to pass on a spec that parsed cleanly. Errors: {result.errors}"
    )


def test_validate_passes_with_empty_guardrails_block(agent_dir: Path) -> None:
    """``guardrails: {}`` → validator still green. The block
    is allowed to be empty (no labels / no policies) —
    agents may opt into guardrails incrementally."""
    _write_config(
        agent_dir,
        """
spec_version: 1
name: empty-guardrails
guardrails: {}
""",
    )
    spec = parse(agent_dir)
    assert spec.guardrails is not None
    # ask_timeout defaulted, labels/policies absent.
    assert spec.guardrails.labels is None
    assert spec.guardrails.policies is None

    result = validate(spec)
    assert result.valid is True


def test_validate_does_not_create_errors_on_policy_names(agent_dir: Path) -> None:
    """Policy names come from YAML keys — YAML parsing already
    dedupes silently. Validator should not raise new errors
    tied to names; if this test starts failing, someone added
    a names-related validator rule without updating the
    parser to reject duplicates first."""
    _write_config(
        agent_dir,
        """
spec_version: 1
name: named-policies
guardrails:
  policies:
    policy_a:
      type: label
      on: [input]
      action: allow
    policy_b:
      type: label
      on: [input]
      action: allow
""",
    )
    result = validate(parse(agent_dir))
    assert result.errors == []
