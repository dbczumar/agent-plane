"""Tests for the ANSI escape-stripping helper.

These exercise the full stripping pipeline (CSI, OSC, charset-
designate, leading-CR) without any Shell or PTY involvement.
"""

from __future__ import annotations

from agent_plane.terminals.ansi import strip_ansi


def test_plain_text_passes_through() -> None:
    """Bytes containing no escape sequences decode unchanged."""
    assert strip_ansi(b"hello world\n") == "hello world\n"


def test_strips_csi_color_codes() -> None:
    """ANSI color codes (SGR) are removed; visible text preserved.

    ``\\x1b[31m`` is red, ``\\x1b[0m`` is reset — common terminal
    output from tools like pytest, git, ls --color.
    """
    colored = b"\x1b[31merror\x1b[0m: something broke"
    assert strip_ansi(colored) == "error: something broke"


def test_strips_csi_cursor_movement() -> None:
    """Cursor-movement escapes (``\\x1b[K`` etc.) are removed.

    Clear-line (``\\x1b[K``) and similar show up when programs
    redraw progress bars or spinners in place.
    """
    spinner = b"working\x1b[K done"
    assert strip_ansi(spinner) == "working done"


def test_strips_osc_hyperlink() -> None:
    """OSC 8 hyperlinks (``ESC ] 8 ; ; url BEL text ESC ] 8 ; ; BEL``) strip out."""
    link = b"\x1b]8;;https://example.com\x07link text\x1b]8;;\x07"
    # Stripping removes both OSC sequences; visible text remains.
    assert strip_ansi(link) == "link text"


def test_strips_osc_633_markers() -> None:
    """Any OSC 633 markers that escape the Shell's own parser are stripped.

    The production Shell parses and splits on D before handing bytes
    to strip_ansi, so C markers in particular are what would land
    here (they're not used for parsing, only D is). This test
    ensures they don't leak into agent-visible output.
    """
    with_marker = b"foo\x1b]633;C\x07bar"
    assert strip_ansi(with_marker) == "foobar"


def test_strips_csi_with_semicolon_parameters() -> None:
    """Multi-parameter CSI sequences (``\\x1b[1;31m``) are handled."""
    bold_red = b"\x1b[1;31mERROR\x1b[0m"
    assert strip_ansi(bold_red) == "ERROR"


def test_strips_charset_designate() -> None:
    """Charset-designate sequences (``ESC ( B``, ``ESC ) 0``) are removed."""
    data = b"\x1b(B\x1b)0hello"
    assert strip_ansi(data) == "hello"


def test_strips_leading_carriage_return_per_line() -> None:
    """``\\r\\n`` artefacts at line starts become ``\\n``.

    Common PTY artefact: the terminal driver echoes ``\\r\\n`` even
    when the program sent ``\\n``; stripping leading ``\\r`` keeps
    the text readable without losing internal progress-bar ``\\r``
    that programs intentionally emit.
    """
    data = b"line1\n\rline2\n\rline3"
    assert strip_ansi(data) == "line1\nline2\nline3"


def test_preserves_internal_carriage_return() -> None:
    """A CR not at a line start is kept (progress-bar use case).

    ``progress\\rprogress!`` is how many CLIs update a single line;
    stripping the internal CR would destroy this.
    """
    data = b"working...\rworking! done"
    # Internal \r preserved; no leading-CR pattern to strip.
    assert strip_ansi(data) == "working...\rworking! done"


def test_invalid_utf8_replaced_not_crashed() -> None:
    """Undecodable bytes produce the Unicode replacement char, not an exception.

    Defends against partial multi-byte UTF-8 chars split across
    read boundaries.
    """
    # 0xFF is never valid in UTF-8.
    data = b"valid\xffmore"
    result = strip_ansi(data)
    # Replacement character (U+FFFD) is what `errors="replace"` inserts.
    assert "\ufffd" in result
    assert "valid" in result
    assert "more" in result


def test_real_world_pytest_like_output() -> None:
    """A realistic chunk of colored pytest output strips to readable text."""
    # Compressed simulation of what pytest emits: "PASSED" in green,
    # test name, "FAILED" in red, traceback markers.
    raw = (
        b"test_foo.py::test_one \x1b[32mPASSED\x1b[0m\n"
        b"test_foo.py::test_two \x1b[31mFAILED\x1b[0m\n"
    )
    result = strip_ansi(raw)
    assert "PASSED" in result
    assert "FAILED" in result
    # No escape characters should remain.
    assert "\x1b" not in result
