r"""ANSI escape-sequence stripping for terminal output.

Called on the read path between the ring buffer and the tool
result. See ``designs/PERSISTENT_TERMINAL_RESEARCH.md`` §6.7
"ANSI stripping: always, at read time."

**Ordering matters**: OSC 633 markers are themselves OSC sequences;
if we strip ANSI before parsing OSC 633, we lose the markers. The
Shell parses OSC 633 first (extracting command boundaries and exit
codes), then passes the remainder here for stripping.

This module strips:

- CSI sequences (``ESC [ ... letter``) — colors, cursor movement,
  clear screen, etc.
- OSC sequences (``ESC ] ... BEL`` or ``ESC ] ... ESC \``) — any
  remaining after OSC 633 parsing (titles, hyperlinks, etc.).
- Designate-charset sequences (``ESC ( X`` / ``ESC ) X``) — rare
  but occasionally emitted.
- Bare ``\r`` at line starts — PTYs often emit carriage returns
  before newlines; stripping these keeps output readable.

This is a best-effort stripper, not a VT100 emulator. A pyte
``Screen`` can render more faithfully but is deferred per §6.2
("pyte optional").
"""

from __future__ import annotations

import re

# CSI (Control Sequence Introducer): ESC [ then any number of
# parameter bytes (0x30-0x3F), intermediate bytes (0x20-0x2F),
# ending with a final byte (0x40-0x7E). This covers colors,
# cursor movement, clear-screen, etc.
_CSI_RE = re.compile(rb"\x1b\[[\x30-\x3f]*[\x20-\x2f]*[\x40-\x7e]")

# OSC (Operating System Command): ESC ] ... BEL or ESC ] ... ESC \
# Used for titles, hyperlinks, and VS Code's shell integration
# (any OSC 633 markers that escape the Shell's own parser land here).
_OSC_RE = re.compile(rb"\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)")

# Designate G0/G1 charset: ESC ( X  or  ESC ) X (X is a single byte).
# Occasionally emitted by legacy programs (e.g. older curses apps).
_CHARSET_RE = re.compile(rb"\x1b[()][\x20-\x7e]")

# Bare CR at line starts (common PTY artefact where "\r\n" gets split
# and the \r becomes visible). We strip leading \r on each line; we
# preserve internal \r because some commands emit \r for progress
# updates that the agent may care about.
_LEADING_CR_RE = re.compile(rb"(?:\A|\n)\r+")


def strip_ansi(data: bytes) -> str:
    """Strip ANSI control sequences and decode to UTF-8 text.

    Applies in order: CSI, OSC, charset-designate, leading-CR
    stripping, then UTF-8 decode (with ``errors="replace"`` for
    undecodable bytes — tool output occasionally contains partial
    multi-byte chars split across reads).

    :param data: Raw bytes captured from the PTY, after OSC 633
        markers have been extracted by the caller.
    :returns: A Unicode string with ANSI escapes removed, ready for
        inclusion in a tool result.
    """
    data = _CSI_RE.sub(b"", data)
    data = _OSC_RE.sub(b"", data)
    data = _CHARSET_RE.sub(b"", data)

    # Strip leading \r on each line (keep the \n, drop the \r before it).
    data = _LEADING_CR_RE.sub(lambda m: m.group(0).replace(b"\r", b""), data)

    return data.decode("utf-8", errors="replace")
