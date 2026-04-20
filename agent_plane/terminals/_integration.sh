# Agent-plane terminal OSC 633 shell-integration snippet.
#
# Sourced via `bash --rcfile` when the terminal tool spawns a new shell.
# Emits an OSC 633 D marker after each command completes, encoding the
# command's exit code. The terminal tool parses these markers to detect
# command completion reliably — far more robust than prompt-pattern
# scraping (Claude Code's fd-3 approach) or timing heuristics.
#
# OSC 633 is VS Code's superset of the vendor-neutral OSC 133; the D
# marker has identical semantics in both. See
# designs/PERSISTENT_TERMINAL_RESEARCH.md §6.3.
#
# Marker format:
#   ESC ] 633 ; D ; <exit_code> BEL
#   i.e. \x1b]633;D;<exit_code>\x07
#
# PROMPT_COMMAND runs just before bash displays each prompt — after a
# command has finished executing. We use it to emit the D marker with
# $? (the most-recent exit code).
#
# PS1 is set to empty so no prompt text appears between D markers; the
# terminal tool reads bytes between commands and expects exactly
# zero-or-more bytes of our own marker output.

__ap_postexec() {
    printf '\e]633;D;%d\a' "$?"
}

PROMPT_COMMAND='__ap_postexec'
PS1=''
