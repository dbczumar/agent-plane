#!/usr/bin/env node
/**
 * PTY-compatible srt sandbox wrapper for the persistent terminal tool.
 *
 * Uses ``spawn(..., stdio: 'inherit')`` so the child stays attached
 * to the Python parent's PTY — stdin flows in, stdout comes back,
 * window size propagates, OSC 633 markers emerge cleanly. An
 * ``execSync``-based wrapper (the natural one-shot pattern) won't
 * work here: bash is expected to live as long as the conversation
 * does. Individual command timeouts are enforced Python-side by
 * ``Shell.run_sync``'s ``timeout_ms``, independent of this wrapper.
 *
 * Usage:
 *   node _srt_shell.mjs <config-json> <argv-json>
 *
 * - config-json: {"filesystem": {"allowRead": [...], "denyRead": [...],
 *                                "allowWrite": [...], "denyWrite": [...]}}
 * - argv-json:   JSON-encoded array, first element is the executable,
 *                rest are args. e.g. ["bash", "--rcfile", "/path/to/snippet"]
 *
 * Exit code:
 *   - Child's exit code on normal exit
 *   - 128 + signal on signal termination (bash convention)
 *   - 1 on srt init / spawn failure
 *
 * See ``designs/PERSISTENT_TERMINAL_RESEARCH.md`` §6.5 + §6.8.
 */

import { spawn, execFileSync } from "child_process";
import { realpathSync } from "fs";
import { dirname, join } from "path";

function resolveSrtLibrary() {
  // Find srt's package root via `which srt` → symlink target → two
  // dirs up → dist/index.js. Keeps us from needing NODE_PATH or a
  // local install.
  const srtBin = execFileSync("which", ["srt"], { encoding: "utf8" }).trim();
  const realBin = realpathSync(srtBin);
  const pkgRoot = dirname(dirname(realBin));
  return join(pkgRoot, "dist", "index.js");
}

const { SandboxManager } = await import(resolveSrtLibrary());

const args = process.argv.slice(2);
if (args.length < 2) {
  process.stderr.write(
    "Usage: _srt_shell.mjs <config-json> <argv-json>\n"
  );
  process.exit(1);
}

const config = JSON.parse(args[0]);
const argv = JSON.parse(args[1]);

if (!Array.isArray(argv) || argv.length === 0) {
  process.stderr.write("argv-json must be a non-empty JSON array\n");
  process.exit(1);
}

// Two-step init: get a schema-valid config accepted first, then
// updateConfig() with no allowedDomains to disable network
// restriction entirely. srt's validation path requires
// allowedDomains at initialize() time, but updateConfig() can
// drop it — the runtime honors the final config, which leaves
// the network unrestricted (agents need it for pip/npm).
await SandboxManager.initialize({
  network: { allowedDomains: [], deniedDomains: [] },
  filesystem: config.filesystem,
});
SandboxManager.updateConfig({
  network: { deniedDomains: [] },
  filesystem: config.filesystem,
});

// SandboxManager.wrapWithSandbox takes a shell command STRING and
// returns the wrapped command string (prefixed with bwrap/sandbox-exec
// and whatever other plumbing srt needs). We build that string from
// the argv array, quoting each element for shell safety.
function shellQuote(s) {
  // Single-quote everything; escape embedded single-quotes via
  // '\'' (close, escaped quote, reopen). This is the standard
  // bash trick for literal-quoting arbitrary content.
  return "'" + String(s).replace(/'/g, "'\\''") + "'";
}

const commandString = argv.map(shellQuote).join(" ");
const wrapped = await SandboxManager.wrapWithSandbox(commandString);

// spawn with shell: true so the wrapped string (which is a shell
// command invoking bwrap/sandbox-exec + bash) parses correctly.
// stdio: 'inherit' connects child stdin/stdout/stderr to the Python
// parent's PTY. The sandbox (bwrap/sandbox-exec) inherits node's
// stdio; bash inside the sandbox inherits from that. Chain works.
const child = spawn(wrapped, [], {
  shell: true,
  stdio: "inherit",
});

child.on("exit", async (code, signal) => {
  try { await SandboxManager.reset(); } catch (_) {}
  // Use bash exit-code convention: 128 + signal for signal
  // termination, otherwise the child's exit code.
  process.exit(signal ? 128 + toSignalNumber(signal) : (code ?? 0));
});

child.on("error", async (err) => {
  process.stderr.write(`_srt_shell.mjs: spawn error: ${err.message}\n`);
  try { await SandboxManager.reset(); } catch (_) {}
  process.exit(1);
});

// Proxy SIGINT/SIGTERM from Python parent to child so Ctrl-C
// through the PTY reaches bash, which then forwards to the
// foreground command.
for (const sig of ["SIGINT", "SIGTERM", "SIGHUP"]) {
  process.on(sig, () => {
    if (!child.killed) {
      child.kill(sig);
    }
  });
}

function toSignalNumber(sig) {
  // Node signal names → POSIX numbers. Only the ones bash sees
  // matter (SIGINT=2 is the most common).
  const map = {
    SIGHUP: 1, SIGINT: 2, SIGQUIT: 3, SIGILL: 4, SIGTRAP: 5,
    SIGABRT: 6, SIGBUS: 7, SIGFPE: 8, SIGKILL: 9, SIGUSR1: 10,
    SIGSEGV: 11, SIGUSR2: 12, SIGPIPE: 13, SIGALRM: 14, SIGTERM: 15,
  };
  return map[sig] ?? 0;
}
