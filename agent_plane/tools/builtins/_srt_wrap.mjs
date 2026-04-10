#!/usr/bin/env node
/**
 * Wrapper around srt's SandboxManager library API that provides
 * filesystem isolation WITHOUT network restriction.
 *
 * srt's CLI requires ``network.allowedDomains`` in the settings
 * file (schema-validated), and rejects broad wildcards like
 * ``*`` or ``*.com``. There is no "allow all network" option.
 *
 * This wrapper bypasses that limitation:
 * 1. Initialize SandboxManager with a valid config (passes schema).
 * 2. Call updateConfig() with a config that omits allowedDomains.
 *    updateConfig() does NOT re-validate — so allowedDomains
 *    becomes undefined, and the ``hasNetworkConfig`` check
 *    evaluates to false, disabling network restriction.
 * 3. Wrap and execute the command with filesystem-only sandboxing.
 *
 * Usage:
 *   node _srt_wrap.mjs <config-json> <command-string>
 *
 * Where config-json is a JSON object with:
 *   {
 *     "filesystem": {
 *       "denyRead": [...],
 *       "allowRead": [...],
 *       "allowWrite": [...],
 *       "denyWrite": [...]
 *     }
 *   }
 */

import { execSync, execFileSync } from "child_process";
import { realpathSync } from "fs";
import { dirname, join } from "path";

// srt is installed globally — resolve the package from the srt
// binary's location so we don't need NODE_PATH or a local install.
function resolveSrtLibrary() {
  // `which srt` → /.../bin/srt → symlink to /.../lib/node_modules/@anthropic-ai/sandbox-runtime/dist/cli.js
  const srtBin = execFileSync("which", ["srt"], { encoding: "utf8" }).trim();
  const realBin = realpathSync(srtBin);
  // realBin = /.../lib/node_modules/@anthropic-ai/sandbox-runtime/dist/cli.js
  // package root = two dirs up from dist/cli.js
  const pkgRoot = dirname(dirname(realBin));
  return join(pkgRoot, "dist", "index.js");
}

const { SandboxManager } = await import(resolveSrtLibrary());

const args = process.argv.slice(2);
if (args.length < 2) {
  process.stderr.write(
    "Usage: _srt_wrap.mjs <config-json> <command-string>\n"
  );
  process.exit(1);
}

const config = JSON.parse(args[0]);
const command = args[1];

// Step 1: Initialize with a schema-valid config. The
// allowedDomains value here doesn't matter — we'll remove it
// in the next step.
await SandboxManager.initialize({
  network: { allowedDomains: [], deniedDomains: [] },
  filesystem: config.filesystem,
});

// Step 2: Replace the config with one that has NO
// allowedDomains key. updateConfig() skips schema validation,
// so config.network.allowedDomains becomes undefined. The
// runtime check `allowedDomains !== undefined` then evaluates
// to false, disabling the network proxy entirely.
SandboxManager.updateConfig({
  network: { deniedDomains: [] },
  filesystem: config.filesystem,
});

// Step 3: Wrap the command with filesystem-only sandboxing.
const wrapped = await SandboxManager.wrapWithSandbox(command);

// Step 4: Execute. Inherit stdio so output flows back to the
// Python parent process via its stdout pipe.
try {
  const output = execSync(wrapped, {
    shell: true,
    stdio: ["ignore", "pipe", "pipe"],
    timeout: 300_000,
  });
  process.stdout.write(output);
} catch (err) {
  // execSync throws on non-zero exit. Forward stdout/stderr
  // and exit with the child's code.
  if (err.stdout) process.stdout.write(err.stdout);
  if (err.stderr) process.stderr.write(err.stderr);
  process.exit(err.status ?? 1);
} finally {
  await SandboxManager.reset();
}
