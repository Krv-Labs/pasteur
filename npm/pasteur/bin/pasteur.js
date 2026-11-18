#!/usr/bin/env node
// Launcher for the `pasteur-cli` binary. The binary itself ships in a
// per-platform package (@krv-labs/pasteur-<platform>) that the package manager
// installed as an optional dependency; nothing is downloaded at install or run
// time.
"use strict";

const { spawn } = require("node:child_process");
const path = require("node:path");

const SCOPE = "@krv-labs";

function platformSuffix() {
  const { platform, arch } = process;
  const cpu = { arm64: "arm64", x64: "x64" }[arch];
  if (!cpu) return null;
  if (platform === "darwin") return `darwin-${cpu}`;
  if (platform === "win32" && cpu === "x64") return "win32-x64-msvc";
  if (platform === "linux") return `linux-${cpu}-gnu`;
  return null;
}

function resolveBinary() {
  if (process.env.PASTEUR_BINARY) return process.env.PASTEUR_BINARY;
  const suffix = platformSuffix();
  if (!suffix) {
    fail(`unsupported platform ${process.platform}/${process.arch}`);
  }
  const pkg = `${SCOPE}/pasteur-${suffix}`;
  try {
    const manifest = require.resolve(`${pkg}/package.json`);
    const exe = process.platform === "win32" ? "pasteur-cli.exe" : "pasteur-cli";
    return path.join(path.dirname(manifest), "bin", exe);
  } catch {
    fail(
      `the platform package ${pkg} is not installed.\n` +
        "Reinstall without --no-optional / --omit=optional, or set PASTEUR_BINARY " +
        "to a pasteur-cli executable. (glibc Linux only; musl is not supported.)",
    );
  }
}

function fail(message) {
  process.stderr.write(`pasteur: ${message}\n`);
  process.exit(1);
}

const child = spawn(resolveBinary(), process.argv.slice(2), { stdio: "inherit" });
child.on("error", (err) => fail(`could not start the binary: ${err.message}`));
child.on("exit", (code, signal) => {
  if (signal) process.kill(process.pid, signal);
  else process.exit(code ?? 1);
});
for (const sig of ["SIGINT", "SIGTERM", "SIGHUP"]) {
  process.on(sig, () => child.kill(sig));
}
