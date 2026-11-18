#!/usr/bin/env node
// Generate the npm packages for a release from built binaries:
//   @krv-labs/pasteur               launcher + optionalDependencies
//   @krv-labs/pasteur-<platform>    one prebuilt pasteur-cli each (esbuild/biome pattern)
//
//   node npm/scripts/build-packages.mjs --binaries <dir> --out <dir>
//        [--version X.Y.Z] [--targets suffix,suffix] [--require-all]
//
// <dir>/<rust-target>/pasteur-cli[.exe] is expected for every platform built.
// The version defaults to the workspace version in Cargo.toml.
import { chmodSync, copyFileSync, existsSync, mkdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { parseArgs } from "node:util";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../..");
const SCOPE = "@krv-labs";

// No musl: ort's download-binaries has no prebuilt ONNX Runtime for it.
export const PLATFORMS = [
  { suffix: "darwin-arm64", target: "aarch64-apple-darwin", os: "darwin", cpu: "arm64" },
  { suffix: "darwin-x64", target: "x86_64-apple-darwin", os: "darwin", cpu: "x64" },
  { suffix: "linux-x64-gnu", target: "x86_64-unknown-linux-gnu", os: "linux", cpu: "x64", libc: "glibc" },
  { suffix: "linux-arm64-gnu", target: "aarch64-unknown-linux-gnu", os: "linux", cpu: "arm64", libc: "glibc" },
  { suffix: "win32-x64-msvc", target: "x86_64-pc-windows-msvc", os: "win32", cpu: "x64" },
];

const cargoToml = () => readFileSync(path.join(ROOT, "Cargo.toml"), "utf8");

function workspaceField(name) {
  const section = cargoToml().split(/^\[workspace\.package\]$/m)[1]?.split(/^\[/m)[0] ?? "";
  const m = section.match(new RegExp(`^${name}\\s*=\\s*"([^"]+)"`, "m"));
  if (!m) throw new Error(`no [workspace.package] ${name} in Cargo.toml`);
  return m[1];
}

export const workspaceVersion = () => workspaceField("version");

const common = (version) => {
  const repository = workspaceField("repository");
  return {
    version,
    license: workspaceField("license"),
    homepage: workspaceField("homepage"),
    repository: { type: "git", url: `git+${repository}.git` },
    publishConfig: { access: "public" },
  };
};

export function buildPackages({ binaries, out, version, targets, requireAll }) {
  const wanted = PLATFORMS.filter((p) => !targets || targets.includes(p.suffix));
  const built = [];
  for (const p of wanted) {
    const exe = p.os === "win32" ? "pasteur-cli.exe" : "pasteur-cli";
    const src = path.join(binaries, p.target, exe);
    if (!existsSync(src)) {
      if (requireAll) throw new Error(`missing binary for ${p.suffix}: ${src}`);
      continue;
    }
    const dir = path.join(out, `pasteur-${p.suffix}`);
    rmSync(dir, { recursive: true, force: true });
    mkdirSync(path.join(dir, "bin"), { recursive: true });
    copyFileSync(src, path.join(dir, "bin", exe));
    chmodSync(path.join(dir, "bin", exe), 0o755);
    const manifest = {
      name: `${SCOPE}/pasteur-${p.suffix}`,
      description: `pasteur-cli binary for ${p.suffix}. Install ${SCOPE}/pasteur instead.`,
      ...common(version),
      os: [p.os],
      cpu: [p.cpu],
      ...(p.libc ? { libc: [p.libc] } : {}),
      files: ["bin"],
    };
    writeFileSync(path.join(dir, "package.json"), JSON.stringify(manifest, null, 2) + "\n");
    built.push(p);
  }
  if (built.length === 0) throw new Error("no binaries found; nothing to package");

  const rootDir = path.join(out, "pasteur");
  rmSync(rootDir, { recursive: true, force: true });
  mkdirSync(path.join(rootDir, "bin"), { recursive: true });
  copyFileSync(path.join(ROOT, "npm/pasteur/bin/pasteur.js"), path.join(rootDir, "bin/pasteur.js"));
  chmodSync(path.join(rootDir, "bin/pasteur.js"), 0o755);
  copyFileSync(path.join(ROOT, "npm/pasteur/README.md"), path.join(rootDir, "README.md"));
  const rootManifest = {
    name: `${SCOPE}/pasteur`,
    description: "Pasteur CLI: simulate data shifts, score ONNX models, compare, and render dataset cards.",
    ...common(version),
    keywords: ["clinical-ai", "robustness", "onnx", "cli", "evaluation"],
    bin: { "pasteur-cli": "bin/pasteur.js", pasteur: "bin/pasteur.js" },
    files: ["bin", "README.md"],
    engines: { node: ">=18" },
    optionalDependencies: Object.fromEntries(
      built.map((p) => [`${SCOPE}/pasteur-${p.suffix}`, version]),
    ),
  };
  writeFileSync(path.join(rootDir, "package.json"), JSON.stringify(rootManifest, null, 2) + "\n");
  return { root: rootDir, platforms: built.map((p) => path.join(out, `pasteur-${p.suffix}`)), built };
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  const { values } = parseArgs({
    options: {
      binaries: { type: "string" },
      out: { type: "string" },
      version: { type: "string" },
      targets: { type: "string" },
      "require-all": { type: "boolean", default: false },
    },
  });
  if (!values.binaries || !values.out) {
    console.error("usage: build-packages.mjs --binaries <dir> --out <dir> [--version X] [--targets a,b] [--require-all]");
    process.exit(2);
  }
  const res = buildPackages({
    binaries: path.resolve(values.binaries),
    out: path.resolve(values.out),
    version: values.version ?? workspaceVersion(),
    targets: values.targets?.split(","),
    requireAll: values["require-all"],
  });
  console.log(`built ${res.built.length} platform package(s) + launcher in ${values.out}`);
}
