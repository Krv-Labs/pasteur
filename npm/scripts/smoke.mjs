#!/usr/bin/env node
// End-to-end check of the npm packaging on the host platform: build the
// packages from target/release/pasteur-cli, `npm pack` them, install the
// tarballs into a clean project (no registry), then run the launcher:
// --help, --version, and a simulate + evaluate against an ONNX fixture to prove
// the packaged binary runs ONNX Runtime with nothing else installed.
//
// The parquet inputs are written by Python with pyarrow (`$PYTHON`, default
// python3); pasteur-cli reads parquet only.
import { execFileSync } from "node:child_process";
import { copyFileSync, mkdirSync, mkdtempSync, readdirSync, writeFileSync } from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { buildPackages, PLATFORMS, workspaceVersion } from "./build-packages.mjs";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../..");
const win = process.platform === "win32";
const npm = win ? "npm.cmd" : "npm";
const targetDir = process.env.CARGO_TARGET_DIR ?? path.join(ROOT, "target");

function hostPlatform() {
  const cpu = { arm64: "arm64", x64: "x64" }[process.arch];
  const suffix =
    process.platform === "darwin" ? `darwin-${cpu}`
    : process.platform === "win32" ? "win32-x64-msvc"
    : `linux-${cpu}-gnu`;
  const p = PLATFORMS.find((x) => x.suffix === suffix);
  if (!p) throw new Error(`no packaging target for ${suffix}`);
  return p;
}

const tmp = mkdtempSync(path.join(os.tmpdir(), "pasteur-npm-smoke-"));
const version = workspaceVersion();
const host = hostPlatform();
const exe = win ? "pasteur-cli.exe" : "pasteur-cli";

const binaries = path.join(tmp, "binaries", host.target);
mkdirSync(binaries, { recursive: true });
copyFileSync(path.join(targetDir, "release", exe), path.join(binaries, exe));

const out = path.join(tmp, "packages");
const built = buildPackages({ binaries: path.join(tmp, "binaries"), out, version, targets: [host.suffix] });

const tarballs = path.join(tmp, "tarballs");
mkdirSync(tarballs);
const pack = (dir) =>
  execFileSync(npm, ["pack", dir, "--pack-destination", tarballs, "--silent"], { encoding: "utf8" }).trim().split("\n").pop();
const platformTar = path.join(tarballs, pack(built.platforms[0]));
const rootTar = path.join(tarballs, pack(built.root));

const project = path.join(tmp, "project");
mkdirSync(project);
writeFileSync(
  path.join(project, "package.json"),
  JSON.stringify({
    name: "smoke",
    private: true,
    dependencies: { "@krv-labs/pasteur": `file:${rootTar}` },
    overrides: { [`@krv-labs/pasteur-${host.suffix}`]: `file:${platformTar}` },
  }),
);
execFileSync(npm, ["install", "--no-audit", "--no-fund", "--loglevel=error"], { cwd: project, stdio: "inherit" });

// Run through the installed `pasteur-cli` bin shim, with a scrubbed env so a
// stray ORT_DYLIB_PATH or PASTEUR_BINARY cannot mask a packaging gap.
const { ORT_DYLIB_PATH, PASTEUR_BINARY, ...env } = process.env;
const bin = path.join(project, "node_modules/.bin", win ? "pasteur-cli.cmd" : "pasteur-cli");
const cli = (args) => execFileSync(bin, args, { encoding: "utf8", env, cwd: tmp, shell: win });

const versionOut = cli(["--version"]);
if (!versionOut.includes(version)) throw new Error(`--version printed ${JSON.stringify(versionOut)}, expected ${version}`);
console.log(`launcher ok: ${versionOut.trim()}`);
if (!cli(["--help"]).includes("evaluate")) throw new Error("--help does not list evaluate");
console.log("help ok");

// simulate + evaluate, mirroring cli/tests/cli_local.rs (multiclass flipper).
const work = path.join(tmp, "work");
mkdirSync(path.join(work, "models"), { recursive: true });
execFileSync(process.env.PYTHON ?? "python3", ["-", work], {
  stdio: ["pipe", "inherit", "inherit"],
  input: `
import sys, pyarrow as pa, pyarrow.parquet as pq
d = sys.argv[1]
pq.write_table(pa.table({
    "node_id": [str(i) for i in range(1, 13)],
    "f1": [-1.5 + 0.27 * i for i in range(12)],
    "f2": [((i * 7) % 12) / 6.0 - 1.0 for i in range(12)],
}), f"{d}/clean.parquet")
pq.write_table(pa.table({
    "group_id": pa.array([1, 2, 3], pa.uint32()),
    "member_ids": pa.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], pa.list_(pa.int64())),
}), f"{d}/groups.parquet")
`,
});
const model = path.join(work, "models/multiclass_tensor.onnx");
copyFileSync(path.join(ROOT, "model/tests/fixtures/multiclass_tensor.onnx"), model);
writeFileSync(
  path.join(work, "models/metadata.json"),
  JSON.stringify({
    task_type: "multiclass",
    input: { feature_order: ["f1", "f2"], absent_sentinel: 0.0 },
    output: { classes: ["low", "mid", "high"] },
  }),
);
const groups = ["1", "2", "3"].flatMap((g) => ["--positive-group-id", g]);
const labels = path.join(work, "groups.parquet");
const simRoot = path.join(work, "sim");
cli([
  "simulate", "--input", path.join(work, "clean.parquet"), "--output", simRoot,
  "--feature", "f1", "--labels", labels, "--task", "multiclass",
  "--jitter-iters", "2", "--flipper-pairs", "6", "--flipper-steps", "5", ...groups,
]);
const result = JSON.parse(cli([
  "evaluate", "flipper", "--sim-root", simRoot, "--labels", labels,
  "--model", model, "--task", "multiclass", ...groups,
]));
if (result.multi?.per_label?.length !== 3) throw new Error(`unexpected evaluate output: ${JSON.stringify(result).slice(0, 300)}`);
console.log("evaluate ok: ONNX model scored through the packaged binary");
console.log(`packages: ${readdirSync(tarballs).join(", ")}`);
