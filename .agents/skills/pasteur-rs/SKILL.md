---
name: pasteur-rs
description: Local Pasteur simulation and evaluation CLI. Use with the hf CLI skill for Hub download/upload. Complements hf skills — does not replace them.
---

# Pasteur Rust (pasteur-cli)

Use this skill together with `hf skills add`. **All Hugging Face Hub I/O goes through `hf`.** `pasteur-cli` only reads/writes local files.

## When to use which tool

| Goal | Command |
|------|---------|
| Download dataset/model | `hf download ...` |
| Upload simulation bundle | `hf repos create ... --private` + `hf upload` |
| Run simulators (blackout/jitter/flipper) | `pasteur-cli simulate --input <local.parquet> --labels <groups.parquet>` |
| Generate README | `pasteur-cli card --input ./output --source-dataset ... --dest-repo ...` |
| Evaluate/compare models | `pasteur-cli evaluate/compare <sim_type> --sim-root ./sim --labels ... --model ...` |
| Flipper stability | `pasteur-cli evaluate flipper --sim-root ./sim --labels ... --model ... --flip-threshold 0.5` |

## Required layout before upload

After `simulate --labels <groups.parquet>`, the bundle must look like:

```
output/
  clean/clean.parquet
  blackout/blackout.parquet
  jitter/jitter_*.parquet
  flipper/flipper.parquet
```

Input schemas (clean parquet, `groups.parquet` labels) are documented in [Data Contracts](../../../docs/source/data-contracts.rst).

`flipper/` is only written when `--labels` is passed to `simulate` (flipper needs labels to sample cross-label pairs; blackout/jitter don't). Its parquet also carries meta columns (`pair_id`, `step`, `t`, `source_a_row`, `source_b_row`, `label_a`, `label_b`) alongside the interpolated features — `evaluate`/`compare sim_type=flipper` strip these before scoring.

Run `pasteur-cli card` to add `README.md`, then upload each folder with `hf upload`.

## Full workflows

See [AGENTS.md](../../../AGENTS.md) in the pasteur-core repo root for copy-paste agent workflows (simulate → publish, evaluate, compare + predictions).

## Build

```bash
cargo build -p pasteur-cli
```
