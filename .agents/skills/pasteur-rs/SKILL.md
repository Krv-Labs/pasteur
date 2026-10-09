---
name: pasteur-rs
description: Local Pasteur simulation and evaluation CLI (pasteur-cli) for binary, multiclass, multilabel, and regression ONNX models. Use with the hf-cli skill for Hub download/upload; pasteur-cli never touches the network.
---

# Pasteur Rust (pasteur-cli)

`pasteur-cli` only reads and writes local files. **All Hugging Face Hub I/O goes through `hf`** ([hf-cli](../hf-cli/SKILL.md)): download inputs first, pass local paths, upload results afterwards.

## Commands

| Goal | Command |
|------|---------|
| Download a dataset or model | `hf download ...` |
| Simulate blackout/jitter/flipper | `pasteur-cli simulate --input <clean.parquet> --labels <groups.parquet> --output ./sim` |
| Score one model | `pasteur-cli evaluate <sim_type> --sim-root ./sim --labels <groups.parquet> --model <model.onnx>` |
| Score several models | `pasteur-cli compare <sim_type> --sim-root ./sim --labels ... --model a.onnx --model b.onnx [--predictions-out preds.parquet]` |
| Write the dataset card | `pasteur-cli card --input ./sim --source-dataset <id> --dest-repo <id>` |
| Publish a bundle | `hf repos create <id> --type dataset --private`, then `hf upload` |

`<sim_type>` is `blackout`, `jitter`, or `flipper`. Flags and defaults: [cli/README.md](../../../cli/README.md).

## Task types

Pass the same `--task` to `simulate`, `evaluate`, and `compare`; it must match the model's `metadata.json` `task_type`.

| `--task` | Truth | `--positive-group-id` | `--flip-threshold` |
|----------|-------|-----------------------|--------------------|
| `binary` (default) | `--labels` | The positive cohort | Probability cutoff, default 0.5 |
| `multiclass` | `--labels` | One per class, in `output.classes` order; cohorts must partition the rows | Unused (argmax) |
| `multilabel` | `--labels` | One per label, in `output.classes` order | Default for labels without `output.thresholds`, 0.5 |
| `regression` | `--targets <parquet> --target-col <col>` | Unused | Clinical cutoff in target units. No default; pass the same value to `simulate` and to `evaluate`/`compare` |

For classifiers, `--flip-threshold` belongs to `evaluate`/`compare` only; `simulate` rejects it. Cohorts are matched to model outputs by position; Pasteur cannot check that they line up.

## Bundle layout

```
sim/
  clean/clean.parquet
  blackout/blackout.parquet
  jitter/jitter_<i>.parquet
  flipper/flipper.parquet   # classifiers: needs --labels; regression: --targets and --flip-threshold
```

Schemas: inputs and `metadata.json` in [Data Contracts](../../../docs/source/data-contracts.rst), outputs in [Output Layout](../../../docs/source/outputs.rst), metrics in [Reading the results](../../../docs/source/metrics.rst).

## Publishing

Only publish simulations of public or synthetic data. Run `pasteur-cli card` to add `README.md`, then upload the bundle with `hf upload` to a private repo.

## Build

```bash
cargo build -p pasteur-cli
```
