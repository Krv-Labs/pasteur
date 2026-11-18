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
| Simulate blackout/jitter/flipper/sweep | `pasteur-cli simulate --input <clean.parquet> --labels <groups.parquet> [--sweep <name=cutoff> --sweep-range <lo:hi>] --output ./sim` |
| Add scramble / units (opt-in) | `... --scramble-rate 0.2` / `... --unit-rate 0.2 --unit-preset <name>` (or `--unit-factor F [--unit-offset O]`) |
| Add a site (opt-in) | `... --site site.json [--site-draws N]`: several blackout/scramble/units steps on different features as one variant; `pasteur-cli catalog` prints the format, presets and encodings as JSON |
| Score one model or predictions table | `pasteur-cli evaluate <sim_type> --sim-root ./sim --labels <groups.parquet> (--model <model.onnx> \| --predictions <preds.parquet>)` |
| Score several models / prediction tables | `pasteur-cli compare <sim_type> --sim-root ./sim --labels ... --model a.onnx --predictions rules=b.parquet [--predictions-out preds.parquet]` |
| Write the dataset card | `pasteur-cli card --input ./sim --source-dataset <id> --dest-repo <id>` |
| Publish a bundle | `hf repos create <id> --type dataset --private`, then `hf upload` |

`<sim_type>` is `blackout`, `jitter`, `flipper`, `sweep`, `scramble`, `units`, or `site`; resiliency is measured on whichever of blackout, scramble, units or site is loaded. Check `parse_loss` in scramble and site results: values the loader silently nulled. Classifier results also carry `stress`: per variant, `rows_changed` and `decision_change_rate` (changed rows whose decision moved). With site draws, `generalizability` is the worst draw's resiliency. Pass `--oracle-labels <parquet>` in addition to `--labels` when perturbations can change the true label. Flags and defaults: [cli/README.md](../../../cli/README.md).

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
  scramble/scramble.parquet # only with --scramble-rate
  units/units.parquet       # only with --unit-rate
  site/site.parquet         # only with --site; site_<i>.parquet with --site-draws
```

Schemas: inputs and `metadata.json` in [Data Contracts](../../../docs/source/data-contracts.rst), outputs in [Output Layout](../../../docs/source/outputs.rst), metrics in [Reading the results](../../../docs/source/metrics.rst).

## Generalizability protocol

For "will this model hold on another site's export?", on public or synthetic data only:

1. **Source.** Find a public dataset with the `hf` CLI, read its card and licence, and download a pinned revision into its own empty directory. Use parquet or CSV files only; never run a dataset's loading script. Refuse gated or credentialed sources (MIMIC, eICU) and anything under a data-use agreement.
2. **Fit.** Map its columns to the model's `metadata.json` `feature_order` and write the clean parquet and labels. `evaluate` refuses a feature-order mismatch, so a wrong mapping fails loudly.
3. **Control.** Run a site whose only step is `units` with `factor: 1`. `resiliency` must be 1.0 and `stress[0].rows_changed` 0. If not, stop: the harness moved, not the model.
4. **Attribute.** Run one site file per kind of mess (unit change, censored text, an assay never ordered) to see what each costs.
5. **Compose.** Put the steps in one site file with a `chance` each and run `--site-draws 20`. Read `generalizability` (worst draw) next to `resiliency` (mean), and the highest `decision_change_rate`.
6. **External cohort.** If a second public dataset has the same features, score the model on it clean. That is external validation; the simulated sites are not.
7. **Report.** One table, plus provenance: dataset id and revision, licence, site files, `--random-state`, `pasteur-cli --version`.

Simulated sites are declared perturbations of one cohort, not other hospitals' patients. Say so in the report, and upload nothing.

## Publishing

Only publish simulations of public or synthetic data. Run `pasteur-cli card` to add `README.md`, then upload the bundle with `hf upload` to a private repo.

## Build

```bash
cargo build -p pasteur-cli
```
