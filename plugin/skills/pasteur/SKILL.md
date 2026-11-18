---
name: pasteur
description: Stress-test clinical ONNX models (binary, multiclass, multilabel, regression) with the local Pasteur CLI - simulate blackout, jitter, flipper, sweep, scramble and unit-conversion data and score models or prediction tables against them. Use when asked to evaluate model robustness, missingness, messy exports, unit mix-ups or decision boundaries with Pasteur.
---

# Pasteur

Run the CLI as `npx -y @krv-labs/pasteur <command> ...`. If `pasteur-cli` is already on `PATH`, `pasteur-cli <command> ...` is the same program. Below, `pasteur` stands for either.

## Safety

- The CLI only reads and writes the local paths you pass. It makes no network calls; keep it that way.
- All Hugging Face Hub I/O goes through the official `hf` CLI: download inputs first, pass local paths, upload results afterwards.
- Never upload or commit patient data, PHI, real clinical datasets or proprietary models. Only publish simulations of public or synthetic data: bundles keep each patient's row ID, so never upload one built from patient data, even to a private repo.
- Create Hugging Face model and dataset repos as **private** by default (`hf repos create <id> --type dataset --private`).

## Commands

| Goal | Command |
|------|---------|
| Simulate blackout/jitter/flipper | `pasteur simulate --input <clean.parquet> --labels <groups.parquet> --id-col <col> --feature <name> [--blackout-companion <col>] --output ./sim` |
| Add a sweep | `... --sweep <feature>[=<cutoff>] --sweep-range LO:HI [--sweep-relative-to <cutoff>]` |
| Add scramble (opt-in) | `... --scramble-rate 0.2 [--scramble-encoding numeric-text\|censored\|unit-suffix\|sentinel ...]` (default: all four) |
| Add units (opt-in) | `... --unit-rate 0.2 --unit-preset <name>` or `--unit-factor F [--unit-offset O]` |
| Add a site (opt-in) | `... --site site.json [--site-draws N]`: several blackout/scramble/units steps on different features as one variant. Templates: [sites/](sites/README.md) |
| List the toolkit as JSON | `pasteur catalog`: simulators, encodings, unit presets, site-file format |
| Score one model or predictions table | `pasteur evaluate <sim_type> --sim-root ./sim --labels <groups.parquet> (--model m.onnx \| --predictions p.parquet) -o out.json` |
| Score several | `pasteur compare <sim_type> --sim-root ./sim --labels ... --model a.onnx --predictions rules=b.parquet [--predictions-out preds.parquet] -o out.json` |
| Write a dataset card | `pasteur card --input ./sim --source-dataset <id> --dest-repo <id>` |

`<sim_type>` is positional: `blackout`, `jitter`, `flipper`, `sweep`, `scramble`, `units` or `site`. Each run loads one type; metrics for types not loaded sit at neutral values.

- Resiliency (`metric_based.*.resiliency`, stress AUC / clean AUC) is measured on whichever of blackout, scramble, units or site is loaded.
- `stress` (classifiers): per stress variant, `rows_changed` and `decision_change_rate`, the share of changed rows whose decision moved. An AUC ratio can hide these.
- Site draws (`--site-draws N`): `evaluate site` scores every draw; `generalizability` is the worst draw's resiliency and `resiliency` their mean. For "will this hold on another site's export?" run `/pasteur:generalizability`.
- Scramble: also read `parse_loss`, the values the loader silently nulled. High parse loss means scramble behaves like blackout. Sentinels (`-999`) parse, to wrong values.
- Units presets: `glucose_mgdl_to_mmoll`, `creatinine_mgdl_to_umoll`, `cholesterol_mgdl_to_mmoll`, `triglycerides_mgdl_to_mmoll`, `hemoglobin_gdl_to_gl`, `ft4_ngdl_to_pmoll`, `hba1c_ngsp_to_ifcc`, `temp_f_to_c`, `weight_lb_to_kg`. A factor-of-ten slip is `--unit-factor 10`; `--unit-factor 1` is an identity control.
- `--predictions [NAME=]PATH`: predictions scored outside Pasteur, keyed by `(variant_name, source_row_id)`, one Float64 column per output. Mix with `--model` in `compare`.
- `--oracle-labels <parquet>` instead of `--labels` when perturbations can change the true label (required for a scored `sweep`). `--abstain-class <class>` names a "cannot decide" output (multiclass/multilabel with oracle labels).
- sklearn-exported models reject NaN: pass `--null-fill 0` or declare `input.absent_sentinel` in `metadata.json`.
- `--id-col`, `--feature` and `--positive-group-id` default to the example thyroid dataset; always pass them. `compare --output` needs an existing directory.

## Task types

Pass the same `--task` to `simulate`, `evaluate` and `compare`; it must match the model's `metadata.json` `task_type`.

| `--task` | Truth | `--positive-group-id` | `--flip-threshold` |
|----------|-------|-----------------------|--------------------|
| `binary` (default) | `--labels` | The positive cohort | Probability cutoff, default 0.5 |
| `multiclass` | `--labels` | One per class, in `output.classes` order; cohorts partition the rows | Unused (argmax) |
| `multilabel` | `--labels` | One per label, in `output.classes` order | Default for labels without `output.thresholds`, 0.5 |
| `regression` | `--targets <parquet> --target-col <col>` | Unused | Clinical cutoff in target units, no default; pass the same value to `simulate` and `evaluate`/`compare` |

For classifiers, `--flip-threshold` belongs to `evaluate`/`compare` only. Cohorts are matched to model outputs by position; Pasteur cannot check that they line up.

## Bundle layout

```
sim/
  clean/clean.parquet
  blackout/blackout.parquet
  jitter/jitter_<i>.parquet
  flipper/flipper.parquet   # classifiers: needs --labels; regression: --targets and --flip-threshold
  sweep/sweep.parquet       # only with --sweep
  scramble/scramble.parquet # only with --scramble-rate
  units/units.parquet       # only with --unit-rate
  site/site.parquet         # only with --site; site_<i>.parquet with --site-draws
```

Pasteur reports evidence about behaviour under declared perturbations, not a pass/fail badge. Docs: https://docs.krv.ai/pasteur/ (metrics: `metrics.html`, simulators: `simulations.html`).
