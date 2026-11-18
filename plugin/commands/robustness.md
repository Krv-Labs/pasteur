---
description: Stress one feature of an ONNX model with Pasteur's scramble and unit-conversion simulators and summarise resiliency and parse loss
argument-hint: <features.parquet> <labels.parquet> <model.onnx> <feature> [extra pasteur flags]
allowed-tools: Bash(npx -y @krv-labs/pasteur:*), Bash(pasteur-cli:*), Read
---

Run a Pasteur robustness check with these arguments: $ARGUMENTS

They are, in order: the features parquet, the labels (`groups.parquet`), the ONNX model, the feature to stress, then any extra Pasteur flags (for example `--id-col`, `--positive-group-id`, `--task`, `--null-fill`). `<extra>` below means the flags each command accepts: `--positive-group-id` and `--task` go to every command, `--id-col` to `simulate` only, `--null-fill` to `compare` only. If any of the first four is missing, or the id column and positive group are not given, ask before running: the defaults (`node_id`, `103`) belong to the example dataset.

Use `npx -y @krv-labs/pasteur` as `PASTEUR`, or `pasteur-cli` if `pasteur-cli --version` works. Everything stays local; do not upload anything.

1. Scramble with all encodings, plus an identity unit control (`--unit-factor 1`, which should change nothing):

   ```
   PASTEUR simulate --input <features> --labels <labels> --feature <feature> --output ./pasteur-robustness/control --scramble-rate 0.2 --unit-rate 0.2 --unit-factor 1 <extra>
   ```

2. A factor-of-ten unit slip (if the user named a matching `--unit-preset`, run that too in its own output directory):

   ```
   PASTEUR simulate --input <features> --labels <labels> --feature <feature> --output ./pasteur-robustness/units-x10 --unit-rate 0.2 --unit-factor 10 <extra>
   ```

3. Score each stress variant:

   ```
   PASTEUR compare scramble --sim-root ./pasteur-robustness/control --labels <labels> --model <model> -o ./pasteur-robustness/control/scramble.json <extra>
   PASTEUR compare units --sim-root ./pasteur-robustness/control --labels <labels> --model <model> -o ./pasteur-robustness/control/units.json <extra>
   PASTEUR compare units --sim-root ./pasteur-robustness/units-x10 --labels <labels> --model <model> -o ./pasteur-robustness/units-x10/units.json <extra>
   ```

   If scoring fails on NaN input, rerun with `--null-fill 0` and say so.

4. Read the three JSON files and report a short table: clean ROC AUC, `metric_based.roc_auc.resiliency` per run, and `parse_loss` for scramble. Then give takeaways:
   - Identity control: resiliency should be 1.0. Anything else means the harness, not the model, moved; flag it and distrust the other numbers.
   - Scramble: high `parse_loss` means the loader silently nulled values, so the model saw missingness; compare its resiliency with what a blackout would cost. Sentinels (`-999`) parse to wrong values and do not show in `parse_loss`.
   - Units x10: resiliency well below 1.0 means a site exporting the feature in another unit would shift predictions; recommend a unit check at ingestion.

   Call a run "holds" when resiliency is at least 0.95 and "degrades" otherwise, and state that the 0.95 line is a reading aid, not a Pasteur verdict. Pasteur reports evidence under declared perturbations, not a pass/fail certification.
