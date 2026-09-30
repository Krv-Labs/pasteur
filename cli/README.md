# pasteur-cli

Local-only CLI for Pasteur tabular simulation and evaluation. It makes **no network calls**: it reads and writes only the local paths you pass. For a full worked example, see [Choosing between models](../docs/source/model-selection.rst); for metric definitions, see [Reading the results](../docs/source/metrics.rst).

Install:

```bash
cargo install pasteur-cli
```

Input schemas (clean parquet, labels table) are documented in [Data Contracts](../docs/source/data-contracts.rst). For security and data handling, see [Security](../docs/source/security.rst).

Build from source:

```bash
cargo build -p pasteur-cli
# or
cargo run -p pasteur-cli -- --help
```

## Tests

- **Unit tests**: `#[cfg(test)] mod tests` at the bottom of `cli/src/**/*.rs` (when a module needs them).
- **Integration tests**: `cli/tests/` (e.g. `cli_local.rs`) for subprocess/CLI workflows.

Run: `cargo test -p pasteur-cli`.

## Commands

### `simulate`

Run Blackout, Jitter, and Flipper simulators over a local clean parquet.

```bash
pasteur-cli simulate \
  --input ./input/clean.parquet \
  --labels ./labels/groups.parquet \
  --output ./output \
  --id-col node_id \
  --feature TSH \
  --blackout-companion TSH_measured \
  --blackout-rate 0.1 \
  --jitter-iters 3 \
  --flipper-pairs 200 \
  --flipper-steps 20
```

| Flag | Default | Description |
|------|---------|-------------|
| `--input` | *(required)* | Local clean parquet |
| `--labels` | — | Group-membership parquet ([schema](../docs/source/data-contracts.rst)) for flipper pair sampling (omit to skip flipper) |
| `--output` | `output` | Root for sim-type folders |
| `--dataset-name` | — | Provenance label for logs |
| `--id-col` | `node_id` | Row id column (becomes `source_row_id`) |
| `--feature` | `TSH` | Feature for blackout/jitter |
| `--blackout-companion` | — | Column nulled together with `--feature` (repeatable). Without it a blacked-out assay keeps its `_measured == 1` flag — a row no EHR emits |
| `--blackout-rate` | `0.1` | Blackout mask rate |
| `--jitter-scale` | `1.0` | Jitter noise scale |
| `--jitter-iters` | `3` | Number of `jitter_<i>` variants |
| `--positive-group-id` | `103` | `group_id` for positive cohort (flipper) |
| `--flipper-pairs` | `200` | Cross-label pairs to sample |
| `--flipper-steps` | `20` | Interpolation steps per pair |
| `--random-state` | `42` | RNG seed |

The `--id-col`, `--feature`, and `--positive-group-id` defaults come from the example thyroid dataset; always pass them for your own data. `--jitter-scale` is in units of the feature's standard deviation.

**Output:** `output/<sim_type>/<variant>.parquet` — see [Output Layout](../docs/source/outputs.rst).

---

### `card`

Write a Hugging Face dataset card (`README.md`) for a simulation bundle.

```bash
hf datasets info org/source-dataset --format json > source-info.json

pasteur-cli card \
  --input ./output \
  --source-dataset org/source-dataset \
  --dest-repo org/source-dataset-simulations \
  --source-info source-info.json
```

| Flag | Description |
|------|-------------|
| `--input` | Simulation bundle root (default: `output`) |
| `--source-dataset` | Source HF dataset id (provenance link) |
| `--dest-repo` | Destination repo id (README title) |
| `--source-info` | JSON from `hf datasets info --format json` |
| `--tag` | Repeatable; alternative to `--source-info` tags |
| `--description` | Alternative to JSON description field |
| `--output-readme` | Default: `<input>/README.md` |

Uploading is a separate, explicit `hf upload` step outside Pasteur. Only publish simulations of public or synthetic data; bundles made from patient data contain patient identifiers and must not be uploaded (see [AGENTS.md](../AGENTS.md)).

---

### `evaluate`

Score one local ONNX model against a local simulation bundle.

```bash
pasteur-cli evaluate blackout \
  --sim-root ./sim \
  --labels ./labels/groups.parquet \
  --model ./models/model.onnx \
  --positive-group-id 103 \
  --output ./scores.json
```

| Flag | Default | Description |
|------|---------|-------------|
| `sim_type` | *(positional)* | e.g. `blackout`, `jitter`, `flipper` |
| `--sim-root` | *(required)* | Local bundle with `clean/` and `<sim_type>/` |
| `--labels` | *(required)* | Local `groups.parquet` for label derivation |
| `--model` | *(required)* | Local `.onnx` path |
| `--contract` | — | Explicit `metadata.json` path (default: next to `--model`) |
| `--input-name` | `input` | ONNX input tensor name |
| `--null-fill` | *(see below)* | What a null becomes on the way into the model |
| `--positive-class-index` | `1` | Positive class column in prob output |
| `--positive-group-id` | `103` | `group_id` for positive cohort |
| `--dataset-name` | `simulation` | Internal dataset key in JSON |
| `--flip-threshold` | `0.5` | Decision threshold for flipper stability |
| `-o, --output` | stdout | Write `EvaluationResult` JSON to file |

---

### `compare`

Score multiple local ONNX models; optionally write predictions parquet.

```bash
pasteur-cli compare blackout \
  --sim-root ./sim \
  --labels ./labels/groups.parquet \
  --model ./models/a.onnx \
  --model ./models/b.onnx \
  --predictions-out ./predictions/blackout.parquet \
  --output ./comparison.json
```

Same flags as `evaluate`, except:

- `--model` is repeatable (one per model). Models are labelled by file stem, so give each a different file name.
- `--predictions-out` writes a local per-patient predictions parquet
- A single `--contract` applies to every model

**Predictions schema:** `source_row_id`, `variant_name`, `label`, one f64 column per model (stem of model path), `all_agree` (all models on the same side of 0.5).

The `--output` directory must already exist.

---

### `cache`

Manage the local Pasteur staging cache (`$PASTEUR_RS_CACHE_DIR`, OS cache `…/pasteur-rs`, or `./.pasteur-rs-cache`). Separate from the Hugging Face Hub cache (`hf cache …`).

```bash
pasteur-cli cache list
pasteur-cli cache list --cache-dir /tmp/my-cache

pasteur-cli cache rm stub.onnx --kind models
pasteur-cli cache rm run-a --kind auto

pasteur-cli cache clear
pasteur-cli cache clear --kind simulations
```

| Subcommand | Description |
|------------|-------------|
| `list` | Show cache root and entries under `models/` and `simulations/` |
| `rm <name>` | Delete one entry (`--kind models\|simulations\|auto`, default `auto`) |
| `clear` | Delete all entries (`--kind` optional; default both buckets) |

---

## JSON schemas

Defined in [`core/src/schema.rs`](../core/src/schema.rs):

- **`EvaluationResult`**: `baselines`, `evaluations.metric_invariant`, `evaluations.metric_based`, `created_at`
- **`ComparisonResult`**: `evaluations: [{ model_label, evaluation }]`

---

### Model contract (`metadata.json`)

`evaluate` and `compare` derive the feature order from the simulation bundle's
own `clean/` parquet — nothing in the ONNX graph says what those columns *mean*.
A wrong width errors inside ONNX Runtime; a wrong *order* at the right width does
not, and every patient silently gets scored against the wrong thresholds.

Pass `--contract /path/to/metadata.json` when the sidecar is not next to the
model (for example, when ONNX files and their metadata are stored in separate
directories). Otherwise `{model_dir}/metadata.json` is used when present:

```json
{
  "input": {
    "feature_order": ["TSH", "T3", "T4"],
    "absent_sentinel": 0.0
  }
}
```

| Flag / key | Effect |
|------------|--------|
| `--contract` | Explicit path to the contract JSON; overrides the default `{model_dir}/metadata.json` lookup |
| `input.feature_order` | Compared element-wise against the order derived from the data; a mismatch aborts the run and names the first differing index |
| `input.absent_sentinel` | Default for `--null-fill` — the value the model was *fit* with standing in for "absent" |

Both keys are optional and unknown keys are ignored, so an existing model
metadata file can carry them alongside its own fields. A model with no sidecar still loads; only its width
gets checked, and `--null-fill` falls back to NaN. A sidecar that exists but does
not parse is an error — a contract nobody can read looks like protection that
isn't there.

**Why NaN and not `0.0`:** in a clinical feature space `0.0` TSH is not absence,
it is profoundly suppressed — a hyperthyroid signal — so filling zeros lets
blackout push patients toward the positive class. NaN-native models handle
missingness as missingness; anything exported from sklearn-onnx returns NaN
probabilities instead, which `predict_proba` rejects outright rather than letting
them poison ROC AUC. Pass `--null-fill 0` (or declare `absent_sentinel`) for those.
