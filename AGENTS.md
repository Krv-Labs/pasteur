# Pasteur Rust — Agent Guide

> **CRITICAL SECURITY MANDATE:** By default, all models and datasets MUST be uploaded to Hugging Face as **private** repositories (using `--private`) to protect sensitive patient features and proprietary model configurations.

`pasteur-cli` performs **local compute only** (simulate, evaluate, compare, dataset cards, local cache). All Hugging Face Hub I/O is done by agents with the official [`hf` CLI](https://huggingface.co/docs/huggingface_hub/guides/cli) and the workspace agent skills: [Hugging Face Hub CLI (`hf-cli`)](.agents/skills/hf-cli/SKILL.md) and [Hugging Face Dataset Viewer (`huggingface-datasets`)](.agents/skills/hf-datasets/SKILL.md).

## Available Workspace Skills

This workspace provides specialized agent skills to optimize Hugging Face Hub integration and local simulation workflows:

- **[Hugging Face CLI (`hf-cli`)](.agents/skills/hf-cli/SKILL.md)**: Activates Hugging Face Hub CLI commands for downloading/uploading datasets, models, spaces, managing local cache, buckets, and jobs.
- **[Hugging Face Dataset Viewer (`huggingface-datasets`)](.agents/skills/hf-datasets/SKILL.md)**: Integrates ready-to-use API queries for retrieving dataset subsets/splits, previewing and paginating rows, filtering, and checking dataset validity without full downloads.
- **[Pasteur Rust (`pasteur-rs`)](.agents/skills/pasteur-rs/SKILL.md)**: Coordinates local simulation, model evaluation, dataset card rendering, and model comparisons. Complements other Hugging Face skills.

## Prerequisites

```bash
# Install the Hugging Face CLI
curl -LsSf https://hf.co/cli/install.sh | bash
hf auth login

# Give your agent Hub and Pasteur command context (activates all workspace skills)
hf skills add            # Codex, Cursor, OpenCode, Pi
hf skills add --claude   # includes Claude Code
```

## Boundary: `hf` vs `pasteur-cli`

| Task | Tool |
|------|------|
| Download datasets/models | `hf download` |
| Explore/paginate/search datasets without download | Hugging Face Dataset Viewer (`huggingface-datasets`) |
| Create repos, upload files, branches | `hf repos create`, `hf upload` |
| Dataset metadata, tags, collections | `hf datasets info`, `hf` resource commands |
| Run simulators (blackout/jitter/flipper) | `pasteur-cli simulate` |
| Generate README dataset card | `pasteur-cli card` |
| Score ONNX models on simulation bundle | `pasteur-cli evaluate` / `compare` |
| Write predictions parquet locally | `pasteur-cli compare --predictions-out` |
| **Clear/list Pasteur local staging cache** | **`pasteur-cli cache`** |
| **Clear/list Hugging Face Hub download cache** | **`hf cache`** |

`pasteur-cli` never opens network connections. If a path does not exist locally, the agent must check its validity with the Dataset Viewer skill and then `hf download` first.

## Local cache: two systems (do not confuse them)

Agents often need to reclaim disk space. There are **two independent caches**:

| Cache | Tool | What it stores | Typical location |
|-------|------|----------------|------------------|
| **Hugging Face Hub cache** | `hf cache …` | Files downloaded by `hf download` (models, datasets, spaces) | Hub default (see `hf cache list --cache-dir`) |
| **Pasteur staging cache** | `pasteur-cli cache …` | Locally staged ONNX models and simulation run dirs (Pasteur-specific layout) | `$PASTEUR_RS_CACHE_DIR`, or OS cache `…/pasteur-rs`, or `./.pasteur-rs-cache` |

**Use `hf cache` for Hub downloads — not `pasteur-cli cache`.**  
**Use `pasteur-cli cache` for Pasteur’s own staging dirs — not `hf cache`.**

### `pasteur-cli cache` (Pasteur staging)

Backed by [`pasteur-hf` `cache`](hf/src/cache.rs). Buckets: `models/` and `simulations/`.

```bash
# Show cache root and entries with sizes
pasteur-cli cache list
pasteur-cli cache list --cache-dir /path/to/cache

# Remove one entry (default: search both buckets)
pasteur-cli cache rm stub.onnx
pasteur-cli cache rm run-a --kind simulations

# Remove all entries (default: both buckets)
pasteur-cli cache clear
pasteur-cli cache clear --kind models
```

Override root with `--cache-dir` or `PASTEUR_RS_CACHE_DIR`. This cache is **deliberately separate** from the Python CLI’s `.pasteur/` project layout.

### `hf cache` (Hub downloads)

For artifacts fetched with `hf download`:

```bash
hf cache list
hf cache prune          # detached / incomplete revisions
hf cache rm <TARGETS>   # specific repos or revisions
```

See [hf-cli skill](.agents/skills/hf-cli/SKILL.md) for full `hf cache` options.

## Inputs

`pasteur-cli` works on your own local parquet files. There is no bundled dataset.

**Clean data** (`simulate --input`): one row per sample.

| Column | Type | Notes |
|--------|------|-------|
| id column (`--id-col`, default `node_id`) | any | Becomes `source_row_id` (string) |
| `--feature` column (default `TSH`) | numeric | Target of blackout and jitter |
| companion columns (`--blackout-companion`) | any | Nulled together with the feature, e.g. a `_measured` flag |
| other features | numeric | Carried through and scored by `evaluate`/`compare` |

**Labels** (`--labels`): a group-membership table. It's required for `evaluate`/`compare` and for flipper; `simulate` skips flipper without it.

| Column | Type | Notes |
|--------|------|-------|
| `group_id` | u32 | One row per group |
| `member_ids` | list of integers | Row ids that belong to the group |

A row is labeled `1` when its id (parsed as an integer) is in the `member_ids` of the group whose `group_id` equals `--positive-group-id`. Every other row is labeled `0`. Ids that don't parse as integers are labeled `0`.

## On-disk output contract

After `pasteur-cli simulate --output ./output`:

```
output/
  clean/
    clean.parquet
  blackout/
    blackout.parquet
  jitter/
    jitter_0.parquet
    jitter_1.parquet
    ...
  flipper/
    flipper.parquet
```

Flipper parquets also include meta columns: `pair_id`, `step`, `t`, `source_a_row`, `source_b_row`, `label_a`, `label_b`. For flipper rows, `source_row_id` is synthetic (`pair_<id>_step_<n>`).

Every simulation parquet includes:

| Column | Type | Notes |
|--------|------|-------|
| `source_row_id` | string | Original row id; never int-coerced |
| `row_ordinal` | u32 | Clean-run row index (0..N-1) |
| *(features)* | numeric | Feature columns used by simulators |

After `pasteur-cli card`, also:

```
output/README.md   # HF dataset card (YAML front matter + layout)
```

After `pasteur-cli compare --predictions-out ./predictions/blackout.parquet`:

| Column | Type |
|--------|------|
| `source_row_id` | string |
| `variant_name` | string (`clean`, `blackout`, `jitter_0`, ...) |
| `label` | i64 |
| `<model_stem>` | f64 (probability per model) |
| `all_agree` | bool |

Publish predictions at `predictions/<sim_type>.parquet` in the HF dataset repo.

### JSON evaluation output

`evaluate` and `compare` emit `EvaluationResult` / `ComparisonResult` JSON (stdout or `--output`):

```json
{
  "baselines": { "roc_auc": 0.85 },
  "evaluations": {
    "metric_invariant": { "jitter_stability": 0.92, "flipper_stability": 0.88 },
    "metric_based": {
      "roc_auc": { "resiliency": 0.88, "generalizability": 1.0 }
    }
  },
  "created_at": "2026-07-10T12:00:00Z"
}
```

`flipper_stability` is populated when evaluating with `sim_type=flipper`; it is `null` for `blackout`/`jitter` runs.

See [`cli/README.md`](cli/README.md) for full command reference.

---

## Workflow 1: Simulate → card → publish

```bash
# 1. Download source clean data and labels (see "Inputs" above for the schemas)
hf download org/source-dataset clean.parquet --repo-type dataset --local-dir ./input
hf download org/source-dataset groups.parquet --repo-type dataset --local-dir ./labels

# 2. Local simulation
cargo run -p pasteur-cli -- simulate \
  --input ./input/clean.parquet \
  --labels ./labels/groups.parquet \
  --output ./output \
  --dataset-name org/source-dataset

# 3. Source metadata for provenance (tags/description)
hf datasets info org/source-dataset --format json > source-info.json

cargo run -p pasteur-cli -- card \
  --input ./output \
  --source-dataset org/source-dataset \
  --dest-repo org/source-dataset-simulations \
  --source-info source-info.json

# 4. Agent publishes as a private repository by default (MANDATORY DEFAULT)
hf repos create org/source-dataset-simulations --repo-type dataset --exist-ok --private
hf upload org/source-dataset-simulations ./output/README.md README.md --repo-type dataset
hf upload org/source-dataset-simulations ./output/clean clean --repo-type dataset
hf upload org/source-dataset-simulations ./output/blackout blackout --repo-type dataset
hf upload org/source-dataset-simulations ./output/jitter jitter --repo-type dataset
hf upload org/source-dataset-simulations ./output/flipper flipper --repo-type dataset
```

Follow `hf` stderr hints for next commands (e.g. branch/revision flags).

---

## Workflow 2: Evaluate a published bundle

```bash
hf download org/source-dataset-simulations --repo-type dataset --local-dir ./sim
hf download org/source-dataset groups.parquet --repo-type dataset --local-dir ./labels
hf download org/my-model model.onnx --local-dir ./models

cargo run -p pasteur-cli -- evaluate blackout \
  --sim-root ./sim \
  --labels ./labels/groups.parquet \
  --model ./models/model.onnx \
  --output ./scores.json
```

---

## Workflow 3: Compare models + publish predictions

```bash
hf download org/source-dataset-simulations --repo-type dataset --local-dir ./sim
hf download org/model-a model.onnx --local-dir ./models/a
hf download org/model-b model.onnx --local-dir ./models/b
hf download org/source-dataset groups.parquet --repo-type dataset --local-dir ./labels

cargo run -p pasteur-cli -- compare blackout \
  --sim-root ./sim \
  --labels ./labels/groups.parquet \
  --model ./models/a/model.onnx \
  --model ./models/b/model.onnx \
  --predictions-out ./predictions/blackout.parquet \
  --output ./comparison.json

hf upload org/source-dataset-simulations \
  ./predictions/blackout.parquet predictions/blackout.parquet --repo-type dataset
```

---

## Workflow 4: Evaluate flipper stability

```bash
hf download org/source-dataset-simulations --repo-type dataset --local-dir ./sim
hf download org/source-dataset groups.parquet --repo-type dataset --local-dir ./labels
hf download org/my-model model.onnx --local-dir ./models

cargo run -p pasteur-cli -- evaluate flipper \
  --sim-root ./sim \
  --labels ./labels/groups.parquet \
  --model ./models/model.onnx \
  --flip-threshold 0.5 \
  --output ./flipper-scores.json
```

---

## Annotations and metadata

Agents can enrich published datasets using standard `hf` commands:

- Edit `README.md` tags/description before upload, or open a PR on the Hub
- Add dataset metadata, collections, or linked models via `hf` resource commands
- Use `hf datasets info` to verify what landed after upload

`pasteur-cli card` only generates an initial README from source-dataset provenance. Further annotation is agent-driven via `hf`.
