# pasteur-hf

Pasteur-specific **artifact contracts** for Hugging Face dataset publishing — not a Hub client.

## Purpose

This crate defines:

- **Layout** — on-disk simulation bundle structure (`layout`)
- **Dataset cards** — README generation for simulation-output repos (`dataset_card`)
- **Local cache** — optional staging dirs under `PASTEUR_RS_CACHE_DIR` (`cache`)

## Public API

| Module | Functions |
|--------|-----------|
| `layout` | `list_sim_types`, `list_parquet_files`, `validate_sim_bundle` |
| `dataset_card` | `render_dataset_card`, `SourceDatasetInfo` |
| `cache` | `cache_root`, `models_dir`, `simulations_dir`, `list_dir_entries`, `delete_entry` |

## Non-goals

Do **not** add to this crate:

- Hub download/upload/auth
- REST API clients (`ureq`, `hf-hub`)
- Subprocess wrappers around `hf`

Agents perform all Hub operations with the official `hf` CLI and the workspace skills: [hf-cli](../.agents/skills/hf-cli/SKILL.md) and [huggingface-datasets](../.agents/skills/hf-datasets/SKILL.md). See [AGENTS.md](../AGENTS.md).

## Layout contract

```
<bundle>/
  clean/clean.parquet
  blackout/blackout.parquet
  jitter/jitter_0.parquet
  flipper/flipper.parquet
  README.md          # from pasteur-cli card
  predictions/       # optional, from pasteur-cli compare
    blackout.parquet
```

Every variant parquet must include `source_row_id` (string) and `row_ordinal` (u32).
