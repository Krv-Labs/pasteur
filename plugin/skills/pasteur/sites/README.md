# Site templates

Starting points for `pasteur simulate --site <file>`. Copy one, replace every `FEATURE` (and `FEATURE_A`, `FEATURE_B`) with a column of the clean parquet and every `PRESET` with a unit preset for that analyte, then run it. `pasteur catalog` lists the presets, encodings and the file format as JSON. An unreplaced placeholder fails before anything is written, as a missing column or unknown preset.

| Template | What it simulates | Expected of a robust model and harness |
|---|---|---|
| `identity-control.json` | Nothing: a unit conversion by 1 | `resiliency` 1.0 and `rows_changed` 0. Anything else means the harness moved, not the model. Run this first |
| `si-units.json` | The site reports the analyte in SI units, unlabelled | A large drop: the model reads a different quantity. The fix is a unit check at ingestion |
| `eu-export.json` | SI units, then numbers written as text (`5,2`, `1,200.5`, padded, `5.2e0`) | High `parse_loss`: most text does not parse and becomes missing |
| `censored-labs.json` | A fifth of results censored (`<5.2`, `ND`) or replaced by a sentinel (`-999`) | Censored values show in `parse_loss`. Sentinels parse to wrong numbers and do not |
| `assay-not-run.json` | The site never orders this assay | The cost of losing the feature entirely. Add its `_measured` flag to `companions` if the model takes one |
| `composed.json` | Several of the above, each kept with a `chance` | Use with `--site-draws 20`; read `generalizability` (worst draw) |

Rules a site file must follow:

- Each step names one `feature`. Repeat a step to cover more columns.
- Nothing may touch a feature after a `scramble` step. Put scramble last for that feature.
- A `units` step takes `preset`, or `factor` with optional `offset`, never both.
- Mass-to-molar presets are per analyte. `glucose_mgdl_to_mmoll` on a creatinine column is a wrong test, not a hard one.
- No comments or extra keys: unknown fields are an error.
