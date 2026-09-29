# Changelog

## Unreleased

## 0.3.0

### Breaking

- **Removed Cloth scaffolding and Node bindings.** Deleted `python-runtime-sdist`
  (`pypasteur-sdist`), `node-runtime-sdist`, `node-bindings`, the unused
  `AUTH_CALLBACK` hook in `pypasteur`, the runtime Dockerfile, and the GCS /
  Artifact Registry release steps. None of it enforced anything: the auth hook
  was never set and the Node login accepted any credentials. Use `pypasteur`;
  release artifacts now attach to the GitHub Release only.

- **Python module renamed** `pypasteur_bindings` → `pypasteur`, so
  `pip install pypasteur` is used with `import pypasteur`.

### Changed

- Releases publish abi3 wheels (Linux x86_64/aarch64, macOS x86_64/arm64,
  Windows x64) and an sdist to PyPI, the four crates to crates.io, and all of
  it to the GitHub Release.
- Licensed under **BSD-3-Clause**. Crates carry crates.io metadata and inherit
  version, license, and MSRV from the workspace.
- Builds on **stable** Rust ≥ 1.95 instead of nightly. 1.95 is the floor
  polars 0.54 actually needs (`polars-ooc` uses APIs stabilized in 1.95), even
  though no dependency declares it.

## 0.2.0

Three contract bugs where pasteur-core produced plausible, wrong numbers instead
of an error. All three were being compensated for in erdos-reyni's Python layer,
so callers invoking `pasteur-cli` directly were unprotected.

### Breaking

- **Blackout masks companion columns** (#7, #9). `BlackoutConfig` gained a
  required `companions: Vec<String>`; `apply_mask` nulls the feature and its
  companions together, and errors on a companion that isn't in the frame.
  Previously a blacked-out assay kept its `{assay}_measured == 1` flag — a row
  no EHR emits — so flip rates measured extrapolation, not robustness.
  New: `pasteur-cli simulate --blackout-companion <COL>` (repeatable).
- **Null fill defaults to NaN, not `0.0`** (#6, #10). `OnnxModel::from_file`
  gained a `null_fill` parameter, and `evaluate`/`compare` gained `--null-fill`.
  Resolution order: flag → the model's `metadata.json` `input.absent_sentinel` →
  NaN. `0.0` TSH is not absence, it is profoundly suppressed, so the old
  hardcoded fill could push patients toward the positive class.
  **Scores change** for any model that does not declare `absent_sentinel`;
  non-NaN-native models now error rather than returning NaN probabilities that
  silently poison ROC AUC.
- **`pasteur-storage` removed** (#8, #11). Nothing referenced it; the CLI
  persists variants via `cli/src/io/parquet.rs::write_variant`.

### Fixed

- **Feature order is validated against the loaded model** (#5, #10).
  `from_file` resolves the input name against the session's actual inputs,
  checks the input width, and — when a `metadata.json` sits beside the `.onnx` —
  compares `input.feature_order` element-wise. Wrong width already errored
  inside ONNX Runtime; wrong *order* at the right width did not, so every
  patient was scored against the wrong thresholds with no warning.
- `hf/README.md` documented `flipper/flipper_pos_0.parquet`; the CLI writes
  `flipper/flipper.parquet` (#8, #11).

### Upgrading

- Add `companions: vec![]` to any `BlackoutConfig` literal, or pass the
  companion columns you want co-masked. Python and Node binding constructors
  take `companions` as a trailing optional argument, so positional callers are
  unaffected.
- Pass `null_fill` to `OnnxModel::from_file` (`None` keeps the sidecar-then-NaN
  default).
- To reproduce pre-0.2.0 numbers, declare `"absent_sentinel": 0.0` under
  `input` in the model's `metadata.json` — preferred, since it records the
  encoding next to the model it belongs to — or pass `--null-fill 0`.

## 0.1.0

Initial `pasteur-core`, `pasteur-cli`, `pasteur-hf`, `pasteur-model`,
`pasteur-storage` crates.
