# Changelog

## Unreleased

### Docs

- New pages for hospital IT and model reviewers: security and data handling
  (data flow, network behaviour, offline install, PHI in outputs), metric
  definitions and caveats, and an end-to-end model-selection walkthrough.
- One-page usage overview PDF for IT review.
- Fixed broken schema links and removed internal references from the READMEs
  and `--help` text.

## 0.3.0

First open-source release. Versions 0.1 and 0.2 were internal and are not in
this repository's history.

### Added

- `pasteur-core`, `pasteur-cli`, `pasteur-hf`, and `pasteur-model`.
- `pypasteur` (`import pypasteur`). Wheels are abi3 for CPython ≥ 3.12:
  Linux x86_64/aarch64, macOS x86_64/arm64, and Windows x64, plus an sdist.
- A release publishes those artifacts to PyPI, the four crates to crates.io,
  and the GitHub Release.

### Behavior

- **Blackout masks companion columns.** `BlackoutConfig.companions` is
  required. `apply_mask` nulls the feature and its companions together, and
  errors if a companion is not in the frame. A blacked-out assay that kept
  `{assay}_measured == 1` is not a row an EHR emits, so flip rates would
  measure extrapolation rather than robustness.
  `pasteur-cli simulate --blackout-companion <COL>` is repeatable.
- **Null fill defaults to NaN.** `OnnxModel::from_file` takes `null_fill`,
  and `evaluate` / `compare` take `--null-fill`. Resolution order: flag →
  the model's `metadata.json` `input.absent_sentinel` → NaN. `0.0` TSH is
  not absence, it is profoundly suppressed. Models that are not NaN-native
  error instead of returning NaN probabilities that poison ROC AUC.
  A model trained with `0.0` as the missing-value token should declare
  `"absent_sentinel": 0.0` under `input` in `metadata.json`, or pass
  `--null-fill 0`.
- **Feature order is validated against the loaded model.** `from_file`
  resolves the input name against the session's actual inputs, checks the
  input width, and — when a `metadata.json` sits beside the `.onnx` —
  compares `input.feature_order` element-wise. Wrong width already errored
  inside ONNX Runtime; wrong order at the right width did not.

### Changed

- Licensed under **BSD-3-Clause**. Crates carry crates.io metadata and inherit
  version, license, and MSRV from the workspace.
- Builds on **stable** Rust ≥ 1.95. 1.95 is the floor polars 0.54 actually
  needs (`polars-ooc` uses APIs stabilized in 1.95), even though no
  dependency declares it.
