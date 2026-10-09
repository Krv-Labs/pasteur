# Changelog

## Unreleased

### Fixed

- `JitterSimulator.transform` before `fit` now raises instead of returning the
  data unchanged, which would have scored as perfect jitter stability. This
  affects the Rust `JitterSimulator` and `pypasteur.JitterSimulator`.
- The `pasteur-model` README example passes `Some(1)` for the positive class
  index, matching the 0.4.0 `Option<usize>` signature.
- Positive-class-index errors from `OnnxModel::from_file` name the parameter
  as well as the CLI flag.

### Added

- `pasteur-cli --version`.
- `pypasteur.__version__`.

## 0.4.0

### Added

- **Multiclass and multilabel models.** `metadata.json` gains `task_type`
  (`binary` | `multiclass` | `multilabel`) and `output.{name, classes,
  thresholds}`. `simulate`, `evaluate`, and `compare` take `--task`, and
  `--positive-group-id` is repeatable (one cohort per model output, in
  `output.classes` order). All sklearn-onnx probability formats are read:
  `[n, K]` tensors, ZipMaps with int or string keys (matched to `classes` by
  name), and `MultiOutputClassifier`'s per-label tensor sequence.
- Every metric is computed per class/label; headline fields are macro
  averages. A new `multi` section in the result JSON reports micro ROC AUC,
  worst-label resiliency, a jitter decision-flip rate, the share of flipper
  pairs that never flip, the multiclass detour rate, labels excluded for
  having one class, and per-label scores. Binary output has no `multi` key.
- Multiclass jitter stability sums the per-class prediction variances and
  halves them, so it equals the binary score for 2 classes and does not
  depend on how many classes the model has. Multilabel averages per label.
- Multiclass flipper finds where the pairwise margin `p[b] − p[a]` crosses 0;
  multilabel flipper samples pairs per label (new `pair_label` grid column)
  and crosses each label's own threshold.
- **Single-target regression models.** `task_type: "regression"` in
  `metadata.json` (default output `variable`, then `predictions`), and
  `--task regression` with true values from `--targets <parquet>
  --target-col <name>`, joined by ID as text. `baselines` are `rmse`, `mae`
  and `r2`; resiliency is blackout R² ÷ clean R² (the share of skill kept)
  under `metric_based.r2`; jitter variance is scaled by the target's variance.
  Flipper pairs patients across a clinical cutoff (`--flip-threshold`, in
  target units, passed to both `simulate` and `evaluate`) and finds where the
  prediction crosses it. A new `regression` section reports these in target
  units; classifier JSON is unchanged.

### Changed

- **Breaking (library):** `Model::predict_proba` returns a `DataFrame` with
  one column per output, `SimulationVariant.y` is a `DataFrame` label matrix,
  `EvaluationConfig` gains `task`, and `OnnxModel::from_file` takes
  `positive_class_index: Option<usize>`. `TaskType` gains `Regression` and
  `EvaluationResult` gains an optional `regression` field. CLI usage for
  binary models is unchanged.
- `evaluate`/`compare` `--labels` is no longer required at parse time; it is
  required for classifiers and refused for regression. `--flip-threshold`
  still defaults to 0.5 for classifiers.
- **ROC AUC handles tied scores.** Ties count half (Mann–Whitney) instead of
  depending on row order. Results without ties agree with the previous
  implementation to floating-point rounding.
- The probability output is resolved when the model loads. A model without
  `output_probability`/`probabilities` (or the contract's `output.name`) now
  fails at load with the list of its real outputs, instead of at first
  prediction.
- `compare` refuses two `--model` files with the same file name instead of
  panicking on duplicate prediction columns.

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
