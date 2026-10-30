use std::path::{Path, PathBuf};

use clap::{Parser, Subcommand, ValueEnum};
use pasteur_core::TaskType;

#[derive(Parser)]
#[command(
    name = "pasteur-cli",
    version,
    about = "Local Pasteur simulation, evaluation, and dataset-card generation (no Hub I/O)"
)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Command,
}

#[derive(Subcommand)]
pub enum Command {
    /// Run Blackout/Jitter/Flipper simulators over a local clean parquet file.
    Simulate(SimulateArgs),
    /// Write a Hugging Face dataset card (README.md) for a simulation bundle.
    Card(CardArgs),
    /// Score one local ONNX model, or precomputed predictions, against a local
    /// simulation bundle.
    Evaluate(EvaluateArgs),
    /// Score ONNX models and precomputed predictions side by side; optionally
    /// write predictions parquet.
    Compare(CompareArgs),
    /// List, remove, or clear the local Pasteur staging cache.
    Cache(CacheArgs),
}

/// Mirrors `pasteur_core::TaskType` for clap.
#[derive(Clone, Copy, Debug, ValueEnum)]
pub enum TaskArg {
    /// One positive cohort; the model scores one positive-class column.
    Binary,
    /// One cohort per class; every row belongs to exactly one.
    Multiclass,
    /// One cohort per label; rows may belong to any number.
    Multilabel,
    /// One continuous target per row, from --targets; the model predicts it.
    Regression,
}

impl From<TaskArg> for TaskType {
    fn from(t: TaskArg) -> Self {
        match t {
            TaskArg::Binary => TaskType::Binary,
            TaskArg::Multiclass => TaskType::Multiclass,
            TaskArg::Multilabel => TaskType::Multilabel,
            TaskArg::Regression => TaskType::Regression,
        }
    }
}

/// Where a regression model's true values come from. Classifiers take
/// `--labels` instead.
#[derive(clap::Args, Clone)]
pub struct TargetArgs {
    /// Regression only: local parquet with one row per patient, holding an ID
    /// column and the true target value.
    #[arg(long)]
    pub targets: Option<PathBuf>,
    /// Regression only: the target column in --targets (e.g. `hba1c`).
    #[arg(long)]
    pub target_col: Option<String>,
    /// Regression only: the ID column in --targets, matched as text to the
    /// clean input's --id-col values.
    #[arg(long, default_value = "node_id")]
    pub targets_id_col: String,
}

/// Models scored outside Pasteur.
#[derive(clap::Args, Clone)]
pub struct PredictionsArgs {
    /// Precomputed predictions parquet, in place of an ONNX model: one row per
    /// evaluated row, keyed by `variant_name` and `source_row_id`, plus one
    /// Float64 column per output (one for binary and regression, one per class
    /// in class order for multiclass and multilabel). A single-model
    /// `--predictions-out` file is accepted as-is. The model is named after
    /// the file stem; `NAME=PATH` names it. `compare` takes it repeatedly.
    #[arg(long = "predictions", value_name = "[NAME=]PATH", value_parser = parse_predictions)]
    pub predictions: Vec<PredictionsSource>,
}

/// One `--predictions` value.
#[derive(Clone, Debug)]
pub struct PredictionsSource {
    pub label: Option<String>,
    pub path: PathBuf,
}

/// `NAME=PATH`, or a bare path; an existing file whose name holds `=` is
/// taken as a path.
fn parse_predictions(value: &str) -> Result<PredictionsSource, String> {
    match value.split_once('=') {
        Some((label, path)) if !label.is_empty() && !Path::new(value).exists() => {
            Ok(PredictionsSource {
                label: Some(label.to_string()),
                path: path.into(),
            })
        }
        _ => Ok(PredictionsSource {
            label: None,
            path: value.into(),
        }),
    }
}

/// Per-row true labels for decision models, whose correct answer can change
/// under a perturbation.
#[derive(clap::Args)]
pub struct OracleArgs {
    /// Local parquet with the correct label for every simulated row: the row
    /// key `source_row_id`, `variant_name` (as `compare --predictions-out`
    /// writes it), and the label columns Pasteur derives from --labels
    /// (`label` for binary, `group_<id>` per class or label). Replaces the
    /// labels copied from the clean row and adds the `oracle` section.
    #[arg(long)]
    pub oracle_labels: Option<PathBuf>,
    /// The model's output class meaning "cannot decide" (e.g.
    /// `cannot_determine`). Adds blackout abstention metrics.
    #[arg(long, requires = "oracle_labels")]
    pub abstain_class: Option<String>,
}

#[derive(clap::Args)]
pub struct CacheArgs {
    #[command(subcommand)]
    pub command: CacheCommand,
}

#[derive(Subcommand)]
pub enum CacheCommand {
    /// List cached models and simulation runs.
    List(CacheListArgs),
    /// Remove one cached entry by name.
    Rm(CacheRmArgs),
    /// Remove all cached entries (models and/or simulations).
    Clear(CacheClearArgs),
}

#[derive(clap::Args)]
pub struct CacheListArgs {
    #[command(flatten)]
    pub common: CacheCommonArgs,
}

#[derive(clap::Args)]
pub struct CacheRmArgs {
    #[command(flatten)]
    pub common: CacheCommonArgs,
    /// Cache bucket: `models`, `simulations`, or `auto` (try both).
    #[arg(long, default_value = "auto")]
    pub kind: String,
    /// File or directory name under the bucket.
    pub name: String,
}

#[derive(clap::Args)]
pub struct CacheClearArgs {
    #[command(flatten)]
    pub common: CacheCommonArgs,
    /// Clear only this bucket (`models` or `simulations`). Default: both.
    #[arg(long)]
    pub kind: Option<String>,
}

#[derive(clap::Args)]
pub struct CacheCommonArgs {
    /// Override cache root (default: `$PASTEUR_RS_CACHE_DIR` or OS cache).
    #[arg(long)]
    pub cache_dir: Option<PathBuf>,
}

#[derive(clap::Args)]
pub struct SimulateArgs {
    /// Local clean parquet (agent: `hf download ...` first).
    #[arg(long)]
    pub input: PathBuf,
    /// Provenance label for logs only.
    #[arg(long)]
    pub dataset_name: Option<String>,
    #[arg(long, default_value = "node_id")]
    pub id_col: String,
    #[arg(long, default_value = "TSH")]
    pub feature: String,
    /// Column nulled together with --feature during blackout (repeatable),
    /// e.g. `--blackout-companion TSH_measured`. Without it a blacked-out
    /// assay keeps claiming it was observed.
    #[arg(long = "blackout-companion")]
    pub blackout_companions: Vec<String>,
    #[arg(long, default_value_t = 0.1)]
    pub blackout_rate: f64,
    #[arg(long, default_value_t = 1.0)]
    pub jitter_scale: f64,
    #[arg(long, default_value_t = 3)]
    pub jitter_iters: u32,
    #[arg(long, default_value_t = 42)]
    pub random_state: u64,
    /// Group/cohort membership table used to derive labels for flipper pair
    /// sampling. Classification only; regression takes --targets.
    #[arg(long)]
    pub labels: Option<PathBuf>,
    #[command(flatten)]
    pub targets: TargetArgs,
    /// Cohort `group_id` defining the labels. Binary takes one (the positive
    /// cohort); multiclass and multilabel take one per model output, repeated
    /// in the model's `output.classes` order. Unused for regression.
    #[arg(long = "positive-group-id", default_values_t = [103u32])]
    pub positive_group_ids: Vec<u32>,
    /// `binary`, `multiclass`, `multilabel`, or `regression`; must match each
    /// model's `metadata.json` `task_type`.
    #[arg(long, value_enum, default_value_t = TaskArg::Binary)]
    pub task: TaskArg,
    /// Regression only: the clinical cutoff, in the target's units, that
    /// flipper pairs are sampled across (one patient below it, one at or
    /// above). Without it the flipper grid is skipped for regression.
    #[arg(long)]
    pub flip_threshold: Option<f64>,
    #[arg(long, default_value_t = 200)]
    pub flipper_pairs: usize,
    #[arg(long, default_value_t = 20)]
    pub flipper_steps: usize,
    #[command(flatten)]
    pub sweep: SweepArgs,
    #[arg(long, default_value = "output")]
    pub output: PathBuf,
}

/// The single-feature sweep grid. Without `--sweep`, `simulate` skips it.
#[derive(clap::Args)]
pub struct SweepArgs {
    /// Feature to sweep, optionally with its cutoff (`feature=cutoff`).
    /// Repeatable, one grid per feature.
    #[arg(long = "sweep", value_name = "FEATURE[=CUTOFF]")]
    pub features: Vec<String>,
    /// Grid range `lo:hi`: absolute, or a multiplier of each feature's
    /// cutoff when --sweep-relative-to is set. Required with --sweep.
    #[arg(long, value_name = "LO:HI", allow_hyphen_values = true)]
    pub sweep_range: Option<String>,
    /// Makes --sweep-range relative, and is the cutoff for every --sweep
    /// feature that does not give its own.
    #[arg(long, value_name = "CUTOFF")]
    pub sweep_relative_to: Option<f64>,
    /// Grid points over --sweep-range, both ends included.
    #[arg(long, default_value_t = 25)]
    pub sweep_steps: usize,
    /// Adds --sweep-steps more points within ±FRAC of the cutoff (e.g. 0.1).
    #[arg(long, value_name = "FRAC")]
    pub sweep_near: Option<f64>,
}

#[derive(clap::Args)]
pub struct CardArgs {
    #[arg(long, default_value = "output")]
    pub input: PathBuf,
    /// Source HF dataset id (provenance link in README).
    #[arg(long)]
    pub source_dataset: String,
    /// Destination HF dataset repo id (title in README).
    #[arg(long)]
    pub dest_repo: String,
    /// JSON from `hf datasets info <source> --format json`.
    #[arg(long)]
    pub source_info: Option<PathBuf>,
    #[arg(long = "tag")]
    pub tags: Vec<String>,
    #[arg(long)]
    pub description: Option<String>,
    /// Where to write README.md (default: `<input>/README.md`).
    #[arg(long)]
    pub output_readme: Option<PathBuf>,
}

#[derive(clap::Args)]
pub struct EvaluateArgs {
    /// Simulation type folder name (e.g. "blackout").
    pub sim_type: String,
    /// Local root with `clean/` and `<sim_type>/` subfolders.
    #[arg(long)]
    pub sim_root: PathBuf,
    /// Group/cohort membership table. Required for classification; regression
    /// takes --targets.
    #[arg(long)]
    pub labels: Option<PathBuf>,
    #[command(flatten)]
    pub targets: TargetArgs,
    #[arg(
        long,
        required_unless_present = "predictions",
        conflicts_with = "predictions"
    )]
    pub model: Option<PathBuf>,
    #[command(flatten)]
    pub predictions: PredictionsArgs,
    /// Model contract sidecar (`metadata.json`). Defaults to
    /// `{model_dir}/metadata.json`; pass an explicit path when the ONNX and
    /// contract live in different directories (e.g. a separate model registry).
    #[arg(long)]
    pub contract: Option<PathBuf>,
    #[arg(long, default_value = "input")]
    pub input_name: String,
    /// What a null becomes on the way into the model. Defaults to the
    /// model's `metadata.json` `input.absent_sentinel` if it declares one,
    /// else NaN — `0.0` TSH is not absence, it is a hyperthyroid signal.
    #[arg(long)]
    pub null_fill: Option<f32>,
    /// Binary models only: which output column is the positive class
    /// (default 1).
    #[arg(long)]
    pub positive_class_index: Option<usize>,
    /// Cohort `group_id` defining the labels. Binary takes one (the positive
    /// cohort); multiclass and multilabel take one per model output, repeated
    /// in the model's `output.classes` order. Unused for regression.
    #[arg(long = "positive-group-id", default_values_t = [103u32])]
    pub positive_group_ids: Vec<u32>,
    /// `binary`, `multiclass`, `multilabel`, or `regression`; must match each
    /// model's `metadata.json` `task_type`.
    #[arg(long, value_enum, default_value_t = TaskArg::Binary)]
    pub task: TaskArg,
    #[arg(long, default_value = "simulation")]
    pub dataset_name: String,
    /// Decision threshold for flipper crossings: the positive-class
    /// probability for binary models, every label without a contract
    /// `output.thresholds` for multilabel. Unused for multiclass (argmax).
    /// Defaults to 0.5. For regression it is the clinical cutoff in the
    /// target's units, has no default, and must match `simulate`'s.
    #[arg(long)]
    pub flip_threshold: Option<f64>,
    #[command(flatten)]
    pub oracle: OracleArgs,
    #[arg(short, long)]
    pub output: Option<PathBuf>,
}

#[derive(clap::Args)]
pub struct CompareArgs {
    pub sim_type: String,
    #[arg(long)]
    pub sim_root: PathBuf,
    /// Group/cohort membership table. Required for classification; regression
    /// takes --targets.
    #[arg(long)]
    pub labels: Option<PathBuf>,
    #[command(flatten)]
    pub targets: TargetArgs,
    #[arg(long = "model", required_unless_present = "predictions")]
    pub models: Vec<PathBuf>,
    #[command(flatten)]
    pub predictions: PredictionsArgs,
    /// Model contract sidecar (`metadata.json`). Defaults to
    /// `{model_dir}/metadata.json`; pass an explicit path when the ONNX and
    /// contract live in different directories (e.g. a separate model registry).
    #[arg(long)]
    pub contract: Option<PathBuf>,
    #[arg(long, default_value = "input")]
    pub input_name: String,
    /// What a null becomes on the way into the model. Defaults to the
    /// model's `metadata.json` `input.absent_sentinel` if it declares one,
    /// else NaN — `0.0` TSH is not absence, it is a hyperthyroid signal.
    #[arg(long)]
    pub null_fill: Option<f32>,
    /// Binary models only: which output column is the positive class
    /// (default 1).
    #[arg(long)]
    pub positive_class_index: Option<usize>,
    /// Cohort `group_id` defining the labels. Binary takes one (the positive
    /// cohort); multiclass and multilabel take one per model output, repeated
    /// in the model's `output.classes` order. Unused for regression.
    #[arg(long = "positive-group-id", default_values_t = [103u32])]
    pub positive_group_ids: Vec<u32>,
    /// `binary`, `multiclass`, `multilabel`, or `regression`; must match each
    /// model's `metadata.json` `task_type`.
    #[arg(long, value_enum, default_value_t = TaskArg::Binary)]
    pub task: TaskArg,
    #[arg(long, default_value = "simulation")]
    pub dataset_name: String,
    /// Decision threshold for flipper crossings: the positive-class
    /// probability for binary models, every label without a contract
    /// `output.thresholds` for multilabel. Unused for multiclass (argmax).
    /// Defaults to 0.5. For regression it is the clinical cutoff in the
    /// target's units, has no default, and must match `simulate`'s.
    #[arg(long)]
    pub flip_threshold: Option<f64>,
    #[command(flatten)]
    pub oracle: OracleArgs,
    #[arg(short, long)]
    pub output: Option<PathBuf>,
    /// Local path for per-patient predictions parquet (agent uploads with `hf`).
    #[arg(long)]
    pub predictions_out: Option<PathBuf>,
}
