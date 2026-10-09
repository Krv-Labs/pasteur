use chrono::{DateTime, Utc};
use polars::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlackoutConfig {
    pub feature: String,
    /// Columns nulled together with `feature` (e.g. `TSH_measured`), so a
    /// blacked-out assay does not keep claiming it was observed. A null assay
    /// beside a `_measured == 1` flag appears in no real record and in no
    /// training set, so leaving companions untouched makes the resulting flip
    /// rate measure extrapolation rather than robustness.
    #[serde(default)]
    pub companions: Vec<String>,
    pub rate: f64,
    pub window_frac: f64,
    pub patient_id_col: Option<String>,
    pub time_col: Option<String>,
    pub random_state: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JitterConfig {
    pub feature: String,
    pub scale: f64,
    pub random_state: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CohortConfig {
    pub query: Option<String>,
}

/// Samples `n_pairs` pairs of rows with different label values and, for
/// each pair, linearly interpolates their raw feature vectors across
/// `n_steps` points from one to the other (`t=0`..`t=1`, inclusive) — no PCA,
/// no inverse transform, since the interpolation already lives in the same
/// space the model takes as input. `flip_threshold`/`positive_class_label`
/// aren't consulted here; they're for the evaluation-side flip finder that
/// consumes the generated grid.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlipperConfig {
    pub n_pairs: usize,
    pub n_steps: usize,
    pub flip_threshold: f64,
    pub positive_class_label: Option<serde_json::Value>,
    pub random_state: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SimulationConfig {
    pub blackout: Option<BlackoutConfig>,
    pub jitter: Option<JitterConfig>,
    pub cohort: Option<CohortConfig>,
    pub flipper: Option<FlipperConfig>,
}

/// What a model's probability output means. Binary models emit one
/// positive-class column; multiclass models emit K mutually exclusive class
/// probabilities (one row sums to 1); multilabel models emit K independent
/// per-label probabilities.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TaskType {
    #[default]
    Binary,
    Multiclass,
    Multilabel,
}

impl TaskType {
    pub fn is_binary(&self) -> bool {
        *self == TaskType::Binary
    }
}

impl std::fmt::Display for TaskType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            TaskType::Binary => "binary",
            TaskType::Multiclass => "multiclass",
            TaskType::Multilabel => "multilabel",
        })
    }
}

/// `y` is a label matrix with one 0/1 column per model output column, in the
/// model's class order: one column for binary, a one-hot row for multiclass,
/// a multi-hot row for multilabel.
#[derive(Debug, Clone)]
pub struct SimulationVariant {
    pub x: DataFrame,
    pub y: DataFrame,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvaluationConfig {
    pub n_jitter_iters: usize,
    pub allow_partial: bool,
    // Note: Callables (metrics) are harder in Rust to serialize,
    // we might just use a registry or predefined names.
    pub metrics: Vec<String>,
    /// Probability threshold for flipper stability (decision-boundary crossing).
    /// For multilabel models without per-label `output.thresholds`, applies
    /// to every label. Unused for multiclass, whose decision is the argmax.
    pub flip_threshold: f64,
    #[serde(default)]
    pub task: TaskType,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricInvariantScores {
    pub jitter_stability: f64,
    pub flipper_stability: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricBasedScores {
    pub resiliency: f64,
    pub generalizability: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvaluationScores {
    pub metric_invariant: MetricInvariantScores,
    pub metric_based: std::collections::HashMap<String, MetricBasedScores>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvaluationResult {
    pub baselines: std::collections::HashMap<String, f64>,
    pub evaluations: EvaluationScores,
    pub created_at: DateTime<Utc>,
    /// Multiclass/multilabel detail. The headline fields above hold macro
    /// averages over labels; this section holds what the averages hide.
    /// Absent for binary models, so binary output is unchanged.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub multi: Option<MultiOutputScores>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiOutputScores {
    pub task: TaskType,
    /// ROC AUC over every (patient, label) pair pooled. Dominated by common
    /// labels; the macro average in `baselines.roc_auc` weighs labels equally.
    pub roc_auc_micro: Option<f64>,
    /// Lowest per-label resiliency. Blacking out one assay often wrecks a
    /// single label while the macro average barely moves.
    pub worst_label_resiliency: Option<f64>,
    /// Share of patients whose decision (argmax class for multiclass, set of
    /// labels over threshold for multilabel) differs between jitter draws.
    pub decision_flip_rate: Option<f64>,
    /// Share of flipper pairs whose prediction never crossed the boundary.
    /// Those pairs count as 1.0 in `flipper_stability`, so read them together.
    pub flipper_never_flipped: Option<f64>,
    /// Multiclass only: share of flipper pairs whose path passes through a
    /// third class (argmax is neither endpoint's class at some step).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub flipper_detour_rate: Option<f64>,
    /// Labels left out of the macro averages because the clean cohort has
    /// only one class for them, so their ROC AUC is undefined.
    pub excluded_labels: Vec<String>,
    pub per_label: Vec<LabelScores>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LabelScores {
    pub label: String,
    pub roc_auc: Option<f64>,
    pub resiliency: Option<f64>,
    pub jitter_stability: f64,
    pub flipper_stability: Option<f64>,
}

/// One model's evaluation, tagged with a caller-supplied label (e.g. an HF
/// repo id) so a list of these is self-describing about which is which.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LabeledEvaluation {
    pub model_label: String,
    pub evaluation: EvaluationResult,
}

/// Any number of models evaluated against the same clean/simulated
/// datasets, side by side.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComparisonResult {
    pub evaluations: Vec<LabeledEvaluation>,
}
