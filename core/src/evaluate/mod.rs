pub mod flipper;
mod metrics;
mod multi;

pub use flipper::{
    calculate_flipper_stability, find_flip_points, flipper_model_features, score_flipper,
    score_flipper_grid, FlipperPairResult, FlipperScores,
};
pub use metrics::{
    calculate_jitter_stability, calculate_roc_auc, mean_jitter_variance, roc_auc,
    JITTER_STABILITY_ALPHA,
};

use crate::error::CoreError;
use crate::schema::{
    EvaluationConfig, EvaluationResult, EvaluationScores, MetricBasedScores, MetricInvariantScores,
    MultiOutputScores, SimulationVariant,
};
use chrono::Utc;
use metrics::{jitter_stability_from_variance, series_to_f64};
use multi::{
    average_multi, decision_thresholds, headline_jitter_variance, jitter_variances, label_matrix,
    label_resiliency, mean_some, multi_output_scores, per_label_auc, Columns, LabelBreakdown,
};
use polars::prelude::*;
use std::collections::HashMap;

pub trait Model {
    /// One `Float64` column per output, in the model's class order: a single
    /// positive-class column for binary models, K columns for multiclass and
    /// multilabel models. Column names label the outputs in reports.
    fn predict_proba(&self, df: &DataFrame) -> Result<DataFrame, CoreError>;

    /// Per-label decision thresholds for multilabel models (from the model's
    /// contract). `None` falls back to `EvaluationConfig::flip_threshold`.
    fn decision_thresholds(&self) -> Option<Vec<f64>> {
        None
    }
}

/// Predicts and returns one `Vec<f64>` per output column (nulls as NaN),
/// refusing a model whose width or height does not match what the labels
/// say it should be — a 3-class model scored against 2 label groups would
/// otherwise pair every column with the wrong label.
pub(crate) fn predict_matrix(
    model: &dyn Model,
    x: &DataFrame,
    n_outputs: usize,
) -> Result<Columns, CoreError> {
    predict_named(model, x, n_outputs).map(|(_, cols)| cols)
}

/// `predict_matrix`, plus the model's output column names.
fn predict_named(
    model: &dyn Model,
    x: &DataFrame,
    n_outputs: usize,
) -> Result<(Vec<String>, Columns), CoreError> {
    let probs = model.predict_proba(x)?;
    if probs.width() != n_outputs {
        return Err(CoreError::InvalidConfig(format!(
            "model returns {} output column(s) but the labels have {n_outputs}; pass one \
             label group per model output, in the model's class order",
            probs.width()
        )));
    }
    if probs.height() != x.height() {
        return Err(CoreError::Generic(format!(
            "model returned {} rows for {} input rows",
            probs.height(),
            x.height()
        )));
    }
    let names = probs
        .get_column_names()
        .iter()
        .map(|s| s.to_string())
        .collect();
    let cols = probs
        .columns()
        .iter()
        .map(|c| series_to_f64(c.as_materialized_series()))
        .collect::<Result<_, _>>()?;
    Ok((names, cols))
}

pub struct Evaluator;

impl Default for Evaluator {
    fn default() -> Self {
        Self
    }
}

impl Evaluator {
    pub fn new() -> Self {
        Self
    }

    pub fn run(
        &self,
        model: &dyn Model,
        clean_datasets: &HashMap<String, SimulationVariant>,
        simulation_variants: &HashMap<String, HashMap<String, SimulationVariant>>,
        config: &EvaluationConfig,
    ) -> Result<EvaluationResult, CoreError> {
        let mut per_dataset_scores = Vec::new();

        for (dataset_name, clean_dataset) in clean_datasets {
            let dataset_variants: HashMap<String, SimulationVariant> = simulation_variants
                .iter()
                .filter(|(variant_name, _)| *variant_name != "clean")
                .filter_map(|(variant_name, named)| {
                    let variant = named.get(dataset_name)?;
                    Some((variant_name.clone(), variant.clone()))
                })
                .collect();
            per_dataset_scores.push(self.run_single_dataset(
                model,
                clean_dataset,
                &dataset_variants,
                config,
            )?);
        }

        // Each headline number is the mean over datasets where it is defined.
        let mean = |f: fn(&SingleDatasetScores) -> Option<f64>| {
            mean_some(per_dataset_scores.iter().map(f))
        };
        let jitter_stability = mean(|s| Some(s.jitter_stability)).unwrap_or(1.0);
        let flipper_stability = mean(|s| s.flipper_stability);
        let resiliency = mean(|s| Some(s.resiliency)).unwrap_or(1.0);
        let baseline = mean(|s| Some(s.baseline)).unwrap_or(0.0);
        let multi = average_multi(
            per_dataset_scores
                .into_iter()
                .filter_map(|s| s.multi)
                .collect(),
        );

        // Only ROC AUC is computed so far; generalizability is a placeholder.
        let metric_based = HashMap::from([(
            "roc_auc".to_string(),
            MetricBasedScores {
                resiliency,
                generalizability: 1.0,
            },
        )]);
        let baselines = HashMap::from([("roc_auc".to_string(), baseline)]);

        Ok(EvaluationResult {
            baselines,
            evaluations: EvaluationScores {
                metric_invariant: MetricInvariantScores {
                    jitter_stability,
                    flipper_stability,
                },
                metric_based,
            },
            created_at: Utc::now(),
            multi,
        })
    }

    fn run_single_dataset(
        &self,
        model: &dyn Model,
        clean_dataset: &SimulationVariant,
        simulated_variants: &HashMap<String, SimulationVariant>,
        config: &EvaluationConfig,
    ) -> Result<SingleDatasetScores, CoreError> {
        let task = config.task;
        let y = label_matrix(&clean_dataset.y, task)?;
        let k = y.len();
        let (names, probs) = predict_named(model, &clean_dataset.x, k)?;
        let thresholds = decision_thresholds(model, task, k, config.flip_threshold)?;

        let auc = per_label_auc(&y, &probs);
        // Binary keeps its documented 0.5-for-one-class fallback; multi-output
        // averages only the labels where AUC is defined.
        let baseline = mean_some(auc.iter().copied()).unwrap_or(0.5);

        let resiliency = match simulated_variants.get("blackout") {
            Some(blackout) => {
                let stressed = per_label_auc(&y, &predict_matrix(model, &blackout.x, k)?);
                label_resiliency(task, &auc, &stressed)
            }
            None => vec![None; k],
        };

        let jitter_draws: Vec<Columns> = simulated_variants
            .iter()
            .filter(|(name, _)| name.starts_with("jitter"))
            .map(|(_, variant)| predict_matrix(model, &variant.x, k))
            .collect::<Result<_, _>>()?;
        let jitter_variance = jitter_variances(&jitter_draws, k)?;

        let flipper = simulated_variants
            .get("flipper")
            .map(|flipper| score_flipper(model, &flipper.x, task, k, &thresholds))
            .transpose()?;

        let multi = (!task.is_binary()).then(|| {
            let labels = LabelBreakdown {
                names: &names,
                y: &y,
                probs: &probs,
                auc: &auc,
                resiliency: &resiliency,
                jitter_variance: &jitter_variance,
                jitter_draws: &jitter_draws,
                thresholds: &thresholds,
            };
            multi_output_scores(task, &labels, flipper.as_ref())
        });

        Ok(SingleDatasetScores {
            baseline,
            resiliency: mean_some(resiliency.iter().copied()).unwrap_or(1.0),
            jitter_stability: jitter_stability_from_variance(headline_jitter_variance(
                task,
                &jitter_variance,
            )),
            flipper_stability: flipper.map(|f| f.stability),
            multi,
        })
    }
}

struct SingleDatasetScores {
    baseline: f64,
    resiliency: f64,
    jitter_stability: f64,
    flipper_stability: Option<f64>,
    multi: Option<MultiOutputScores>,
}
