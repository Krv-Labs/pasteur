pub mod flipper;
mod metrics;

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
    EvaluationConfig, EvaluationResult, EvaluationScores, LabelScores, MetricBasedScores,
    MetricInvariantScores, MultiOutputScores, SimulationVariant, TaskType,
};
use chrono::Utc;
use metrics::series_to_f64;
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
) -> Result<Vec<Vec<f64>>, CoreError> {
    predict_named(model, x, n_outputs).map(|(_, cols)| cols)
}

/// `predict_matrix`, plus the model's output column names.
fn predict_named(
    model: &dyn Model,
    x: &DataFrame,
    n_outputs: usize,
) -> Result<(Vec<String>, Vec<Vec<f64>>), CoreError> {
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
            let mut dataset_variants = HashMap::new();
            for (variant_name, named_variants) in simulation_variants {
                if variant_name == "clean" {
                    continue;
                }
                if let Some(variant) = named_variants.get(dataset_name) {
                    dataset_variants.insert(variant_name.clone(), variant.clone());
                }
            }

            let scores =
                self.run_single_dataset(model, clean_dataset, &dataset_variants, config)?;
            per_dataset_scores.push(scores);
        }

        // Aggregate scores (simplified)
        let jitter_stability = if !per_dataset_scores.is_empty() {
            per_dataset_scores
                .iter()
                .map(|s| s.jitter_stability)
                .sum::<f64>()
                / per_dataset_scores.len() as f64
        } else {
            1.0
        };

        let flipper_stability = if !per_dataset_scores.is_empty() {
            let scores: Vec<f64> = per_dataset_scores
                .iter()
                .filter_map(|s| s.flipper_stability)
                .collect();
            if scores.is_empty() {
                None
            } else {
                Some(scores.iter().sum::<f64>() / scores.len() as f64)
            }
        } else {
            None
        };

        let mut metric_based = HashMap::new();
        let mut baselines = HashMap::new();

        // Assume "roc_auc" for now
        let avg_resiliency = if !per_dataset_scores.is_empty() {
            per_dataset_scores.iter().map(|s| s.resiliency).sum::<f64>()
                / per_dataset_scores.len() as f64
        } else {
            1.0
        };

        let avg_baseline = if !per_dataset_scores.is_empty() {
            per_dataset_scores.iter().map(|s| s.baseline).sum::<f64>()
                / per_dataset_scores.len() as f64
        } else {
            0.0
        };

        metric_based.insert(
            "roc_auc".to_string(),
            MetricBasedScores {
                resiliency: avg_resiliency,
                generalizability: 1.0, // Simplified
            },
        );

        baselines.insert("roc_auc".to_string(), avg_baseline);

        let multi = average_multi(
            per_dataset_scores
                .into_iter()
                .filter_map(|s| s.multi)
                .collect(),
        );

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
        let (label_names, probs_clean) = predict_named(model, &clean_dataset.x, k)?;
        let thresholds = decision_thresholds(model, task, k, config.flip_threshold)?;

        let aucs: Vec<Option<f64>> = y
            .iter()
            .zip(&probs_clean)
            .map(|(y, p)| roc_auc(y, p))
            .collect();
        // Binary keeps its documented 0.5-for-one-class fallback; multi-output
        // averages only the labels where AUC is defined.
        let baseline = mean_some(&aucs).unwrap_or(0.5);

        let mut resiliency = 1.0;
        let mut label_resiliency = vec![None; k];
        if let Some(blackout) = simulated_variants.get("blackout") {
            let probs_stressed = predict_matrix(model, &blackout.x, k)?;
            let stressed: Vec<Option<f64>> = y
                .iter()
                .zip(&probs_stressed)
                .map(|(y, p)| roc_auc(y, p))
                .collect();
            if task.is_binary() {
                let stressed = stressed[0].unwrap_or(0.5);
                if baseline > 0.0 {
                    resiliency = stressed / baseline;
                }
                label_resiliency[0] = Some(resiliency);
            } else {
                // A ratio against chance-level discrimination is noise, not
                // robustness: only labels the model actually separates count.
                for (j, (clean, stressed)) in aucs.iter().zip(&stressed).enumerate() {
                    if let (Some(c), Some(s)) = (clean, stressed) {
                        if *c > 0.5 {
                            label_resiliency[j] = Some(s / c);
                        }
                    }
                }
                resiliency = mean_some(&label_resiliency).unwrap_or(1.0);
            }
        }

        let jitter_preds: Vec<Vec<Vec<f64>>> = simulated_variants
            .iter()
            .filter(|(name, _)| name.starts_with("jitter"))
            .map(|(_, variant)| predict_matrix(model, &variant.x, k))
            .collect::<Result<_, _>>()?;
        let label_jitter_var: Vec<Option<f64>> = (0..k)
            .map(|j| {
                let draws: Vec<Series> = jitter_preds
                    .iter()
                    .map(|d| Series::new("p".into(), d[j].clone()))
                    .collect();
                mean_jitter_variance(&draws)
            })
            .collect::<Result<_, _>>()?;
        let squash = |v: Option<f64>| v.map_or(1.0, |v| 1.0 / (1.0 + JITTER_STABILITY_ALPHA * v));
        let jitter_stability = squash(mean_some(&label_jitter_var));

        let flipper = if let Some(flipper) = simulated_variants.get("flipper") {
            Some(score_flipper(model, &flipper.x, task, k, &thresholds)?)
        } else {
            None
        };

        let multi = (!task.is_binary()).then(|| MultiOutputScores {
            task,
            roc_auc_micro: roc_auc(&y.concat(), &probs_clean.concat()),
            worst_label_resiliency: label_resiliency.iter().flatten().copied().reduce(f64::min),
            decision_flip_rate: decision_flip_rate(&jitter_preds, task, &thresholds),
            flipper_never_flipped: flipper.as_ref().map(|f| f.never_flipped),
            flipper_detour_rate: flipper.as_ref().and_then(|f| f.detour_rate),
            excluded_labels: label_names
                .iter()
                .zip(&aucs)
                .filter(|(_, auc)| auc.is_none())
                .map(|(name, _)| name.clone())
                .collect(),
            per_label: (0..k)
                .map(|j| LabelScores {
                    label: label_names[j].clone(),
                    roc_auc: aucs[j],
                    resiliency: label_resiliency[j],
                    jitter_stability: squash(label_jitter_var[j]),
                    flipper_stability: flipper.as_ref().and_then(|f| f.per_label[j]),
                })
                .collect(),
        });

        Ok(SingleDatasetScores {
            baseline,
            resiliency,
            jitter_stability,
            flipper_stability: flipper.map(|f| f.stability),
            multi,
        })
    }
}

/// The label matrix as one `Vec<f64>` per column, checked against the task:
/// binary is one column, multiclass rows are one-hot, every value is 0 or 1.
fn label_matrix(y: &DataFrame, task: TaskType) -> Result<Vec<Vec<f64>>, CoreError> {
    let cols: Vec<Vec<f64>> = y
        .columns()
        .iter()
        .map(|c| series_to_f64(c.as_materialized_series()))
        .collect::<Result<_, _>>()?;
    match (task, cols.len()) {
        (_, 0) => return Err(CoreError::InvalidConfig("no label columns".to_string())),
        (TaskType::Binary, 1) => return Ok(cols),
        (TaskType::Binary, n) => {
            return Err(CoreError::InvalidConfig(format!(
                "binary evaluation takes one label column, got {n}"
            )))
        }
        (TaskType::Multiclass, 1) => {
            return Err(CoreError::InvalidConfig(
                "multiclass evaluation needs at least two classes".to_string(),
            ))
        }
        _ => {}
    }
    for i in 0..y.height() {
        let row: Vec<f64> = cols.iter().map(|c| c[i]).collect();
        if row.iter().any(|v| *v != 0.0 && *v != 1.0) {
            return Err(CoreError::InvalidConfig(format!(
                "label row {i} holds {row:?}; {task} labels must be 0/1 indicators"
            )));
        }
        if task == TaskType::Multiclass && row.iter().sum::<f64>() != 1.0 {
            return Err(CoreError::InvalidConfig(format!(
                "label row {i} belongs to {} classes; multiclass rows need exactly one",
                row.iter().sum::<f64>()
            )));
        }
    }
    Ok(cols)
}

fn decision_thresholds(
    model: &dyn Model,
    task: TaskType,
    k: usize,
    flip_threshold: f64,
) -> Result<Vec<f64>, CoreError> {
    match (task, model.decision_thresholds()) {
        (TaskType::Multilabel, Some(t)) if t.len() != k => Err(CoreError::InvalidConfig(format!(
            "model declares {} decision thresholds for {k} labels",
            t.len()
        ))),
        (TaskType::Multilabel, Some(t)) => Ok(t),
        _ => Ok(vec![flip_threshold; k]),
    }
}

/// Share of rows whose decision differs between any two jitter draws: the
/// argmax class for multiclass, the set of labels at or over threshold for
/// multilabel. `None` with fewer than two draws (or for binary, whose output
/// predates this field).
fn decision_flip_rate(draws: &[Vec<Vec<f64>>], task: TaskType, thresholds: &[f64]) -> Option<f64> {
    if task.is_binary() || draws.len() < 2 {
        return None;
    }
    let n_rows = draws[0].first().map_or(0, Vec::len);
    if n_rows == 0 {
        return None;
    }
    let decision = |draw: &[Vec<f64>], i: usize| -> Vec<bool> {
        match task {
            TaskType::Multiclass => {
                let best = (0..draw.len())
                    .max_by(|&a, &b| draw[a][i].total_cmp(&draw[b][i]))
                    .unwrap_or(0);
                (0..draw.len()).map(|j| j == best).collect()
            }
            _ => (0..draw.len())
                .map(|j| draw[j][i] >= thresholds[j])
                .collect(),
        }
    };
    let flipped = (0..n_rows)
        .filter(|&i| {
            let first = decision(&draws[0], i);
            draws[1..].iter().any(|d| decision(d, i) != first)
        })
        .count();
    Some(flipped as f64 / n_rows as f64)
}

fn mean_some(xs: &[Option<f64>]) -> Option<f64> {
    let vals: Vec<f64> = xs.iter().flatten().copied().collect();
    (!vals.is_empty()).then(|| vals.iter().sum::<f64>() / vals.len() as f64)
}

/// Averages multi-output detail across datasets the same way the headline
/// numbers are averaged: field by field, over the datasets where defined.
fn average_multi(scores: Vec<MultiOutputScores>) -> Option<MultiOutputScores> {
    if scores.len() <= 1 {
        return scores.into_iter().next();
    }
    let avg = |f: &dyn Fn(&MultiOutputScores) -> Option<f64>| {
        mean_some(&scores.iter().map(f).collect::<Vec<_>>())
    };
    let first = &scores[0];
    let per_label = (0..first.per_label.len())
        .map(|j| {
            let label = |f: &dyn Fn(&LabelScores) -> Option<f64>| {
                mean_some(
                    &scores
                        .iter()
                        .map(|s| s.per_label.get(j).and_then(f))
                        .collect::<Vec<_>>(),
                )
            };
            LabelScores {
                label: first.per_label[j].label.clone(),
                roc_auc: label(&|l| l.roc_auc),
                resiliency: label(&|l| l.resiliency),
                jitter_stability: label(&|l| Some(l.jitter_stability)).unwrap_or(1.0),
                flipper_stability: label(&|l| l.flipper_stability),
            }
        })
        .collect();
    let mut excluded: Vec<String> = scores
        .iter()
        .flat_map(|s| s.excluded_labels.iter().cloned())
        .collect();
    excluded.sort();
    excluded.dedup();
    Some(MultiOutputScores {
        task: first.task,
        roc_auc_micro: avg(&|s| s.roc_auc_micro),
        worst_label_resiliency: avg(&|s| s.worst_label_resiliency),
        decision_flip_rate: avg(&|s| s.decision_flip_rate),
        flipper_never_flipped: avg(&|s| s.flipper_never_flipped),
        flipper_detour_rate: avg(&|s| s.flipper_detour_rate),
        excluded_labels: excluded,
        per_label,
    })
}

struct SingleDatasetScores {
    baseline: f64,
    resiliency: f64,
    jitter_stability: f64,
    flipper_stability: Option<f64>,
    multi: Option<MultiOutputScores>,
}
