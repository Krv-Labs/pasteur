//! Per-label pieces of an evaluation. A binary model is the one-column case
//! of everything here; multiclass and multilabel models also get the
//! `multi` detail section built from them.

use super::flipper::FlipperScores;
use super::metrics::{
    jitter_stability_from_variance, mean_jitter_variance, roc_auc, series_to_f64,
};
use super::Model;
use crate::error::CoreError;
use crate::schema::{LabelScores, MultiOutputScores, TaskType};
use polars::prelude::*;

/// One `Vec` per output column: `columns[label][row]`.
pub(crate) type Columns = Vec<Vec<f64>>;

/// The label matrix as one `Vec<f64>` per column, checked against the task:
/// binary is one column, multiclass rows are one-hot, every value is 0 or 1.
pub(crate) fn label_matrix(y: &DataFrame, task: TaskType) -> Result<Columns, CoreError> {
    let cols: Columns = y
        .columns()
        .iter()
        .map(|c| series_to_f64(c.as_materialized_series()))
        .collect::<Result<_, _>>()?;
    let invalid = |msg: String| Err(CoreError::InvalidConfig(msg));
    match (task, cols.len()) {
        (_, 0) => return invalid("no label columns".to_string()),
        (TaskType::Binary, 1) => return Ok(cols),
        (TaskType::Binary, n) => {
            return invalid(format!("binary evaluation takes one label column, got {n}"))
        }
        (TaskType::Multiclass, 1) => {
            return invalid("multiclass evaluation needs at least two classes".to_string())
        }
        (TaskType::Multiclass | TaskType::Multilabel, _) => {}
        (TaskType::Regression, _) => {
            return invalid("regression targets are scored by `regression::target_vector`".into())
        }
    }
    for i in 0..y.height() {
        let row: Vec<f64> = cols.iter().map(|c| c[i]).collect();
        if row.iter().any(|v| *v != 0.0 && *v != 1.0) {
            return invalid(format!(
                "label row {i} holds {row:?}; {task} labels must be 0/1 indicators"
            ));
        }
        let classes = row.iter().sum::<f64>();
        if task == TaskType::Multiclass && classes != 1.0 {
            return invalid(format!(
                "label row {i} belongs to {classes} classes; multiclass rows need exactly one"
            ));
        }
    }
    Ok(cols)
}

/// One decision threshold per output: the model's own for multilabel when
/// its contract declares them, else `flip_threshold`. Multiclass decides by
/// argmax and never reads these.
pub(crate) fn decision_thresholds(
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
        (TaskType::Multilabel, None)
        | (TaskType::Binary | TaskType::Multiclass | TaskType::Regression, _) => {
            Ok(vec![flip_threshold; k])
        }
    }
}

pub(crate) fn per_label_auc(y: &Columns, probs: &Columns) -> Vec<Option<f64>> {
    y.iter().zip(probs).map(|(y, p)| roc_auc(y, p)).collect()
}

/// Stressed ÷ clean AUC per label. Binary keeps its original rule (any
/// positive baseline, an undefined AUC read as 0.5). Multi-output skips
/// labels at or below chance: a ratio against chance-level discrimination is
/// noise, not robustness.
pub(crate) fn label_resiliency(
    task: TaskType,
    clean: &[Option<f64>],
    stressed: &[Option<f64>],
) -> Vec<Option<f64>> {
    match task {
        TaskType::Binary => {
            let (c, s) = (clean[0].unwrap_or(0.5), stressed[0].unwrap_or(0.5));
            vec![(c > 0.0).then(|| s / c)]
        }
        TaskType::Multiclass | TaskType::Multilabel => clean
            .iter()
            .zip(stressed)
            .map(|(c, s)| match (c, s) {
                (Some(c), Some(s)) if *c > 0.5 => Some(s / c),
                _ => None,
            })
            .collect(),
        // Regression resiliency is an RMSE ratio, computed in `regression`.
        TaskType::Regression => vec![None; clean.len()],
    }
}

/// The variance behind the headline `jitter_stability`, from each output
/// column's mean jitter variance.
///
/// Multilabel labels are separate binary decisions, so their variances are
/// averaged like every other per-label score. A multiclass probability vector
/// moves as one: mass leaving one class arrives in another. Its variance is
/// therefore *summed* over classes and halved, which is the binary variance
/// for two classes and does not change when a class the model never predicts
/// is added. Averaging over K instead would score the same movement as K/2
/// times more stable for every extra class.
pub(crate) fn headline_jitter_variance(task: TaskType, per_column: &[Option<f64>]) -> Option<f64> {
    match task {
        // Regression scores its one column in `regression`; it never gets here.
        TaskType::Binary | TaskType::Multilabel | TaskType::Regression => {
            mean_some(per_column.iter().copied())
        }
        TaskType::Multiclass => per_column
            .iter()
            .copied()
            .sum::<Option<f64>>()
            .map(|total| total / 2.0),
    }
}

/// Mean jitter variance of each output column across the draws.
pub(crate) fn jitter_variances(draws: &[Columns], k: usize) -> Result<Vec<Option<f64>>, CoreError> {
    (0..k)
        .map(|j| {
            let column: Vec<Series> = draws
                .iter()
                .map(|draw| Series::new("p".into(), draw[j].as_slice()))
                .collect();
            mean_jitter_variance(&column)
        })
        .collect()
}

/// Share of rows whose decision differs between any two jitter draws: the
/// argmax class for multiclass, the set of labels at or over threshold for
/// multilabel. `None` with fewer than two draws, and for binary, whose
/// output predates this field.
fn decision_flip_rate(draws: &[Columns], task: TaskType, thresholds: &[f64]) -> Option<f64> {
    let decide = match task {
        TaskType::Binary | TaskType::Regression => return None,
        TaskType::Multiclass => argmax_decision,
        TaskType::Multilabel => threshold_decision,
    };
    let (first, rest) = draws.split_first()?;
    let n_rows = first.first().map_or(0, Vec::len);
    if rest.is_empty() || n_rows == 0 {
        return None;
    }
    let flipped = (0..n_rows)
        .filter(|&i| {
            let baseline = decide(first, i, thresholds);
            rest.iter()
                .any(|draw| decide(draw, i, thresholds) != baseline)
        })
        .count();
    Some(flipped as f64 / n_rows as f64)
}

/// Row `i`'s decision as a one-hot over classes: the most probable one.
fn argmax_decision(draw: &[Vec<f64>], i: usize, _thresholds: &[f64]) -> Vec<bool> {
    let best = (0..draw.len())
        .max_by(|&a, &b| draw[a][i].total_cmp(&draw[b][i]))
        .unwrap_or(0);
    (0..draw.len()).map(|j| j == best).collect()
}

/// Row `i`'s decision as the set of labels at or over their threshold.
fn threshold_decision(draw: &[Vec<f64>], i: usize, thresholds: &[f64]) -> Vec<bool> {
    (0..draw.len())
        .map(|j| draw[j][i] >= thresholds[j])
        .collect()
}

/// Mean of the defined values; `None` when there are none.
pub(crate) fn mean_some(xs: impl IntoIterator<Item = Option<f64>>) -> Option<f64> {
    let (sum, n) = xs
        .into_iter()
        .flatten()
        .fold((0.0, 0usize), |(sum, n), x| (sum + x, n + 1));
    (n > 0).then(|| sum / n as f64)
}

/// Everything known per label once a dataset has been scored.
pub(crate) struct LabelBreakdown<'a> {
    pub names: &'a [String],
    pub y: &'a Columns,
    pub probs: &'a Columns,
    pub auc: &'a [Option<f64>],
    pub resiliency: &'a [Option<f64>],
    pub jitter_variance: &'a [Option<f64>],
    pub jitter_draws: &'a [Columns],
    pub thresholds: &'a [f64],
}

pub(crate) fn multi_output_scores(
    task: TaskType,
    labels: &LabelBreakdown,
    flipper: Option<&FlipperScores>,
) -> MultiOutputScores {
    MultiOutputScores {
        task,
        roc_auc_micro: roc_auc(&labels.y.concat(), &labels.probs.concat()),
        worst_label_resiliency: labels.resiliency.iter().flatten().copied().reduce(f64::min),
        decision_flip_rate: decision_flip_rate(labels.jitter_draws, task, labels.thresholds),
        flipper_never_flipped: flipper.map(|f| f.never_flipped),
        flipper_detour_rate: flipper.and_then(|f| f.detour_rate),
        excluded_labels: labels
            .names
            .iter()
            .zip(labels.auc)
            .filter(|(_, auc)| auc.is_none())
            .map(|(name, _)| name.clone())
            .collect(),
        per_label: labels
            .names
            .iter()
            .enumerate()
            .map(|(j, name)| LabelScores {
                label: name.clone(),
                roc_auc: labels.auc[j],
                resiliency: labels.resiliency[j],
                jitter_stability: jitter_stability_from_variance(labels.jitter_variance[j]),
                flipper_stability: flipper.and_then(|f| f.per_label[j]),
            })
            .collect(),
    }
}

/// Averages multi-output detail across datasets the same way the headline
/// numbers are averaged: field by field, over the datasets where defined.
pub(crate) fn average_multi(scores: Vec<MultiOutputScores>) -> Option<MultiOutputScores> {
    if scores.len() <= 1 {
        return scores.into_iter().next();
    }
    let avg = |f: &dyn Fn(&MultiOutputScores) -> Option<f64>| mean_some(scores.iter().map(f));
    let first = &scores[0];
    let per_label = first
        .per_label
        .iter()
        .enumerate()
        .map(|(j, label)| {
            let mean = |f: &dyn Fn(&LabelScores) -> Option<f64>| {
                mean_some(scores.iter().map(|s| s.per_label.get(j).and_then(f)))
            };
            LabelScores {
                label: label.label.clone(),
                roc_auc: mean(&|l| l.roc_auc),
                resiliency: mean(&|l| l.resiliency),
                jitter_stability: mean(&|l| Some(l.jitter_stability)).unwrap_or(1.0),
                flipper_stability: mean(&|l| l.flipper_stability),
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn multiclass_jitter_does_not_depend_on_the_number_of_classes() {
        let v = |xs: &[f64]| xs.iter().copied().map(Some).collect::<Vec<_>>();
        // 2-class softmax: both columns move together, so the headline equals
        // the binary positive-class variance.
        assert_eq!(
            headline_jitter_variance(TaskType::Multiclass, &v(&[0.01, 0.01])),
            Some(0.01)
        );
        // The same movement with never-predicted classes added scores the same.
        let three = headline_jitter_variance(TaskType::Multiclass, &v(&[0.01, 0.01, 0.0]));
        let six =
            headline_jitter_variance(TaskType::Multiclass, &v(&[0.01, 0.01, 0.0, 0.0, 0.0, 0.0]));
        assert_eq!(three, Some(0.01));
        assert_eq!(six, Some(0.01));
        // Multilabel labels are independent decisions: averaged.
        assert_eq!(
            headline_jitter_variance(TaskType::Multilabel, &v(&[0.02, 0.0])),
            Some(0.01)
        );
        // Fewer than two draws: nothing to measure.
        assert_eq!(
            headline_jitter_variance(TaskType::Multiclass, &[None, None]),
            None
        );
    }
}
