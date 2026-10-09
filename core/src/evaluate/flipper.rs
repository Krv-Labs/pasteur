use super::metrics::series_to_f64;
use super::{predict_matrix, Model};
use crate::error::CoreError;
use crate::schema::TaskType;
use crate::simulate::flipper::{
    is_flipper_meta_column, LABEL_A_COL, LABEL_B_COL, PAIR_ID_COL, PAIR_LABEL_COL,
    SOURCE_A_ROW_COL, SOURCE_B_ROW_COL, STEP_COL, T_COL,
};
use polars::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// One sampled pair's outcome: does the model's prediction cross
/// `flip_threshold` somewhere along the interpolation from `source_a_row`
/// to `source_b_row`, and if so, where.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlipperPairResult {
    pub pair_id: u32,
    pub source_a_row: u32,
    pub source_b_row: u32,
    pub label_a: f64,
    pub label_b: f64,
    /// Interpolation fraction where the model's predicted class first
    /// crosses `flip_threshold`, linearly refined between the two
    /// bracketing grid steps. `None` if the prediction never crosses the
    /// threshold across the sampled steps for this pair.
    pub flip_t: Option<f64>,
}

/// `grid` must be a `FlipperSimulator::generate` output (has `pair_id`/
/// `step`/`t`/`source_a_row`/`source_b_row`/`label_a`/`label_b` columns);
/// `predictions` is the model's positive-class probability for each row of
/// `grid`, in the same order. Returns one result per distinct `pair_id`,
/// sorted by `pair_id`.
pub fn find_flip_points(
    grid: &DataFrame,
    predictions: &Series,
    flip_threshold: f64,
) -> Result<Vec<FlipperPairResult>, CoreError> {
    let pair_id = grid.column(PAIR_ID_COL)?.u32()?;
    let step = grid.column(STEP_COL)?.u32()?;
    let t = grid.column(T_COL)?.f64()?;
    let source_a_row = grid.column(SOURCE_A_ROW_COL)?.u32()?;
    let source_b_row = grid.column(SOURCE_B_ROW_COL)?.u32()?;
    let label_a = grid.column(LABEL_A_COL)?.f64()?;
    let label_b = grid.column(LABEL_B_COL)?.f64()?;
    let preds = predictions.cast(&DataType::Float64)?;
    let preds = preds.f64()?;

    struct PairMeta {
        source_a_row: u32,
        source_b_row: u32,
        label_a: f64,
        label_b: f64,
        steps: Vec<(u32, f64, f64)>, // (step, t, prediction)
    }

    let mut by_pair: HashMap<u32, PairMeta> = HashMap::new();
    for i in 0..grid.height() {
        let pid = pair_id.get(i).unwrap_or(0);
        by_pair
            .entry(pid)
            .or_insert_with(|| PairMeta {
                source_a_row: source_a_row.get(i).unwrap_or(0),
                source_b_row: source_b_row.get(i).unwrap_or(0),
                label_a: label_a.get(i).unwrap_or(f64::NAN),
                label_b: label_b.get(i).unwrap_or(f64::NAN),
                steps: Vec::new(),
            })
            .steps
            .push((
                step.get(i).unwrap_or(0),
                t.get(i).unwrap_or(0.0),
                preds.get(i).unwrap_or(f64::NAN),
            ));
    }

    let mut pair_ids: Vec<u32> = by_pair.keys().copied().collect();
    pair_ids.sort_unstable();

    let mut results = Vec::with_capacity(pair_ids.len());
    for pid in pair_ids {
        let mut meta = by_pair.remove(&pid).unwrap();
        meta.steps.sort_by_key(|(s, _, _)| *s);

        let mut flip_t = None;
        for window in meta.steps.windows(2) {
            let (_, t0, p0) = window[0];
            let (_, t1, p1) = window[1];
            let side0 = p0 >= flip_threshold;
            let side1 = p1 >= flip_threshold;
            if side0 != side1 {
                // Linear refinement of the threshold crossing between the
                // two bracketing grid steps ("zeroing in" on the flip area
                // without extra model calls).
                let frac = if (p1 - p0).abs() > f64::EPSILON {
                    ((flip_threshold - p0) / (p1 - p0)).clamp(0.0, 1.0)
                } else {
                    0.5
                };
                flip_t = Some(t0 + frac * (t1 - t0));
                break;
            }
        }

        results.push(FlipperPairResult {
            pair_id: pid,
            source_a_row: meta.source_a_row,
            source_b_row: meta.source_b_row,
            label_a: meta.label_a,
            label_b: meta.label_b,
            flip_t,
        });
    }
    Ok(results)
}

/// Feature columns from a persisted flipper grid (drops flipper meta columns).
pub fn flipper_model_features(grid: &DataFrame) -> Result<DataFrame, CoreError> {
    let names: Vec<String> = grid
        .get_column_names()
        .iter()
        .map(|s| s.to_string())
        .filter(|c| !is_flipper_meta_column(c))
        .collect();
    if names.is_empty() {
        return Err(CoreError::InvalidConfig(
            "flipper grid has no feature columns".to_string(),
        ));
    }
    Ok(grid.select(names)?)
}

/// Run model inference on a flipper grid and return flipper stability, for a
/// binary model. See `score_flipper` for multi-output models.
pub fn score_flipper_grid(
    model: &dyn Model,
    grid: &DataFrame,
    flip_threshold: f64,
) -> Result<f64, CoreError> {
    Ok(score_flipper(model, grid, TaskType::Binary, 1, &[flip_threshold])?.stability)
}

#[derive(Debug, Clone)]
pub struct FlipperScores {
    pub stability: f64,
    /// Share of pairs that never crossed (and so counted as 1.0 above).
    pub never_flipped: f64,
    /// Per output column: stability over the pairs about that label (the
    /// pair's `pair_label` for multilabel, either endpoint's class for
    /// multiclass). `None` when no pair involved it.
    pub per_label: Vec<Option<f64>>,
    /// Multiclass only: share of pairs whose argmax visits a third class.
    pub detour_rate: Option<f64>,
}

/// Flipper stability for any task, reduced to the binary flip finder by
/// turning each grid row into a score that crosses 0 where the decision
/// changes:
///
/// - binary: `p − flip_threshold`, which is the original computation;
/// - multiclass: the pairwise margin `p[class_b] − p[class_a]`, since the
///   decision is an argmax and no single threshold applies;
/// - multilabel: `p[k] − threshold[k]` for the label `k` the pair was sampled
///   for, read from the grid's `pair_label` column;
/// - regression: `prediction − cutoff`, where the cutoff is `thresholds[0]`
///   in the target's units. Every pair must have one endpoint below the
///   cutoff and one at or above it.
///
/// `thresholds` holds one entry per output column for binary, multilabel and
/// regression, and is ignored for multiclass.
pub fn score_flipper(
    model: &dyn Model,
    grid: &DataFrame,
    task: TaskType,
    n_outputs: usize,
    thresholds: &[f64],
) -> Result<FlipperScores, CoreError> {
    let has_pair_label = grid.get_column_index(PAIR_LABEL_COL).is_some();
    if has_pair_label != (task == TaskType::Multilabel) {
        return Err(CoreError::InvalidConfig(format!(
            "flipper grid was generated for {} pairs but this is a {task} evaluation; \
             regenerate it with `simulate --task {task}` and the same label groups",
            if has_pair_label {
                "multilabel"
            } else {
                "binary/multiclass"
            }
        )));
    }
    let features = flipper_model_features(grid)?;
    let mut probs = predict_matrix(model, &features, n_outputs)?;

    // Each pair's labels and, per grid row, a score that crosses 0 exactly
    // where the decision changes.
    let (pair_labels, crossing): (Vec<Vec<usize>>, Vec<f64>) = match task {
        TaskType::Binary | TaskType::Regression => {
            if task == TaskType::Regression {
                check_pairs_straddle(grid, thresholds[0])?;
            }
            let preds = Series::new("pred".into(), probs.swap_remove(0));
            let results = find_flip_points(grid, &preds, thresholds[0])?;
            let stability = calculate_flipper_stability(&results);
            return Ok(FlipperScores {
                stability,
                never_flipped: never_flipped(&results),
                per_label: vec![Some(stability)],
                detour_rate: None,
            });
        }
        TaskType::Multiclass => {
            let class_a = class_indices(grid, LABEL_A_COL, n_outputs)?;
            let class_b = class_indices(grid, LABEL_B_COL, n_outputs)?;
            class_a
                .into_iter()
                .zip(class_b)
                .enumerate()
                .map(|(row, (a, b))| (vec![a, b], probs[b][row] - probs[a][row]))
                .unzip()
        }
        TaskType::Multilabel => class_indices(grid, PAIR_LABEL_COL, n_outputs)?
            .into_iter()
            .enumerate()
            .map(|(row, k)| (vec![k], probs[k][row] - thresholds[k]))
            .unzip(),
    };
    let results = find_flip_points(grid, &Series::new("crossing".into(), crossing), 0.0)?;

    // Every row of a pair carries the same labels; take the first.
    let pair_id = grid.column(PAIR_ID_COL)?.u32()?;
    let mut labels_of_pair: HashMap<u32, &[usize]> = HashMap::new();
    for (i, labels) in pair_labels.iter().enumerate() {
        labels_of_pair
            .entry(pair_id.get(i).unwrap_or(0))
            .or_insert(labels);
    }
    let per_label = (0..n_outputs)
        .map(|k| {
            let about_k: Vec<FlipperPairResult> = results
                .iter()
                .filter(|r| labels_of_pair[&r.pair_id].contains(&k))
                .cloned()
                .collect();
            (!about_k.is_empty()).then(|| calculate_flipper_stability(&about_k))
        })
        .collect();

    let detour_rate = (task == TaskType::Multiclass)
        .then(|| detour_rate(&probs, pair_id, &pair_labels, results.len()));

    Ok(FlipperScores {
        stability: calculate_flipper_stability(&results),
        never_flipped: never_flipped(&results),
        per_label,
        detour_rate,
    })
}

/// A regression grid pairs a patient below the cutoff with one at or above
/// it. A pair that does not straddle `cutoff` came from a grid generated for
/// another cutoff or another task, and its crossing would mean nothing.
fn check_pairs_straddle(grid: &DataFrame, cutoff: f64) -> Result<(), CoreError> {
    let label_a = series_to_f64(grid.column(LABEL_A_COL)?.as_materialized_series())?;
    let label_b = series_to_f64(grid.column(LABEL_B_COL)?.as_materialized_series())?;
    let bad = label_a
        .iter()
        .zip(&label_b)
        .filter(|(a, b)| !(**a < cutoff && **b >= cutoff))
        .count();
    if bad == 0 {
        return Ok(());
    }
    Err(CoreError::InvalidConfig(format!(
        "{bad} flipper grid row(s) do not pair a target below {cutoff} with one at or above \
         it; the grid was generated for another cutoff or task. Regenerate it with \
         `simulate --task regression --flip-threshold {cutoff}`"
    )))
}

fn never_flipped(results: &[FlipperPairResult]) -> f64 {
    if results.is_empty() {
        return 0.0;
    }
    results.iter().filter(|r| r.flip_t.is_none()).count() as f64 / results.len() as f64
}

/// Share of pairs where, at some step, the argmax class is neither endpoint's
/// class — the straight path from class a to class b runs through class c.
fn detour_rate(
    probs: &[Vec<f64>],
    pair_id: &UInt32Chunked,
    pair_labels: &[Vec<usize>],
    n_pairs: usize,
) -> f64 {
    if n_pairs == 0 {
        return 0.0;
    }
    let mut detoured = std::collections::HashSet::new();
    for (i, ends) in pair_labels.iter().enumerate() {
        let argmax = (0..probs.len())
            .max_by(|&a, &b| probs[a][i].total_cmp(&probs[b][i]))
            .unwrap_or(0);
        if !ends.contains(&argmax) {
            detoured.insert(pair_id.get(i).unwrap_or(0));
        }
    }
    detoured.len() as f64 / n_pairs as f64
}

/// A grid column of class or label indices, each checked to be a whole
/// number below `n_outputs`.
fn class_indices(grid: &DataFrame, col: &str, n_outputs: usize) -> Result<Vec<usize>, CoreError> {
    series_to_f64(grid.column(col)?.as_materialized_series())?
        .into_iter()
        .map(|v| {
            if v.fract() == 0.0 && v >= 0.0 && (v as usize) < n_outputs {
                Ok(v as usize)
            } else {
                Err(CoreError::InvalidConfig(format!(
                    "flipper grid `{col}` holds {v}, not a class index below {n_outputs}; \
                     was the grid generated with the same --task and label groups?"
                )))
            }
        })
        .collect()
}

/// Mean flip fraction across all pairs — pairs that never flip are treated
/// as maximally stable (`t=1.0`, i.e. the model never changed its mind
/// across the entire sampled path). Already normalized to `t` in `[0,1]` by
/// construction, unlike the Python reference's `mean(flip_std)/max_std`
/// (there's no `max_std` here — the interpolation is between two real rows,
/// not steps along a fixed axis).
pub fn calculate_flipper_stability(results: &[FlipperPairResult]) -> f64 {
    if results.is_empty() {
        return 1.0;
    }
    let sum: f64 = results.iter().map(|r| r.flip_t.unwrap_or(1.0)).sum();
    sum / results.len() as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    fn grid_for_one_pair(preds: &[f64]) -> (DataFrame, Series) {
        let n = preds.len();
        let t: Vec<f64> = (0..n).map(|i| i as f64 / (n - 1) as f64).collect();
        let columns: Vec<Column> = vec![
            Series::new(PAIR_ID_COL.into(), vec![0u32; n]).into(),
            Series::new(STEP_COL.into(), (0..n as u32).collect::<Vec<_>>()).into(),
            Series::new(T_COL.into(), t).into(),
            Series::new(SOURCE_A_ROW_COL.into(), vec![1u32; n]).into(),
            Series::new(SOURCE_B_ROW_COL.into(), vec![7u32; n]).into(),
            Series::new(LABEL_A_COL.into(), vec![0.0f64; n]).into(),
            Series::new(LABEL_B_COL.into(), vec![1.0f64; n]).into(),
        ];
        let df = DataFrame::new(n, columns).unwrap();
        (df, Series::new("pred".into(), preds))
    }

    #[test]
    fn finds_flip_between_bracketing_steps() {
        let (grid, preds) = grid_for_one_pair(&[0.1, 0.2, 0.4, 0.6, 0.9]);
        let results = find_flip_points(&grid, &preds, 0.5).unwrap();
        assert_eq!(results.len(), 1);
        let r = &results[0];
        assert_eq!(r.pair_id, 0);
        assert_eq!(r.source_a_row, 1);
        assert_eq!(r.source_b_row, 7);
        let flip_t = r.flip_t.expect("expected a flip");
        assert!((flip_t - 0.625).abs() < 1e-9, "got {flip_t}");
    }

    #[test]
    fn no_flip_when_never_crossing_threshold() {
        let (grid, preds) = grid_for_one_pair(&[0.1, 0.2, 0.3, 0.35, 0.4]);
        let results = find_flip_points(&grid, &preds, 0.5).unwrap();
        assert_eq!(results.len(), 1);
        assert!(results[0].flip_t.is_none());
    }

    #[test]
    fn stability_is_perfect_when_no_pair_ever_flips() {
        let (grid, preds) = grid_for_one_pair(&[0.1, 0.2, 0.3]);
        let results = find_flip_points(&grid, &preds, 0.9).unwrap();
        assert_eq!(calculate_flipper_stability(&results), 1.0);
    }

    #[test]
    fn stability_drops_when_a_pair_flips() {
        let (grid, preds) = grid_for_one_pair(&[0.3, 0.6, 0.9]);
        let results = find_flip_points(&grid, &preds, 0.5).unwrap();
        let stability = calculate_flipper_stability(&results);
        assert!(stability < 1.0, "expected < 1.0, got {stability}");
    }
}
