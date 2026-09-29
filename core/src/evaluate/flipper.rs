use super::Model;
use crate::error::CoreError;
use crate::simulate::flipper::{
    is_flipper_meta_column, LABEL_A_COL, LABEL_B_COL, PAIR_ID_COL, SOURCE_A_ROW_COL,
    SOURCE_B_ROW_COL, STEP_COL, T_COL,
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

/// Run model inference on a flipper grid and return flipper stability.
pub fn score_flipper_grid(
    model: &dyn Model,
    grid: &DataFrame,
    flip_threshold: f64,
) -> Result<f64, CoreError> {
    let features = flipper_model_features(grid)?;
    let predictions = model.predict_proba(&features)?;
    let results = find_flip_points(grid, &predictions, flip_threshold)?;
    Ok(calculate_flipper_stability(&results))
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
