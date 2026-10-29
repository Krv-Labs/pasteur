//! Metrics for `SweepSimulator` grids. A *path* is one source row swept over
//! one feature: the grid rows sharing `(sweep_case_row, sweep_feature)`, in
//! `sweep_step` order. A *flip* is a step where the decision differs from the
//! step before; its *flip value* is the midpoint of the two bracketing grid
//! values.
//!
//! With oracle decisions (the correct decision for every grid row), the
//! metrics measure whether the model's boundary sits where the true one does.
//! Without them, only the descriptive ones are filled: the model's own flip
//! value, `never_flipped` over all paths, and `extra_flips_rate`.

use super::multi::{argmax_decision, label_matrix, threshold_decision};
use super::{predict_matrix, Model};
use crate::error::CoreError;
use crate::schema::TaskType;
use crate::simulate::sweep::{
    SWEEP_CASE_ROW_COL, SWEEP_CUTOFF_COL, SWEEP_FEATURE_COL, SWEEP_STEP_COL, SWEEP_VALUE_COL,
};
use polars::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Median and interquartile range of a sample.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Spread {
    pub median: f64,
    pub q25: f64,
    pub q75: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StepAccuracy {
    pub step: u32,
    pub value: f64,
    pub accuracy: f64,
}

/// One swept feature's scores. Rates are shares of paths.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SweepScores {
    pub feature: String,
    pub cutoff: Option<f64>,
    pub n_paths: usize,
    /// Whether the oracle fields below were scored against oracle labels.
    pub oracle: bool,
    /// Oracle only: `(model flip value − oracle flip value) / |cutoff|` over
    /// monotonic paths. Positive means the model flips later on the grid.
    /// `None` without a cutoff or without monotonic paths.
    pub boundary_offset: Option<Spread>,
    /// Oracle only: the same offset in the feature's units.
    pub boundary_offset_units: Option<Spread>,
    /// Oracle only: paths where the model and the oracle each flip exactly
    /// once, to the same decision.
    pub monotonic_rate: Option<f64>,
    /// With oracle labels: among paths where the oracle flips, the share
    /// where the model never does ("ignores this feature"); `None` when the
    /// oracle never flips. Without them: share of all paths where the
    /// model's decision never changes, so both endpoints agree.
    pub never_flipped: Option<f64>,
    /// Paths where the model flips more than once.
    pub extra_flips_rate: f64,
    /// The model's flip value, in feature units, over paths where it flips
    /// exactly once.
    pub model_flip_value: Option<Spread>,
    /// Oracle only: accuracy over every grid row of this feature.
    pub accuracy: Option<f64>,
    /// Oracle only: accuracy at each grid step, across paths.
    pub accuracy_by_step: Option<Vec<StepAccuracy>>,
}

/// Predicts on a sweep grid and scores it. `thresholds` holds one decision
/// threshold per output (ignored for multiclass, whose decision is the
/// argmax). `oracle_y` is the correct label matrix for every grid row, in
/// grid order, when the caller has one.
pub fn score_sweep(
    model: &dyn Model,
    grid: &DataFrame,
    task: TaskType,
    n_outputs: usize,
    thresholds: &[f64],
    oracle_y: Option<&DataFrame>,
) -> Result<Vec<SweepScores>, CoreError> {
    let probs = predict_matrix(model, grid, n_outputs)?;
    let decide = match task {
        TaskType::Multiclass => argmax_decision,
        _ => threshold_decision,
    };
    let decisions: Vec<Vec<bool>> = (0..grid.height())
        .map(|i| decide(&probs, i, thresholds))
        .collect();
    let oracle = oracle_y
        .map(|y| {
            // 0/1 labels: over 0.5 is the label set, and for multiclass the
            // one-hot row is its own argmax.
            let y = label_matrix(y, task)?;
            let half = vec![0.5; y.len()];
            Ok::<_, CoreError>(
                (0..grid.height())
                    .map(|i| threshold_decision(&y, i, &half))
                    .collect::<Vec<_>>(),
            )
        })
        .transpose()?;
    sweep_scores(grid, &decisions, oracle.as_deref())
}

/// The pure scoring step: `decisions[i]` and `oracle[i]` are grid row `i`'s
/// decisions (any comparable encoding). One entry per feature, sorted by name.
pub fn sweep_scores<D: PartialEq>(
    grid: &DataFrame,
    decisions: &[D],
    oracle: Option<&[D]>,
) -> Result<Vec<SweepScores>, CoreError> {
    let n = grid.height();
    if decisions.len() != n || oracle.is_some_and(|o| o.len() != n) {
        return Err(CoreError::InvalidConfig(format!(
            "sweep grid has {n} rows but got {} decisions and {} oracle decisions",
            decisions.len(),
            oracle.map_or(0, <[D]>::len)
        )));
    }
    let case = grid.column(SWEEP_CASE_ROW_COL)?.u32()?;
    let feature = grid.column(SWEEP_FEATURE_COL)?.str()?;
    let step = grid.column(SWEEP_STEP_COL)?.u32()?;
    let value = grid.column(SWEEP_VALUE_COL)?.f64()?;
    let cutoff = grid.column(SWEEP_CUTOFF_COL)?.f64()?;

    // feature -> case -> [(step, row)]
    let mut paths: BTreeMap<&str, BTreeMap<u32, Vec<(u32, usize)>>> = BTreeMap::new();
    for i in 0..n {
        let (Some(f), Some(c), Some(s)) = (feature.get(i), case.get(i), step.get(i)) else {
            return Err(CoreError::InvalidConfig(format!(
                "sweep grid row {i} is missing its feature, case or step"
            )));
        };
        paths
            .entry(f)
            .or_default()
            .entry(c)
            .or_default()
            .push((s, i));
    }

    let mut out = Vec::with_capacity(paths.len());
    for (name, by_case) in paths {
        let first_row = by_case.values().next().and_then(|p| p.first()).map(|p| p.1);
        let cut = first_row.and_then(|i| cutoff.get(i));
        let mut acc = Tally::default();
        let mut by_step: BTreeMap<u32, (f64, usize, usize)> = BTreeMap::new();
        for mut path in by_case.into_values() {
            path.sort_unstable();
            let rows: Vec<usize> = path.iter().map(|p| p.1).collect();
            let vals: Vec<f64> = rows
                .iter()
                .map(|&i| value.get(i).unwrap_or(f64::NAN))
                .collect();
            let mid = |k: usize| (vals[k - 1] + vals[k]) / 2.0;
            let m: Vec<&D> = rows.iter().map(|&i| &decisions[i]).collect();
            let fm = flips(&m);
            acc.paths += 1;
            acc.extra += usize::from(fm.len() > 1);
            if let [k] = fm[..] {
                acc.model_flip.push(mid(k));
            }
            let Some(oracle) = oracle else {
                acc.never += usize::from(fm.is_empty());
                acc.never_of += 1;
                continue;
            };
            let o: Vec<&D> = rows.iter().map(|&i| &oracle[i]).collect();
            for ((&(s, _), v), (a, b)) in path.iter().zip(&vals).zip(m.iter().zip(&o)) {
                let e = by_step.entry(s).or_insert((*v, 0, 0));
                e.1 += usize::from(a == b);
                e.2 += 1;
            }
            let fo = flips(&o);
            if !fo.is_empty() {
                acc.never_of += 1;
                acc.never += usize::from(fm.is_empty());
            }
            if let ([km], [ko]) = (&fm[..], &fo[..]) {
                if m[*km] == o[*ko] {
                    acc.monotonic += 1;
                    acc.offsets.push(mid(*km) - mid(*ko));
                }
            }
        }

        let has_oracle = oracle.is_some();
        let rate = |k: usize, of: usize| (of > 0).then(|| k as f64 / of as f64);
        let (correct, total) = by_step
            .values()
            .fold((0, 0), |(c, t), &(_, sc, st)| (c + sc, t + st));
        out.push(SweepScores {
            feature: name.to_string(),
            cutoff: cut,
            n_paths: acc.paths,
            oracle: has_oracle,
            boundary_offset: cut
                .filter(|c| *c != 0.0)
                .and_then(|c| spread(acc.offsets.iter().map(|o| o / c.abs()).collect())),
            boundary_offset_units: spread(acc.offsets),
            monotonic_rate: has_oracle.then(|| acc.monotonic as f64 / acc.paths as f64),
            never_flipped: rate(acc.never, acc.never_of),
            extra_flips_rate: acc.extra as f64 / acc.paths as f64,
            model_flip_value: spread(acc.model_flip),
            accuracy: has_oracle.then(|| correct as f64 / total as f64),
            accuracy_by_step: has_oracle.then(|| {
                by_step
                    .into_iter()
                    .map(|(step, (value, c, t))| StepAccuracy {
                        step,
                        value,
                        accuracy: c as f64 / t as f64,
                    })
                    .collect()
            }),
        });
    }
    Ok(out)
}

#[derive(Default)]
struct Tally {
    paths: usize,
    monotonic: usize,
    never: usize,
    /// Denominator of `never`: paths where the oracle flips, or every path
    /// without an oracle.
    never_of: usize,
    extra: usize,
    offsets: Vec<f64>,
    model_flip: Vec<f64>,
}

/// Indices `k` where `xs[k] != xs[k - 1]`.
fn flips<D: PartialEq>(xs: &[D]) -> Vec<usize> {
    (1..xs.len()).filter(|&k| xs[k] != xs[k - 1]).collect()
}

/// Median (mean of the middle two for even counts) and quartiles taken as
/// `sorted[min(n − 1, ⌊q·n⌋)]`, matching the reference scripts.
fn spread(mut xs: Vec<f64>) -> Option<Spread> {
    xs.retain(|x| x.is_finite());
    if xs.is_empty() {
        return None;
    }
    xs.sort_by(f64::total_cmp);
    let n = xs.len();
    let q = |f: f64| xs[((f * n as f64) as usize).min(n - 1)];
    let median = if n % 2 == 1 {
        xs[n / 2]
    } else {
        (xs[n / 2 - 1] + xs[n / 2]) / 2.0
    };
    Some(Spread {
        median,
        q25: q(0.25),
        q75: q(0.75),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::{SweepConfig, SweepFeature};
    use crate::simulate::SweepSimulator;

    const CUTOFF: f64 = 10.0;

    /// Three cases, one feature `f`, 31 steps from 0.5× to 2× the cutoff:
    /// grid spacing is 0.05 of the cutoff.
    fn grid() -> DataFrame {
        let x = df!["f" => [1.0, 2.0, 3.0], "g" => [0.0, 1.0, 2.0]].unwrap();
        SweepSimulator::new(SweepConfig {
            features: vec![SweepFeature {
                name: "f".into(),
                cutoff: Some(CUTOFF),
            }],
            range: (0.5, 2.0),
            relative_to: Some(CUTOFF),
            n_steps: 31,
            near: None,
        })
        .generate(&x)
        .unwrap()
    }

    fn decide(grid: &DataFrame, at: f64) -> Vec<bool> {
        let v = grid.column(SWEEP_VALUE_COL).unwrap().f64().unwrap();
        v.into_no_null_iter().map(|v| v >= at).collect()
    }

    #[test]
    fn perfect_predictor_has_zero_offset_and_is_monotonic() {
        let g = grid();
        let oracle = decide(&g, CUTOFF);
        let s = &sweep_scores(&g, &oracle, Some(&oracle)).unwrap()[0];
        assert_eq!((s.feature.as_str(), s.n_paths), ("f", 3));
        assert_eq!(s.boundary_offset.as_ref().unwrap().median, 0.0);
        assert_eq!(s.monotonic_rate, Some(1.0));
        assert_eq!(s.never_flipped, Some(0.0));
        assert_eq!(s.extra_flips_rate, 0.0);
        assert_eq!(s.accuracy, Some(1.0));
        let steps = s.accuracy_by_step.as_ref().unwrap();
        assert_eq!(steps.len(), 31);
        assert!(steps.iter().all(|a| a.accuracy == 1.0));
    }

    #[test]
    fn constant_predictor_never_flips() {
        let g = grid();
        let oracle = decide(&g, CUTOFF);
        let constant = vec![false; g.height()];
        let s = &sweep_scores(&g, &constant, Some(&oracle)).unwrap()[0];
        assert_eq!(s.never_flipped, Some(1.0));
        assert_eq!(s.monotonic_rate, Some(0.0));
        assert!(s.boundary_offset.is_none());
        assert!(s.model_flip_value.is_none());
    }

    #[test]
    fn shifted_predictor_has_a_positive_offset() {
        let g = grid();
        let oracle = decide(&g, CUTOFF);
        let shifted = decide(&g, CUTOFF * 1.1);
        let s = &sweep_scores(&g, &shifted, Some(&oracle)).unwrap()[0];
        let offset = s.boundary_offset.as_ref().unwrap();
        assert!((offset.median - 0.1).abs() <= 0.05, "{offset:?}");
        assert!((offset.q25 - offset.q75).abs() < 1e-12);
        assert_eq!(s.monotonic_rate, Some(1.0));
        assert!(s.accuracy.unwrap() < 1.0);
    }

    #[test]
    fn without_an_oracle_only_descriptive_fields_are_filled() {
        let g = grid();
        let mut wobbly = decide(&g, CUTOFF);
        wobbly[0] = true; // case 0 starts high, drops, then rises again
        let s = &sweep_scores(&g, &wobbly, None).unwrap()[0];
        assert!(!s.oracle);
        assert!(s.monotonic_rate.is_none() && s.accuracy_by_step.is_none());
        assert!(s.boundary_offset.is_none());
        assert_eq!(s.never_flipped, Some(0.0));
        assert!((s.extra_flips_rate - 1.0 / 3.0).abs() < 1e-12);
        // Midpoint of 9.5 and 10.0 on the two single-flip paths.
        assert!((s.model_flip_value.as_ref().unwrap().median - 9.75).abs() < 1e-9);
    }

    #[test]
    fn rejects_mismatched_lengths() {
        let g = grid();
        assert!(sweep_scores(&g, &[true], None).is_err());
    }
}
