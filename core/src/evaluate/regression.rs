//! Scoring for regression models: one predicted value per patient, compared
//! against a continuous target. The headline fields keep their classifier
//! meaning where one carries over (resiliency is "how much of the model's
//! skill is kept under blackout"), and the `regression` section reports the
//! same numbers in the target's units.

use super::flipper::{score_flipper, FlipperScores};
use super::metrics::series_to_f64;
use super::multi::mean_some;
use super::{predict_matrix, predict_named, Model};
use crate::error::CoreError;
use crate::schema::{RegressionScores, SimulationVariant, TaskType};
use polars::prelude::*;
use std::collections::HashMap;

/// The keys of `baselines` for a regression model, in report order.
pub(crate) const BASELINE_KEYS: [&str; 3] = ["rmse", "mae", "r2"];

/// Everything one dataset contributes to a regression evaluation.
pub(crate) struct RegressionDataset {
    /// `(rmse, mae, r2)` on the clean cohort.
    pub baselines: [f64; 3],
    pub resiliency: Option<f64>,
    /// Mean per-patient prediction variance across jitter draws, divided by
    /// the target's variance.
    pub jitter_variance: Option<f64>,
    pub flipper: Option<FlipperScores>,
    pub detail: RegressionScores,
}

pub(crate) fn score_dataset(
    model: &dyn Model,
    clean: &SimulationVariant,
    variants: &HashMap<String, SimulationVariant>,
    flip_threshold: f64,
) -> Result<RegressionDataset, CoreError> {
    let (target_name, y) = target_vector(&clean.y)?;
    let target_variance = variance(&y);
    if target_variance == 0.0 {
        return Err(CoreError::InvalidConfig(format!(
            "target {target_name:?} has the same value for every clean row; a constant target \
             cannot tell a good regression model from a bad one"
        )));
    }
    let (_, mut clean_pred) = predict_named(model, &clean.x, 1)?;
    let clean_pred = clean_pred.swap_remove(0);
    let clean_rmse = rmse(&y, &clean_pred);
    let clean_r2 = r2(&y, &clean_pred, target_variance);

    let blackout_rmse = variants
        .get("blackout")
        .map(|blackout| -> Result<f64, CoreError> {
            Ok(rmse(&y, &predict_column(model, &blackout.x)?))
        })
        .transpose()?;
    let blackout_r2 = blackout_rmse.map(|stressed| 1.0 - stressed.powi(2) / target_variance);

    let jitter_draws: Vec<Vec<f64>> = variants
        .iter()
        .filter(|(name, _)| name.starts_with("jitter"))
        .map(|(_, variant)| predict_column(model, &variant.x))
        .collect::<Result<_, _>>()?;
    let jitter = jitter_spread(&jitter_draws);

    let flipper = variants
        .get("flipper")
        .map(|flipper| {
            score_flipper(
                model,
                &flipper.x,
                TaskType::Regression,
                1,
                &[flip_threshold],
            )
        })
        .transpose()?;

    Ok(RegressionDataset {
        baselines: [clean_rmse, mae(&y, &clean_pred), clean_r2],
        resiliency: blackout_r2.and_then(|stressed| r2_resiliency(clean_r2, stressed)),
        jitter_variance: jitter.map(|j| j.variance / target_variance),
        detail: RegressionScores {
            target: target_name,
            n_rows: y.len(),
            target_sd: target_variance.sqrt(),
            blackout_rmse,
            blackout_r2,
            jitter_prediction_sd: jitter.map(|j| j.sd),
            flip_threshold: flipper.as_ref().map(|_| flip_threshold),
            flipper_never_flipped: flipper.as_ref().map(|f| f.never_flipped),
        },
        flipper,
    })
}

/// The single target column, which must be finite on every row: a missing
/// target has no error to measure, and dropping the row silently would score
/// the model on a different cohort than the one asked for.
pub(crate) fn target_vector(y: &DataFrame) -> Result<(String, Vec<f64>), CoreError> {
    let [column] = y.columns() else {
        return Err(CoreError::InvalidConfig(format!(
            "regression evaluation takes one target column, got {}",
            y.width()
        )));
    };
    let values = series_to_f64(column.as_materialized_series())?;
    let missing = values.iter().filter(|v| !v.is_finite()).count();
    if missing > 0 {
        return Err(CoreError::InvalidConfig(format!(
            "target {:?} is missing or non-finite on {missing} of {} rows",
            column.name(),
            values.len()
        )));
    }
    Ok((column.name().to_string(), values))
}

fn predict_column(model: &dyn Model, x: &DataFrame) -> Result<Vec<f64>, CoreError> {
    Ok(predict_matrix(model, x, 1)?.swap_remove(0))
}

/// Root-mean-square error. A non-finite prediction makes it NaN rather than
/// being skipped: the model failed on that patient, and that is an error too.
pub(crate) fn rmse(y: &[f64], pred: &[f64]) -> f64 {
    mean(y.iter().zip(pred).map(|(y, p)| (p - y).powi(2))).sqrt()
}

pub(crate) fn mae(y: &[f64], pred: &[f64]) -> f64 {
    mean(y.iter().zip(pred).map(|(y, p)| (p - y).abs()))
}

/// Coefficient of determination, `1 − SSE / SST`. 0 means no better than
/// predicting the mean; it is negative for a model worse than that.
pub(crate) fn r2(y: &[f64], pred: &[f64], target_variance: f64) -> f64 {
    let mse = mean(y.iter().zip(pred).map(|(y, p)| (p - y).powi(2)));
    1.0 - mse / target_variance
}

/// Blackout R² ÷ clean R²: the share of the model's skill over predicting
/// the cohort mean that survives blackout. 1.0 means blackout cost nothing,
/// 0 means the model fell back to no better than the mean, and it is
/// negative when blackout made the model worse than the mean. It exceeds 1.0
/// when blackout happens to help.
///
/// A ratio of errors (clean RMSE ÷ blackout RMSE) would reward a model for
/// having little accuracy to lose: its floor is `sqrt(1 − R²)`, so a weak
/// model that collapses below the mean can outscore a strong one that keeps
/// most of its skill.
///
/// `None` when the clean R² is not positive: a model with no skill has none
/// to keep, just as a classifier label at chance has no resiliency.
pub(crate) fn r2_resiliency(clean: f64, stressed: f64) -> Option<f64> {
    (clean > 0.0).then(|| stressed / clean)
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct JitterSpread {
    /// Mean per-patient variance of the prediction across draws.
    pub variance: f64,
    /// Mean per-patient standard deviation, in target units.
    pub sd: f64,
}

/// Spread of each patient's prediction across jitter draws, averaged over
/// patients. `None` with fewer than two draws.
pub(crate) fn jitter_spread(draws: &[Vec<f64>]) -> Option<JitterSpread> {
    let n_rows = draws.first()?.len();
    if draws.len() < 2 || n_rows == 0 {
        return None;
    }
    let per_row: Vec<f64> = (0..n_rows)
        .map(|i| variance(&draws.iter().map(|d| d[i]).collect::<Vec<_>>()))
        .collect();
    Some(JitterSpread {
        variance: mean(per_row.iter().copied()),
        sd: mean(per_row.iter().map(|v| v.sqrt())),
    })
}

/// Population variance.
fn variance(xs: &[f64]) -> f64 {
    let m = mean(xs.iter().copied());
    mean(xs.iter().map(|x| (x - m).powi(2)))
}

fn mean(xs: impl IntoIterator<Item = f64>) -> f64 {
    let (sum, n) = xs
        .into_iter()
        .fold((0.0, 0usize), |(sum, n), x| (sum + x, n + 1));
    if n == 0 {
        f64::NAN
    } else {
        sum / n as f64
    }
}

/// Averages regression detail across datasets field by field, as the
/// headline numbers are averaged.
pub(crate) fn average_regression(scores: Vec<RegressionScores>) -> Option<RegressionScores> {
    if scores.len() <= 1 {
        return scores.into_iter().next();
    }
    let avg = |f: &dyn Fn(&RegressionScores) -> Option<f64>| mean_some(scores.iter().map(f));
    let first = &scores[0];
    Some(RegressionScores {
        target: first.target.clone(),
        n_rows: scores.iter().map(|s| s.n_rows).sum(),
        target_sd: avg(&|s| Some(s.target_sd)).unwrap_or(0.0),
        blackout_rmse: avg(&|s| s.blackout_rmse),
        blackout_r2: avg(&|s| s.blackout_r2),
        jitter_prediction_sd: avg(&|s| s.jitter_prediction_sd),
        flip_threshold: first.flip_threshold,
        flipper_never_flipped: avg(&|s| s.flipper_never_flipped),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-12
    }

    #[test]
    fn error_metrics_match_hand_computation() {
        let y = [1.0, 2.0, 3.0, 4.0];
        let p = [1.5, 2.0, 2.0, 5.0];
        // Errors 0.5, 0, -1, 1: squared 0.25, 0, 1, 1 → MSE 0.5625.
        assert!(close(rmse(&y, &p), 0.5625f64.sqrt()));
        assert!(close(mae(&y, &p), 0.625));
        // Var(y) = 1.25, so R² = 1 − 0.5625 / 1.25.
        assert!(close(r2(&y, &p, variance(&y)), 0.55));
    }

    #[test]
    fn resiliency_is_the_share_of_r2_kept() {
        assert!(close(r2_resiliency(0.8, 0.4).unwrap(), 0.5));
        assert!(close(r2_resiliency(0.4, 0.8).unwrap(), 2.0));
        // Worse than predicting the mean under blackout: negative.
        assert!(r2_resiliency(0.3, -0.1).unwrap() < 0.0);
        // No skill to keep.
        assert_eq!(r2_resiliency(0.0, -0.5), None);
        assert_eq!(r2_resiliency(-0.2, -0.2), None);
    }

    #[test]
    fn a_weak_model_that_collapses_does_not_outscore_a_strong_one() {
        // Target variance 1. Strong: R² 0.66 → 0.27. Weak: R² 0.26 → −0.09.
        // The RMSE ratio ranks the weak model as more resilient (0.82 vs
        // 0.69); the R² ratio does not.
        let rmse_of = |r2: f64| (1.0 - r2).sqrt();
        let strong = (0.66, 0.27);
        let weak = (0.26, -0.09);
        assert!(rmse_of(weak.0) / rmse_of(weak.1) > rmse_of(strong.0) / rmse_of(strong.1));
        assert!(
            r2_resiliency(weak.0, weak.1).unwrap() < r2_resiliency(strong.0, strong.1).unwrap()
        );
    }

    #[test]
    fn jitter_spread_needs_two_draws() {
        assert_eq!(jitter_spread(&[vec![1.0, 2.0]]), None);
        // Row 0 moves by ±1 (variance 1, sd 1); row 1 does not move.
        let spread = jitter_spread(&[vec![0.0, 5.0], vec![2.0, 5.0]]).unwrap();
        assert!(close(spread.variance, 0.5));
        assert!(close(spread.sd, 0.5));
    }

    #[test]
    fn target_must_be_one_finite_column() {
        let two = df!["a" => [1.0], "b" => [2.0]].unwrap();
        assert!(target_vector(&two).is_err());
        let nan = df!["a" => [1.0, f64::NAN]].unwrap();
        let err = target_vector(&nan).unwrap_err().to_string();
        assert!(err.contains("1 of 2 rows"), "{err}");
        let null = df!["a" => [Some(1.0), None]].unwrap();
        assert!(target_vector(&null).is_err());
    }
}
