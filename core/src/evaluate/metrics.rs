use crate::error::CoreError;
use polars::prelude::*;

pub const JITTER_STABILITY_ALPHA: f64 = 100.0;

/// Mean per-row variance across N independent jitter draws' predictions,
/// squashed into (0, 1] — `1 / (1 + alpha * mean_variance)`. Needs at least
/// two jitter variants to have any variance to measure; with fewer, there's
/// nothing to compare and stability is reported as perfect (1.0).
pub fn calculate_jitter_stability(preds: &[Series], alpha: f64) -> Result<f64, CoreError> {
    Ok(match mean_jitter_variance(preds)? {
        Some(v) => 1.0 / (1.0 + alpha * v),
        None => 1.0,
    })
}

/// The `mean_variance` inside `calculate_jitter_stability`, exposed so that
/// multi-output models can average it over labels *before* squashing: the
/// mean over the two columns of a 2-class softmax equals the binary
/// positive-class variance, so multiclass scores stay on the binary scale.
/// `None` with fewer than two draws.
pub fn mean_jitter_variance(preds: &[Series]) -> Result<Option<f64>, CoreError> {
    if preds.len() < 2 {
        return Ok(None);
    }
    let cols: Vec<Float64Chunked> = preds
        .iter()
        .map(|s| s.cast(&DataType::Float64).map(|s| s.f64().unwrap().clone()))
        .collect::<Result<_, _>>()?;
    let n_rows = cols[0].len();
    let n = cols.len() as f64;

    let mut total_variance = 0.0;
    for i in 0..n_rows {
        let vals: Vec<f64> = cols.iter().filter_map(|c| c.get(i)).collect();
        if vals.is_empty() {
            continue;
        }
        let mean = vals.iter().sum::<f64>() / n;
        let var = vals.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;
        total_variance += var;
    }
    Ok(Some(total_variance / n_rows as f64))
}

/// ROC AUC with a 0.5 fallback when the labels hold only one class, which
/// `baselines.roc_auc` documents for binary models. Multi-output code calls
/// `roc_auc` instead so that undefined labels can be excluded rather than
/// averaged in as chance.
pub fn calculate_roc_auc(y_true: &Series, y_scores: &Series) -> Result<f64, CoreError> {
    Ok(roc_auc(&series_to_f64(y_true)?, &series_to_f64(y_scores)?).unwrap_or(0.5))
}

/// ROC AUC as the Mann–Whitney statistic: the probability that a random
/// positive (label `1.0`) outscores a random negative, with ties counting
/// half. Tied scores get their average rank, so the result does not depend on
/// the order a sort happens to leave them in — trees and softmax heads emit
/// many exact ties. Rows with a non-finite score or label are skipped.
/// `None` when either class is absent.
pub fn roc_auc(y_true: &[f64], y_scores: &[f64]) -> Option<f64> {
    let mut rows: Vec<(f64, bool)> = y_true
        .iter()
        .zip(y_scores)
        .filter(|(y, s)| y.is_finite() && s.is_finite())
        .map(|(y, s)| (*s, *y == 1.0))
        .collect();
    let n_pos = rows.iter().filter(|(_, pos)| *pos).count() as f64;
    let n_neg = rows.len() as f64 - n_pos;
    if n_pos == 0.0 || n_neg == 0.0 {
        return None;
    }

    rows.sort_by(|a, b| a.0.total_cmp(&b.0));
    let mut pos_rank_sum = 0.0;
    let mut i = 0;
    while i < rows.len() {
        let mut j = i;
        while j < rows.len() && rows[j].0 == rows[i].0 {
            j += 1;
        }
        // Ranks are 1-based: rows i..j share the mean of ranks i+1..=j.
        let avg_rank = (i + 1 + j) as f64 / 2.0;
        let tied_pos = rows[i..j].iter().filter(|(_, pos)| *pos).count() as f64;
        pos_rank_sum += avg_rank * tied_pos;
        i = j;
    }
    Some((pos_rank_sum - n_pos * (n_pos + 1.0) / 2.0) / (n_pos * n_neg))
}

/// Cast to `f64`, nulls becoming NaN.
pub(crate) fn series_to_f64(s: &Series) -> Result<Vec<f64>, CoreError> {
    let cast = s.cast(&DataType::Float64)?;
    Ok(cast.f64()?.iter().map(|v| v.unwrap_or(f64::NAN)).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn jitter_stability_is_one_with_fewer_than_two_variants() {
        let preds = vec![Series::new("p".into(), &[0.1f64, 0.2, 0.3])];
        assert_eq!(calculate_jitter_stability(&preds, 100.0).unwrap(), 1.0);
        assert_eq!(calculate_jitter_stability(&[], 100.0).unwrap(), 1.0);
    }

    #[test]
    fn jitter_stability_decreases_as_predictions_disagree_more() {
        let stable = vec![
            Series::new("p".into(), &[0.5f64, 0.5, 0.5]),
            Series::new("p".into(), &[0.5f64, 0.5, 0.5]),
        ];
        let unstable = vec![
            Series::new("p".into(), &[0.1f64, 0.1, 0.1]),
            Series::new("p".into(), &[0.9f64, 0.9, 0.9]),
        ];
        let stable_score = calculate_jitter_stability(&stable, JITTER_STABILITY_ALPHA).unwrap();
        let unstable_score = calculate_jitter_stability(&unstable, JITTER_STABILITY_ALPHA).unwrap();
        assert_eq!(stable_score, 1.0);
        assert!(unstable_score < stable_score);
    }

    /// The pre-rank-based implementation, kept to show the rewrite agrees
    /// with it wherever it was well defined (no tied scores).
    fn trapezoid_auc(y: &[f64], s: &[f64]) -> f64 {
        let mut rows: Vec<(f64, f64)> = s.iter().copied().zip(y.iter().copied()).collect();
        rows.sort_by(|a, b| b.0.total_cmp(&a.0));
        let n_pos: f64 = y.iter().sum();
        let n_neg = y.len() as f64 - n_pos;
        let (mut auc, mut prev_fpr, mut prev_tpr, mut pos, mut neg) = (0.0, 0.0, 0.0, 0.0, 0.0);
        for (_, v) in rows {
            if v == 1.0 {
                pos += 1.0;
            } else {
                neg += 1.0;
            }
            let (tpr, fpr) = (pos / n_pos, neg / n_neg);
            auc += (fpr - prev_fpr) * (tpr + prev_tpr) / 2.0;
            prev_fpr = fpr;
            prev_tpr = tpr;
        }
        auc
    }

    #[test]
    fn roc_auc_matches_trapezoid_without_ties() {
        let y = [0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0];
        let s = [0.1, 0.8, 0.35, 0.3, 0.9, 0.2, 0.75, 0.6];
        let got = roc_auc(&y, &s).unwrap();
        assert!((got - trapezoid_auc(&y, &s)).abs() < 1e-12, "got {got}");
        assert!((got - 0.8125).abs() < 1e-12, "got {got}");
    }

    #[test]
    fn tied_scores_count_half_regardless_of_row_order() {
        // Every score tied: no discrimination at all, whatever the order.
        let s = [0.5; 4];
        assert_eq!(roc_auc(&[1.0, 1.0, 0.0, 0.0], &s), Some(0.5));
        assert_eq!(roc_auc(&[0.0, 0.0, 1.0, 1.0], &s), Some(0.5));
        // Of the 4 positive/negative pairs: 2 wins, 1 loss, 1 tie at 0.9.
        let y = [1.0, 0.0, 1.0, 0.0];
        let s = [0.9, 0.9, 0.5, 0.1];
        assert_eq!(roc_auc(&y, &s), Some(0.625));
    }

    #[test]
    fn roc_auc_is_undefined_with_one_class() {
        assert_eq!(roc_auc(&[1.0, 1.0], &[0.2, 0.8]), None);
        assert_eq!(roc_auc(&[0.0, 0.0], &[0.2, 0.8]), None);
        let y = Series::new("y".into(), &[0i64, 0]);
        let s = Series::new("s".into(), &[0.2f64, 0.8]);
        assert_eq!(calculate_roc_auc(&y, &s).unwrap(), 0.5);
    }
}
