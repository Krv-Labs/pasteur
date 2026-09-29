use crate::error::CoreError;
use polars::prelude::*;

pub const JITTER_STABILITY_ALPHA: f64 = 100.0;

/// Mean per-row variance across N independent jitter draws' predictions,
/// squashed into (0, 1] — `1 / (1 + alpha * mean_variance)`. Needs at least
/// two jitter variants to have any variance to measure; with fewer, there's
/// nothing to compare and stability is reported as perfect (1.0).
pub fn calculate_jitter_stability(preds: &[Series], alpha: f64) -> Result<f64, CoreError> {
    if preds.len() < 2 {
        return Ok(1.0);
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
    let mean_variance = total_variance / n_rows as f64;
    Ok(1.0 / (1.0 + alpha * mean_variance))
}

pub fn calculate_roc_auc(y_true: &Series, y_scores: &Series) -> Result<f64, CoreError> {
    // Simple implementation of ROC AUC
    // 1. Sort by scores descending
    // 2. Iterate and calculate trapezoidal area

    let df = df![
        "y_true" => y_true,
        "y_scores" => y_scores
    ]?;

    let sorted = df.sort(
        ["y_scores"],
        SortMultipleOptions {
            descending: vec![true],
            ..Default::default()
        },
    )?;

    let y_true_sorted = sorted.column("y_true")?.cast(&DataType::Float64)?;
    let y_true_vec = y_true_sorted.f64()?;

    let n_pos = y_true_vec.sum().unwrap_or(0.0);
    let n_neg = y_true_vec.len() as f64 - n_pos;

    if n_pos == 0.0 || n_neg == 0.0 {
        return Ok(0.5); // Or error?
    }

    let mut auc = 0.0;
    let mut prev_fpr = 0.0;
    let mut prev_tpr = 0.0;
    let mut current_pos = 0.0;
    let mut current_neg = 0.0;

    for v in y_true_vec.iter().flatten() {
        if v == 1.0 {
            current_pos += 1.0;
        } else {
            current_neg += 1.0;
        }

        let tpr = current_pos / n_pos;
        let fpr = current_neg / n_neg;

        // Area = (fpr - prev_fpr) * (tpr + prev_tpr) / 2
        auc += (fpr - prev_fpr) * (tpr + prev_tpr) / 2.0;

        prev_fpr = fpr;
        prev_tpr = tpr;
    }

    Ok(auc)
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
}
