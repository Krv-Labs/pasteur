//! What a stress variant did to the model's decisions, row by row. It needs
//! no oracle labels: the reference is the model's own decision on the clean
//! row. An AUC ratio can stay near 1.0 while individual patients flip; this
//! counts them.

use super::metrics::series_to_f64;
use crate::error::CoreError;
use polars::prelude::*;
use serde::{Deserialize, Serialize};

/// One stress variant (blackout, scramble, units, site, or one site draw).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StressScores {
    pub variant: String,
    /// ROC AUC on this variant against the clean rows' labels, averaged over
    /// the labels where it is defined.
    pub roc_auc: Option<f64>,
    /// This variant's share of the clean ROC AUC.
    pub resiliency: Option<f64>,
    pub rows: usize,
    /// Rows where at least one feature reaches the model with a different
    /// value than in clean, after the same lenient numeric cast scoring uses.
    /// Text that still parses to the clean number (`5.2e0`) is not a change;
    /// text that parses to nothing (`<0.01`) is.
    pub rows_changed: usize,
    /// Changed rows whose decision differs from the decision on the clean row.
    pub decisions_changed: usize,
    /// `decisions_changed / rows_changed`; `None` when no row changed.
    pub decision_change_rate: Option<f64>,
    /// As `decision_change_rate`, not counting rows the model answered with
    /// the abstain class: a different answer given without a warning. Present
    /// only when an abstain class is set.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub silent_change_rate: Option<f64>,
}

/// Per row, whether any feature differs between `clean` and `variant` as the
/// model would read it. Rows are matched by position, as every other metric
/// matches them. Nulls and unparseable text both read as missing.
pub(crate) fn changed_rows(clean: &DataFrame, variant: &DataFrame) -> Result<Vec<bool>, CoreError> {
    if clean.height() != variant.height() {
        return Err(CoreError::InvalidConfig(format!(
            "a stress variant has {} rows but the clean data has {}",
            variant.height(),
            clean.height()
        )));
    }
    let mut changed = vec![false; clean.height()];
    for column in clean
        .columns()
        .iter()
        .filter(|c| !matches!(c.name().as_str(), "variant_name" | "source_row_id"))
    {
        let Ok(other) = variant.column(column.name().as_str()) else {
            continue;
        };
        // A column that cannot be cast at all is not a model input.
        let (Ok(a), Ok(b)) = (
            series_to_f64(column.as_materialized_series()),
            series_to_f64(other.as_materialized_series()),
        ) else {
            continue;
        };
        for (flag, (a, b)) in changed.iter_mut().zip(a.iter().zip(&b)) {
            *flag |= a != b && !(a.is_nan() && b.is_nan());
        }
    }
    Ok(changed)
}

/// `(decisions_changed, silently_changed)` over the changed rows. A decision
/// is one bool per output, as `oracle::decisions` returns it. The second count
/// leaves out rows where the stressed decision is the `abstain` output.
pub(crate) fn decision_changes(
    changed: &[bool],
    clean: &[Vec<bool>],
    stressed: &[Vec<bool>],
    abstain: Option<usize>,
) -> (usize, Option<usize>) {
    let flipped: Vec<&Vec<bool>> = changed
        .iter()
        .zip(clean.iter().zip(stressed))
        .filter(|(&c, (a, b))| c && a != b)
        .map(|(_, (_, b))| b)
        .collect();
    let silent = abstain.map(|j| flipped.iter().filter(|d| !d[j]).count());
    (flipped.len(), silent)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn changed_rows_compare_what_the_model_reads() {
        let clean = df![
            "a" => [1.0, 2.0, 3.0, 4.0, 5.0],
            "b" => [Some(1.0), None, Some(3.0), Some(4.0), Some(5.0)],
            "site" => ["x"; 5],
            "variant_name" => ["clean"; 5],
        ]
        .unwrap();
        let variant = df![
            // Same number as text, unparseable text, same, a sentinel, same.
            "a" => ["1e0", "<2", "3", "-999", "5"],
            // Same, still missing, blacked out, same, same.
            "b" => [Some(1.0), None, None, Some(4.0), Some(5.0)],
            "site" => ["x"; 5],
            "variant_name" => ["scramble"; 5],
        ]
        .unwrap();
        assert_eq!(
            changed_rows(&clean, &variant).unwrap(),
            vec![false, true, true, true, false]
        );
        assert!(changed_rows(&clean, &variant.head(Some(2))).is_err());
    }

    #[test]
    fn only_changed_rows_count_and_abstaining_is_not_silent() {
        let d = |a: bool, b: bool| vec![a, b];
        // Outputs: [positive, abstain].
        let clean = vec![
            d(true, false),
            d(true, false),
            d(true, false),
            d(false, false),
        ];
        let stressed = vec![
            d(false, false),
            d(false, true),
            d(true, false),
            d(true, false),
        ];
        let changed = [true, true, true, false];
        assert_eq!(
            decision_changes(&changed, &clean, &stressed, None),
            (2, None)
        );
        assert_eq!(
            decision_changes(&changed, &clean, &stressed, Some(1)),
            (2, Some(1))
        );
    }
}
