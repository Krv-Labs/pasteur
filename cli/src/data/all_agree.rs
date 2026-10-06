use pasteur_core::TaskType;

/// `per_model_preds[model][column][row]`. Each model's decision per row is
/// the positive class at a fixed 0.5 for binary (so `--flip-threshold` does
/// not move it), the argmax for multiclass, the set of labels at or over
/// that model's threshold for multilabel, and the side of the cutoff
/// (`per_model_thresholds[model][0]`) for regression.
pub fn build_all_agree_column(
    per_model_preds: &[Vec<Vec<f64>>],
    per_model_thresholds: &[Vec<f64>],
    task: TaskType,
    n_rows: usize,
) -> Vec<bool> {
    let mut all_agree = vec![true; n_rows];
    let mut prev_class: Option<Vec<Vec<bool>>> = None;
    for (preds, thresholds) in per_model_preds.iter().zip(per_model_thresholds) {
        let classes = decisions(preds, thresholds, task, n_rows);
        if let Some(prev) = &prev_class {
            mark_disagreements(&mut all_agree, prev, &classes);
        }
        prev_class = Some(classes);
    }
    all_agree
}

fn decisions(
    preds: &[Vec<f64>],
    thresholds: &[f64],
    task: TaskType,
    n_rows: usize,
) -> Vec<Vec<bool>> {
    (0..n_rows)
        .map(|i| match task {
            TaskType::Binary => vec![preds[0][i] >= 0.5],
            TaskType::Regression => vec![preds[0][i] >= thresholds[0]],
            TaskType::Multiclass => {
                let best = (0..preds.len())
                    .max_by(|&a, &b| preds[a][i].total_cmp(&preds[b][i]))
                    .unwrap_or(0);
                (0..preds.len()).map(|j| j == best).collect()
            }
            TaskType::Multilabel => (0..preds.len())
                .map(|j| preds[j][i] >= thresholds[j])
                .collect(),
        })
        .collect()
}

fn mark_disagreements(all_agree: &mut [bool], prev: &[Vec<bool>], curr: &[Vec<bool>]) {
    for i in 0..all_agree.len() {
        if prev[i] != curr[i] {
            all_agree[i] = false;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn multiclass_agreement_is_on_the_argmax() {
        // Row 0: both pick class 1. Row 1: class 0 vs class 2.
        let a = vec![vec![0.2, 0.5], vec![0.7, 0.1], vec![0.1, 0.4]];
        let b = vec![vec![0.1, 0.3], vec![0.6, 0.2], vec![0.3, 0.5]];
        let t = vec![vec![], vec![]];
        let agree = build_all_agree_column(&[a, b], &t, TaskType::Multiclass, 2);
        assert_eq!(agree, vec![true, false]);
    }

    #[test]
    fn regression_agreement_is_the_side_of_the_cutoff() {
        // Row 0: 6.0 and 6.4 both under 6.5. Row 1: 6.4 vs 6.6 straddle it.
        let a = vec![vec![6.0, 6.4]];
        let b = vec![vec![6.4, 6.6]];
        let t = vec![vec![6.5], vec![6.5]];
        let agree = build_all_agree_column(&[a, b], &t, TaskType::Regression, 2);
        assert_eq!(agree, vec![true, false]);
    }

    #[test]
    fn multilabel_agreement_uses_each_models_thresholds() {
        let a = vec![vec![0.6], vec![0.2]];
        let b = vec![vec![0.6], vec![0.2]];
        let same = build_all_agree_column(
            &[a.clone(), b.clone()],
            &[vec![0.5, 0.5], vec![0.5, 0.5]],
            TaskType::Multilabel,
            1,
        );
        assert_eq!(same, vec![true]);
        let differ = build_all_agree_column(
            &[a, b],
            &[vec![0.5, 0.5], vec![0.7, 0.5]],
            TaskType::Multilabel,
            1,
        );
        assert_eq!(differ, vec![false]);
    }
}
