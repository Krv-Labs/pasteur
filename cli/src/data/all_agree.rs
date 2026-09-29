pub fn build_all_agree_column(per_model_preds: &[Vec<f64>], n_rows: usize) -> Vec<bool> {
    let mut all_agree = vec![true; n_rows];
    let mut prev_class: Option<Vec<i64>> = None;
    for preds in per_model_preds {
        let classes = classes_from_probs(preds);
        if let Some(prev) = &prev_class {
            mark_disagreements(&mut all_agree, prev, &classes);
        }
        prev_class = Some(classes);
    }
    all_agree
}

fn classes_from_probs(preds: &[f64]) -> Vec<i64> {
    preds.iter().map(|p| i64::from(*p >= 0.5)).collect()
}

fn mark_disagreements(all_agree: &mut [bool], prev: &[i64], curr: &[i64]) {
    for i in 0..all_agree.len() {
        if prev[i] != curr[i] {
            all_agree[i] = false;
        }
    }
}
