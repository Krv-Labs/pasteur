use crate::error::CoreError;
use polars::prelude::*;
use rand::prelude::*;
use std::collections::HashMap;

use super::rng::stable_unit_floats;

type PatientRowGroups = (Vec<String>, HashMap<String, Vec<usize>>);

pub(super) fn group_rows_by_patient(
    df: &DataFrame,
    patient_id_col: &str,
) -> Result<PatientRowGroups, CoreError> {
    let n_rows = df.height();
    let patient_series = df.column(patient_id_col)?;
    let mut patient_order: Vec<String> = Vec::new();
    let mut patient_rows: HashMap<String, Vec<usize>> = HashMap::new();
    for i in 0..n_rows {
        let key = format!("{}", patient_series.get(i)?);
        patient_rows
            .entry(key.clone())
            .or_insert_with(|| {
                patient_order.push(key.clone());
                Vec::new()
            })
            .push(i);
    }
    Ok((patient_order, patient_rows))
}

pub(super) fn selected_patient_keys(
    patient_order: &[String],
    rate: f64,
    rng: &mut StdRng,
) -> Vec<String> {
    let n_patients = patient_order.len();
    let mut n_blackout = (n_patients as f64 * rate) as usize;
    if n_blackout == 0 && rate > 0.0 && n_patients > 0 {
        n_blackout = 1;
    }
    n_blackout = n_blackout.min(n_patients);

    let mut order_idxs: Vec<usize> = (0..n_patients).collect();
    order_idxs.shuffle(rng);
    order_idxs[0..n_blackout]
        .iter()
        .map(|&i| patient_order[i].clone())
        .collect()
}

pub(super) fn time_sort_keys(
    df: &DataFrame,
    n_rows: usize,
    time_col: Option<&str>,
) -> Option<Vec<f64>> {
    time_col.and_then(|tc| {
        let s = df.column(tc).ok()?;
        let as_f64 = s
            .cast(&DataType::Int64)
            .and_then(|s2| s2.cast(&DataType::Float64))
            .or_else(|_| s.cast(&DataType::Float64))
            .ok()?;
        let ca = as_f64.f64().ok()?;
        Some((0..n_rows).map(|i| ca.get(i).unwrap_or(f64::NAN)).collect())
    })
}

pub(super) fn apply_patient_window_mask(
    blackout_mask: &mut [bool],
    patient_rows: &HashMap<String, Vec<usize>>,
    pid: &str,
    sort_keys: Option<&[f64]>,
    window_frac: f64,
    random_state: Option<u64>,
) {
    let mut rows = patient_rows.get(pid).cloned().unwrap_or_default();
    if let Some(keys) = sort_keys {
        rows.sort_by(|&a, &b| {
            keys[a]
                .partial_cmp(&keys[b])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    }
    let n_samples = rows.len();
    if n_samples == 0 {
        return;
    }

    let window_size = if window_frac > 0.0 {
        ((n_samples as f64 * window_frac).floor() as usize).max(1)
    } else {
        0
    };
    let window_size = window_size.min(n_samples);
    if window_size == 0 {
        return;
    }

    let (u1, _) = stable_unit_floats(random_state, &format!("patient:{pid}"));
    let max_start = n_samples - window_size;
    let start_idx = ((u1 * (max_start as f64 + 1.0)) as usize).min(max_start);

    for &row_idx in &rows[start_idx..start_idx + window_size] {
        blackout_mask[row_idx] = true;
    }
}
