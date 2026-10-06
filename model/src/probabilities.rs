//! Reading a classifier's probability output into one column per class,
//! or a regressor's predicted value into one column, from every shape
//! sklearn-onnx exports, and refusing outputs that cannot be scored honestly.

use crate::contract::{OutputSpec, CONTRACT_FILENAME};
use ort::value::{DynMapValueType, DynSequenceValueType, DynTensorValueType, DynValue};
use pasteur_core::{CoreError, TaskType};

/// The probabilities `spec` selects from `value`, one `Vec` per output
/// column, checked against the input row count.
pub(crate) fn read_probabilities(
    value: &DynValue,
    spec: &OutputSpec,
    n_rows: usize,
) -> Result<Vec<Vec<f32>>, CoreError> {
    let columns = match spec.task {
        TaskType::Binary => vec![extract_positive_class_probabilities(
            value,
            spec.positive_class_index,
        )?],
        TaskType::Multiclass | TaskType::Multilabel => {
            extract_all_probabilities(value, spec.task, &spec.columns)?
        }
        TaskType::Regression => vec![extract_prediction(value)?],
    };
    check_columns(&columns, spec.task, n_rows)?;
    Ok(columns)
}

fn check_columns(columns: &[Vec<f32>], task: TaskType, n_rows: usize) -> Result<(), CoreError> {
    let total: usize = columns.iter().map(Vec::len).sum();
    let nans = columns.iter().flatten().filter(|p| p.is_nan()).count();
    let noun = match task {
        TaskType::Binary | TaskType::Multiclass | TaskType::Multilabel => "probabilities",
        TaskType::Regression => "predictions",
    };
    if nans > 0 {
        // A NaN fill only means "missingness" to a NaN-native model.
        // Anything exported from sklearn-onnx propagates it straight
        // through, and a NaN probability silently poisons ROC AUC rather
        // than erroring. Refuse.
        return Err(CoreError::Generic(format!(
            "{nans}/{total} {noun} came back NaN — this model is not NaN-native. \
             Pass the value it was trained with (e.g. --null-fill 0), or declare \
             `input.absent_sentinel` in its {CONTRACT_FILENAME}."
        )));
    }
    if let Some(c) = columns.iter().find(|c| c.len() != n_rows) {
        return Err(CoreError::Generic(format!(
            "model returned {} rows for {n_rows} input rows",
            c.len()
        )));
    }
    match task {
        TaskType::Multiclass => check_rows_sum_to_one(columns),
        TaskType::Binary | TaskType::Multilabel | TaskType::Regression => Ok(()),
    }
}

/// A regressor's output: a `tensor(float)` of shape `[n]` or `[n, 1]`. The
/// value is in the target's units, so nothing is range-checked.
fn extract_prediction(value: &DynValue) -> Result<Vec<f32>, CoreError> {
    let (shape, data) = value.try_extract_tensor::<f32>().map_err(|e| {
        CoreError::Generic(format!(
            "a regression model's output must be a float tensor of shape [n] or [n, 1]: {e}"
        ))
    })?;
    match **shape {
        [_] | [_, 1] => Ok(data.to_vec()),
        _ => Err(CoreError::Generic(format!(
            "prediction tensor has shape {shape:?}; a single-target regression model \
             returns [n] or [n, 1]"
        ))),
    }
}

/// Handles both common shapes for a classifier's probability output:
/// a plain `tensor(float)` of shape `[n, n_classes]` (the common case when a
/// model is exported with `zipmap=False`), or — if the model still routes
/// through an ONNX `ZipMap` node — a `sequence(map(int64, tensor(float)))`.
fn extract_positive_class_probabilities(
    value: &DynValue,
    positive_class_index: usize,
) -> Result<Vec<f32>, CoreError> {
    if let Ok((shape, data)) = value.try_extract_tensor::<f32>() {
        let n_classes = *shape.last().unwrap_or(&1) as usize;
        let n_rows = data.len() / n_classes.max(1);
        let mut out = Vec::with_capacity(n_rows);
        for i in 0..n_rows {
            out.push(data[i * n_classes + positive_class_index]);
        }
        return Ok(out);
    }

    let sequence = value
        .downcast_ref::<DynSequenceValueType>()
        .map_err(|e| CoreError::Generic(format!("unsupported probability output type: {e}")))?;
    let rows = sequence
        .try_extract_sequence::<DynMapValueType>()
        .map_err(|e| CoreError::Generic(format!("expected sequence of maps: {e}")))?;
    let mut out = Vec::with_capacity(rows.len());
    for row in rows {
        let map = row
            .try_extract_map::<i64, f32>()
            .map_err(|e| CoreError::Generic(format!("expected map in output sequence: {e}")))?;
        out.push(
            *map.get(&(positive_class_index as i64)).ok_or_else(|| {
                CoreError::Generic("positive class not in output map".to_string())
            })?,
        );
    }
    Ok(out)
}

/// Every class's probabilities, one `Vec` per output column, from any of
/// the shapes sklearn-onnx emits:
///
/// - a `tensor(float)` of shape `[n, K]` (`zipmap=False`; also a plain
///   sigmoid/softmax head);
/// - a ZipMap `sequence(map(int64|string, float))`, matched to `classes` by
///   key name rather than position, since ZipMap's key order is the
///   training labels' sort order and not necessarily the contract's;
/// - multilabel only: a `sequence(tensor(float))` of K `[n, 2]` tensors,
///   which is how `MultiOutputClassifier` exports — column 1 of each is the
///   label's positive probability.
fn extract_all_probabilities(
    value: &DynValue,
    task: TaskType,
    classes: &[String],
) -> Result<Vec<Vec<f32>>, CoreError> {
    if let Ok((shape, data)) = value.try_extract_tensor::<f32>() {
        return split_columns(shape, data, classes.len());
    }

    let sequence = value
        .downcast_ref::<DynSequenceValueType>()
        .map_err(|e| CoreError::Generic(format!("unsupported probability output type: {e}")))?;

    if let Ok(rows) = sequence.try_extract_sequence::<DynMapValueType>() {
        let rows: Vec<Vec<(String, f32)>> = rows
            .iter()
            .map(|row| {
                if let Ok(m) = row.try_extract_key_values::<i64, f32>() {
                    return Ok(m.into_iter().map(|(k, v)| (k.to_string(), v)).collect());
                }
                row.try_extract_key_values::<String, f32>().map_err(|e| {
                    CoreError::Generic(format!("expected int64 or string keyed map: {e}"))
                })
            })
            .collect::<Result<_, _>>()?;
        return columns_from_maps(&rows, classes);
    }

    if task != TaskType::Multilabel {
        return Err(CoreError::Generic(format!(
            "a sequence of tensors is the multilabel MultiOutputClassifier format; this \
             contract declares a {task} model"
        )));
    }
    let tensors = sequence
        .try_extract_sequence::<DynTensorValueType>()
        .map_err(|e| CoreError::Generic(format!("unsupported probability sequence: {e}")))?;
    let per_label: Vec<(Vec<i64>, Vec<f32>)> = tensors
        .iter()
        .map(|t| {
            t.try_extract_tensor::<f32>()
                .map(|(shape, data)| (shape.to_vec(), data.to_vec()))
                .map_err(|e| CoreError::Generic(format!("expected float tensors: {e}")))
        })
        .collect::<Result<_, _>>()?;
    columns_from_label_tensors(&per_label, classes.len())
}

/// Row-major `[n, k]` → k columns of n.
fn split_columns(shape: &[i64], data: &[f32], k: usize) -> Result<Vec<Vec<f32>>, CoreError> {
    if shape.len() != 2 || shape[1] as usize != k {
        return Err(CoreError::Generic(format!(
            "probability tensor has shape {shape:?}; expected [n, {k}] to match output.classes"
        )));
    }
    let n = shape[0] as usize;
    Ok((0..k)
        .map(|j| (0..n).map(|i| data[i * k + j]).collect())
        .collect())
}

fn columns_from_maps(
    rows: &[Vec<(String, f32)>],
    classes: &[String],
) -> Result<Vec<Vec<f32>>, CoreError> {
    let mut columns = vec![Vec::with_capacity(rows.len()); classes.len()];
    for row in rows {
        if row.len() != classes.len() {
            return Err(CoreError::Generic(format!(
                "probability map has {} entries but output.classes lists {}",
                row.len(),
                classes.len()
            )));
        }
        for (key, p) in row {
            let j = classes.iter().position(|c| c == key).ok_or_else(|| {
                CoreError::Generic(format!(
                    "model output class {key:?} is not in output.classes {classes:?}"
                ))
            })?;
            columns[j].push(*p);
        }
    }
    Ok(columns)
}

fn columns_from_label_tensors(
    per_label: &[(Vec<i64>, Vec<f32>)],
    k: usize,
) -> Result<Vec<Vec<f32>>, CoreError> {
    if per_label.len() != k {
        return Err(CoreError::Generic(format!(
            "model returns {} per-label tensors but output.classes lists {k}",
            per_label.len()
        )));
    }
    per_label
        .iter()
        .map(|(shape, data)| {
            if shape.len() != 2 || shape[1] != 2 {
                return Err(CoreError::Generic(format!(
                    "per-label probability tensor has shape {shape:?}; expected [n, 2]"
                )));
            }
            Ok(data.chunks(2).map(|row| row[1]).collect())
        })
        .collect()
}

/// A multiclass head's rows are a distribution. Independent sigmoids are not,
/// and scoring them as one would read every argmax off unrelated scales.
fn check_rows_sum_to_one(columns: &[Vec<f32>]) -> Result<(), CoreError> {
    let n = columns.first().map_or(0, Vec::len);
    for i in 0..n {
        let sum: f32 = columns.iter().map(|c| c[i]).sum();
        if (sum - 1.0).abs() > 1e-3 {
            return Err(CoreError::InvalidConfig(format!(
                "row {i} probabilities sum to {sum}, not 1; a model with independent \
                 per-label outputs is `task_type: multilabel`, not multiclass"
            )));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn row_major_tensor_splits_into_columns() {
        let cols = split_columns(&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 3).unwrap();
        assert_eq!(cols, vec![vec![1.0, 4.0], vec![2.0, 5.0], vec![3.0, 6.0]]);
        assert!(split_columns(&[2, 3], &[0.0; 6], 2).is_err());
    }
}
