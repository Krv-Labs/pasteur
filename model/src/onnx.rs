use crate::contract::{
    load_contract, resolve_contract_path, resolve_output_spec, validate_feature_order, OutputSpec,
};
use ort::session::Session;
use ort::value::{DynMapValueType, DynSequenceValueType, DynTensorValueType, DynValue, ValueType};
use pasteur_core::{CoreError, Model, TaskType};
use polars::prelude::*;
use std::path::Path;
use std::sync::Mutex;

pub use crate::contract::{BINARY_OUTPUT_COLUMN, CONTRACT_FILENAME};

/// In-process ONNX Runtime model, loaded from a locally-cached (pulled) file.
/// No HTTP server — this is a direct `Model` impl for scoring, per the
/// "in-process runner" decision. `Session::run` needs `&mut self` but
/// `Model::predict_proba` takes `&self`, so the session is behind a `Mutex`
/// (which also correctly reflects that a single ORT session isn't meant to
/// run concurrent inferences from multiple threads).
pub struct OnnxModel {
    session: Mutex<Session>,
    input_name: String,
    feature_order: Vec<String>,
    null_fill: f32,
    output: OutputSpec,
}

impl OnnxModel {
    /// `null_fill` is what a null becomes on the way into the tensor. `None`
    /// resolves to the sidecar's `absent_sentinel` if there is one, else NaN.
    /// NaN is the honest default: `0.0` TSH is not absence, it is profoundly
    /// suppressed — a hyperthyroid signal — so silently filling zeros lets
    /// blackout push patients toward the positive class.
    ///
    /// `positive_class_index` picks the scored column of a binary model
    /// (default 1) and must be `None` for multiclass and multilabel models.
    pub fn from_file(
        path: &Path,
        input_name: impl Into<String>,
        feature_order: Vec<String>,
        positive_class_index: Option<usize>,
        null_fill: Option<f32>,
        contract: Option<&Path>,
    ) -> Result<Self, CoreError> {
        let input_name = input_name.into();
        let contract_path = resolve_contract_path(path, contract);
        let contract_label = contract_path
            .as_ref()
            .map(|p| p.display().to_string())
            .unwrap_or_else(|| CONTRACT_FILENAME.to_string());
        let contract = load_contract(path, contract)?;
        let mut builder = Session::builder().map_err(|e| CoreError::Generic(e.to_string()))?;
        let session = builder
            .commit_from_file(path)
            .map_err(|e| CoreError::Generic(e.to_string()))?;

        validate_feature_order(
            &contract_label,
            contract.as_ref().map(|c| c.input.feature_order.as_slice()),
            declared_width(&session, &input_name)?,
            &feature_order,
        )?;

        let outputs: Vec<(String, Option<usize>)> = session
            .outputs()
            .iter()
            .map(|o| (o.name().to_string(), static_width(o.dtype())))
            .collect();
        let output = resolve_output_spec(
            &contract_label,
            contract.as_ref(),
            positive_class_index,
            &outputs,
        )?;

        let null_fill = null_fill
            .or_else(|| contract.as_ref().and_then(|c| c.input.absent_sentinel))
            .unwrap_or(f32::NAN);

        Ok(Self {
            session: Mutex::new(session),
            input_name,
            feature_order,
            null_fill,
            output,
        })
    }

    pub fn task(&self) -> TaskType {
        self.output.task
    }

    /// Output column names, in the order `predict_proba` returns them.
    pub fn output_columns(&self) -> &[String] {
        &self.output.columns
    }
}

/// The trailing dimension of the named input, or `None` when it is dynamic
/// (`-1`) or the input is not a plain tensor — in those cases the graph tells
/// us nothing about width and there is nothing to check.
fn declared_width(session: &Session, input_name: &str) -> Result<Option<usize>, CoreError> {
    let outlet = session
        .inputs()
        .iter()
        .find(|o| o.name() == input_name)
        .ok_or_else(|| {
            let actual: Vec<&str> = session.inputs().iter().map(|o| o.name()).collect();
            CoreError::InvalidConfig(format!(
                "model has no input named {input_name:?}; its inputs are {actual:?} \
                 (set --input-name)"
            ))
        })?;
    Ok(match outlet.dtype() {
        ValueType::Tensor { shape, .. } => match shape.last() {
            Some(&n) if n > 0 => Some(n as usize),
            _ => None,
        },
        _ => None,
    })
}

/// Trailing dimension of a plain tensor output, when the graph fixes it.
fn static_width(dtype: &ValueType) -> Option<usize> {
    match dtype {
        ValueType::Tensor { shape, .. } if shape.len() >= 2 => match shape.last() {
            Some(&n) if n > 0 => Some(n as usize),
            _ => None,
        },
        _ => None,
    }
}

impl Model for OnnxModel {
    fn predict_proba(&self, df: &DataFrame) -> Result<DataFrame, CoreError> {
        let n_rows = df.height();
        let n_features = self.feature_order.len();
        let mut data = vec![0f32; n_rows * n_features];
        for (j, name) in self.feature_order.iter().enumerate() {
            let series = df
                .column(name)?
                .as_materialized_series()
                .cast(&DataType::Float64)
                .map_err(CoreError::Polars)?;
            let ca = series.f64().map_err(CoreError::Polars)?;
            for i in 0..n_rows {
                data[i * n_features + j] = match ca.get(i) {
                    Some(v) => v as f32,
                    None => self.null_fill,
                };
            }
        }

        let input = ort::value::Tensor::from_array(([n_rows, n_features], data.into_boxed_slice()))
            .map_err(|e| CoreError::Generic(e.to_string()))?;

        let mut session = self
            .session
            .lock()
            .map_err(|_| CoreError::Generic("ONNX session lock poisoned".to_string()))?;
        let outputs = session
            .run(ort::inputs![self.input_name.as_str() => input])
            .map_err(|e| CoreError::Generic(e.to_string()))?;

        let value = outputs
            .get(self.output.output_name.as_str())
            .ok_or_else(|| {
                CoreError::Generic(format!(
                    "model produced no {:?} output",
                    self.output.output_name
                ))
            })?;
        let columns: Vec<Vec<f32>> = if self.output.task.is_binary() {
            vec![extract_positive_class_probabilities(
                value,
                self.output.positive_class_index,
            )?]
        } else {
            extract_all_probabilities(value, self.output.task, &self.output.columns)?
        };

        let total: usize = columns.iter().map(Vec::len).sum();
        let nans = columns.iter().flatten().filter(|p| p.is_nan()).count();
        if nans > 0 {
            // A NaN fill only means "missingness" to a NaN-native model.
            // Anything exported from sklearn-onnx propagates it straight
            // through, and a NaN probability silently poisons ROC AUC rather
            // than erroring. Refuse.
            return Err(CoreError::Generic(format!(
                "{nans}/{total} probabilities came back NaN — this model is not NaN-native. \
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
        if self.output.task == TaskType::Multiclass {
            check_rows_sum_to_one(&columns)?;
        }

        let columns: Vec<Column> = self
            .output
            .columns
            .iter()
            .zip(columns)
            .map(|(name, col)| {
                let col: Vec<f64> = col.into_iter().map(f64::from).collect();
                Series::new(name.as_str().into(), col).into()
            })
            .collect();
        Ok(DataFrame::new(n_rows, columns)?)
    }

    fn decision_thresholds(&self) -> Option<Vec<f64>> {
        self.output.thresholds.clone()
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
