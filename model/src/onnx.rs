use ort::session::Session;
use ort::value::{DynMapValueType, DynSequenceValueType, DynTensorValueType, DynValue, ValueType};
use pasteur_core::{CoreError, Model, TaskType};
use polars::prelude::*;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

/// Sidecar written next to the `.onnx` by whoever trained it, as
/// `metadata.json`. It is the only thing that can
/// verify *column order*: the ONNX graph knows how many inputs it takes, not
/// what they mean. Every field is optional — a model without a sidecar still
/// loads, just with order unverifiable.
pub const CONTRACT_FILENAME: &str = "metadata.json";

#[derive(Debug, Default, serde::Deserialize)]
struct ContractInput {
    #[serde(default)]
    feature_order: Vec<String>,
    /// The value the model was *fit* with standing in for "absent". Overrides
    /// the NaN default, because reproducing the training encoding beats any
    /// guess this crate could make.
    #[serde(default)]
    absent_sentinel: Option<f32>,
}

#[derive(Debug, Default, serde::Deserialize)]
struct ContractOutput {
    /// Graph output holding the probabilities. Defaults to the first of
    /// `output_probability`/`probabilities` the graph has.
    #[serde(default)]
    name: Option<String>,
    /// One name per output column, in the model's column order. Required for
    /// multiclass and multilabel models: it is the only record of what each
    /// column means, and its length is checked against the graph.
    #[serde(default)]
    classes: Option<Vec<String>>,
    /// Multilabel only: per-label decision thresholds, in `classes` order.
    #[serde(default)]
    thresholds: Option<Vec<f64>>,
}

#[derive(Debug, Default, serde::Deserialize)]
struct ModelContract {
    #[serde(default)]
    task_type: TaskType,
    #[serde(default)]
    input: ContractInput,
    #[serde(default)]
    output: ContractOutput,
}

const DEFAULT_PROBABILITY_OUTPUTS: [&str; 2] = ["output_probability", "probabilities"];
/// Column name of a binary model's single positive-class output.
pub const BINARY_OUTPUT_COLUMN: &str = "proba";

/// What `predict_proba` reads and how it labels it, settled at load time so a
/// mismatched contract fails before any patient is scored.
#[derive(Debug, Clone, PartialEq)]
struct OutputSpec {
    task: TaskType,
    output_name: String,
    /// Binary only.
    positive_class_index: usize,
    /// Output column names: `["proba"]` for binary, `output.classes` otherwise.
    columns: Vec<String>,
    thresholds: Option<Vec<f64>>,
}

/// `outputs` lists the graph's outputs with their static trailing dimension
/// when the graph declares one.
fn resolve_output_name(
    contract_label: &str,
    declared: Option<&ContractOutput>,
    outputs: &[(String, Option<usize>)],
) -> Result<String, CoreError> {
    match declared.and_then(|o| o.name.clone()) {
        Some(name) if outputs.iter().any(|(n, _)| *n == name) => Ok(name),
        Some(name) => Err(CoreError::InvalidConfig(format!(
            "{contract_label} names output {name:?} but the model's outputs are {:?}",
            outputs.iter().map(|(n, _)| n).collect::<Vec<_>>()
        ))),
        None => match DEFAULT_PROBABILITY_OUTPUTS
            .iter()
            .find(|d| outputs.iter().any(|(n, _)| n == **d))
        {
            Some(name) => Ok(name.to_string()),
            None => Err(CoreError::InvalidConfig(format!(
                "model has no probability output (looked for {DEFAULT_PROBABILITY_OUTPUTS:?}; \
                 its outputs are {:?}). Set `output.name` in {contract_label}.",
                outputs.iter().map(|(n, _)| n).collect::<Vec<_>>()
            ))),
        },
    }
}

fn resolve_binary_spec(
    contract_label: &str,
    task: TaskType,
    output_name: String,
    positive_class_index: Option<usize>,
    width: Option<usize>,
    classes: Option<Vec<String>>,
    thresholds: Option<Vec<f64>>,
) -> Result<OutputSpec, CoreError> {
    if thresholds.is_some() {
        return Err(CoreError::InvalidConfig(format!(
            "{contract_label} sets `output.thresholds`, which applies to multilabel models; \
             a binary model's threshold is --flip-threshold"
        )));
    }
    let positive_class_index = positive_class_index.unwrap_or(1);
    if let Some(n) = classes.as_ref().map(Vec::len).or(width) {
        if positive_class_index >= n {
            return Err(CoreError::InvalidConfig(format!(
                "--positive-class-index {positive_class_index} is out of range for a model \
                 with {n} output columns"
            )));
        }
    }
    Ok(OutputSpec {
        task,
        output_name,
        positive_class_index,
        columns: vec![BINARY_OUTPUT_COLUMN.to_string()],
        thresholds: None,
    })
}

fn resolve_multi_spec(
    contract_label: &str,
    task: TaskType,
    output_name: String,
    positive_class_index: Option<usize>,
    classes: Option<Vec<String>>,
    thresholds: Option<Vec<f64>>,
) -> Result<OutputSpec, CoreError> {
    if positive_class_index.is_some() {
        return Err(CoreError::InvalidConfig(format!(
            "--positive-class-index selects one column of a binary model; this is a {task} \
             model and every column is scored"
        )));
    }
    let Some(classes) = classes.filter(|c| !c.is_empty()) else {
        return Err(CoreError::InvalidConfig(format!(
            "{contract_label} declares task_type {task} but no `output.classes`; list one \
             name per output column, in the model's order"
        )));
    };
    let mut unique = classes.clone();
    unique.sort();
    unique.dedup();
    if unique.len() != classes.len() {
        return Err(CoreError::InvalidConfig(format!(
            "{contract_label} `output.classes` has duplicate names: {classes:?}"
        )));
    }
    if task == TaskType::Multiclass && classes.len() < 2 {
        return Err(CoreError::InvalidConfig(format!(
            "{contract_label} declares a multiclass model with {} class",
            classes.len()
        )));
    }
    match (&thresholds, task) {
        (Some(_), TaskType::Multiclass) => {
            return Err(CoreError::InvalidConfig(format!(
                "{contract_label} sets `output.thresholds`, but a multiclass decision is the \
                 argmax; thresholds apply to multilabel models"
            )));
        }
        (Some(t), _) if t.len() != classes.len() => {
            return Err(CoreError::InvalidConfig(format!(
                "{contract_label} has {} thresholds for {} labels",
                t.len(),
                classes.len()
            )));
        }
        (Some(t), _) if t.iter().any(|v| !(0.0..=1.0).contains(v)) => {
            return Err(CoreError::InvalidConfig(format!(
                "{contract_label} thresholds must lie in [0, 1], got {t:?}"
            )));
        }
        _ => {}
    }
    Ok(OutputSpec {
        task,
        output_name,
        positive_class_index: 0,
        columns: classes,
        thresholds,
    })
}

/// `outputs` lists the graph's outputs with their static trailing dimension
/// when the graph declares one.
fn resolve_output_spec(
    contract_label: &str,
    contract: Option<&ModelContract>,
    positive_class_index: Option<usize>,
    outputs: &[(String, Option<usize>)],
) -> Result<OutputSpec, CoreError> {
    let task = contract.map(|c| c.task_type).unwrap_or_default();
    let declared = contract.map(|c| &c.output);

    let output_name = resolve_output_name(contract_label, declared, outputs)?;

    let width = outputs
        .iter()
        .find(|(n, _)| *n == output_name)
        .and_then(|(_, w)| *w);
    let classes = declared.and_then(|o| o.classes.clone());
    let thresholds = declared.and_then(|o| o.thresholds.clone());

    if let (Some(w), Some(c)) = (width, &classes) {
        if w != c.len() {
            return Err(CoreError::InvalidConfig(format!(
                "{contract_label} lists {} classes but output {output_name:?} has {w} columns",
                c.len()
            )));
        }
    }

    if task.is_binary() {
        resolve_binary_spec(
            contract_label,
            task,
            output_name,
            positive_class_index,
            width,
            classes,
            thresholds,
        )
    } else {
        resolve_multi_spec(
            contract_label,
            task,
            output_name,
            positive_class_index,
            classes,
            thresholds,
        )
    }
}

/// Resolves the contract path: an explicit `--contract` wins; otherwise
/// `{model_dir}/metadata.json`.
fn resolve_contract_path(onnx_path: &Path, contract: Option<&Path>) -> Option<PathBuf> {
    contract
        .map(Path::to_path_buf)
        .or_else(|| onnx_path.parent().map(|dir| dir.join(CONTRACT_FILENAME)))
}

/// Reads a model contract sidecar. Absent is fine when the path was inferred;
/// an explicit path that does not exist is an error; present-but-malformed
/// is always an error — a contract nobody can parse is worse than no contract.
fn load_contract(
    onnx_path: &Path,
    contract: Option<&Path>,
) -> Result<Option<ModelContract>, CoreError> {
    let Some(path) = resolve_contract_path(onnx_path, contract) else {
        return Ok(None);
    };
    if !path.exists() {
        if contract.is_some() {
            return Err(CoreError::InvalidConfig(format!(
                "contract not found at {}",
                path.display()
            )));
        }
        return Ok(None);
    }
    let text = std::fs::read_to_string(&path)?;
    serde_json::from_str(&text)
        .map(Some)
        .map_err(|e| CoreError::InvalidConfig(format!("{} is not readable: {e}", path.display())))
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

/// Refuse to score when the column order cannot be trusted.
///
/// Wrong *width* errors inside ORT anyway. Wrong *order* at the right width
/// does not: every patient gets scored against the wrong thresholds and the
/// run looks clean. That is the case this exists for.
fn validate_feature_order(
    contract_label: &str,
    declared: Option<&[String]>,
    model_width: Option<usize>,
    requested: &[String],
) -> Result<(), CoreError> {
    if let Some(width) = model_width {
        if width != requested.len() {
            return Err(CoreError::InvalidConfig(format!(
                "model takes {width} features but {} were derived from the data",
                requested.len()
            )));
        }
    }

    let Some(declared) = declared.filter(|d| !d.is_empty()) else {
        // No sidecar feature_order: width is all we can check. Documented
        // limitation, not a failure.
        return Ok(());
    };

    if declared.len() != requested.len() {
        return Err(CoreError::InvalidConfig(format!(
            "{contract_label} declares {} features but {} were derived from the data",
            declared.len(),
            requested.len()
        )));
    }
    if let Some((i, (want, got))) = declared
        .iter()
        .zip(requested)
        .enumerate()
        .find(|(_, (a, b))| a != b)
    {
        return Err(CoreError::InvalidConfig(format!(
            "feature order disagrees with {contract_label} at index {i}: model expects \
             {want:?}, data supplies {got:?}. Scoring anyway would compare every patient \
             against the wrong thresholds."
        )));
    }
    Ok(())
}

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

    fn names(xs: &[&str]) -> Vec<String> {
        xs.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn width_mismatch_is_rejected() {
        let err = validate_feature_order("metadata.json", None, Some(3), &names(&["a", "b"]))
            .unwrap_err();
        assert!(
            err.to_string().contains("takes 3 features but 2"),
            "got {err}"
        );
    }

    #[test]
    fn dynamic_or_unknown_width_is_not_checked() {
        validate_feature_order("metadata.json", None, None, &names(&["a", "b"])).unwrap();
    }

    #[test]
    fn same_width_wrong_order_is_rejected() {
        // The whole point: ORT accepts this silently.
        let declared = names(&["TSH", "T3", "T4"]);
        let err = validate_feature_order(
            "metadata.json",
            Some(&declared),
            Some(3),
            &names(&["TSH", "T4", "T3"]),
        )
        .unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("index 1"), "got {msg}");
        assert!(
            msg.contains("\"T3\"") && msg.contains("\"T4\""),
            "got {msg}"
        );
    }

    #[test]
    fn matching_order_passes() {
        let declared = names(&["TSH", "T3", "T4"]);
        validate_feature_order("metadata.json", Some(&declared), Some(3), &declared).unwrap();
    }

    #[test]
    fn absent_declaration_passes_on_width_alone() {
        // No sidecar (None) and an empty declaration are both "unverifiable",
        // not "wrong" — a model with no contract must still be loadable.
        validate_feature_order("metadata.json", None, Some(2), &names(&["a", "b"])).unwrap();
        validate_feature_order("metadata.json", Some(&[]), Some(2), &names(&["a", "b"])).unwrap();
    }

    #[test]
    fn declaration_of_a_different_length_than_the_graph_is_rejected() {
        let declared = names(&["a", "b", "c"]);
        let err =
            validate_feature_order("metadata.json", Some(&declared), None, &names(&["a", "b"]))
                .unwrap_err();
        assert!(err.to_string().contains("declares 3 features"), "got {err}");
    }

    #[test]
    fn contract_is_optional_but_must_parse_when_present() {
        let dir = std::env::temp_dir().join(format!("pasteur-model-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let onnx = dir.join("m.onnx");

        // Absent sidecar -> None, no error.
        assert!(load_contract(&onnx, None).unwrap().is_none());

        std::fs::write(
            dir.join(CONTRACT_FILENAME),
            r#"{"input": {"feature_order": ["a", "b"], "absent_sentinel": 0.0}}"#,
        )
        .unwrap();
        let c = load_contract(&onnx, None).unwrap().unwrap();
        assert_eq!(c.input.feature_order, names(&["a", "b"]));
        assert_eq!(c.input.absent_sentinel, Some(0.0));

        // A contract that looks like protection but isn't must not pass silently.
        std::fs::write(dir.join(CONTRACT_FILENAME), "{not json").unwrap();
        assert!(load_contract(&onnx, None).is_err());

        // Unrelated keys are fine; this is a shared file with other consumers.
        std::fs::write(
            dir.join(CONTRACT_FILENAME),
            r#"{"target": {"column": "y"}}"#,
        )
        .unwrap();
        let c = load_contract(&onnx, None).unwrap().unwrap();
        assert!(c.input.feature_order.is_empty());
        assert_eq!(c.input.absent_sentinel, None);

        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn explicit_contract_path_overrides_model_directory() {
        let dir =
            std::env::temp_dir().join(format!("pasteur-model-contract-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let onnx = dir.join("models/m.onnx");
        std::fs::create_dir_all(onnx.parent().unwrap()).unwrap();
        let external = dir.join("staging/metadata.json");
        std::fs::create_dir_all(external.parent().unwrap()).unwrap();
        std::fs::write(
            &external,
            r#"{"input": {"feature_order": ["a", "b"], "absent_sentinel": 0.0}}"#,
        )
        .unwrap();

        let c = load_contract(&onnx, Some(&external)).unwrap().unwrap();
        assert_eq!(c.input.feature_order, names(&["a", "b"]));

        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn missing_explicit_contract_is_an_error() {
        let dir =
            std::env::temp_dir().join(format!("pasteur-model-missing-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let onnx = dir.join("m.onnx");
        let missing = dir.join("no-such/metadata.json");
        let err = load_contract(&onnx, Some(&missing)).unwrap_err();
        assert!(err.to_string().contains("contract not found"), "got {err}");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    fn contract(json: &str) -> ModelContract {
        serde_json::from_str(json).unwrap()
    }

    fn outputs(width: Option<usize>) -> Vec<(String, Option<usize>)> {
        vec![
            ("label".to_string(), None),
            ("probabilities".to_string(), width),
        ]
    }

    fn resolve_err(json: &str, pci: Option<usize>) -> String {
        resolve_output_spec("metadata.json", Some(&contract(json)), pci, &outputs(None))
            .unwrap_err()
            .to_string()
    }

    #[test]
    fn no_contract_is_binary_on_the_default_output() {
        let spec = resolve_output_spec("metadata.json", None, None, &outputs(Some(2))).unwrap();
        assert_eq!(spec.task, TaskType::Binary);
        assert_eq!(spec.output_name, "probabilities");
        assert_eq!(spec.positive_class_index, 1);
        assert_eq!(spec.columns, vec![BINARY_OUTPUT_COLUMN.to_string()]);
    }

    #[test]
    fn binary_positive_class_index_is_range_checked() {
        let err = resolve_output_spec("metadata.json", None, Some(2), &outputs(Some(2)))
            .unwrap_err()
            .to_string();
        assert!(err.contains("out of range"), "got {err}");
    }

    #[test]
    fn thresholds_belong_to_multilabel_only() {
        let err = resolve_err(r#"{"output": {"thresholds": [0.5]}}"#, None);
        assert!(err.contains("--flip-threshold"), "got {err}");
        let err = resolve_err(
            r#"{"task_type": "multiclass",
                "output": {"classes": ["a", "b"], "thresholds": [0.5, 0.5]}}"#,
            None,
        );
        assert!(err.contains("argmax"), "got {err}");
    }

    #[test]
    fn multilabel_thresholds_must_match_labels_and_be_probabilities() {
        let err = resolve_err(
            r#"{"task_type": "multilabel",
                "output": {"classes": ["a", "b"], "thresholds": [0.5]}}"#,
            None,
        );
        assert!(err.contains("1 thresholds for 2 labels"), "got {err}");
        let err = resolve_err(
            r#"{"task_type": "multilabel",
                "output": {"classes": ["a", "b"], "thresholds": [0.5, 1.5]}}"#,
            None,
        );
        assert!(err.contains("[0, 1]"), "got {err}");
    }

    #[test]
    fn multi_output_needs_unique_classes() {
        let err = resolve_err(r#"{"task_type": "multilabel"}"#, None);
        assert!(err.contains("no `output.classes`"), "got {err}");
        let err = resolve_err(
            r#"{"task_type": "multiclass", "output": {"classes": ["a", "a"]}}"#,
            None,
        );
        assert!(err.contains("duplicate"), "got {err}");
        let err = resolve_err(
            r#"{"task_type": "multiclass", "output": {"classes": ["a"]}}"#,
            None,
        );
        assert!(err.contains("1 class"), "got {err}");
    }

    #[test]
    fn row_major_tensor_splits_into_columns() {
        let cols = split_columns(&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 3).unwrap();
        assert_eq!(cols, vec![vec![1.0, 4.0], vec![2.0, 5.0], vec![3.0, 6.0]]);
        assert!(split_columns(&[2, 3], &[0.0; 6], 2).is_err());
    }
}
