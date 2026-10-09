use crate::contract::{
    load_contract, resolve_contract_path, resolve_output_spec, validate_feature_order, OutputSpec,
};
use crate::probabilities::read_probabilities;
use ort::session::Session;
use ort::value::ValueType;
use pasteur_core::{CoreError, Model, TaskType};
use polars::prelude::*;
use std::path::Path;
use std::sync::Mutex;

pub use crate::contract::{BINARY_OUTPUT_COLUMN, CONTRACT_FILENAME, REGRESSION_OUTPUT_COLUMN};

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
    /// (default 1) and must be `None` for multiclass, multilabel and
    /// regression models.
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
    /// For a regression model the one column is the predicted value, in the
    /// target's units, not a probability.
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
        let columns = read_probabilities(value, &self.output, n_rows)?;

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
