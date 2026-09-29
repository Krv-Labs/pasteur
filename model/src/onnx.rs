use ort::session::Session;
use ort::value::{DynMapValueType, DynSequenceValueType, ValueType};
use pasteur_core::{CoreError, Model};
use polars::prelude::*;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

/// Sidecar written next to the `.onnx` by whoever trained it (erdos-reyni's
/// `er train` emits this as `metadata.json`). It is the only thing that can
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
struct ModelContract {
    #[serde(default)]
    input: ContractInput,
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
    positive_class_index: usize,
    null_fill: f32,
}

impl OnnxModel {
    /// `null_fill` is what a null becomes on the way into the tensor. `None`
    /// resolves to the sidecar's `absent_sentinel` if there is one, else NaN.
    /// NaN is the honest default: `0.0` TSH is not absence, it is profoundly
    /// suppressed — a hyperthyroid signal — so silently filling zeros lets
    /// blackout push patients toward the positive class.
    pub fn from_file(
        path: &Path,
        input_name: impl Into<String>,
        feature_order: Vec<String>,
        positive_class_index: usize,
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

        let null_fill = null_fill
            .or_else(|| contract.as_ref().and_then(|c| c.input.absent_sentinel))
            .unwrap_or(f32::NAN);

        Ok(Self {
            session: Mutex::new(session),
            input_name,
            feature_order,
            positive_class_index,
            null_fill,
        })
    }
}

impl Model for OnnxModel {
    fn predict_proba(&self, df: &DataFrame) -> Result<Series, CoreError> {
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

        let probs = extract_positive_class_probabilities(&outputs, self.positive_class_index)?;

        // A NaN fill only means "missingness" to a NaN-native model. Anything
        // exported from sklearn-onnx propagates it straight through, and a NaN
        // probability silently poisons ROC AUC rather than erroring. Refuse.
        let nans = probs.iter().filter(|p| p.is_nan()).count();
        if nans > 0 {
            return Err(CoreError::Generic(format!(
                "{nans}/{} probabilities came back NaN — this model is not NaN-native. \
                 Pass the value it was trained with (e.g. --null-fill 0), or declare \
                 `input.absent_sentinel` in its {CONTRACT_FILENAME}.",
                probs.len()
            )));
        }
        Ok(Series::new("proba".into(), probs))
    }
}

/// Handles both common shapes for a classifier's probability output:
/// a plain `tensor(float)` of shape `[n, n_classes]` (the common case when a
/// model is exported with `zipmap=False`), or — if the model still routes
/// through an ONNX `ZipMap` node — a `sequence(map(int64, tensor(float)))`.
fn extract_positive_class_probabilities(
    outputs: &ort::session::SessionOutputs,
    positive_class_index: usize,
) -> Result<Vec<f32>, CoreError> {
    let value = outputs
        .get("output_probability")
        .or_else(|| outputs.get("probabilities"))
        .ok_or_else(|| CoreError::Generic("model has no probability output".to_string()))?;

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
}
