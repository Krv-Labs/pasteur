//! The model contract: the `metadata.json` sidecar, and everything settled
//! from it before a model scores anyone — feature order, the probability
//! output, and what each output column means.

use pasteur_core::{CoreError, TaskType};
use std::path::{Path, PathBuf};

/// Sidecar written next to the `.onnx` by whoever trained it, as
/// `metadata.json`. It is the only thing that can
/// verify *column order*: the ONNX graph knows how many inputs it takes, not
/// what they mean. Every field is optional — a model without a sidecar still
/// loads, just with order unverifiable.
pub const CONTRACT_FILENAME: &str = "metadata.json";

#[derive(Debug, Default, serde::Deserialize)]
pub(crate) struct ContractInput {
    #[serde(default)]
    pub(crate) feature_order: Vec<String>,
    /// The value the model was *fit* with standing in for "absent". Overrides
    /// the NaN default, because reproducing the training encoding beats any
    /// guess this crate could make.
    #[serde(default)]
    pub(crate) absent_sentinel: Option<f32>,
}

#[derive(Debug, Default, serde::Deserialize)]
pub(crate) struct ContractOutput {
    /// Graph output holding the probabilities (or, for regression, the
    /// predicted value). Defaults to the first of
    /// `output_probability`/`probabilities` the graph has, or of
    /// `variable`/`predictions` for regression.
    #[serde(default)]
    pub(crate) name: Option<String>,
    /// One name per output column, in the model's column order. Required for
    /// multiclass and multilabel models: it is the only record of what each
    /// column means, and its length is checked against the graph. Optional
    /// for regression, where its one entry names the target.
    #[serde(default)]
    pub(crate) classes: Option<Vec<String>>,
    /// Multilabel only: per-label decision thresholds, in `classes` order.
    #[serde(default)]
    pub(crate) thresholds: Option<Vec<f64>>,
}

#[derive(Debug, Default, serde::Deserialize)]
pub(crate) struct ModelContract {
    #[serde(default)]
    pub(crate) task_type: TaskType,
    #[serde(default)]
    pub(crate) input: ContractInput,
    #[serde(default)]
    pub(crate) output: ContractOutput,
}

const DEFAULT_PROBABILITY_OUTPUTS: [&str; 2] = ["output_probability", "probabilities"];
/// sklearn-onnx names a regressor's output `variable`.
const DEFAULT_PREDICTION_OUTPUTS: [&str; 2] = ["variable", "predictions"];
/// Column name of a binary model's single positive-class output.
pub const BINARY_OUTPUT_COLUMN: &str = "proba";
/// Column name of a regression model's output when `output.classes` does not
/// name it.
pub const REGRESSION_OUTPUT_COLUMN: &str = "prediction";

/// What `predict_proba` reads and how it labels it, settled at load time so a
/// mismatched contract fails before any patient is scored.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct OutputSpec {
    pub(crate) task: TaskType,
    pub(crate) output_name: String,
    /// Binary only.
    pub(crate) positive_class_index: usize,
    /// Output column names: `["proba"]` for binary, `output.classes` for
    /// multi-output, and one name for regression.
    pub(crate) columns: Vec<String>,
    pub(crate) thresholds: Option<Vec<f64>>,
}

/// `outputs` lists the graph's outputs with their static trailing dimension
/// when the graph declares one.
fn resolve_output_name(
    contract_label: &str,
    task: TaskType,
    declared: Option<&ContractOutput>,
    outputs: &[(String, Option<usize>)],
) -> Result<String, CoreError> {
    let (kind, defaults) = match task {
        TaskType::Binary | TaskType::Multiclass | TaskType::Multilabel => {
            ("probability", DEFAULT_PROBABILITY_OUTPUTS)
        }
        TaskType::Regression => ("prediction", DEFAULT_PREDICTION_OUTPUTS),
    };
    match declared.and_then(|o| o.name.clone()) {
        Some(name) if outputs.iter().any(|(n, _)| *n == name) => Ok(name),
        Some(name) => Err(CoreError::InvalidConfig(format!(
            "{contract_label} names output {name:?} but the model's outputs are {:?}",
            outputs.iter().map(|(n, _)| n).collect::<Vec<_>>()
        ))),
        None => match defaults
            .iter()
            .find(|d| outputs.iter().any(|(n, _)| n == **d))
        {
            Some(name) => Ok(name.to_string()),
            None => Err(CoreError::InvalidConfig(format!(
                "model has no {kind} output (looked for {defaults:?}; its outputs are {:?}). \
                 Set `output.name` in {contract_label}.",
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
                "positive class index {positive_class_index} (--positive-class-index) is out \
                 of range for a model with {n} output columns"
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
            "a positive class index (--positive-class-index) selects one column of a binary \
             model; this is a {task} model and every column is scored"
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

fn resolve_regression_spec(
    contract_label: &str,
    output_name: String,
    positive_class_index: Option<usize>,
    width: Option<usize>,
    classes: Option<Vec<String>>,
    thresholds: Option<Vec<f64>>,
) -> Result<OutputSpec, CoreError> {
    let invalid = |msg: String| Err(CoreError::InvalidConfig(msg));
    if positive_class_index.is_some() {
        return invalid(
            "a positive class index (--positive-class-index) selects one column of a binary \
             model; a regression model has one output, its predicted value"
                .to_string(),
        );
    }
    if thresholds.is_some() {
        return invalid(format!(
            "{contract_label} sets `output.thresholds`, which applies to multilabel models; \
             a regression model's flipper cutoff is --flip-threshold, in the target's units"
        ));
    }
    if let Some(w) = width.filter(|w| *w != 1) {
        return invalid(format!(
            "output {output_name:?} has {w} columns; Pasteur scores single-target \
             regression models, which predict one value per row"
        ));
    }
    let column = match classes.as_deref() {
        None => REGRESSION_OUTPUT_COLUMN.to_string(),
        Some([name]) => name.clone(),
        Some(names) => {
            return invalid(format!(
                "{contract_label} lists {} `output.classes` for a regression model; give one \
                 name for the target, or leave it out",
                names.len()
            ))
        }
    };
    Ok(OutputSpec {
        task: TaskType::Regression,
        output_name,
        positive_class_index: 0,
        columns: vec![column],
        thresholds: None,
    })
}

pub(crate) fn resolve_output_spec(
    contract_label: &str,
    contract: Option<&ModelContract>,
    positive_class_index: Option<usize>,
    outputs: &[(String, Option<usize>)],
) -> Result<OutputSpec, CoreError> {
    let task = contract.map(|c| c.task_type).unwrap_or_default();
    let declared = contract.map(|c| &c.output);

    let output_name = resolve_output_name(contract_label, task, declared, outputs)?;

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

    match task {
        TaskType::Binary => resolve_binary_spec(
            contract_label,
            task,
            output_name,
            positive_class_index,
            width,
            classes,
            thresholds,
        ),
        TaskType::Multiclass | TaskType::Multilabel => resolve_multi_spec(
            contract_label,
            task,
            output_name,
            positive_class_index,
            classes,
            thresholds,
        ),
        TaskType::Regression => resolve_regression_spec(
            contract_label,
            output_name,
            positive_class_index,
            width,
            classes,
            thresholds,
        ),
    }
}

/// Resolves the contract path: an explicit `--contract` wins; otherwise
/// `{model_dir}/metadata.json`.
pub(crate) fn resolve_contract_path(onnx_path: &Path, contract: Option<&Path>) -> Option<PathBuf> {
    contract
        .map(Path::to_path_buf)
        .or_else(|| onnx_path.parent().map(|dir| dir.join(CONTRACT_FILENAME)))
}

/// Reads a model contract sidecar. Absent is fine when the path was inferred;
/// an explicit path that does not exist is an error; present-but-malformed
/// is always an error — a contract nobody can parse is worse than no contract.
pub(crate) fn load_contract(
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

/// Refuse to score when the column order cannot be trusted.
///
/// Wrong *width* errors inside ORT anyway. Wrong *order* at the right width
/// does not: every patient gets scored against the wrong thresholds and the
/// run looks clean. That is the case this exists for.
pub(crate) fn validate_feature_order(
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

    fn regression_outputs(width: Option<usize>) -> Vec<(String, Option<usize>)> {
        vec![("variable".to_string(), width)]
    }

    fn resolve_regression(
        json: &str,
        pci: Option<usize>,
        width: Option<usize>,
    ) -> Result<OutputSpec, String> {
        resolve_output_spec(
            "metadata.json",
            Some(&contract(json)),
            pci,
            &regression_outputs(width),
        )
        .map_err(|e| e.to_string())
    }

    #[test]
    fn regression_defaults_to_the_variable_output() {
        let spec = resolve_regression(r#"{"task_type": "regression"}"#, None, Some(1)).unwrap();
        assert_eq!(spec.task, TaskType::Regression);
        assert_eq!(spec.output_name, "variable");
        assert_eq!(spec.columns, vec![REGRESSION_OUTPUT_COLUMN.to_string()]);
        let named = resolve_regression(
            r#"{"task_type": "regression", "output": {"classes": ["hba1c"]}}"#,
            None,
            Some(1),
        )
        .unwrap();
        assert_eq!(named.columns, vec!["hba1c".to_string()]);
    }

    #[test]
    fn regression_without_a_prediction_output_names_what_it_looked_for() {
        let err = resolve_output_spec(
            "metadata.json",
            Some(&contract(r#"{"task_type": "regression"}"#)),
            None,
            &outputs(Some(2)),
        )
        .unwrap_err()
        .to_string();
        assert!(
            err.contains("no prediction output") && err.contains("variable"),
            "{err}"
        );
    }

    #[test]
    fn regression_is_single_output_without_thresholds_or_positive_class() {
        let json = r#"{"task_type": "regression"}"#;
        let err = resolve_regression(json, None, Some(3)).unwrap_err();
        assert!(err.contains("3 columns"), "{err}");
        let err = resolve_regression(json, Some(1), Some(1)).unwrap_err();
        assert!(err.contains("--positive-class-index"), "{err}");
        let err = resolve_regression(
            r#"{"task_type": "regression", "output": {"thresholds": [6.5]}}"#,
            None,
            Some(1),
        )
        .unwrap_err();
        assert!(err.contains("--flip-threshold"), "{err}");
        let err = resolve_regression(
            r#"{"task_type": "regression", "output": {"classes": ["a", "b"]}}"#,
            None,
            None,
        )
        .unwrap_err();
        assert!(err.contains("2 `output.classes`"), "{err}");
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
}
