//! Loads each synthetic fixture from `tests/fixtures` (see `generate.py`)
//! through `OnnxModel` and checks the probabilities land in the contract's
//! column order, against what onnxruntime reported when they were generated.

use std::path::{Path, PathBuf};

use pasteur_core::{Model, TaskType};
use pasteur_model::OnnxModel;
use polars::prelude::*;
use serde_json::Value;

fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

fn expected() -> Value {
    let text = std::fs::read_to_string(fixtures().join("expected.json")).unwrap();
    serde_json::from_str(&text).unwrap()
}

fn probe() -> DataFrame {
    let rows = expected()["probe"].clone();
    let col = |j: usize| -> Vec<f64> {
        rows.as_array()
            .unwrap()
            .iter()
            .map(|r| r[j].as_f64().unwrap())
            .collect()
    };
    df!["f1" => col(0), "f2" => col(1)].unwrap()
}

/// Writes `contract` to its own temp file so tests do not share sidecars.
fn load(
    fixture: &str,
    contract: &str,
    positive_class_index: Option<usize>,
) -> Result<OnnxModel, String> {
    let dir = std::env::temp_dir().join(format!(
        "pasteur-onnx-outputs-{fixture}-{}-{}",
        std::process::id(),
        contract.len()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let contract_path = dir.join("metadata.json");
    std::fs::write(&contract_path, contract).unwrap();
    let model = OnnxModel::from_file(
        &fixtures().join(format!("{fixture}.onnx")),
        "input",
        vec!["f1".to_string(), "f2".to_string()],
        positive_class_index,
        None,
        Some(&contract_path),
    )
    .map_err(|e| e.to_string());
    let _ = std::fs::remove_dir_all(&dir);
    model
}

/// `expected[row][class]`, reading ZipMap rows by `classes` name.
fn load_err(fixture: &str, contract: &str, positive_class_index: Option<usize>) -> String {
    match load(fixture, contract, positive_class_index) {
        Ok(_) => panic!("{fixture} loaded with contract {contract}"),
        Err(e) => e,
    }
}

fn expected_matrix(fixture: &str, classes: &[&str]) -> Vec<Vec<f64>> {
    expected()[fixture]["probabilities"]
        .as_array()
        .unwrap()
        .iter()
        .map(|row| match row {
            Value::Object(m) => classes.iter().map(|c| m[*c].as_f64().unwrap()).collect(),
            Value::Array(a) => a.iter().map(|v| v.as_f64().unwrap()).collect(),
            other => panic!("unexpected row {other}"),
        })
        .collect()
}

fn assert_probabilities(model: &OnnxModel, fixture: &str, classes: &[&str]) {
    let got = model.predict_proba(&probe()).unwrap();
    let names: Vec<String> = got
        .get_column_names()
        .iter()
        .map(|s| s.to_string())
        .collect();
    assert_eq!(names, classes);
    let want = expected_matrix(fixture, classes);
    for (j, col) in got.columns().iter().enumerate() {
        let col = col.f64().unwrap();
        for (i, row) in want.iter().enumerate() {
            let g = col.get(i).unwrap();
            assert!(
                (g - row[j]).abs() < 1e-6,
                "{fixture} row {i} class {}: got {g}, want {}",
                classes[j],
                row[j]
            );
        }
    }
}

#[test]
fn binary_without_contract_fields_still_scores_the_positive_column() {
    let model = load("binary_zipmap", "{}", None).unwrap();
    assert_eq!(model.task(), TaskType::Binary);
    let got = model.predict_proba(&probe()).unwrap();
    assert_eq!(got.get_column_names(), ["proba"]);
    let want = expected_matrix("binary_zipmap", &["0", "1"]);
    let col = got.column("proba").unwrap().f64().unwrap();
    for (i, row) in want.iter().enumerate() {
        assert!((col.get(i).unwrap() - row[1]).abs() < 1e-6);
    }
}

#[test]
fn multiclass_tensor_output() {
    let contract = r#"{"task_type": "multiclass", "output": {"classes": ["c0", "c1", "c2"]}}"#;
    let model = load("multiclass_tensor", contract, None).unwrap();
    assert_probabilities(&model, "multiclass_tensor", &["c0", "c1", "c2"]);
}

#[test]
fn multiclass_zipmap_int_keys_match_by_name() {
    let contract = r#"{"task_type": "multiclass", "output": {"classes": ["0", "1", "2"]}}"#;
    let model = load("multiclass_zipmap_int", contract, None).unwrap();
    assert_probabilities(&model, "multiclass_zipmap_int", &["0", "1", "2"]);
}

#[test]
fn multiclass_zipmap_string_keys_follow_the_contract_order() {
    // ZipMap emits keys sorted (high, low, mid); the contract's order wins.
    let contract = r#"{"task_type": "multiclass", "output": {"classes": ["low", "mid", "high"]}}"#;
    let model = load("multiclass_zipmap_str", contract, None).unwrap();
    assert_probabilities(&model, "multiclass_zipmap_str", &["low", "mid", "high"]);
}

#[test]
fn multiclass_zipmap_key_missing_from_contract_is_an_error() {
    let contract =
        r#"{"task_type": "multiclass", "output": {"classes": ["low", "mid", "elevated"]}}"#;
    let model = load("multiclass_zipmap_str", contract, None).unwrap();
    let err = model.predict_proba(&probe()).unwrap_err().to_string();
    assert!(
        err.contains("\"high\" is not in output.classes"),
        "got {err}"
    );
}

#[test]
fn multilabel_multioutput_sequence_of_tensors() {
    let contract = r#"{"task_type": "multilabel", "output": {"classes": ["a", "b", "c"]}}"#;
    let model = load("multilabel_multioutput", contract, None).unwrap();
    assert_probabilities(&model, "multilabel_multioutput", &["a", "b", "c"]);
}

#[test]
fn multilabel_tensor_output_with_thresholds() {
    let contract = r#"{"task_type": "multilabel",
        "output": {"classes": ["a", "b", "c"], "thresholds": [0.5, 0.3, 0.7]}}"#;
    let model = load("multilabel_tensor", contract, None).unwrap();
    assert_probabilities(&model, "multilabel_tensor", &["a", "b", "c"]);
    assert_eq!(model.decision_thresholds(), Some(vec![0.5, 0.3, 0.7]));
}

#[test]
fn class_count_disagreeing_with_the_graph_fails_at_load() {
    let contract = r#"{"task_type": "multiclass", "output": {"classes": ["a", "b"]}}"#;
    let err = load_err("multiclass_tensor", contract, None);
    assert!(err.contains("lists 2 classes but output"), "got {err}");
}

#[test]
fn independent_sigmoids_declared_multiclass_are_refused() {
    // Same width as a 3-class head, so only the row sums give it away.
    let contract = r#"{"task_type": "multiclass", "output": {"classes": ["a", "b", "c"]}}"#;
    let model = load("multilabel_tensor", contract, None).unwrap();
    let err = model.predict_proba(&probe()).unwrap_err().to_string();
    assert!(err.contains("task_type: multilabel"), "got {err}");
}

#[test]
fn multi_output_models_reject_positive_class_index() {
    let contract = r#"{"task_type": "multiclass", "output": {"classes": ["a", "b", "c"]}}"#;
    let err = load_err("multiclass_tensor", contract, Some(1));
    assert!(err.contains("--positive-class-index"), "got {err}");
}

#[test]
fn unknown_output_name_lists_the_real_ones() {
    let contract = r#"{"task_type": "multiclass",
        "output": {"name": "probs", "classes": ["a", "b", "c"]}}"#;
    let err = load_err("multiclass_tensor", contract, None);
    assert!(
        err.contains("\"label\"") && err.contains("\"probabilities\""),
        "got {err}"
    );
}

#[test]
fn unknown_task_type_names_the_valid_ones() {
    let err = load_err("multiclass_tensor", r#"{"task_type": "multi-label"}"#, None);
    assert!(
        err.contains("binary") && err.contains("multiclass") && err.contains("multilabel"),
        "got {err}"
    );
}
