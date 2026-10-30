use std::path::Path;

use anyhow::Result;
use pasteur_core::{Model, TaskType};
use pasteur_model::OnnxModel;

use crate::args::PredictionsSource;
use crate::data::PrecomputedPredictions;

/// A model to score, ONNX or precomputed, with the name reports key it by
/// and its output column names in class order.
pub struct LoadedModel {
    pub label: String,
    pub output_columns: Vec<String>,
    pub model: Box<dyn Model>,
}

pub fn load_model(
    model_path: &Path,
    input_name: &str,
    feature_order: Vec<String>,
    positive_class_index: Option<usize>,
    null_fill: Option<f32>,
    contract: Option<&Path>,
    task: TaskType,
) -> Result<LoadedModel> {
    if !model_path.exists() {
        anyhow::bail!(
            "model not found at {} (download with `hf download` first)",
            model_path.display()
        );
    }
    let model = OnnxModel::from_file(
        model_path,
        input_name,
        feature_order,
        positive_class_index,
        null_fill,
        contract,
    )?;
    if model.task() != task {
        anyhow::bail!(
            "{} is a {} model (per its metadata.json task_type) but --task is {task}",
            model_path.display(),
            model.task()
        );
    }
    Ok(LoadedModel {
        label: model_label(model_path),
        output_columns: model.output_columns().to_vec(),
        model: Box::new(model),
    })
}

/// `n_outputs` is the number of label columns the predictions must match.
pub fn load_precomputed(
    source: &PredictionsSource,
    task: TaskType,
    n_outputs: usize,
) -> Result<LoadedModel> {
    if !source.path.exists() {
        anyhow::bail!("predictions not found at {}", source.path.display());
    }
    let model = PrecomputedPredictions::from_parquet(&source.path, task, n_outputs)?;
    Ok(LoadedModel {
        label: source
            .label
            .clone()
            .unwrap_or_else(|| model_label(&source.path)),
        output_columns: model.output_columns().to_vec(),
        model: Box::new(model),
    })
}

pub fn model_label(model_path: &Path) -> String {
    model_path
        .file_stem()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| model_path.display().to_string())
}
