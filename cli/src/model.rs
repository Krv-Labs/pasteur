use std::path::Path;

use anyhow::Result;
use pasteur_core::TaskType;
use pasteur_model::OnnxModel;

pub fn load_model(
    model_path: &Path,
    input_name: &str,
    feature_order: Vec<String>,
    positive_class_index: Option<usize>,
    null_fill: Option<f32>,
    contract: Option<&Path>,
    task: TaskType,
) -> Result<OnnxModel> {
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
    Ok(model)
}

pub fn model_label(model_path: &Path) -> String {
    model_path
        .file_stem()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| model_path.display().to_string())
}
