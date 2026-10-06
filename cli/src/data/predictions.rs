use anyhow::Result;
use pasteur_core::{Model, SimulationVariant, TaskType};
use pasteur_model::OnnxModel;
use polars::prelude::*;

use crate::data::all_agree::build_all_agree_column;
use crate::data::eval_inputs::{model_feature_frame, variant_source_row_ids, EvaluationInputs};

pub fn build_predictions_table(
    inputs: &EvaluationInputs,
    models: &[(String, OnnxModel)],
) -> Result<DataFrame> {
    let dataset_key = &inputs.dataset_name;
    let mut rows = PredictionRows::new(models, inputs.eval_config.task);
    // NaN when a regression run was given no cutoff.
    let cutoff = Some(inputs.eval_config.flip_threshold).filter(|t| t.is_finite());
    rows.append_variant("clean", &inputs.clean_datasets[dataset_key], inputs, models)?;
    for (name, per_dataset) in &inputs.simulation_variants {
        rows.append_variant(name, &per_dataset[dataset_key], inputs, models)?;
    }
    Ok(assemble_dataframe(rows, models, cutoff))
}

struct PredictionRows {
    task: TaskType,
    /// Output column names shared by every model (checked at load).
    classes: Vec<String>,
    source_row_id: Vec<String>,
    variant_name: Vec<String>,
    /// `labels[column][row]`: 0/1 for classifiers, the true target for
    /// regression (`None` on synthetic flipper rows).
    labels: Vec<Vec<Option<f64>>>,
    /// `per_model_preds[model][column][row]`.
    per_model_preds: Vec<Vec<Vec<f64>>>,
}

impl PredictionRows {
    fn new(models: &[(String, OnnxModel)], task: TaskType) -> Self {
        let classes = models
            .first()
            .map(|(_, m)| m.output_columns().to_vec())
            .unwrap_or_default();
        Self {
            task,
            source_row_id: Vec::new(),
            variant_name: Vec::new(),
            labels: vec![Vec::new(); classes.len()],
            per_model_preds: vec![vec![Vec::new(); classes.len()]; models.len()],
            classes,
        }
    }

    fn append_variant(
        &mut self,
        name: &str,
        variant: &SimulationVariant,
        inputs: &EvaluationInputs,
        models: &[(String, OnnxModel)],
    ) -> Result<()> {
        let n = variant.x.height();
        let k = self.classes.len();
        if variant.y.width() != k {
            anyhow::bail!(
                "{name} has {} label columns for {k} model outputs",
                variant.y.width()
            );
        }
        let ids = variant_source_row_ids(variant, &inputs.source_row_id)?;
        let id_strings = ids.str()?;
        let x = model_feature_frame(variant, &inputs.feature_order)?;

        for i in 0..n {
            self.source_row_id
                .push(id_strings.get(i).unwrap_or("").to_string());
            self.variant_name.push(name.to_string());
        }
        for (out, col) in self.labels.iter_mut().zip(variant.y.columns()) {
            let col = col.cast(&DataType::Float64)?;
            let col = col.f64()?;
            out.extend((0..n).map(|i| col.get(i)));
        }
        for (preds, (_, model)) in self.per_model_preds.iter_mut().zip(models) {
            let frame = model.predict_proba(&x)?;
            if frame.width() != k {
                anyhow::bail!("model returned {} columns for {k} outputs", frame.width());
            }
            for (out, col) in preds.iter_mut().zip(frame.columns()) {
                let col = col.cast(&DataType::Float64)?;
                let col = col.f64()?;
                out.extend((0..n).map(|i| col.get(i).unwrap_or(f64::NAN)));
            }
        }
        Ok(())
    }
}

/// Binary keeps its original columns (`label`, one column per model).
/// Multi-output models get `label__<class>` and `<model>__<class>` per class.
/// Regression gets `target` and one prediction column per model.
fn assemble_dataframe(
    rows: PredictionRows,
    models: &[(String, OnnxModel)],
    cutoff: Option<f64>,
) -> DataFrame {
    let n_rows = rows.source_row_id.len();
    let single_column = match rows.task {
        TaskType::Binary | TaskType::Regression => true,
        TaskType::Multiclass | TaskType::Multilabel => false,
    };
    let suffix = |prefix: &str, class: &str| {
        if single_column {
            prefix.to_string()
        } else {
            format!("{prefix}__{class}")
        }
    };
    let mut columns: Vec<Column> = vec![
        Series::new("source_row_id".into(), rows.source_row_id).into(),
        Series::new("variant_name".into(), rows.variant_name).into(),
    ];
    for (class, labels) in rows.classes.iter().zip(rows.labels) {
        columns.push(label_column(rows.task, &suffix("label", class), labels).into());
    }
    let thresholds: Vec<Vec<f64>> = models
        .iter()
        .map(|(_, m)| match rows.task {
            TaskType::Binary | TaskType::Multiclass | TaskType::Multilabel => m
                .decision_thresholds()
                .unwrap_or_else(|| vec![0.5; rows.classes.len()]),
            TaskType::Regression => cutoff.into_iter().collect(),
        })
        .collect();
    let all_agree = match (rows.task, cutoff) {
        // Without a cutoff, regression predictions have no decision to agree on.
        (TaskType::Regression, None) => vec![None; n_rows],
        _ => build_all_agree_column(&rows.per_model_preds, &thresholds, rows.task, n_rows)
            .into_iter()
            .map(Some)
            .collect(),
    };
    for ((model_label, _), preds) in models.iter().zip(rows.per_model_preds) {
        for (class, col) in rows.classes.iter().zip(preds) {
            columns.push(Series::new(suffix(model_label, class).as_str().into(), col).into());
        }
    }
    columns.push(Series::new("all_agree".into(), all_agree).into());
    DataFrame::new(n_rows, columns).expect("prediction columns aligned")
}

/// Classifier labels stay `i64` 0/1, as before; regression writes the true
/// target as `target`.
fn label_column(task: TaskType, name: &str, labels: Vec<Option<f64>>) -> Series {
    match task {
        TaskType::Binary | TaskType::Multiclass | TaskType::Multilabel => {
            let labels: Vec<i64> = labels
                .into_iter()
                .map(|v| v.unwrap_or(0.0) as i64)
                .collect();
            Series::new(name.into(), labels)
        }
        TaskType::Regression => Series::new("target".into(), labels),
    }
}
