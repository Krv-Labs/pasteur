use anyhow::Result;
use pasteur_core::{Model, SimulationVariant};
use pasteur_model::OnnxModel;
use polars::prelude::*;

use crate::data::all_agree::build_all_agree_column;
use crate::data::eval_inputs::{model_feature_frame, variant_source_row_ids, EvaluationInputs};

pub fn build_predictions_table(
    inputs: &EvaluationInputs,
    models: &[(String, OnnxModel)],
) -> Result<DataFrame> {
    let dataset_key = &inputs.dataset_name;
    let mut rows = PredictionRows::new(models.len());
    rows.append_variant("clean", &inputs.clean_datasets[dataset_key], inputs, models)?;
    for (name, per_dataset) in &inputs.simulation_variants {
        rows.append_variant(name, &per_dataset[dataset_key], inputs, models)?;
    }
    Ok(assemble_dataframe(rows, models))
}

struct PredictionRows {
    source_row_id: Vec<String>,
    variant_name: Vec<String>,
    label: Vec<i64>,
    per_model_preds: Vec<Vec<f64>>,
}

impl PredictionRows {
    fn new(n_models: usize) -> Self {
        Self {
            source_row_id: Vec::new(),
            variant_name: Vec::new(),
            label: Vec::new(),
            per_model_preds: vec![Vec::new(); n_models],
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
        let ids = variant_source_row_ids(variant, &inputs.source_row_id)?;
        let id_strings = ids.str()?;
        let labels_series = variant.y.cast(&DataType::Int64)?;
        let labels = labels_series.i64()?;
        let x = model_feature_frame(variant, &inputs.feature_order)?;

        for i in 0..n {
            self.source_row_id
                .push(id_strings.get(i).unwrap_or("").to_string());
            self.variant_name.push(name.to_string());
            self.label.push(labels.get(i).unwrap_or(0));
        }
        append_model_pred_rows(&mut self.per_model_preds, models, &x, n)?;
        Ok(())
    }
}

fn assemble_dataframe(rows: PredictionRows, models: &[(String, OnnxModel)]) -> DataFrame {
    let n_rows = rows.source_row_id.len();
    let mut columns: Vec<Column> = vec![
        Series::new("source_row_id".into(), rows.source_row_id).into(),
        Series::new("variant_name".into(), rows.variant_name).into(),
        Series::new("label".into(), rows.label).into(),
    ];
    let all_agree = build_all_agree_column(&rows.per_model_preds, n_rows);
    for ((model_label, _), preds) in models.iter().zip(rows.per_model_preds) {
        columns.push(Series::new(model_label.as_str().into(), preds).into());
    }
    columns.push(Series::new("all_agree".into(), all_agree).into());
    DataFrame::new(n_rows, columns).expect("prediction columns aligned")
}

fn append_model_pred_rows(
    per_model_preds: &mut [Vec<f64>],
    models: &[(String, OnnxModel)],
    x: &DataFrame,
    n: usize,
) -> Result<()> {
    for (mi, (_, model)) in models.iter().enumerate() {
        append_one_model_preds(&mut per_model_preds[mi], model, x, n)?;
    }
    Ok(())
}

fn append_one_model_preds(
    out: &mut Vec<f64>,
    model: &OnnxModel,
    x: &DataFrame,
    n: usize,
) -> Result<()> {
    let pred_series = model.predict_proba(x)?.cast(&DataType::Float64)?;
    let preds = pred_series.f64()?;
    for i in 0..n {
        out.push(preds.get(i).unwrap_or(f64::NAN));
    }
    Ok(())
}
