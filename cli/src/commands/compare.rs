use anyhow::Result;
use pasteur_core::{Comparer, Evaluator, Model};
use pasteur_model::OnnxModel;

use crate::args::CompareArgs;
use crate::data::{
    build_evaluation_inputs, build_predictions_table, flipper_feature_order_from_inputs,
    resolve_flip_threshold, LabelSource,
};
use crate::io::{print_or_write, write_predictions_parquet};
use crate::model::{load_model, model_label};

pub fn run(args: CompareArgs) -> Result<()> {
    let task = args.task.into();
    let source = LabelSource::required(
        task,
        args.labels.as_deref(),
        &args.positive_group_ids,
        &args.targets,
    )?;
    let inputs = build_evaluation_inputs(
        &args.sim_root,
        &args.sim_type,
        &source,
        task,
        &args.dataset_name,
        resolve_flip_threshold(task, &args.sim_type, args.flip_threshold)?,
    )?;

    let feature_order = flipper_feature_order_from_inputs(&inputs, &args.sim_type);
    let models = load_models(&args, &feature_order)?;
    let model_refs: Vec<(String, &dyn Model)> = models
        .iter()
        .map(|(label, model)| (label.clone(), model as &dyn Model))
        .collect();

    let evaluator = Evaluator::new();
    let comparer = Comparer::new();
    let result = comparer.run(
        &model_refs,
        &inputs.clean_datasets,
        &inputs.simulation_variants,
        &inputs.eval_config,
        &evaluator,
    )?;
    print_or_write(&result, args.output.as_deref())?;

    if let Some(out_path) = &args.predictions_out {
        let predictions = build_predictions_table(&inputs, &models)?;
        write_predictions_parquet(predictions, out_path)?;
    }

    Ok(())
}

fn load_models(args: &CompareArgs, feature_order: &[String]) -> Result<Vec<(String, OnnxModel)>> {
    let mut models: Vec<(String, OnnxModel)> = Vec::with_capacity(args.models.len());
    for model_path in &args.models {
        let label = model_label(model_path);
        let model = load_model(
            model_path,
            &args.input_name,
            feature_order.to_vec(),
            args.positive_class_index,
            args.null_fill,
            args.contract.as_deref(),
            args.task.into(),
        )?;
        check_comparable(&models, &label, &model)?;
        models.push((label, model));
    }
    Ok(models)
}

/// Results and prediction columns are keyed by model file name and read in
/// one class order, so a model joining the comparison must have a new name
/// and the same output columns as the models already loaded.
fn check_comparable(loaded: &[(String, OnnxModel)], label: &str, model: &OnnxModel) -> Result<()> {
    if loaded.iter().any(|(existing, _)| existing == label) {
        anyhow::bail!(
            "two --model files are both named {label:?}; per-model columns and results are \
             keyed by file name, so rename one"
        );
    }
    match loaded.first() {
        Some((first_label, first)) if model.output_columns() != first.output_columns() => {
            anyhow::bail!(
                "{label} outputs {:?} but {first_label} outputs {:?}; compared models must \
                 share one class order",
                model.output_columns(),
                first.output_columns()
            )
        }
        _ => Ok(()),
    }
}
