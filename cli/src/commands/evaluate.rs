use anyhow::Result;
use pasteur_core::Evaluator;

use crate::args::EvaluateArgs;
use crate::data::{
    build_evaluation_inputs, flipper_feature_order_from_inputs, resolve_flip_threshold, LabelSource,
};
use crate::io::print_or_write;
use crate::model::{load_model, load_precomputed};

pub fn run(args: EvaluateArgs) -> Result<()> {
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

    let model = match (&args.model, args.predictions.predictions.as_slice()) {
        (Some(path), []) => load_model(
            path,
            &args.input_name,
            flipper_feature_order_from_inputs(&inputs, &args.sim_type),
            args.positive_class_index,
            args.null_fill,
            args.contract.as_deref(),
            task,
        )?,
        (None, [predictions]) => load_precomputed(
            predictions,
            task,
            inputs.clean_datasets[&inputs.dataset_name].y.width(),
        )?,
        _ => anyhow::bail!(
            "evaluate scores one model: pass --model or a single --predictions (use `compare` \
             for several)"
        ),
    };

    let evaluator = Evaluator::new();
    let result = evaluator.run(
        model.model.as_ref(),
        &inputs.clean_datasets,
        &inputs.simulation_variants,
        &inputs.eval_config,
    )?;

    print_or_write(&result, args.output.as_deref())
}
