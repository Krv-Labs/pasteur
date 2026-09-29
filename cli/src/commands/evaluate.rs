use anyhow::Result;
use pasteur_core::Evaluator;

use crate::args::EvaluateArgs;
use crate::data::{build_evaluation_inputs, flipper_feature_order_from_inputs};
use crate::io::print_or_write;
use crate::model::load_model;

pub fn run(args: EvaluateArgs) -> Result<()> {
    let inputs = build_evaluation_inputs(
        &args.sim_root,
        &args.sim_type,
        &args.labels,
        args.positive_group_id,
        &args.dataset_name,
        args.flip_threshold,
    )?;

    let feature_order = flipper_feature_order_from_inputs(&inputs, &args.sim_type);
    let model = load_model(
        &args.model,
        &args.input_name,
        feature_order,
        args.positive_class_index,
        args.null_fill,
        args.contract.as_deref(),
    )?;

    let evaluator = Evaluator::new();
    let result = evaluator.run(
        &model,
        &inputs.clean_datasets,
        &inputs.simulation_variants,
        &inputs.eval_config,
    )?;

    print_or_write(&result, args.output.as_deref())
}
