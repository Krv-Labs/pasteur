mod all_agree;
mod eval_inputs;
mod labels;
mod predictions;
mod source;
mod targets;

pub use eval_inputs::{build_evaluation_inputs, flipper_feature_order_from_inputs};
pub use labels::class_indices;
pub use predictions::build_predictions_table;
pub use source::{resolve_flip_threshold, LabelSource};
