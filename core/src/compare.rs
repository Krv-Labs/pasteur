use crate::error::CoreError;
use crate::evaluate::{Evaluator, Model};
use crate::schema::{ComparisonResult, EvaluationConfig, LabeledEvaluation, SimulationVariant};
use std::collections::HashMap;

/// Runs the same `Evaluator` pass against any number of models over the same
/// clean/simulated datasets, so their scores can be read side by side —
/// e.g. a list of candidate ONNX models pulled from different HF repos.
pub struct Comparer;

impl Comparer {
    pub fn new() -> Self {
        Self
    }

    pub fn run(
        &self,
        models: &[(String, &dyn Model)],
        clean_datasets: &HashMap<String, SimulationVariant>,
        simulation_variants: &HashMap<String, HashMap<String, SimulationVariant>>,
        config: &EvaluationConfig,
        evaluator: &Evaluator,
    ) -> Result<ComparisonResult, CoreError> {
        let mut evaluations = Vec::with_capacity(models.len());
        for (model_label, model) in models {
            let evaluation = evaluator.run(*model, clean_datasets, simulation_variants, config)?;
            evaluations.push(LabeledEvaluation {
                model_label: model_label.clone(),
                evaluation,
            });
        }
        Ok(ComparisonResult { evaluations })
    }
}

impl Default for Comparer {
    fn default() -> Self {
        Self::new()
    }
}
