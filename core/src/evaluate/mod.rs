pub mod flipper;
mod metrics;

pub use flipper::{
    calculate_flipper_stability, find_flip_points, flipper_model_features, score_flipper_grid,
    FlipperPairResult,
};
pub use metrics::{calculate_jitter_stability, calculate_roc_auc, JITTER_STABILITY_ALPHA};

use crate::error::CoreError;
use crate::schema::{
    EvaluationConfig, EvaluationResult, EvaluationScores, MetricBasedScores, MetricInvariantScores,
    SimulationVariant,
};
use chrono::Utc;
use polars::prelude::*;
use std::collections::HashMap;

pub trait Model {
    fn predict_proba(&self, df: &DataFrame) -> Result<Series, CoreError>;
}

pub struct Evaluator;

impl Default for Evaluator {
    fn default() -> Self {
        Self
    }
}

impl Evaluator {
    pub fn new() -> Self {
        Self
    }

    pub fn run(
        &self,
        model: &dyn Model,
        clean_datasets: &HashMap<String, SimulationVariant>,
        simulation_variants: &HashMap<String, HashMap<String, SimulationVariant>>,
        config: &EvaluationConfig,
    ) -> Result<EvaluationResult, CoreError> {
        let mut per_dataset_scores = Vec::new();

        for (dataset_name, clean_dataset) in clean_datasets {
            let mut dataset_variants = HashMap::new();
            for (variant_name, named_variants) in simulation_variants {
                if variant_name == "clean" {
                    continue;
                }
                if let Some(variant) = named_variants.get(dataset_name) {
                    dataset_variants.insert(variant_name.clone(), variant.clone());
                }
            }

            let scores =
                self.run_single_dataset(model, clean_dataset, &dataset_variants, config)?;
            per_dataset_scores.push(scores);
        }

        // Aggregate scores (simplified)
        let jitter_stability = if !per_dataset_scores.is_empty() {
            per_dataset_scores
                .iter()
                .map(|s| s.jitter_stability)
                .sum::<f64>()
                / per_dataset_scores.len() as f64
        } else {
            1.0
        };

        let flipper_stability = if !per_dataset_scores.is_empty() {
            let scores: Vec<f64> = per_dataset_scores
                .iter()
                .filter_map(|s| s.flipper_stability)
                .collect();
            if scores.is_empty() {
                None
            } else {
                Some(scores.iter().sum::<f64>() / scores.len() as f64)
            }
        } else {
            None
        };

        let mut metric_based = HashMap::new();
        let mut baselines = HashMap::new();

        // Assume "roc_auc" for now
        let avg_resiliency = if !per_dataset_scores.is_empty() {
            per_dataset_scores.iter().map(|s| s.resiliency).sum::<f64>()
                / per_dataset_scores.len() as f64
        } else {
            1.0
        };

        let avg_baseline = if !per_dataset_scores.is_empty() {
            per_dataset_scores.iter().map(|s| s.baseline).sum::<f64>()
                / per_dataset_scores.len() as f64
        } else {
            0.0
        };

        metric_based.insert(
            "roc_auc".to_string(),
            MetricBasedScores {
                resiliency: avg_resiliency,
                generalizability: 1.0, // Simplified
            },
        );

        baselines.insert("roc_auc".to_string(), avg_baseline);

        Ok(EvaluationResult {
            baselines,
            evaluations: EvaluationScores {
                metric_invariant: MetricInvariantScores {
                    jitter_stability,
                    flipper_stability,
                },
                metric_based,
            },
            created_at: Utc::now(),
        })
    }

    fn run_single_dataset(
        &self,
        model: &dyn Model,
        clean_dataset: &SimulationVariant,
        simulated_variants: &HashMap<String, SimulationVariant>,
        config: &EvaluationConfig,
    ) -> Result<SingleDatasetScores, CoreError> {
        let probs_clean = model.predict_proba(&clean_dataset.x)?;
        let y_true = clean_dataset.y.clone();

        let baseline_roc_auc = calculate_roc_auc(&y_true, &probs_clean)?;

        let mut resiliency = 1.0;
        if let Some(blackout) = simulated_variants.get("blackout") {
            let probs_stressed = model.predict_proba(&blackout.x)?;
            let stressed_roc_auc = calculate_roc_auc(&y_true, &probs_stressed)?;
            if baseline_roc_auc > 0.0 {
                resiliency = stressed_roc_auc / baseline_roc_auc;
            }
        }

        let jitter_preds: Vec<Series> = simulated_variants
            .iter()
            .filter(|(name, _)| name.starts_with("jitter"))
            .map(|(_, variant)| model.predict_proba(&variant.x))
            .collect::<Result<_, _>>()?;
        let jitter_stability = calculate_jitter_stability(&jitter_preds, JITTER_STABILITY_ALPHA)?;

        let flipper_stability = if let Some(flipper) = simulated_variants.get("flipper") {
            Some(score_flipper_grid(
                model,
                &flipper.x,
                config.flip_threshold,
            )?)
        } else {
            None
        };

        Ok(SingleDatasetScores {
            baseline: baseline_roc_auc,
            resiliency,
            jitter_stability,
            flipper_stability,
        })
    }
}

struct SingleDatasetScores {
    baseline: f64,
    resiliency: f64,
    jitter_stability: f64,
    flipper_stability: Option<f64>,
}
