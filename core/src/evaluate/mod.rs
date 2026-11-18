pub mod flipper;
mod metrics;
mod multi;
pub mod oracle;
mod regression;
mod stress;
pub mod sweep;

pub use stress::StressScores;

pub use flipper::{
    calculate_flipper_stability, find_flip_points, flipper_model_features, score_flipper,
    score_flipper_grid, FlipperPairResult, FlipperScores,
};
pub use metrics::{
    calculate_jitter_stability, calculate_roc_auc, mean_jitter_variance, parse_loss, roc_auc,
    JITTER_STABILITY_ALPHA,
};
pub use oracle::{OracleConfig, OracleScores};
pub use sweep::{score_sweep, sweep_scores, SweepScores};

use crate::error::CoreError;
use crate::schema::{
    EvaluationConfig, EvaluationResult, EvaluationScores, MetricBasedScores, MetricInvariantScores,
    MultiOutputScores, RegressionScores, SimulationVariant, TaskType,
};
use chrono::Utc;
use metrics::{jitter_stability_from_variance, series_to_f64};
use multi::{
    average_multi, decision_thresholds, headline_jitter_variance, jitter_variances, label_matrix,
    label_resiliency, mean_some, multi_output_scores, per_label_auc, Columns, LabelBreakdown,
};
use polars::prelude::*;
use std::collections::{BTreeMap, HashMap};

pub trait Model {
    /// One `Float64` column per output, in the model's class order: a single
    /// positive-class column for binary models, K columns for multiclass and
    /// multilabel models. Column names label the outputs in reports.
    fn predict_proba(&self, df: &DataFrame) -> Result<DataFrame, CoreError>;

    /// Per-label decision thresholds for multilabel models (from the model's
    /// contract). `None` falls back to `EvaluationConfig::flip_threshold`.
    fn decision_thresholds(&self) -> Option<Vec<f64>> {
        None
    }
}

/// Predicts and returns one `Vec<f64>` per output column (nulls as NaN),
/// refusing a model whose width or height does not match what the labels
/// say it should be — a 3-class model scored against 2 label groups would
/// otherwise pair every column with the wrong label.
pub(crate) fn predict_matrix(
    model: &dyn Model,
    x: &DataFrame,
    n_outputs: usize,
) -> Result<Columns, CoreError> {
    predict_named(model, x, n_outputs).map(|(_, cols)| cols)
}

/// `predict_matrix`, plus the model's output column names.
pub(crate) fn predict_named(
    model: &dyn Model,
    x: &DataFrame,
    n_outputs: usize,
) -> Result<(Vec<String>, Columns), CoreError> {
    let probs = model.predict_proba(x)?;
    if probs.width() != n_outputs {
        return Err(CoreError::InvalidConfig(format!(
            "model returns {} output column(s) but the labels have {n_outputs}; pass one \
             label group per model output, in the model's class order",
            probs.width()
        )));
    }
    if probs.height() != x.height() {
        return Err(CoreError::Generic(format!(
            "model returned {} rows for {} input rows",
            probs.height(),
            x.height()
        )));
    }
    let names = probs
        .get_column_names()
        .iter()
        .map(|s| s.to_string())
        .collect();
    let cols = probs
        .columns()
        .iter()
        .map(|c| series_to_f64(c.as_materialized_series()))
        .collect::<Result<_, _>>()?;
    Ok((names, cols))
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
        let mut parse_losses: BTreeMap<String, BTreeMap<String, usize>> = BTreeMap::new();

        for (dataset_name, clean_dataset) in clean_datasets {
            let dataset_variants: HashMap<String, SimulationVariant> = simulation_variants
                .iter()
                .filter(|(variant_name, _)| *variant_name != "clean")
                .filter_map(|(variant_name, named)| {
                    let variant = named.get(dataset_name)?;
                    Some((variant_name.clone(), variant.clone()))
                })
                .collect();
            // Only the stress variant can carry re-encoded text; flipper grids
            // keep id columns like `pair_0_step_3` that would count as lost.
            // Loss clean already has (a text column like `site`) is the data's,
            // not the variant's, so only the excess is reported.
            let clean_loss = parse_loss(&clean_dataset.x);
            for (variant_name, variant) in stress_variants(&dataset_variants)? {
                for (column, lost) in parse_loss(&variant.x) {
                    let extra = lost.saturating_sub(clean_loss.get(&column).copied().unwrap_or(0));
                    if extra == 0 {
                        continue;
                    }
                    *parse_losses
                        .entry(variant_name.to_string())
                        .or_default()
                        .entry(column)
                        .or_default() += extra;
                }
            }
            per_dataset_scores.push(self.run_single_dataset(
                model,
                clean_dataset,
                &dataset_variants,
                config,
            )?);
        }

        // Each headline number is the mean over datasets where it is defined.
        let mean = |f: &dyn Fn(&SingleDatasetScores) -> Option<f64>| {
            mean_some(per_dataset_scores.iter().map(f))
        };
        let jitter_stability = mean(&|s| Some(s.jitter_stability)).unwrap_or(1.0);
        let flipper_stability = mean(&|s| s.flipper_stability);
        let resiliency = mean(&|s| Some(s.resiliency)).unwrap_or(1.0);
        // Defined only when site draws were scored; neutral otherwise.
        let generalizability = mean(&|s| s.generalizability).unwrap_or(1.0);
        let (metric, keys) = baseline_keys(config.task);
        let baselines = keys
            .iter()
            .enumerate()
            .map(|(j, key)| {
                let value = mean(&|s| s.baselines.get(j).copied()).unwrap_or(0.0);
                (key.to_string(), value)
            })
            .collect();
        // Sweep and oracle detail is per dataset and not averaged, so
        // more than one dataset with it is refused rather than one picked in
        // HashMap order. The CLI scores one dataset; add a per-dataset map if
        // a library caller needs several.
        for (what, n) in [
            (
                "sweep",
                per_dataset_scores
                    .iter()
                    .filter(|s| s.sweep.is_some())
                    .count(),
            ),
            (
                "oracle",
                per_dataset_scores
                    .iter()
                    .filter(|s| s.oracle.is_some())
                    .count(),
            ),
        ] {
            if n > 1 {
                return Err(CoreError::InvalidConfig(format!(
                    "{what} results are reported for one dataset, but {n} datasets have them; \
                     evaluate each dataset separately"
                )));
            }
        }
        let sweep = per_dataset_scores.iter_mut().find_map(|s| s.sweep.take());
        let oracle = per_dataset_scores.iter_mut().find_map(|s| s.oracle.take());
        // Row counts from different datasets do not add up to one answer, so
        // `stress` is reported only when a single dataset has it.
        let with_stress = per_dataset_scores
            .iter()
            .filter(|s| s.stress.is_some())
            .count();
        let stress = per_dataset_scores
            .iter_mut()
            .find_map(|s| s.stress.take())
            .filter(|_| with_stress == 1);
        let (multi, regression): (Vec<_>, Vec<_>) = per_dataset_scores
            .into_iter()
            .map(|s| (s.multi, s.regression))
            .unzip();
        let multi = average_multi(multi.into_iter().flatten().collect());
        let regression = regression::average_regression(regression.into_iter().flatten().collect());

        let metric_based = HashMap::from([(
            metric.to_string(),
            MetricBasedScores {
                resiliency,
                generalizability,
            },
        )]);

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
            multi,
            regression,
            oracle,
            sweep,
            parse_loss: (!parse_losses.is_empty()).then_some(parse_losses),
            stress,
        })
    }

    fn run_single_dataset(
        &self,
        model: &dyn Model,
        clean_dataset: &SimulationVariant,
        simulated_variants: &HashMap<String, SimulationVariant>,
        config: &EvaluationConfig,
    ) -> Result<SingleDatasetScores, CoreError> {
        match config.task {
            TaskType::Binary | TaskType::Multiclass | TaskType::Multilabel => {
                self.score_classifier(model, clean_dataset, simulated_variants, config)
            }
            TaskType::Regression => {
                let scored = regression::score_dataset(
                    model,
                    clean_dataset,
                    simulated_variants,
                    config.flip_threshold,
                )?;
                Ok(SingleDatasetScores {
                    baselines: scored.baselines.to_vec(),
                    resiliency: scored.resiliency.unwrap_or(1.0),
                    jitter_stability: jitter_stability_from_variance(scored.jitter_variance),
                    flipper_stability: scored.flipper.map(|f| f.stability),
                    multi: None,
                    regression: Some(scored.detail),
                    sweep: None,
                    oracle: None,
                    generalizability: None,
                    stress: None,
                })
            }
        }
    }

    fn score_classifier(
        &self,
        model: &dyn Model,
        clean_dataset: &SimulationVariant,
        simulated_variants: &HashMap<String, SimulationVariant>,
        config: &EvaluationConfig,
    ) -> Result<SingleDatasetScores, CoreError> {
        let task = config.task;
        let y = label_matrix(&clean_dataset.y, task)?;
        let k = y.len();
        let (names, probs) = predict_named(model, &clean_dataset.x, k)?;
        let thresholds = decision_thresholds(model, task, k, config.flip_threshold)?;

        let auc = per_label_auc(&y, &probs);
        // Binary keeps its documented 0.5-for-one-class fallback; multi-output
        // averages only the labels where AUC is defined.
        let baseline = mean_some(auc.iter().copied()).unwrap_or(0.5);

        let mut variant_probs: HashMap<String, Columns> = HashMap::new();
        // One stress variant, or several site draws. Each is scored on its
        // own; per-label resiliency is the mean over them.
        let abstain = config
            .oracle
            .as_ref()
            .and_then(|o| o.abstain_class.as_deref())
            .map(|class| oracle::abstain_index(task, &names, class))
            .transpose()?;
        let clean_decisions = oracle::decisions(task, &probs, &thresholds);
        let mut per_variant: Vec<Vec<Option<f64>>> = Vec::new();
        let mut stress_scores: Vec<StressScores> = Vec::new();
        for (name, variant) in stress_variants(simulated_variants)? {
            let p = predict_matrix(model, &variant.x, k)?;
            let stressed = per_label_auc(&y, &p);
            let label_scores = label_resiliency(task, &auc, &stressed);
            let changed = stress::changed_rows(&clean_dataset.x, &variant.x)?;
            let rows_changed = changed.iter().filter(|&&c| c).count();
            let (decisions_changed, silent) = stress::decision_changes(
                &changed,
                &clean_decisions,
                &oracle::decisions(task, &p, &thresholds),
                abstain,
            );
            let rate = |n: usize| (rows_changed > 0).then(|| n as f64 / rows_changed as f64);
            stress_scores.push(StressScores {
                variant: name.to_string(),
                roc_auc: mean_some(stressed.iter().copied()),
                resiliency: mean_some(label_scores.iter().copied()),
                rows: changed.len(),
                rows_changed,
                decisions_changed,
                decision_change_rate: rate(decisions_changed),
                silent_change_rate: silent.and_then(rate),
            });
            variant_probs.insert(name.to_string(), p);
            per_variant.push(label_scores);
        }
        let resiliency: Vec<Option<f64>> = (0..k)
            .map(|j| mean_some(per_variant.iter().map(|scores| scores[j])))
            .collect();
        // The worst simulated site, not the average one: a mean over sites
        // hides the site the model fails at.
        let generalizability = stress_scores
            .iter()
            .filter(|s| is_site_draw(&s.variant))
            .filter_map(|s| s.resiliency)
            .reduce(f64::min);

        let mut jitter_draws: Vec<Columns> = Vec::new();
        for (name, variant) in simulated_variants
            .iter()
            .filter(|(name, _)| name.starts_with("jitter"))
        {
            let p = predict_matrix(model, &variant.x, k)?;
            if config.oracle.is_some() {
                variant_probs.insert(name.clone(), p.clone());
            }
            jitter_draws.push(p);
        }
        let jitter_variance = jitter_variances(&jitter_draws, k)?;

        // Flipper and sweep predictions are kept for oracle scoring too, so
        // each grid is predicted once.
        let flipper = match simulated_variants.get("flipper") {
            Some(flipper) => {
                let p = predict_matrix(model, &flipper_model_features(&flipper.x)?, k)?;
                if config.oracle.is_some() {
                    variant_probs.insert("flipper".to_string(), p.clone());
                }
                Some(flipper::score_flipper_probs(
                    &flipper.x,
                    task,
                    k,
                    &thresholds,
                    p,
                )?)
            }
            None => None,
        };

        // With `--oracle-labels` every variant's `y` is the oracle's, grid
        // rows included; without it sweep is scored descriptively.
        let sweep = match simulated_variants.get("sweep") {
            Some(grid) => {
                let p = predict_matrix(model, &grid.x, k)?;
                let oracle_y = config.oracle.as_ref().map(|_| &grid.y);
                let scores = sweep::score_sweep_probs(&grid.x, task, &thresholds, oracle_y, &p)?;
                if config.oracle.is_some() {
                    variant_probs.insert("sweep".to_string(), p);
                }
                Some(scores)
            }
            None => None,
        };
        let oracle = oracle::score_oracle(
            model,
            clean_dataset,
            simulated_variants,
            config,
            &names,
            &thresholds,
            Some(&probs),
            &variant_probs,
        )?;

        let multi = (!task.is_binary()).then(|| {
            let labels = LabelBreakdown {
                names: &names,
                y: &y,
                probs: &probs,
                auc: &auc,
                resiliency: &resiliency,
                jitter_variance: &jitter_variance,
                jitter_draws: &jitter_draws,
                thresholds: &thresholds,
            };
            multi_output_scores(task, &labels, flipper.as_ref())
        });

        Ok(SingleDatasetScores {
            baselines: vec![baseline],
            resiliency: mean_some(resiliency.iter().copied()).unwrap_or(1.0),
            jitter_stability: jitter_stability_from_variance(headline_jitter_variance(
                task,
                &jitter_variance,
            )),
            flipper_stability: flipper.map(|f| f.stability),
            multi,
            regression: None,
            sweep,
            oracle,
            generalizability,
            stress: (!stress_scores.is_empty()).then_some(stress_scores),
        })
    }
}

/// A random site draw, `site_<i>`, as `simulate --site-draws` names them.
fn is_site_draw(name: &str) -> bool {
    name.strip_prefix("site_")
        .is_some_and(|i| !i.is_empty() && i.bytes().all(|b| b.is_ascii_digit()))
}

/// The variants resiliency is measured on: whichever are not clean, jitter,
/// flipper or sweep (blackout, scramble, units or site). Evaluation loads one
/// simulation type at a time, so there is at most one, except for site draws
/// (`site_0`, `site_1`, ...), which are returned in draw order. Any other
/// group of several (say, two parquet files in `scramble/`) is an error
/// rather than a hash-order pick.
pub(crate) fn stress_variants(
    variants: &HashMap<String, SimulationVariant>,
) -> Result<Vec<(&str, &SimulationVariant)>, CoreError> {
    let mut stress: Vec<(&str, &SimulationVariant)> = variants
        .iter()
        .map(|(name, variant)| (name.as_str(), variant))
        .filter(|(name, _)| {
            *name != "clean"
                && *name != "flipper"
                && !name.starts_with("jitter")
                && !name.starts_with("sweep")
        })
        .collect();
    // By length first so `site_2` sorts before `site_10`.
    stress.sort_unstable_by_key(|(name, _)| (name.len(), *name));
    if stress.len() > 1 && !stress.iter().all(|(name, _)| is_site_draw(name)) {
        let mut names: Vec<&str> = stress.iter().map(|(name, _)| *name).collect();
        names.sort_unstable();
        return Err(CoreError::InvalidConfig(format!(
            "expected one stress variant (blackout, scramble, units or site) or several site \
             draws (site_0, site_1, ...), found {names:?}"
        )));
    }
    Ok(stress)
}

/// `stress_variants` for scoring that takes one variant: several site draws
/// are refused.
pub(crate) fn stress_variant(
    variants: &HashMap<String, SimulationVariant>,
) -> Result<Option<(&str, &SimulationVariant)>, CoreError> {
    let mut stress = stress_variants(variants)?;
    if stress.len() > 1 {
        return Err(CoreError::InvalidConfig(format!(
            "{} site draws were loaded, and several draws are scored for classifiers only; \
             evaluate a regression model on one site at a time (`simulate --site` without \
             `--site-draws`)",
            stress.len()
        )));
    }
    Ok(stress.pop())
}

/// The metric that `metric_based` is keyed by, and the `baselines` keys, in
/// the order `SingleDatasetScores::baselines` holds them.
fn baseline_keys(task: TaskType) -> (&'static str, &'static [&'static str]) {
    match task {
        TaskType::Binary | TaskType::Multiclass | TaskType::Multilabel => ("roc_auc", &["roc_auc"]),
        TaskType::Regression => ("r2", &regression::BASELINE_KEYS),
    }
}

struct SingleDatasetScores {
    /// One value per `baseline_keys(task)` entry.
    baselines: Vec<f64>,
    resiliency: f64,
    jitter_stability: f64,
    flipper_stability: Option<f64>,
    multi: Option<MultiOutputScores>,
    regression: Option<RegressionScores>,
    sweep: Option<Vec<SweepScores>>,
    oracle: Option<OracleScores>,
    /// Worst site draw's resiliency; `None` unless site draws were scored.
    generalizability: Option<f64>,
    stress: Option<Vec<StressScores>>,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Scores `val` directly, the way an ONNX model casts it: leniently,
    /// unparseable text becoming missing (here 0.0).
    struct ValModel;
    impl Model for ValModel {
        fn predict_proba(&self, df: &DataFrame) -> Result<DataFrame, CoreError> {
            let v = df.column("val")?.cast(&DataType::Float64)?;
            let p: Vec<f64> = v.f64()?.iter().map(|x| x.unwrap_or(0.0)).collect();
            Ok(df!["p" => p]?)
        }
    }

    fn run_with(name: &str, x: DataFrame) -> EvaluationResult {
        run_against(df!["val" => [0.1, 0.8, 0.2, 0.9]].unwrap(), name, x)
    }

    fn run_against(clean_x: DataFrame, name: &str, x: DataFrame) -> EvaluationResult {
        run_variants(clean_x, vec![(name, x)])
    }

    fn run_variants(clean_x: DataFrame, variants: Vec<(&str, DataFrame)>) -> EvaluationResult {
        let y = df!["target" => [0, 1, 0, 1]].unwrap();
        let clean = SimulationVariant {
            x: clean_x,
            y: y.clone(),
        };
        let config = EvaluationConfig {
            n_jitter_iters: 0,
            allow_partial: false,
            metrics: vec!["roc_auc".into()],
            flip_threshold: 0.5,
            task: TaskType::Binary,
            oracle: None,
        };
        Evaluator::new()
            .run(
                &ValModel,
                &HashMap::from([("d".to_string(), clean)]),
                &variants
                    .into_iter()
                    .map(|(name, x)| {
                        let variant = SimulationVariant { x, y: y.clone() };
                        (
                            name.to_string(),
                            HashMap::from([("d".to_string(), variant)]),
                        )
                    })
                    .collect(),
                &config,
            )
            .unwrap()
    }

    #[test]
    fn stress_section_counts_changed_rows_and_decisions() {
        let x = df!["val" => ["0.1", "ND", "0.2", "<0.01"]].unwrap();
        let r = run_with("scramble", x);
        let stress = r.stress.unwrap();
        assert_eq!(stress.len(), 1);
        let s = &stress[0];
        assert_eq!(
            (s.variant.as_str(), s.rows, s.rows_changed),
            ("scramble", 4, 2)
        );
        // Both positives lost their value and fell under the threshold.
        assert_eq!(s.decisions_changed, 2);
        assert_eq!(s.decision_change_rate, Some(1.0));
        assert_eq!(s.silent_change_rate, None);
        // One stress variant is not a set of sites.
        assert_eq!(r.evaluations.metric_based["roc_auc"].generalizability, 1.0);
    }

    #[test]
    fn identity_variant_changes_no_row() {
        let r = run_with("units", df!["val" => [0.1, 0.8, 0.2, 0.9]].unwrap());
        let s = &r.stress.as_ref().unwrap()[0];
        assert_eq!((s.rows_changed, s.decision_change_rate), (0, None));
        assert_eq!(resiliency(&r), 1.0);
    }

    #[test]
    fn generalizability_is_the_worst_site_draw() {
        let clean = || df!["val" => [0.1, 0.8, 0.2, 0.9]].unwrap();
        // site_1 loses one positive: it scores 0.0, under both negatives.
        let worse = df!["val" => [Some(0.1), None, Some(0.2), Some(0.9)]].unwrap();
        let r = run_variants(clean(), vec![("site_1", worse), ("site_0", clean())]);
        let scores = &r.evaluations.metric_based["roc_auc"];
        assert_eq!(scores.generalizability, 0.5);
        assert_eq!(scores.resiliency, 0.75);
        let stress = r.stress.unwrap();
        let names: Vec<&str> = stress.iter().map(|s| s.variant.as_str()).collect();
        assert_eq!(names, ["site_0", "site_1"]);
        assert_eq!(stress[1].resiliency, Some(0.5));
        assert_eq!(
            (stress[1].rows_changed, stress[1].decisions_changed),
            (1, 1)
        );
    }

    #[test]
    fn site_draws_sort_numerically_and_do_not_mix_with_other_variants() {
        let v = || SimulationVariant {
            x: df!["val" => [1.0]].unwrap(),
            y: df!["target" => [0]].unwrap(),
        };
        let mut variants: HashMap<String, SimulationVariant> = ["site_10", "site_2", "site_0"]
            .map(|n| (n.to_string(), v()))
            .into();
        let names: Vec<&str> = stress_variants(&variants)
            .unwrap()
            .into_iter()
            .map(|(name, _)| name)
            .collect();
        assert_eq!(names, ["site_0", "site_2", "site_10"]);
        assert!(stress_variant(&variants)
            .unwrap_err()
            .to_string()
            .contains("3 site draws"));

        variants.insert("scramble".into(), v());
        assert!(stress_variants(&variants).is_err());
    }

    fn resiliency(r: &EvaluationResult) -> f64 {
        r.evaluations.metric_based["roc_auc"].resiliency
    }

    #[test]
    fn scramble_variant_drives_resiliency_and_reports_parse_loss() {
        let x = df!["val" => ["0.1", "ND", "0.2", "<0.01"]].unwrap();
        let r = run_with("scramble", x);
        assert!(resiliency(&r) < 1.0);
        let expected = BTreeMap::from([(
            "scramble".to_string(),
            BTreeMap::from([("val".to_string(), 2)]),
        )]);
        assert_eq!(r.parse_loss, Some(expected));
    }

    #[test]
    fn blackout_run_has_no_parse_loss() {
        let x = df!["val" => [Some(0.1), None, Some(0.2), None]].unwrap();
        let r = run_with("blackout", x);
        assert!(resiliency(&r) < 1.0);
        assert_eq!(r.parse_loss, None);
    }

    #[test]
    fn parse_loss_already_in_clean_is_not_reported() {
        let site = ["A"; 4];
        let clean = || df!["val" => [0.1, 0.8, 0.2, 0.9], "site" => site].unwrap();
        let blackout = df!["val" => [Some(0.1), None, Some(0.2), None], "site" => site].unwrap();
        assert_eq!(run_against(clean(), "blackout", blackout).parse_loss, None);

        let scramble = df!["val" => ["0.1", "ND", "0.2", "<0.01"], "site" => site].unwrap();
        let expected = BTreeMap::from([(
            "scramble".to_string(),
            BTreeMap::from([("val".to_string(), 2)]),
        )]);
        assert_eq!(
            run_against(clean(), "scramble", scramble).parse_loss,
            Some(expected)
        );
    }

    #[test]
    fn flipper_id_columns_are_not_parse_loss() {
        let x = df![
            "val" => [0.1, 0.8, 0.2, 0.9],
            "source_row_id" => ["pair_0_step_0", "pair_0_step_1", "pair_1_step_0", "pair_1_step_1"],
        ]
        .unwrap();
        let mut variants = HashMap::from([(
            "flipper".to_string(),
            SimulationVariant {
                x,
                y: df!["target" => [0, 0, 0, 0]].unwrap(),
            },
        )]);
        assert!(stress_variant(&variants).unwrap().is_none());

        let stress = |x| SimulationVariant {
            x,
            y: df!["target" => [0, 1, 0, 1]].unwrap(),
        };
        variants.insert("scramble".into(), stress(df!["val" => ["1"]].unwrap()));
        variants.insert("units".into(), stress(df!["val" => [1.0]].unwrap()));
        let err = stress_variant(&variants).unwrap_err();
        assert!(
            err.to_string().contains("[\"scramble\", \"units\"]"),
            "{err}"
        );
    }
}
