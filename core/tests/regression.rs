use pasteur_core::evaluate::JITTER_STABILITY_ALPHA;
use pasteur_core::*;
use polars::prelude::*;
use std::collections::HashMap;

/// Predicts feature `x0` unchanged, so each test spells out the model's
/// prediction row by row.
struct PassThrough;

impl Model for PassThrough {
    fn predict_proba(&self, df: &DataFrame) -> Result<DataFrame, CoreError> {
        let mut c = df.column("x0")?.clone();
        c.rename("prediction".into());
        Ok(DataFrame::new(df.height(), vec![c])?)
    }
}

/// A binary model for the "classifier output is unchanged" check.
struct Proba;

impl Model for Proba {
    fn predict_proba(&self, df: &DataFrame) -> Result<DataFrame, CoreError> {
        let mut c = df.column("x0")?.clone();
        c.rename("proba".into());
        Ok(DataFrame::new(df.height(), vec![c])?)
    }
}

fn variant(pred: &[f64], target: &[f64]) -> SimulationVariant {
    SimulationVariant {
        x: df!["x0" => pred.to_vec()].unwrap(),
        y: df!["hba1c" => target.to_vec()].unwrap(),
    }
}

fn run(
    model: &dyn Model,
    clean: SimulationVariant,
    variants: Vec<(&str, SimulationVariant)>,
    task: TaskType,
    flip_threshold: f64,
) -> Result<EvaluationResult, CoreError> {
    let config = EvaluationConfig {
        n_jitter_iters: 0,
        allow_partial: true,
        metrics: vec![],
        flip_threshold,
        task,
    };
    let clean_datasets = HashMap::from([("d".to_string(), clean)]);
    let simulation_variants = variants
        .into_iter()
        .map(|(name, v)| (name.to_string(), HashMap::from([("d".to_string(), v)])))
        .collect();
    Evaluator::new().run(model, &clean_datasets, &simulation_variants, &config)
}

fn close(a: f64, b: f64) -> bool {
    (a - b).abs() < 1e-9
}

const TARGET: [f64; 4] = [1.0, 2.0, 3.0, 4.0];

#[test]
fn clean_baselines_are_rmse_mae_and_r2() {
    // Errors 0.5, 0, -1, 1 → MSE 0.5625; Var(y) = 1.25.
    let r = run(
        &PassThrough,
        variant(&[1.5, 2.0, 2.0, 5.0], &TARGET),
        vec![],
        TaskType::Regression,
        0.0,
    )
    .unwrap();
    assert!(close(r.baselines["rmse"], 0.75));
    assert!(close(r.baselines["mae"], 0.625));
    assert!(close(r.baselines["r2"], 1.0 - 0.5625 / 1.25));
    assert_eq!(r.baselines.len(), 3);
    assert!(r.evaluations.metric_based.contains_key("rmse"));
    assert!(!r.evaluations.metric_based.contains_key("roc_auc"));
    // Nothing simulated: neutral values, as for classifiers.
    assert_eq!(r.evaluations.metric_based["rmse"].resiliency, 1.0);
    assert_eq!(r.evaluations.metric_invariant.jitter_stability, 1.0);
    assert!(r.evaluations.metric_invariant.flipper_stability.is_none());

    let detail = r.regression.expect("regression detail");
    assert_eq!(detail.target, "hba1c");
    assert_eq!(detail.n_rows, 4);
    assert!(close(detail.target_sd, 1.25f64.sqrt()));
    assert!(r.multi.is_none());
}

#[test]
fn resiliency_is_clean_rmse_over_blackout_rmse() {
    // Clean errors all 0.5 (RMSE 0.5); blackout errors all 2 (RMSE 2).
    let r = run(
        &PassThrough,
        variant(&[1.5, 2.5, 3.5, 4.5], &TARGET),
        vec![("blackout", variant(&[3.0, 4.0, 5.0, 6.0], &TARGET))],
        TaskType::Regression,
        0.0,
    )
    .unwrap();
    assert!(close(r.evaluations.metric_based["rmse"].resiliency, 0.25));
    assert!(close(r.regression.unwrap().blackout_rmse.unwrap(), 2.0));
}

#[test]
fn jitter_variance_is_scaled_by_the_target_variance() {
    // Row 0 moves 1.0 ↔ 2.0 (variance 0.25); the rest do not move.
    let r = run(
        &PassThrough,
        variant(&TARGET, &TARGET),
        vec![
            ("jitter_0", variant(&[1.0, 2.0, 3.0, 4.0], &TARGET)),
            ("jitter_1", variant(&[2.0, 2.0, 3.0, 4.0], &TARGET)),
        ],
        TaskType::Regression,
        0.0,
    )
    .unwrap();
    let v = 0.25 / 4.0 / 1.25;
    assert!(close(
        r.evaluations.metric_invariant.jitter_stability,
        1.0 / (1.0 + JITTER_STABILITY_ALPHA * v)
    ));
    // Mean per-patient SD in target units: 0.5 on one of four rows.
    assert!(close(
        r.regression.unwrap().jitter_prediction_sd.unwrap(),
        0.125
    ));
}

#[test]
fn jitter_stability_does_not_depend_on_the_target_unit() {
    let scaled = |k: f64| {
        let t: Vec<f64> = TARGET.iter().map(|v| v * k).collect();
        let draw = |first: f64| {
            let mut p = t.clone();
            p[0] = first * k;
            variant(&p, &t)
        };
        run(
            &PassThrough,
            variant(&t, &t),
            vec![("jitter_0", draw(1.0)), ("jitter_1", draw(2.0))],
            TaskType::Regression,
            0.0,
        )
        .unwrap()
        .evaluations
        .metric_invariant
        .jitter_stability
    };
    // mmol/mol vs %: same model, same score.
    assert!(close(scaled(1.0), scaled(10.93)));
}

/// One pair, `n` steps from a target-5 patient to a target-8 patient.
fn regression_grid(preds: &[f64], label_a: f64, label_b: f64) -> SimulationVariant {
    let n = preds.len();
    let x = df![
        "pair_id" => vec![0u32; n],
        "step" => (0..n as u32).collect::<Vec<_>>(),
        "t" => (0..n).map(|i| i as f64 / (n - 1) as f64).collect::<Vec<_>>(),
        "source_a_row" => vec![0u32; n],
        "source_b_row" => vec![1u32; n],
        "label_a" => vec![label_a; n],
        "label_b" => vec![label_b; n],
        "x0" => preds.to_vec(),
    ]
    .unwrap();
    let y = df!["hba1c" => vec![0.0; n]].unwrap();
    SimulationVariant { x, y }
}

#[test]
fn flipper_finds_where_the_prediction_crosses_the_cutoff() {
    // Predictions 5, 6, 7, 8 at t = 0, 1/3, 2/3, 1 cross 6.5 at t = 0.5.
    let r = run(
        &PassThrough,
        variant(&TARGET, &TARGET),
        vec![("flipper", regression_grid(&[5.0, 6.0, 7.0, 8.0], 5.0, 8.0))],
        TaskType::Regression,
        6.5,
    )
    .unwrap();
    assert!(close(
        r.evaluations.metric_invariant.flipper_stability.unwrap(),
        0.5
    ));
    let detail = r.regression.unwrap();
    assert_eq!(detail.flip_threshold, Some(6.5));
    assert_eq!(detail.flipper_never_flipped, Some(0.0));
}

#[test]
fn flipper_rejects_a_grid_built_for_another_cutoff() {
    // Both endpoints below 9.0: this pair was not sampled across it.
    let err = run(
        &PassThrough,
        variant(&TARGET, &TARGET),
        vec![("flipper", regression_grid(&[5.0, 6.0, 7.0, 8.0], 5.0, 8.0))],
        TaskType::Regression,
        9.0,
    )
    .unwrap_err()
    .to_string();
    assert!(err.contains("--flip-threshold 9"), "{err}");

    // A binary grid's 0/1 labels do not straddle a clinical cutoff either.
    let binary_grid = regression_grid(&[5.0, 6.0, 7.0, 8.0], 0.0, 1.0);
    let err = run(
        &PassThrough,
        variant(&TARGET, &TARGET),
        vec![("flipper", binary_grid)],
        TaskType::Regression,
        6.5,
    );
    assert!(err.is_err());
}

#[test]
fn constant_or_missing_targets_are_rejected() {
    let err = run(
        &PassThrough,
        variant(&[1.0, 2.0], &[3.0, 3.0]),
        vec![],
        TaskType::Regression,
        0.0,
    )
    .unwrap_err()
    .to_string();
    assert!(err.contains("same value for every clean row"), "{err}");
    assert!(run(
        &PassThrough,
        variant(&[1.0, 2.0], &[3.0, f64::NAN]),
        vec![],
        TaskType::Regression,
        0.0,
    )
    .is_err());
}

#[test]
fn a_regression_model_must_return_one_column() {
    struct TwoColumns;
    impl Model for TwoColumns {
        fn predict_proba(&self, df: &DataFrame) -> Result<DataFrame, CoreError> {
            Ok(df!["a" => vec![0.0; df.height()], "b" => vec![0.0; df.height()]]?)
        }
    }
    assert!(run(
        &TwoColumns,
        variant(&TARGET, &TARGET),
        vec![],
        TaskType::Regression,
        0.0
    )
    .is_err());
}

#[test]
fn binary_results_have_no_regression_section() {
    let clean = SimulationVariant {
        x: df!["x0" => [0.1, 0.9, 0.4, 0.6]].unwrap(),
        y: df!["label" => [0i64, 1, 0, 1]].unwrap(),
    };
    let r = run(&Proba, clean, vec![], TaskType::Binary, 0.5).unwrap();
    assert_eq!(r.baselines.keys().collect::<Vec<_>>(), vec!["roc_auc"]);
    assert!(r.regression.is_none());
    let json = serde_json::to_value(&r).unwrap();
    assert!(json.get("regression").is_none());
    assert!(json.get("multi").is_none());
}
