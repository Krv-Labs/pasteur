use pasteur_core::*;
use polars::prelude::*;
use std::collections::HashMap;

struct MockModel;
impl Model for MockModel {
    fn predict_proba(&self, df: &DataFrame) -> Result<Series, CoreError> {
        // Return values proportional to the feature "val"
        let val = df.column("val")?;
        Ok(val.as_materialized_series().clone())
    }
}

#[test]
fn test_full_pipeline() -> Result<(), CoreError> {
    // 1. Create dummy data
    let df = df![
        "val" => [0.1, 0.8, 0.2, 0.9],
        "target" => [0, 1, 0, 1]
    ]?;
    let y = df.column("target")?.as_materialized_series().clone();
    let x = df.drop("target")?;

    let variant = SimulationVariant {
        x: x.clone(),
        y: y.clone(),
    };
    let mut clean_datasets = HashMap::new();
    clean_datasets.insert("test".to_string(), variant.clone());

    // 2. Run Simulator (Blackout)
    let config = BlackoutConfig {
        feature: "val".to_string(),
        companions: vec![],
        rate: 0.5,
        window_frac: 0.0,
        patient_id_col: None,
        time_col: None,
        random_state: Some(42),
    };
    let mut sim = BlackoutSimulator::new(config);
    sim.fit(&x)?;
    let stressed_x = sim.transform(&x)?;

    let mut simulation_variants = HashMap::new();
    let mut blackout_variants = HashMap::new();
    blackout_variants.insert(
        "test".to_string(),
        SimulationVariant {
            x: stressed_x,
            y: y.clone(),
        },
    );
    simulation_variants.insert("blackout".to_string(), blackout_variants);

    // 3. Run Evaluator
    let eval = Evaluator::new();
    let eval_config = EvaluationConfig {
        n_jitter_iters: 0,
        allow_partial: false,
        metrics: vec!["roc_auc".to_string()],
        flip_threshold: 0.5,
    };

    let model = MockModel;
    let result = eval.run(&model, &clean_datasets, &simulation_variants, &eval_config)?;

    assert!(result.evaluations.metric_based.contains_key("roc_auc"));
    let roc_auc = result.evaluations.metric_based.get("roc_auc").unwrap();
    println!("Resiliency: {}", roc_auc.resiliency);
    assert!(result
        .evaluations
        .metric_invariant
        .flipper_stability
        .is_none());

    Ok(())
}

#[test]
fn test_flipper_evaluation() -> Result<(), CoreError> {
    let df = df![
        "f1" => [0.0, 1.0, 0.0, 1.0],
        "f2" => [0.0, 0.0, 1.0, 1.0],
    ]?;
    let y = Series::new("y".into(), &[0i64, 1, 0, 1]);
    let x = df.clone();

    let grid = df![
        "pair_id" => [0u32, 0, 0],
        "step" => [0u32, 1, 2],
        "t" => [0.0f64, 0.5, 1.0],
        "source_a_row" => [0u32, 0, 0],
        "source_b_row" => [1u32, 1, 1],
        "label_a" => [0.0f64, 0.0, 0.0],
        "label_b" => [1.0f64, 1.0, 1.0],
        "f1" => [0.0, 0.5, 1.0],
        "f2" => [0.0, 0.0, 0.0],
    ]?;

    let mut clean_datasets = HashMap::new();
    clean_datasets.insert(
        "test".to_string(),
        SimulationVariant {
            x: x.clone(),
            y: y.clone(),
        },
    );

    let mut simulation_variants = HashMap::new();
    let mut flipper_variants = HashMap::new();
    flipper_variants.insert(
        "test".to_string(),
        SimulationVariant {
            x: grid,
            y: Series::new("y".into(), &[0i64, 0, 0]),
        },
    );
    simulation_variants.insert("flipper".to_string(), flipper_variants);

    struct ThresholdModel;
    impl Model for ThresholdModel {
        fn predict_proba(&self, df: &DataFrame) -> Result<Series, CoreError> {
            let f1 = df.column("f1")?.f64()?;
            let mut out = Vec::with_capacity(df.height());
            for i in 0..df.height() {
                let v = f1.get(i).unwrap_or(0.0);
                out.push(if v >= 0.5 { 1.0 } else { 0.0 });
            }
            Ok(Series::new("p".into(), out))
        }
    }

    let eval = Evaluator::new();
    let eval_config = EvaluationConfig {
        n_jitter_iters: 0,
        allow_partial: false,
        metrics: vec!["roc_auc".to_string()],
        flip_threshold: 0.5,
    };
    let result = eval.run(
        &ThresholdModel,
        &clean_datasets,
        &simulation_variants,
        &eval_config,
    )?;
    let flipper_stability = result
        .evaluations
        .metric_invariant
        .flipper_stability
        .expect("flipper stability should be scored");
    assert!(flipper_stability < 1.0);

    Ok(())
}

#[test]
fn test_jitter_simulator() -> Result<(), CoreError> {
    let df = df![
        "val" => [1.0, 2.0, 3.0, 4.0, 5.0]
    ]?;

    let config = JitterConfig {
        feature: "val".to_string(),
        scale: 0.1,
        random_state: Some(42),
    };

    let mut sim = JitterSimulator::new(config);
    sim.fit(&df)?;
    let jittered = sim.transform(&df)?;

    let original = df.column("val")?.f64()?;
    let transformed = jittered.column("val")?.f64()?;

    assert_ne!(original.get(0), transformed.get(0));

    Ok(())
}
