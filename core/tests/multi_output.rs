use pasteur_core::*;
use polars::prelude::*;
use std::collections::HashMap;

/// Returns the feature columns `x0..x{k-1}` as its probabilities, renamed to
/// `names`, so each test spells out the model's output row by row.
struct PassThrough {
    names: Vec<&'static str>,
    thresholds: Option<Vec<f64>>,
}

impl Model for PassThrough {
    fn predict_proba(&self, df: &DataFrame) -> Result<DataFrame, CoreError> {
        let cols: Vec<Column> = self
            .names
            .iter()
            .enumerate()
            .map(|(j, name)| {
                let mut c = df.column(&format!("x{j}"))?.clone();
                c.rename((*name).into());
                Ok(c)
            })
            .collect::<Result<_, CoreError>>()?;
        Ok(DataFrame::new(df.height(), cols)?)
    }

    fn decision_thresholds(&self) -> Option<Vec<f64>> {
        self.thresholds.clone()
    }
}

fn features(rows: &[[f64; 3]]) -> DataFrame {
    let col = |j: usize| rows.iter().map(|r| r[j]).collect::<Vec<f64>>();
    df!["x0" => col(0), "x1" => col(1), "x2" => col(2)].unwrap()
}

fn labels(cols: &[(&str, &[i64])]) -> DataFrame {
    let n = cols[0].1.len();
    DataFrame::new(
        n,
        cols.iter()
            .map(|(name, v)| Series::new((*name).into(), v.to_vec()).into())
            .collect(),
    )
    .unwrap()
}

fn config(task: TaskType) -> EvaluationConfig {
    EvaluationConfig {
        n_jitter_iters: 0,
        allow_partial: true,
        metrics: vec!["roc_auc".to_string()],
        flip_threshold: 0.5,
        task,
    }
}

fn run(
    model: &dyn Model,
    clean: SimulationVariant,
    variants: Vec<(&str, SimulationVariant)>,
    task: TaskType,
) -> Result<EvaluationResult, CoreError> {
    let clean_datasets = HashMap::from([("d".to_string(), clean)]);
    let simulation_variants = variants
        .into_iter()
        .map(|(name, v)| (name.to_string(), HashMap::from([("d".to_string(), v)])))
        .collect();
    Evaluator::new().run(model, &clean_datasets, &simulation_variants, &config(task))
}

fn close(a: f64, b: f64) -> bool {
    (a - b).abs() < 1e-9
}

/// Flipper grid rows: (pair_id, step, t, label_a, label_b, pair_label, features).
type GridRow = (u32, u32, f64, f64, f64, Option<u32>, [f64; 3]);

fn grid(rows: &[GridRow]) -> DataFrame {
    let n = rows.len();
    let mut df = df![
        "pair_id" => rows.iter().map(|r| r.0).collect::<Vec<_>>(),
        "step" => rows.iter().map(|r| r.1).collect::<Vec<_>>(),
        "t" => rows.iter().map(|r| r.2).collect::<Vec<_>>(),
        "source_a_row" => vec![0u32; n],
        "source_b_row" => vec![1u32; n],
        "label_a" => rows.iter().map(|r| r.3).collect::<Vec<_>>(),
        "label_b" => rows.iter().map(|r| r.4).collect::<Vec<_>>(),
    ]
    .unwrap();
    if rows.iter().any(|r| r.5.is_some()) {
        let pair_label: Vec<u32> = rows.iter().map(|r| r.5.unwrap_or(0)).collect();
        df.with_column(Series::new(PAIR_LABEL_COL.into(), pair_label).into())
            .unwrap();
    }
    df.hstack(features(&rows.iter().map(|r| r.6).collect::<Vec<_>>()).columns())
        .unwrap()
}

fn multiclass_clean() -> SimulationVariant {
    SimulationVariant {
        x: features(&[
            [0.8, 0.1, 0.1],
            [0.6, 0.3, 0.1],
            [0.1, 0.8, 0.1],
            [0.3, 0.6, 0.1],
            [0.1, 0.1, 0.8],
            [0.1, 0.7, 0.2],
        ]),
        y: labels(&[
            ("g0", &[1, 1, 0, 0, 0, 0]),
            ("g1", &[0, 0, 1, 1, 0, 0]),
            ("g2", &[0, 0, 0, 0, 1, 1]),
        ]),
    }
}

fn multiclass_model() -> PassThrough {
    PassThrough {
        names: vec!["c0", "c1", "c2"],
        thresholds: None,
    }
}

#[test]
fn multiclass_auc_is_one_vs_rest_per_class_and_macro_averaged() -> Result<(), CoreError> {
    let result = run(
        &multiclass_model(),
        multiclass_clean(),
        vec![],
        TaskType::Multiclass,
    )?;
    let multi = result.multi.as_ref().expect("multiclass detail");
    let aucs: Vec<Option<f64>> = multi.per_label.iter().map(|l| l.roc_auc).collect();
    // c1: positives 0.8, 0.6 vs negatives 0.1, 0.3, 0.1, 0.7 — 7 of 8 pairs.
    assert_eq!(aucs, vec![Some(1.0), Some(0.875), Some(1.0)]);
    assert!(close(result.baselines["roc_auc"], 2.875 / 3.0));
    assert_eq!(
        multi
            .per_label
            .iter()
            .map(|l| l.label.as_str())
            .collect::<Vec<_>>(),
        ["c0", "c1", "c2"]
    );
    assert!(multi.excluded_labels.is_empty());
    assert!(multi.roc_auc_micro.is_some());
    Ok(())
}

#[test]
fn multiclass_resiliency_is_per_class_with_worst_reported() -> Result<(), CoreError> {
    // Blackout flattens every row to uniform: all scores tie, AUC 0.5 everywhere.
    let blackout = SimulationVariant {
        x: features(&[[1.0 / 3.0; 3]; 6]),
        y: multiclass_clean().y,
    };
    let result = run(
        &multiclass_model(),
        multiclass_clean(),
        vec![("blackout", blackout)],
        TaskType::Multiclass,
    )?;
    let multi = result.multi.unwrap();
    let r: Vec<f64> = multi
        .per_label
        .iter()
        .map(|l| l.resiliency.unwrap())
        .collect();
    assert!(close(r[0], 0.5) && close(r[1], 0.5 / 0.875) && close(r[2], 0.5));
    assert!(close(
        result.evaluations.metric_based["roc_auc"].resiliency,
        r.iter().sum::<f64>() / 3.0
    ));
    assert!(close(multi.worst_label_resiliency.unwrap(), 0.5));
    Ok(())
}

#[test]
fn multiclass_jitter_reports_argmax_flips() -> Result<(), CoreError> {
    let clean = multiclass_clean();
    let mut moved = clean.x.clone();
    // Row 1 tips from c0 to c1; every other row only wobbles.
    moved.with_column(Series::new("x0".into(), [0.8, 0.3, 0.1, 0.3, 0.1, 0.1]).into())?;
    moved.with_column(Series::new("x1".into(), [0.1, 0.6, 0.8, 0.6, 0.1, 0.7]).into())?;
    let result = run(
        &multiclass_model(),
        clean.clone(),
        vec![
            (
                "jitter_0",
                SimulationVariant {
                    x: clean.x.clone(),
                    y: clean.y.clone(),
                },
            ),
            (
                "jitter_1",
                SimulationVariant {
                    x: moved,
                    y: clean.y.clone(),
                },
            ),
        ],
        TaskType::Multiclass,
    )?;
    let multi = result.multi.unwrap();
    assert!(close(multi.decision_flip_rate.unwrap(), 1.0 / 6.0));
    // Row 1 moves 0.3 of probability from c0 to c1: across the two draws each
    // of those columns has variance 0.15² on that row, 0.0225 / 6 averaged
    // over rows. The class variances are summed and halved (0.00375), not
    // averaged over the three classes.
    let stability = result.evaluations.metric_invariant.jitter_stability;
    assert!(
        close(stability, 1.0 / (1.0 + 100.0 * 0.00375)),
        "{stability}"
    );
    // c2 never moved.
    assert_eq!(multi.per_label[2].jitter_stability, 1.0);
    Ok(())
}

#[test]
fn multiclass_flipper_uses_the_pairwise_margin_and_flags_detours() -> Result<(), CoreError> {
    let flipper = grid(&[
        // Pair 0: class 0 -> class 2, passing through class 1 on the way.
        (0, 0, 0.0, 0.0, 2.0, None, [0.8, 0.1, 0.1]),
        (0, 1, 0.5, 0.0, 2.0, None, [0.1, 0.8, 0.1]),
        (0, 2, 1.0, 0.0, 2.0, None, [0.1, 0.1, 0.8]),
        // Pair 1: class 0 -> class 1 directly.
        (1, 0, 0.0, 0.0, 1.0, None, [0.8, 0.1, 0.1]),
        (1, 1, 0.5, 0.0, 1.0, None, [0.45, 0.45, 0.1]),
        (1, 2, 1.0, 0.0, 1.0, None, [0.1, 0.8, 0.1]),
    ]);
    let n = flipper.height();
    let result = run(
        &multiclass_model(),
        multiclass_clean(),
        vec![(
            "flipper",
            SimulationVariant {
                x: flipper,
                y: labels(&[
                    ("g0", &vec![0; n]),
                    ("g1", &vec![0; n]),
                    ("g2", &vec![0; n]),
                ]),
            },
        )],
        TaskType::Multiclass,
    )?;
    // Both margins go -0.7, 0, +0.7: each crosses exactly at t = 0.5.
    assert!(close(
        result
            .evaluations
            .metric_invariant
            .flipper_stability
            .unwrap(),
        0.5
    ));
    let multi = result.multi.unwrap();
    assert_eq!(multi.flipper_never_flipped, Some(0.0));
    assert_eq!(multi.flipper_detour_rate, Some(0.5));
    for l in &multi.per_label {
        assert_eq!(l.flipper_stability, Some(0.5), "{}", l.label);
    }
    Ok(())
}

#[test]
fn multiclass_labels_must_be_one_hot() {
    let mut clean = multiclass_clean();
    clean.y = labels(&[
        ("g0", &[1, 1, 0, 0, 0, 0]),
        ("g1", &[1, 0, 1, 1, 0, 0]),
        ("g2", &[0, 0, 0, 0, 1, 1]),
    ]);
    let err = run(&multiclass_model(), clean, vec![], TaskType::Multiclass)
        .unwrap_err()
        .to_string();
    assert!(err.contains("exactly one"), "got {err}");
}

#[test]
fn model_width_must_match_label_width() {
    let mut clean = multiclass_clean();
    clean.y = labels(&[("g0", &[1, 1, 0, 0, 0, 0]), ("g1", &[0, 0, 1, 1, 1, 1])]);
    let err = run(&multiclass_model(), clean, vec![], TaskType::Multiclass)
        .unwrap_err()
        .to_string();
    assert!(
        err.contains("returns 3 output column(s) but the labels have 2"),
        "got {err}"
    );
}

fn multilabel_clean() -> SimulationVariant {
    SimulationVariant {
        x: features(&[
            [0.9, 0.8, 0.1],
            [0.7, 0.3, 0.2],
            [0.2, 0.6, 0.1],
            [0.1, 0.4, 0.3],
        ]),
        y: labels(&[
            ("ga", &[1, 1, 0, 0]),
            ("gb", &[1, 0, 1, 0]),
            // Nobody has c: its AUC is undefined.
            ("gc", &[0, 0, 0, 0]),
        ]),
    }
}

fn multilabel_model(thresholds: Option<Vec<f64>>) -> PassThrough {
    PassThrough {
        names: vec!["a", "b", "c"],
        thresholds,
    }
}

#[test]
fn multilabel_excludes_single_class_labels_from_the_macro() -> Result<(), CoreError> {
    let result = run(
        &multilabel_model(None),
        multilabel_clean(),
        vec![],
        TaskType::Multilabel,
    )?;
    // a and b separate perfectly. Averaging c in as 0.5 would report 0.833.
    assert_eq!(result.baselines["roc_auc"], 1.0);
    let multi = result.multi.unwrap();
    assert_eq!(multi.excluded_labels, vec!["c".to_string()]);
    assert_eq!(multi.per_label[2].roc_auc, None);
    Ok(())
}

fn multilabel_flip_t(thresholds: Option<Vec<f64>>) -> Result<f64, CoreError> {
    // One pair, sampled across label b: b's probability goes 0.3, 0.6, 0.9.
    let flipper = grid(&[
        (0, 0, 0.0, 0.0, 1.0, Some(1), [0.5, 0.3, 0.1]),
        (0, 1, 0.5, 0.0, 1.0, Some(1), [0.5, 0.6, 0.1]),
        (0, 2, 1.0, 0.0, 1.0, Some(1), [0.5, 0.9, 0.1]),
    ]);
    let result = run(
        &multilabel_model(thresholds),
        multilabel_clean(),
        vec![(
            "flipper",
            SimulationVariant {
                x: flipper,
                y: labels(&[("ga", &[0; 3]), ("gb", &[0; 3]), ("gc", &[0; 3])]),
            },
        )],
        TaskType::Multilabel,
    )?;
    let multi = result.multi.unwrap();
    assert_eq!(
        multi.per_label[0].flipper_stability, None,
        "no pair about a"
    );
    Ok(multi.per_label[1].flipper_stability.unwrap())
}

#[test]
fn multilabel_flipper_crosses_each_labels_own_threshold() -> Result<(), CoreError> {
    // --flip-threshold 0.5: crosses between 0.3 and 0.6, two thirds of the way.
    assert!(close(multilabel_flip_t(None)?, 1.0 / 3.0));
    // Contract threshold 0.7 for b: between 0.6 and 0.9, a third of the way.
    assert!(close(
        multilabel_flip_t(Some(vec![0.5, 0.7, 0.5]))?,
        0.5 + 0.5 / 3.0
    ));
    Ok(())
}

#[test]
fn multilabel_jitter_flip_rate_uses_label_thresholds() -> Result<(), CoreError> {
    let clean = multilabel_clean();
    let mut moved = clean.x.clone();
    // Row 0's b drops from 0.8 to 0.65: over 0.5 either way, under 0.7.
    moved.with_column(Series::new("x1".into(), [0.65, 0.3, 0.6, 0.4]).into())?;
    let variants = || {
        vec![
            (
                "jitter_0",
                SimulationVariant {
                    x: clean.x.clone(),
                    y: clean.y.clone(),
                },
            ),
            (
                "jitter_1",
                SimulationVariant {
                    x: moved.clone(),
                    y: clean.y.clone(),
                },
            ),
        ]
    };
    let rate = |t| -> Result<f64, CoreError> {
        let r = run(
            &multilabel_model(t),
            clean.clone(),
            variants(),
            TaskType::Multilabel,
        )?;
        Ok(r.multi.unwrap().decision_flip_rate.unwrap())
    };
    assert_eq!(rate(None)?, 0.0);
    assert_eq!(rate(Some(vec![0.5, 0.7, 0.5]))?, 0.25);
    Ok(())
}

#[test]
fn flipper_grid_task_must_match_evaluation_task() {
    let flipper = grid(&[
        (0, 0, 0.0, 0.0, 1.0, Some(1), [0.5, 0.3, 0.1]),
        (0, 1, 1.0, 0.0, 1.0, Some(1), [0.5, 0.9, 0.1]),
    ]);
    let err = run(
        &multiclass_model(),
        multiclass_clean(),
        vec![(
            "flipper",
            SimulationVariant {
                x: flipper,
                y: labels(&[("g0", &[0; 2]), ("g1", &[0; 2]), ("g2", &[0; 2])]),
            },
        )],
        TaskType::Multiclass,
    )
    .unwrap_err()
    .to_string();
    assert!(err.contains("regenerate it"), "got {err}");
}

#[test]
fn binary_output_has_no_multi_section() -> Result<(), CoreError> {
    let model = PassThrough {
        names: vec!["proba"],
        thresholds: None,
    };
    let clean = SimulationVariant {
        x: features(&[
            [0.9, 0.0, 0.0],
            [0.2, 0.0, 0.0],
            [0.7, 0.0, 0.0],
            [0.4, 0.0, 0.0],
        ]),
        y: labels(&[("label", &[1, 0, 1, 0])]),
    };
    let result = run(&model, clean, vec![], TaskType::Binary)?;
    assert!(result.multi.is_none());
    let json = serde_json::to_value(&result)?;
    assert!(
        json.get("multi").is_none(),
        "binary JSON gained a key: {json}"
    );
    assert_eq!(result.baselines["roc_auc"], 1.0);
    Ok(())
}

#[test]
fn multilabel_flipper_pairs_cross_one_splittable_label() -> Result<(), CoreError> {
    let x = df!["f1" => [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]]?;
    // Only label 0 has both classes; label 1 is all-negative.
    let y = labels(&[("a", &[1, 1, 1, 0, 0, 0]), ("b", &[0; 6])]);
    let config = FlipperConfig {
        n_pairs: 10,
        n_steps: 3,
        flip_threshold: 0.5,
        positive_class_label: None,
        random_state: Some(3),
    };
    let grid = FlipperSimulator::new(config.clone()).generate_multilabel(&x, &y)?;
    assert_eq!(grid.height(), 30);
    let pair_label = grid.column(PAIR_LABEL_COL)?.u32()?;
    let a = grid.column("source_a_row")?.u32()?;
    let b = grid.column("source_b_row")?.u32()?;
    for i in 0..grid.height() {
        assert_eq!(pair_label.get(i), Some(0));
        assert!(a.get(i).unwrap() >= 3, "a must be negative for label 0");
        assert!(b.get(i).unwrap() < 3, "b must be positive for label 0");
    }
    assert!(is_flipper_meta_column(PAIR_LABEL_COL));
    let again = FlipperSimulator::new(config.clone()).generate_multilabel(&x, &y)?;
    assert!(grid.equals(&again));

    let none = labels(&[("a", &[0; 6]), ("b", &[1; 6])]);
    assert!(FlipperSimulator::new(config)
        .generate_multilabel(&x, &none)
        .is_err());
    Ok(())
}
