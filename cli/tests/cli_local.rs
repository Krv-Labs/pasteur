use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use polars::prelude::*;

fn pasteur_cli() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_pasteur-cli"))
}

fn write_clean_input(path: &Path) {
    let mut df = df![
        "node_id" => ["1", "2", "3", "4", "5"],
        "TSH" => [1.0f64, 2.0, 3.0, 4.0, 5.0],
        // Companion flag, the shape a real assay arrives in: a value plus a
        // "we actually measured this" marker.
        "TSH_measured" => [1i64, 1, 1, 1, 1],
        "f2" => [0.1, 0.2, 0.3, 0.4, 0.5],
    ]
    .unwrap();
    let mut file = fs::File::create(path).unwrap();
    ParquetWriter::new(&mut file).finish(&mut df).unwrap();
}

fn write_labels_input(path: &Path, negative_ids: &[i64], positive_ids: &[i64]) {
    let neg_members = Series::new("m".into(), negative_ids.to_vec());
    let pos_members = Series::new("m".into(), positive_ids.to_vec());
    let member_ids = Series::new("member_ids".into(), vec![neg_members, pos_members]);
    let mut df = DataFrame::new(
        2,
        vec![
            Series::new("group_id".into(), [100u32, 103]).into(),
            member_ids.into(),
        ],
    )
    .unwrap();
    let mut file = fs::File::create(path).unwrap();
    ParquetWriter::new(&mut file).finish(&mut df).unwrap();
}

fn temp_dir(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("pasteur-cli-it-{name}-{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

#[test]
fn simulate_produces_sim_type_folders() {
    let root = temp_dir("simulate");
    let input = root.join("clean.parquet");
    let labels = root.join("groups.parquet");
    let output = root.join("output");
    write_clean_input(&input);
    write_labels_input(&labels, &[1, 3, 5], &[2, 4]);

    let status = Command::new(pasteur_cli())
        .args([
            "simulate",
            "--input",
            input.to_str().unwrap(),
            "--output",
            output.to_str().unwrap(),
            "--labels",
            labels.to_str().unwrap(),
            "--flipper-pairs",
            "2",
            "--flipper-steps",
            "3",
            "--jitter-iters",
            "1",
        ])
        .status()
        .unwrap();
    assert!(status.success());

    for sim_type in ["clean", "blackout", "jitter", "flipper"] {
        assert!(output.join(sim_type).is_dir(), "missing {sim_type}/");
    }
    assert!(output.join("clean/clean.parquet").is_file());
    assert!(output.join("flipper/flipper.parquet").is_file());
    let _ = fs::remove_dir_all(&root);
}

#[test]
fn blackout_companion_is_nulled_with_the_feature() {
    let root = temp_dir("blackout-companion");
    let input = root.join("clean.parquet");
    let output = root.join("output");
    write_clean_input(&input);

    let status = Command::new(pasteur_cli())
        .args([
            "simulate",
            "--input",
            input.to_str().unwrap(),
            "--output",
            output.to_str().unwrap(),
            "--blackout-companion",
            "TSH_measured",
            "--blackout-rate",
            "0.6",
            "--jitter-iters",
            "1",
        ])
        .status()
        .unwrap();
    assert!(status.success());

    let df = ParquetReader::new(fs::File::open(output.join("blackout/blackout.parquet")).unwrap())
        .finish()
        .unwrap();
    let nulls = |name: &str| -> Vec<bool> {
        let c = df.column(name).unwrap();
        (0..df.height())
            .map(|i| c.get(i).unwrap().is_null())
            .collect()
    };
    let tsh = nulls("TSH");
    assert_eq!(
        tsh,
        nulls("TSH_measured"),
        "blacked-out TSH must not keep TSH_measured == 1"
    );
    assert!(tsh.iter().any(|n| *n), "expected some rows blacked out");
    let _ = fs::remove_dir_all(&root);
}

#[test]
fn card_writes_readme() {
    let root = temp_dir("card");
    let input = root.join("bundle");
    let output = input.clone();
    fs::create_dir_all(input.join("clean")).unwrap();
    fs::create_dir_all(input.join("blackout")).unwrap();
    fs::write(input.join("clean/clean.parquet"), b"x").unwrap();
    fs::write(input.join("blackout/blackout.parquet"), b"x").unwrap();

    let status = Command::new(pasteur_cli())
        .args([
            "card",
            "--input",
            input.to_str().unwrap(),
            "--source-dataset",
            "org/source",
            "--dest-repo",
            "org/dest",
            "--tag",
            "clinical",
            "--description",
            "Test dataset",
        ])
        .status()
        .unwrap();
    assert!(status.success());
    let readme = fs::read_to_string(output.join("README.md")).unwrap();
    assert!(readme.contains("org/dest"));
    assert!(readme.contains("pasteur-simulation"));
    assert!(readme.contains("clinical"));
    let _ = fs::remove_dir_all(&root);
}

#[test]
fn cache_list_rm_clear_round_trip() {
    let root = temp_dir("cache");
    let models = root.join("models");
    fs::create_dir_all(&models).unwrap();
    fs::write(models.join("stub.onnx"), b"onnx").unwrap();

    let cli = pasteur_cli();
    let cache_dir = ["--cache-dir", root.to_str().unwrap()];

    let list = Command::new(&cli)
        .args(["cache", "list"])
        .args(cache_dir)
        .output()
        .unwrap();
    assert!(
        list.status.success(),
        "{}",
        String::from_utf8_lossy(&list.stderr)
    );
    let stdout = String::from_utf8_lossy(&list.stdout);
    assert!(stdout.contains("stub.onnx"));

    let rm = Command::new(&cli)
        .args(["cache", "rm", "stub.onnx", "--kind", "models"])
        .args(cache_dir)
        .status()
        .unwrap();
    assert!(rm.success());

    fs::write(models.join("other.onnx"), b"x").unwrap();
    fs::create_dir_all(root.join("simulations/run-a")).unwrap();
    fs::write(root.join("simulations/run-a/manifest.json"), b"{}").unwrap();

    let clear = Command::new(&cli)
        .args(["cache", "clear"])
        .args(cache_dir)
        .status()
        .unwrap();
    assert!(clear.success());

    let list_after = Command::new(&cli)
        .args(["cache", "list"])
        .args(cache_dir)
        .output()
        .unwrap();
    assert!(list_after.status.success());
    let after = String::from_utf8_lossy(&list_after.stdout);
    assert!(after.contains("(empty)"));
    let _ = fs::remove_dir_all(&root);
}

#[test]
fn evaluate_rejects_missing_explicit_contract() {
    let root = temp_dir("missing-contract");
    let input = root.join("clean.parquet");
    let labels = root.join("groups.parquet");
    let sim_root = root.join("sim");
    write_clean_input(&input);
    write_labels_input(&labels, &[1, 3, 5], &[2, 4]);

    let status = Command::new(pasteur_cli())
        .args([
            "simulate",
            "--input",
            input.to_str().unwrap(),
            "--output",
            sim_root.to_str().unwrap(),
            "--blackout-rate",
            "0.3",
            "--jitter-iters",
            "1",
        ])
        .status()
        .unwrap();
    assert!(status.success());

    let models = root.join("models");
    std::fs::create_dir_all(&models).unwrap();
    std::fs::write(models.join("test.onnx"), b"onnx").unwrap();
    let missing = root.join("contracts/metadata.json");

    let output = Command::new(pasteur_cli())
        .args([
            "evaluate",
            "blackout",
            "--sim-root",
            sim_root.to_str().unwrap(),
            "--labels",
            labels.to_str().unwrap(),
            "--model",
            models.join("test.onnx").to_str().unwrap(),
            "--contract",
            missing.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(!output.status.success());
    let err = String::from_utf8_lossy(&output.stderr);
    assert!(err.contains("contract not found"), "got: {err}");
    let _ = fs::remove_dir_all(&root);
}

#[test]
fn evaluate_rejects_malformed_explicit_contract() {
    let root = temp_dir("bad-contract");
    let input = root.join("clean.parquet");
    let labels = root.join("groups.parquet");
    let sim_root = root.join("sim");
    write_clean_input(&input);
    write_labels_input(&labels, &[1, 3, 5], &[2, 4]);

    let status = Command::new(pasteur_cli())
        .args([
            "simulate",
            "--input",
            input.to_str().unwrap(),
            "--output",
            sim_root.to_str().unwrap(),
            "--blackout-rate",
            "0.3",
            "--jitter-iters",
            "1",
        ])
        .status()
        .unwrap();
    assert!(status.success());

    let models = root.join("models");
    std::fs::create_dir_all(&models).unwrap();
    std::fs::write(models.join("test.onnx"), b"onnx").unwrap();
    let contract = root.join("contracts/metadata.json");
    std::fs::create_dir_all(contract.parent().unwrap()).unwrap();
    std::fs::write(&contract, b"{not json").unwrap();

    let output = Command::new(pasteur_cli())
        .args([
            "evaluate",
            "blackout",
            "--sim-root",
            sim_root.to_str().unwrap(),
            "--labels",
            labels.to_str().unwrap(),
            "--model",
            models.join("test.onnx").to_str().unwrap(),
            "--contract",
            contract.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(!output.status.success());
    let err = String::from_utf8_lossy(&output.stderr);
    assert!(err.contains("not readable"), "got: {err}");
    let _ = fs::remove_dir_all(&root);
}

/// Two features, matching the synthetic fixtures in `model/tests/fixtures`.
fn write_two_feature_input(path: &Path) {
    let ids: Vec<String> = (1..=12).map(|i| i.to_string()).collect();
    let f1: Vec<f64> = (0..12).map(|i| -1.5 + 0.27 * i as f64).collect();
    let f2: Vec<f64> = (0..12).map(|i| ((i * 7) % 12) as f64 / 6.0 - 1.0).collect();
    let mut df = df!["node_id" => ids, "f1" => f1, "f2" => f2].unwrap();
    let mut file = fs::File::create(path).unwrap();
    ParquetWriter::new(&mut file).finish(&mut df).unwrap();
}

fn write_groups(path: &Path, groups: &[(u32, &[i64])]) {
    let members: Vec<Series> = groups
        .iter()
        .map(|(_, ids)| Series::new("m".into(), ids.to_vec()))
        .collect();
    let mut df = DataFrame::new(
        groups.len(),
        vec![
            Series::new(
                "group_id".into(),
                groups.iter().map(|(g, _)| *g).collect::<Vec<u32>>(),
            )
            .into(),
            Series::new("member_ids".into(), members).into(),
        ],
    )
    .unwrap();
    let mut file = fs::File::create(path).unwrap();
    ParquetWriter::new(&mut file).finish(&mut df).unwrap();
}

/// Copies a fixture model next to a contract declaring `task` and `classes`.
fn fixture_model(dir: &Path, fixture: &str, task: &str, classes: &[&str]) -> PathBuf {
    let src = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../model/tests/fixtures")
        .join(format!("{fixture}.onnx"));
    fs::create_dir_all(dir).unwrap();
    let model = dir.join(format!("{fixture}.onnx"));
    fs::copy(src, &model).unwrap();
    let contract = format!(
        r#"{{"task_type": "{task}", "input": {{"feature_order": ["f1", "f2"],
            "absent_sentinel": 0.0}}, "output": {{"classes": {classes:?}}}}}"#
    );
    fs::write(dir.join("metadata.json"), contract).unwrap();
    model
}

fn run_cli(args: &[&str]) -> std::process::Output {
    Command::new(pasteur_cli()).args(args).output().unwrap()
}

fn simulate_multi(root: &Path, task: &str, groups: &[&str]) -> (PathBuf, PathBuf) {
    let input = root.join("clean.parquet");
    let labels = root.join("groups.parquet");
    let sim_root = root.join("sim");
    write_two_feature_input(&input);
    write_groups(
        &labels,
        &[
            (1, &[1, 2, 3, 4]),
            (2, &[5, 6, 7, 8]),
            (3, &[9, 10, 11, 12]),
            (11, &[1, 2, 3, 5, 9]),
            (12, &[2, 4, 6, 8, 10, 12]),
        ],
    );
    let mut args = vec![
        "simulate",
        "--input",
        input.to_str().unwrap(),
        "--output",
        sim_root.to_str().unwrap(),
        "--feature",
        "f1",
        "--labels",
        labels.to_str().unwrap(),
        "--task",
        task,
        "--jitter-iters",
        "2",
        "--flipper-pairs",
        "6",
        "--flipper-steps",
        "5",
    ];
    for g in groups {
        args.extend(["--positive-group-id", g]);
    }
    let out = run_cli(&args);
    assert!(
        out.status.success(),
        "simulate failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    (sim_root, labels)
}

fn evaluate_json(
    sim_type: &str,
    sim_root: &Path,
    labels: &Path,
    model: &Path,
    task: &str,
    groups: &[&str],
) -> serde_json::Value {
    let mut args = vec![
        "evaluate",
        sim_type,
        "--sim-root",
        sim_root.to_str().unwrap(),
        "--labels",
        labels.to_str().unwrap(),
        "--model",
        model.to_str().unwrap(),
        "--task",
        task,
    ];
    for g in groups {
        args.extend(["--positive-group-id", g]);
    }
    let out = run_cli(&args);
    assert!(
        out.status.success(),
        "evaluate {sim_type} failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    serde_json::from_slice(&out.stdout).unwrap()
}

#[test]
fn multiclass_simulate_and_evaluate_flipper() {
    let root = temp_dir("multiclass");
    let groups = ["1", "2", "3"];
    let (sim_root, labels) = simulate_multi(&root, "multiclass", &groups);
    let model = fixture_model(
        &root.join("models"),
        "multiclass_tensor",
        "multiclass",
        &["low", "mid", "high"],
    );

    let result = evaluate_json("flipper", &sim_root, &labels, &model, "multiclass", &groups);
    let multi = &result["multi"];
    assert_eq!(multi["task"], "multiclass");
    let per_label = multi["per_label"].as_array().unwrap();
    assert_eq!(per_label.len(), 3);
    assert_eq!(per_label[0]["label"], "low");
    assert!(multi["flipper_detour_rate"].is_number());
    assert!(result["evaluations"]["metric_invariant"]["flipper_stability"].is_number());
    let _ = fs::remove_dir_all(&root);
}

#[test]
fn multilabel_simulate_and_evaluate_jitter() {
    let root = temp_dir("multilabel");
    let groups = ["11", "12"];
    let (sim_root, labels) = simulate_multi(&root, "multilabel", &groups);
    // The fixture has three labels; scoring it against two groups must fail.
    let three = fixture_model(
        &root.join("three"),
        "multilabel_multioutput",
        "multilabel",
        &["a", "b", "c"],
    );
    let mut args = vec![
        "evaluate",
        "jitter",
        "--sim-root",
        sim_root.to_str().unwrap(),
        "--labels",
        labels.to_str().unwrap(),
        "--model",
        three.to_str().unwrap(),
        "--task",
        "multilabel",
    ];
    for g in groups {
        args.extend(["--positive-group-id", g]);
    }
    let out = run_cli(&args);
    assert!(!out.status.success());
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(err.contains("labels have 2"), "got: {err}");

    let (sim_root, labels) = simulate_multi(&root, "multilabel", &["11", "12", "1"]);
    let result = evaluate_json(
        "jitter",
        &sim_root,
        &labels,
        &three,
        "multilabel",
        &["11", "12", "1"],
    );
    assert_eq!(result["multi"]["task"], "multilabel");
    assert!(result["multi"]["decision_flip_rate"].is_number());
    let _ = fs::remove_dir_all(&root);
}

#[test]
fn multiclass_labels_must_partition_the_rows() {
    let root = temp_dir("multiclass-partition");
    let input = root.join("clean.parquet");
    let labels = root.join("groups.parquet");
    write_two_feature_input(&input);
    // Row 4 is in both groups; rows 9..=12 are in neither.
    write_groups(&labels, &[(1, &[1, 2, 3, 4]), (2, &[4, 5, 6, 7, 8])]);
    let out = run_cli(&[
        "simulate",
        "--input",
        input.to_str().unwrap(),
        "--output",
        root.join("sim").to_str().unwrap(),
        "--feature",
        "f1",
        "--labels",
        labels.to_str().unwrap(),
        "--task",
        "multiclass",
        "--positive-group-id",
        "1",
        "--positive-group-id",
        "2",
    ]);
    assert!(!out.status.success());
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(
        err.contains("4 row(s) are in no group") && err.contains("1 row(s) are in several"),
        "got: {err}"
    );
    let _ = fs::remove_dir_all(&root);
}

#[test]
fn binary_takes_exactly_one_group() {
    let root = temp_dir("binary-one-group");
    let input = root.join("clean.parquet");
    let labels = root.join("groups.parquet");
    write_two_feature_input(&input);
    write_groups(&labels, &[(1, &[1, 2]), (2, &[3, 4])]);
    let out = run_cli(&[
        "simulate",
        "--input",
        input.to_str().unwrap(),
        "--output",
        root.join("sim").to_str().unwrap(),
        "--feature",
        "f1",
        "--labels",
        labels.to_str().unwrap(),
        "--positive-group-id",
        "1",
        "--positive-group-id",
        "2",
    ]);
    assert!(!out.status.success());
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(err.contains("--task multiclass"), "got: {err}");
    let _ = fs::remove_dir_all(&root);
}

/// Targets for `write_two_feature_input`'s rows, close to the fixtures'
/// `5.5 + f1 − 0.5 f2` so the models are scored on something they fit.
fn write_regression_targets(path: &Path, skip_id: Option<&str>) {
    let (ids, values): (Vec<String>, Vec<f64>) = (0..12)
        .map(|i| {
            let f1 = -1.5 + 0.27 * i as f64;
            let f2 = ((i * 7) % 12) as f64 / 6.0 - 1.0;
            (
                (i + 1).to_string(),
                5.5 + f1 - 0.5 * f2 + 0.05 * (i % 3) as f64,
            )
        })
        .filter(|(id, _)| Some(id.as_str()) != skip_id)
        .unzip();
    let mut df = df!["node_id" => ids, "hba1c" => values].unwrap();
    let mut file = fs::File::create(path).unwrap();
    ParquetWriter::new(&mut file).finish(&mut df).unwrap();
}

/// Simulates the two-feature cohort for regression, with a flipper grid
/// across `cutoff` when one is given.
fn simulate_regression(root: &Path, cutoff: Option<&str>) -> (PathBuf, PathBuf) {
    let input = root.join("clean.parquet");
    let targets = root.join("targets.parquet");
    let sim_root = root.join("sim");
    write_two_feature_input(&input);
    write_regression_targets(&targets, None);
    let mut args = vec![
        "simulate",
        "--input",
        input.to_str().unwrap(),
        "--output",
        sim_root.to_str().unwrap(),
        "--feature",
        "f1",
        "--task",
        "regression",
        "--targets",
        targets.to_str().unwrap(),
        "--target-col",
        "hba1c",
        "--jitter-iters",
        "2",
        "--flipper-pairs",
        "6",
        "--flipper-steps",
        "5",
    ];
    if let Some(cutoff) = cutoff {
        args.extend(["--flip-threshold", cutoff]);
    }
    let out = run_cli(&args);
    assert!(
        out.status.success(),
        "simulate failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    (sim_root, targets)
}

fn regression_args<'a>(
    command: &'a str,
    sim_type: &'a str,
    sim_root: &'a Path,
    targets: &'a Path,
) -> Vec<&'a str> {
    vec![
        command,
        sim_type,
        "--sim-root",
        sim_root.to_str().unwrap(),
        "--targets",
        targets.to_str().unwrap(),
        "--target-col",
        "hba1c",
        "--task",
        "regression",
    ]
}

#[test]
fn regression_simulate_evaluate_and_compare() {
    let root = temp_dir("regression");
    let (sim_root, targets) = simulate_regression(&root, Some("5.5"));
    let flipper =
        ParquetReader::new(fs::File::open(sim_root.join("flipper/flipper.parquet")).unwrap())
            .finish()
            .unwrap();
    let label_a = flipper.column("label_a").unwrap().f64().unwrap();
    let label_b = flipper.column("label_b").unwrap().f64().unwrap();
    assert!((0..flipper.height())
        .all(|i| label_a.get(i).unwrap() < 5.5 && label_b.get(i).unwrap() >= 5.5));

    let linear = fixture_model(
        &root.join("linear"),
        "regression_linear",
        "regression",
        &["hba1c"],
    );
    let forest = fixture_model(
        &root.join("forest"),
        "regression_forest",
        "regression",
        &["hba1c"],
    );

    for sim_type in ["blackout", "jitter", "flipper"] {
        let mut args = regression_args("evaluate", sim_type, &sim_root, &targets);
        args.extend([
            "--model",
            linear.to_str().unwrap(),
            "--flip-threshold",
            "5.5",
        ]);
        let out = run_cli(&args);
        assert!(
            out.status.success(),
            "evaluate {sim_type} failed: {}",
            String::from_utf8_lossy(&out.stderr)
        );
        let result: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
        assert!(
            result["baselines"]["rmse"].as_f64().unwrap() < 0.2,
            "{result}"
        );
        assert!(
            result["baselines"]["r2"].as_f64().unwrap() > 0.9,
            "{result}"
        );
        assert!(result["evaluations"]["metric_based"]["r2"]["resiliency"].is_number());
        assert!(result.get("multi").is_none());
        let detail = &result["regression"];
        assert_eq!(detail["target"], "hba1c");
        assert_eq!(detail["n_rows"], 12);
        match sim_type {
            "blackout" => {
                assert!(detail["blackout_rmse"].is_number());
                assert!(detail["blackout_r2"].is_number());
            }
            "jitter" => assert!(detail["jitter_prediction_sd"].is_number()),
            _ => {
                assert_eq!(detail["flip_threshold"], 5.5);
                assert!(result["evaluations"]["metric_invariant"]["flipper_stability"].is_number());
            }
        }
    }

    let predictions = root.join("predictions.parquet");
    let comparison = root.join("compare.json");
    let mut args = regression_args("compare", "jitter", &sim_root, &targets);
    args.extend([
        "--model",
        linear.to_str().unwrap(),
        "--model",
        forest.to_str().unwrap(),
        "--flip-threshold",
        "5.5",
        "--predictions-out",
        predictions.to_str().unwrap(),
        "--output",
        comparison.to_str().unwrap(),
    ]);
    let out = run_cli(&args);
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let result: serde_json::Value =
        serde_json::from_slice(&fs::read(&comparison).unwrap()).unwrap();
    let df = ParquetReader::new(fs::File::open(&predictions).unwrap())
        .finish()
        .unwrap();
    assert_eq!(
        df.get_column_names(),
        [
            "source_row_id",
            "variant_name",
            "target",
            "regression_linear",
            "regression_forest",
            "all_agree"
        ]
    );
    // The JSON's clean RMSE is the RMSE of the clean rows in the parquet.
    let clean = df
        .clone()
        .lazy()
        .filter(col("variant_name").eq(lit("clean")))
        .collect()
        .unwrap();
    let target = clean.column("target").unwrap().f64().unwrap();
    let pred = clean.column("regression_linear").unwrap().f64().unwrap();
    let mse: f64 = (0..clean.height())
        .map(|i| (pred.get(i).unwrap() - target.get(i).unwrap()).powi(2))
        .sum::<f64>()
        / clean.height() as f64;
    let reported = result["evaluations"][0]["evaluation"]["baselines"]["rmse"]
        .as_f64()
        .unwrap();
    assert!(
        (mse.sqrt() - reported).abs() < 1e-9,
        "{} vs {reported}",
        mse.sqrt()
    );
    assert_eq!(df.column("all_agree").unwrap().null_count(), 0);
    let _ = fs::remove_dir_all(&root);
}

#[test]
fn regression_flipper_needs_the_cutoff_it_was_simulated_with() {
    let root = temp_dir("regression-cutoff");
    let (sim_root, targets) = simulate_regression(&root, Some("5.5"));
    let model = fixture_model(
        &root.join("m"),
        "regression_linear",
        "regression",
        &["hba1c"],
    );

    let mut args = regression_args("evaluate", "flipper", &sim_root, &targets);
    args.extend(["--model", model.to_str().unwrap()]);
    let out = run_cli(&args);
    assert!(!out.status.success());
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(err.contains("needs --flip-threshold"), "got: {err}");

    args.extend(["--flip-threshold", "9"]);
    let out = run_cli(&args);
    assert!(!out.status.success());
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(err.contains("another cutoff or task"), "got: {err}");
    let _ = fs::remove_dir_all(&root);
}

#[test]
fn regression_without_a_cutoff_skips_flipper() {
    let root = temp_dir("regression-no-cutoff");
    let (sim_root, _) = simulate_regression(&root, None);
    assert!(sim_root.join("jitter").is_dir());
    assert!(!sim_root.join("flipper").exists());
    let _ = fs::remove_dir_all(&root);
}

#[test]
fn regression_rejects_cohort_labels_and_classifier_models() {
    let root = temp_dir("regression-flags");
    let (sim_root, targets) = simulate_regression(&root, None);
    let labels = root.join("groups.parquet");
    write_groups(&labels, &[(103, &[1, 2])]);
    let model = fixture_model(
        &root.join("m"),
        "regression_linear",
        "regression",
        &["hba1c"],
    );

    let mut args = regression_args("evaluate", "jitter", &sim_root, &targets);
    args.extend([
        "--model",
        model.to_str().unwrap(),
        "--labels",
        labels.to_str().unwrap(),
    ]);
    let err = String::from_utf8_lossy(&run_cli(&args).stderr).to_string();
    assert!(
        err.contains("--labels holds cohorts for classifiers"),
        "got: {err}"
    );

    // A binary model scored as regression is refused at load.
    let binary = fixture_model(&root.join("b"), "binary_zipmap", "binary", &["0", "1"]);
    let mut args = regression_args("evaluate", "jitter", &sim_root, &targets);
    args.extend(["--model", binary.to_str().unwrap()]);
    let err = String::from_utf8_lossy(&run_cli(&args).stderr).to_string();
    assert!(err.contains("is a binary model"), "got: {err}");

    // And --targets is refused for a classifier.
    let out = run_cli(&[
        "evaluate",
        "jitter",
        "--sim-root",
        sim_root.to_str().unwrap(),
        "--labels",
        labels.to_str().unwrap(),
        "--targets",
        targets.to_str().unwrap(),
        "--model",
        binary.to_str().unwrap(),
    ]);
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(
        err.contains("--targets and --target-col are for --task regression"),
        "got: {err}"
    );
    let _ = fs::remove_dir_all(&root);
}

#[test]
fn regression_targets_must_cover_the_cohort() {
    let root = temp_dir("regression-missing");
    let (sim_root, _) = simulate_regression(&root, None);
    let partial = root.join("partial.parquet");
    write_regression_targets(&partial, Some("7"));
    let model = fixture_model(
        &root.join("m"),
        "regression_linear",
        "regression",
        &["hba1c"],
    );
    let mut args = regression_args("evaluate", "jitter", &sim_root, &partial);
    args.extend(["--model", model.to_str().unwrap()]);
    let out = run_cli(&args);
    assert!(!out.status.success());
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(
        err.contains("1 row(s) have no row in --targets (e.g. 7)"),
        "got: {err}"
    );
    let _ = fs::remove_dir_all(&root);
}

#[test]
fn version_flag_prints_crate_version() {
    let out = Command::new(pasteur_cli())
        .arg("--version")
        .output()
        .unwrap();
    assert!(out.status.success());
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert_eq!(
        stdout.trim(),
        format!("pasteur-cli {}", env!("CARGO_PKG_VERSION"))
    );
}
