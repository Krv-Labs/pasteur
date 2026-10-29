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
fn simulate_and_evaluate_sweep() {
    let root = temp_dir("sweep");
    let input = root.join("clean.parquet");
    let labels = root.join("groups.parquet");
    let sim_root = root.join("sim");
    write_two_feature_input(&input);
    write_groups(&labels, &[(11, &[1, 2, 3, 5, 9])]);
    #[rustfmt::skip]
    let out = run_cli(&[
        "simulate", "--input", input.to_str().unwrap(), "--output", sim_root.to_str().unwrap(),
        "--feature", "f1", "--jitter-iters", "1",
        "--sweep", "f1=0.5", "--sweep", "f2",
        "--sweep-range", "-2:2", "--sweep-steps", "9",
    ]);
    assert!(
        out.status.success(),
        "simulate failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );

    let grid = ParquetReader::new(fs::File::open(sim_root.join("sweep/sweep.parquet")).unwrap())
        .finish()
        .unwrap();
    assert_eq!(grid.height(), 12 * 2 * 9);
    let ids = grid.column("source_row_id").unwrap().str().unwrap();
    assert_eq!(ids.get(0), Some("sweep_1_f1_step_0"));
    assert_eq!(ids.get(grid.height() - 1), Some("sweep_12_f2_step_8"));

    let model = fixture_model(&root.join("models"), "binary_zipmap", "binary", &["0", "1"]);
    let result = evaluate_json("sweep", &sim_root, &labels, &model, "binary", &["11"]);
    let sweep = result["sweep"].as_array().unwrap();
    assert_eq!(sweep.len(), 2);
    assert_eq!(sweep[0]["feature"], "f1");
    assert_eq!(sweep[0]["cutoff"], 0.5);
    assert_eq!(sweep[1]["feature"], "f2");
    for s in sweep {
        assert_eq!(s["n_paths"], 12);
        assert_eq!(s["oracle"], false);
        assert!(s["never_flipped"].is_number());
        assert!(s["extra_flips_rate"].is_number());
        assert!(s["monotonic_rate"].is_null());
    }

    // Other variants don't report sweep.
    let jitter = evaluate_json("jitter", &sim_root, &labels, &model, "binary", &["11"]);
    assert!(jitter.get("sweep").is_none());

    let err = run_cli(&[
        "simulate",
        "--input",
        input.to_str().unwrap(),
        "--output",
        root.join("bad").to_str().unwrap(),
        "--feature",
        "f1",
        "--sweep",
        "f1",
    ]);
    assert!(!err.status.success());
    assert!(String::from_utf8_lossy(&err.stderr).contains("--sweep-range"));
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

/// ONNX `evaluate` and `compare` output on the fixtures, checked against
/// goldens under `tests/golden/` so a change to how evaluation inputs are
/// built cannot silently change ONNX scores. `created_at` is dropped and
/// prediction rows are sorted by row key; HashMap iteration order would
/// otherwise reorder them run to run. Regenerate with
/// `PASTEUR_UPDATE_GOLDEN=1 cargo test -p pasteur-cli onnx_output_matches_golden`.
#[test]
fn onnx_output_matches_golden() {
    let root = temp_dir("golden");
    let golden_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/golden/onnx_outputs.json");
    let mut actual = serde_json::Map::new();

    let classifiers: [(&str, &str, &[&str], &[&str]); 3] = [
        ("binary", "binary_zipmap", &["0", "1"], &["1"]),
        (
            "multiclass",
            "multiclass_tensor",
            &["low", "mid", "high"],
            &["1", "2", "3"],
        ),
        (
            "multilabel",
            "multilabel_multioutput",
            &["a", "b", "c"],
            &["11", "12", "1"],
        ),
    ];
    for (task, fixture, classes, groups) in classifiers {
        let dir = root.join(task);
        fs::create_dir_all(&dir).unwrap();
        let (sim_root, labels) = simulate_multi(&dir, task, groups);
        let model_a = fixture_model(&dir.join("models"), fixture, task, classes);
        let model_b = model_a.with_file_name(format!("{fixture}_copy.onnx"));
        fs::copy(&model_a, &model_b).unwrap();
        let mut base: Vec<String> = vec![
            "--sim-root".into(),
            sim_root.display().to_string(),
            "--labels".into(),
            labels.display().to_string(),
            "--task".into(),
            task.into(),
        ];
        for g in groups {
            base.extend(["--positive-group-id".into(), g.to_string()]);
        }
        record_golden_runs(&mut actual, task, &dir, &base, &[&model_a, &model_b]);
    }

    let dir = root.join("regression");
    fs::create_dir_all(&dir).unwrap();
    let (sim_root, targets) = simulate_regression(&dir, Some("5.5"));
    let linear = fixture_model(
        &dir.join("linear"),
        "regression_linear",
        "regression",
        &["hba1c"],
    );
    let forest = fixture_model(
        &dir.join("forest"),
        "regression_forest",
        "regression",
        &["hba1c"],
    );
    let base: Vec<String> = [
        "--sim-root",
        sim_root.to_str().unwrap(),
        "--targets",
        targets.to_str().unwrap(),
        "--target-col",
        "hba1c",
        "--task",
        "regression",
        "--flip-threshold",
        "5.5",
    ]
    .map(String::from)
    .to_vec();
    record_golden_runs(&mut actual, "regression", &dir, &base, &[&linear, &forest]);

    // Compared as text: parsing the golden back can move a float by one ULP.
    let actual = serde_json::to_string_pretty(&serde_json::Value::Object(actual)).unwrap() + "\n";
    if std::env::var_os("PASTEUR_UPDATE_GOLDEN").is_some() {
        fs::create_dir_all(golden_path.parent().unwrap()).unwrap();
        fs::write(&golden_path, actual).unwrap();
    } else {
        let expected = fs::read_to_string(&golden_path).unwrap();
        let (a, e): (serde_json::Value, serde_json::Value) = (
            serde_json::from_str(&actual).unwrap(),
            serde_json::from_str(&expected).unwrap(),
        );
        for (key, value) in e.as_object().unwrap() {
            let same =
                serde_json::to_string(&a[key]).unwrap() == serde_json::to_string(value).unwrap();
            assert!(same, "{key} differs from {}", golden_path.display());
        }
        assert!(
            actual == expected,
            "output differs from {}",
            golden_path.display()
        );
    }
    let _ = fs::remove_dir_all(&root);
}

/// Runs `evaluate` (first model) and `compare` (all models, with
/// `--predictions-out`) for each sim type and records the normalised output.
fn record_golden_runs(
    out: &mut serde_json::Map<String, serde_json::Value>,
    task: &str,
    dir: &Path,
    base: &[String],
    models: &[&Path],
) {
    for sim_type in ["blackout", "jitter", "flipper"] {
        let mut args: Vec<String> = vec!["evaluate".into(), sim_type.into()];
        args.extend_from_slice(base);
        args.extend(["--model".into(), models[0].display().to_string()]);
        out.insert(
            format!("{task}/evaluate/{sim_type}"),
            golden_json(&run_golden_cli(&args)),
        );

        let predictions = dir.join(format!("predictions_{sim_type}.parquet"));
        let mut args: Vec<String> = vec!["compare".into(), sim_type.into()];
        args.extend_from_slice(base);
        for model in models {
            args.extend(["--model".into(), model.display().to_string()]);
        }
        let comparison = dir.join(format!("compare_{sim_type}.json"));
        args.extend([
            "--predictions-out".into(),
            predictions.display().to_string(),
            "--output".into(),
            comparison.display().to_string(),
        ]);
        run_golden_cli(&args);
        out.insert(
            format!("{task}/compare/{sim_type}"),
            golden_json(&fs::read(&comparison).unwrap()),
        );
        out.insert(
            format!("{task}/predictions/{sim_type}"),
            predictions_json(&predictions),
        );
    }
}

fn run_golden_cli(args: &[String]) -> Vec<u8> {
    let out = Command::new(pasteur_cli()).args(args).output().unwrap();
    assert!(
        out.status.success(),
        "{args:?} failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    out.stdout
}

fn golden_json(stdout: &[u8]) -> serde_json::Value {
    fn strip(value: &mut serde_json::Value) {
        match value {
            serde_json::Value::Object(map) => {
                map.remove("created_at");
                map.values_mut().for_each(strip);
            }
            serde_json::Value::Array(items) => items.iter_mut().for_each(strip),
            _ => {}
        }
    }
    let mut value: serde_json::Value = serde_json::from_slice(stdout).unwrap();
    strip(&mut value);
    value
}

/// The predictions parquet as `{column: [values]}`, rows sorted by
/// `(variant_name, source_row_id)`.
fn predictions_json(path: &Path) -> serde_json::Value {
    let df = ParquetReader::new(fs::File::open(path).unwrap())
        .finish()
        .unwrap()
        .sort(["variant_name", "source_row_id"], Default::default())
        .unwrap();
    let mut columns = serde_json::Map::new();
    for column in df.columns() {
        let values: Vec<serde_json::Value> = match column.dtype() {
            DataType::String => {
                let ca = column.str().unwrap();
                (0..df.height())
                    .map(|i| serde_json::json!(ca.get(i)))
                    .collect()
            }
            _ => {
                let cast = column.cast(&DataType::Float64).unwrap();
                let ca = cast.f64().unwrap();
                (0..df.height())
                    .map(|i| serde_json::json!(ca.get(i)))
                    .collect()
            }
        };
        columns.insert(column.name().to_string(), values.into());
    }
    columns.into()
}

/// Runs `compare` with `extra` flags and returns the normalised comparison
/// JSON as text.
fn compare_text(dir: &Path, sim_type: &str, base: &[String], extra: &[String]) -> String {
    let output = dir.join("comparison.json");
    let mut args: Vec<String> = vec!["compare".into(), sim_type.into()];
    args.extend_from_slice(base);
    args.extend_from_slice(extra);
    args.extend(["--output".into(), output.display().to_string()]);
    run_golden_cli(&args);
    let text = serde_json::to_string_pretty(&golden_json(&fs::read(&output).unwrap())).unwrap();
    let _ = fs::remove_file(&output);
    text
}

fn classifier_base(sim_root: &Path, labels: &Path, task: &str, groups: &[&str]) -> Vec<String> {
    let mut base: Vec<String> = vec![
        "--sim-root".into(),
        sim_root.display().to_string(),
        "--labels".into(),
        labels.display().to_string(),
        "--task".into(),
        task.into(),
    ];
    for g in groups {
        base.extend(["--positive-group-id".into(), g.to_string()]);
    }
    base
}

/// `compare --model x.onnx --predictions-out p.parquet` then
/// `compare --predictions p.parquet` (and `evaluate`) give the same metrics.
#[test]
fn precomputed_predictions_reproduce_onnx_metrics() {
    let root = temp_dir("precomputed_golden");
    let classifiers: [(&str, &str, &[&str], &[&str]); 3] = [
        ("binary", "binary_zipmap", &["0", "1"], &["1"]),
        (
            "multiclass",
            "multiclass_tensor",
            &["low", "mid", "high"],
            &["1", "2", "3"],
        ),
        (
            "multilabel",
            "multilabel_multioutput",
            &["a", "b", "c"],
            &["11", "12", "1"],
        ),
    ];
    let mut runs: Vec<(PathBuf, String, Vec<String>)> = Vec::new();
    for (task, fixture, classes, groups) in classifiers {
        let dir = root.join(task);
        fs::create_dir_all(&dir).unwrap();
        let (sim_root, labels) = simulate_multi(&dir, task, groups);
        let model = fixture_model(&dir.join("models"), fixture, task, classes);
        runs.push((
            model,
            fixture.into(),
            classifier_base(&sim_root, &labels, task, groups),
        ));
    }
    let dir = root.join("regression");
    fs::create_dir_all(&dir).unwrap();
    let (sim_root, targets) = simulate_regression(&dir, Some("5.5"));
    let linear = fixture_model(
        &dir.join("linear"),
        "regression_linear",
        "regression",
        &["hba1c"],
    );
    let base = [
        "--sim-root",
        sim_root.to_str().unwrap(),
        "--targets",
        targets.to_str().unwrap(),
        "--target-col",
        "hba1c",
        "--task",
        "regression",
        "--flip-threshold",
        "5.5",
    ]
    .map(String::from)
    .to_vec();
    runs.push((linear, "regression_linear".into(), base));

    for (model, label, base) in &runs {
        for sim_type in ["blackout", "jitter", "flipper"] {
            let preds = model.with_file_name(format!("preds_{sim_type}.parquet"));
            let onnx = compare_text(
                &root,
                sim_type,
                base,
                &[
                    "--model".into(),
                    model.display().to_string(),
                    "--predictions-out".into(),
                    preds.display().to_string(),
                ],
            );
            let precomputed = compare_text(
                &root,
                sim_type,
                base,
                &["--predictions".into(), preds.display().to_string()],
            );
            let relabelled =
                onnx.replace(&format!("\"{label}\""), &format!("\"preds_{sim_type}\""));
            assert_eq!(precomputed, relabelled, "{label} {sim_type}");

            let evaluate = |flag: &str, path: &Path| {
                let mut args: Vec<String> = vec!["evaluate".into(), sim_type.into()];
                args.extend_from_slice(base);
                args.extend([flag.into(), path.display().to_string()]);
                serde_json::to_string_pretty(&golden_json(&run_golden_cli(&args))).unwrap()
            };
            assert_eq!(
                evaluate("--predictions", &preds),
                evaluate("--model", model),
                "{label} {sim_type} evaluate"
            );
        }
    }
    let _ = fs::remove_dir_all(&root);
}

#[test]
fn precomputed_predictions_mix_with_onnx_and_refuse_missing_rows() {
    let root = temp_dir("precomputed_mix");
    let groups = ["1", "2", "3"];
    let (sim_root, labels) = simulate_multi(&root, "multiclass", &groups);
    let model = fixture_model(
        &root.join("models"),
        "multiclass_tensor",
        "multiclass",
        &["low", "mid", "high"],
    );
    let base = classifier_base(&sim_root, &labels, "multiclass", &groups);
    let preds = root.join("preds.parquet");
    compare_text(
        &root,
        "jitter",
        &base,
        &[
            "--model".into(),
            model.display().to_string(),
            "--predictions-out".into(),
            preds.display().to_string(),
        ],
    );

    // Mixed: the precomputed copy scores exactly like the ONNX model.
    let mixed_out = root.join("mixed.parquet");
    let mixed: serde_json::Value = serde_json::from_str(&compare_text(
        &root,
        "jitter",
        &base,
        &[
            "--model".into(),
            model.display().to_string(),
            "--predictions".into(),
            format!("scored_elsewhere={}", preds.display()),
            "--predictions-out".into(),
            mixed_out.display().to_string(),
        ],
    ))
    .unwrap();
    let text = serde_json::to_string(&mixed).unwrap();
    assert!(text.contains("scored_elsewhere"), "{text}");
    let written = ParquetReader::new(fs::File::open(&mixed_out).unwrap())
        .finish()
        .unwrap();
    let onnx_low = written.column("multiclass_tensor__low").unwrap();
    let pre_low = written.column("scored_elsewhere__low").unwrap();
    assert!(onnx_low.equals(pre_low));

    // A name clash is refused.
    let mut args: Vec<String> = vec!["compare".into(), "jitter".into()];
    args.extend_from_slice(&base);
    args.extend([
        "--model".into(),
        model.display().to_string(),
        "--predictions".into(),
        format!("multiclass_tensor={}", preds.display()),
    ]);
    let out = Command::new(pasteur_cli()).args(&args).output().unwrap();
    assert!(!out.status.success());
    assert!(String::from_utf8_lossy(&out.stderr).contains("both named"));

    // Drop one row: evaluation must fail rather than fill it.
    let mut df = ParquetReader::new(fs::File::open(&preds).unwrap())
        .finish()
        .unwrap();
    df = df.slice(1, df.height() - 1);
    let short = root.join("short.parquet");
    ParquetWriter::new(&mut fs::File::create(&short).unwrap())
        .finish(&mut df)
        .unwrap();
    let mut args: Vec<String> = vec!["evaluate".into(), "jitter".into()];
    args.extend_from_slice(&base);
    args.extend(["--predictions".into(), short.display().to_string()]);
    let out = Command::new(pasteur_cli()).args(&args).output().unwrap();
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("no prediction for variant"), "{stderr}");

    // --model and --predictions are exclusive on evaluate.
    args.extend(["--model".into(), model.display().to_string()]);
    let out = Command::new(pasteur_cli()).args(&args).output().unwrap();
    assert!(!out.status.success());
    assert!(String::from_utf8_lossy(&out.stderr).contains("cannot be used with"));
    let _ = fs::remove_dir_all(&root);
}

/// `(variant_name, source_row_id, values of cols)` per row of a
/// `--predictions-out` file.
fn prediction_rows(path: &Path, cols: &[&str]) -> Vec<(String, String, Vec<f64>)> {
    let df = ParquetReader::new(fs::File::open(path).unwrap())
        .finish()
        .unwrap();
    let text = |name: &str| df.column(name).unwrap().str().unwrap().clone();
    let (variant, id) = (text("variant_name"), text("source_row_id"));
    let values: Vec<Float64Chunked> = cols
        .iter()
        .map(|c| df.column(c).unwrap().f64().unwrap().clone())
        .collect();
    (0..df.height())
        .map(|i| {
            (
                variant.get(i).unwrap().to_string(),
                id.get(i).unwrap().to_string(),
                values.iter().map(|v| v.get(i).unwrap()).collect(),
            )
        })
        .collect()
}

fn write_oracle(path: &Path, rows: &[(String, String, Vec<i64>)], label_cols: &[&str]) {
    let mut columns: Vec<Column> = vec![
        Series::new(
            "variant_name".into(),
            rows.iter().map(|r| r.0.as_str()).collect::<Vec<_>>(),
        )
        .into(),
        Series::new(
            "source_row_id".into(),
            rows.iter().map(|r| r.1.as_str()).collect::<Vec<_>>(),
        )
        .into(),
    ];
    for (j, name) in label_cols.iter().enumerate() {
        let values: Vec<i64> = rows.iter().map(|r| r.2[j]).collect();
        columns.push(Series::new((*name).into(), values).into());
    }
    let mut df = DataFrame::new(rows.len(), columns).unwrap();
    ParquetWriter::new(&mut fs::File::create(path).unwrap())
        .finish(&mut df)
        .unwrap();
}

/// Runs `args`, asserting success, and parses stdout as JSON.
fn run_json(args: &[&str]) -> serde_json::Value {
    let out = run_cli(args);
    assert!(
        out.status.success(),
        "{args:?} failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    serde_json::from_slice(&out.stdout).unwrap()
}

/// A two-feature binary bundle simulated with strong jitter and six flipper
/// pairs, plus the fixture binary model.
fn binary_oracle_bundle(root: &Path) -> (PathBuf, PathBuf, PathBuf) {
    let input = root.join("clean.parquet");
    let labels = root.join("groups.parquet");
    let sim_root = root.join("sim");
    write_two_feature_input(&input);
    write_groups(&labels, &[(103, &[7, 8, 9, 10, 11, 12])]);
    let out = run_cli(&[
        "simulate",
        "--input",
        input.to_str().unwrap(),
        "--output",
        sim_root.to_str().unwrap(),
        "--feature",
        "f1",
        "--labels",
        labels.to_str().unwrap(),
        "--jitter-iters",
        "3",
        "--jitter-scale",
        "2.0",
        "--flipper-pairs",
        "6",
        "--flipper-steps",
        "9",
    ]);
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let model = fixture_model(&root.join("models"), "binary_zipmap", "binary", &["0", "1"]);
    (sim_root, labels, model)
}

/// Per-row model probabilities for one simulation type, via `compare`.
fn predictions_for(
    root: &Path,
    sim_type: &str,
    common: &[&str],
    cols: &[&str],
) -> Vec<(String, String, Vec<f64>)> {
    let path = root.join(format!("preds_{sim_type}.parquet"));
    let mut args = vec!["compare", sim_type];
    args.extend(common);
    args.extend(["--predictions-out", path.to_str().unwrap()]);
    let out = run_cli(&args);
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    prediction_rows(&path, cols)
}

#[test]
fn oracle_jitter_splits_flips_by_label_change() {
    let root = temp_dir("oracle-jitter");
    let (sim_root, labels, model) = binary_oracle_bundle(&root);
    let common = [
        "--sim-root",
        sim_root.to_str().unwrap(),
        "--labels",
        labels.to_str().unwrap(),
        "--model",
        model.to_str().unwrap(),
    ];
    let preds = predictions_for(&root, "jitter", &common, &["binary_zipmap"]);
    let decide = |p: f64| i64::from(p >= 0.5);
    let oracle = root.join("oracle.parquet");
    let evaluate = |oracle: &Path| {
        let mut args = vec!["evaluate", "jitter"];
        args.extend(common);
        args.extend(["--oracle-labels", oracle.to_str().unwrap()]);
        run_json(&args)["oracle"].clone()
    };

    // The oracle agrees with every decision, so wherever jitter moved a
    // decision the oracle moved too: the model always followed it.
    let agree: Vec<_> = preds
        .iter()
        .map(|(v, id, p)| (v.clone(), id.clone(), vec![decide(p[0])]))
        .collect();
    write_oracle(&oracle, &agree, &["label"]);
    let scores = evaluate(&oracle);
    for (variant, acc) in scores["accuracy"].as_object().unwrap() {
        assert_eq!(acc, 1.0, "{variant}");
    }
    let jitter = &scores["jitter"];
    assert!(jitter["n_label_changed"].as_u64().unwrap() > 0, "{jitter}");
    assert_eq!(jitter["flip_rate_label_changed"], 1.0);
    assert_eq!(jitter["flip_rate_label_preserved"], 0.0);

    // The oracle never changes its clean answer, so every decision jitter
    // moved is a flip the label did not justify.
    let clean: std::collections::HashMap<&str, i64> = preds
        .iter()
        .filter(|(v, _, _)| v == "clean")
        .map(|(_, id, p)| (id.as_str(), decide(p[0])))
        .collect();
    let fixed: Vec<_> = preds
        .iter()
        .map(|(v, id, _)| (v.clone(), id.clone(), vec![clean[id.as_str()]]))
        .collect();
    write_oracle(&oracle, &fixed, &["label"]);
    let draws: Vec<_> = preds.iter().filter(|(v, _, _)| v != "clean").collect();
    let moved = draws
        .iter()
        .filter(|(_, id, p)| decide(p[0]) != clean[id.as_str()])
        .count();
    let jitter = evaluate(&oracle)["jitter"].clone();
    assert_eq!(jitter["n_label_changed"], 0);
    assert_eq!(jitter["n_label_preserved"], draws.len());
    assert!(jitter["flip_rate_label_changed"].is_null());
    let rate = jitter["flip_rate_label_preserved"].as_f64().unwrap();
    assert!((rate - moved as f64 / draws.len() as f64).abs() < 1e-12);

    // A row the oracle does not cover is an error, not a silent default.
    write_oracle(&oracle, &fixed[1..], &["label"]);
    let mut args = vec!["evaluate", "jitter"];
    args.extend(common);
    args.extend(["--oracle-labels", oracle.to_str().unwrap()]);
    let out = run_cli(&args);
    assert!(!out.status.success());
    assert!(String::from_utf8_lossy(&out.stderr).contains("miss 1 of"));
    let _ = fs::remove_dir_all(&root);
}

#[test]
fn oracle_flipper_boundary_error() {
    let root = temp_dir("oracle-flipper");
    let (sim_root, labels, model) = binary_oracle_bundle(&root);
    let common = [
        "--sim-root",
        sim_root.to_str().unwrap(),
        "--labels",
        labels.to_str().unwrap(),
        "--model",
        model.to_str().unwrap(),
    ];
    let preds = predictions_for(&root, "flipper", &common, &["binary_zipmap"]);
    // Nine steps, t = step / 8. The oracle turns positive at step 3, so it
    // flips halfway between t = 2/8 and t = 3/8.
    let rows: Vec<_> = preds
        .iter()
        .map(|(v, id, p)| {
            let label = match id.rsplit_once("_step_") {
                Some((_, step)) => i64::from(step.parse::<u32>().unwrap() >= 3),
                None => i64::from(p[0] >= 0.5),
            };
            (v.clone(), id.clone(), vec![label])
        })
        .collect();
    let oracle = root.join("oracle.parquet");
    write_oracle(&oracle, &rows, &["label"]);
    let mut args = vec!["evaluate", "flipper"];
    args.extend(common);
    args.extend(["--oracle-labels", oracle.to_str().unwrap()]);
    let flipper = run_json(&args)["oracle"]["flipper"].clone();

    // Each pair's model flip_t: where p crosses 0.5, interpolated between
    // the bracketing steps.
    let mut paths: std::collections::BTreeMap<&str, Vec<(u32, f64)>> = Default::default();
    for (_, id, p) in preds.iter().filter(|(v, _, _)| v == "flipper") {
        let (pair, step) = id.rsplit_once("_step_").unwrap();
        paths
            .entry(pair)
            .or_default()
            .push((step.parse().unwrap(), p[0]));
    }
    let mut errors: Vec<f64> = paths
        .values_mut()
        .filter_map(|path| {
            path.sort_by_key(|(step, _)| *step);
            path.windows(2).find_map(|w| {
                let ((s0, p0), (_, p1)) = (w[0], w[1]);
                ((p0 >= 0.5) != (p1 >= 0.5))
                    .then(|| (s0 as f64 + (0.5 - p0) / (p1 - p0)) / 8.0 - 0.3125)
            })
        })
        .collect();
    errors.sort_by(f64::total_cmp);
    assert!(!errors.is_empty(), "no pair flips: {flipper}");
    let quantile = |q: f64| {
        let pos = q * (errors.len() - 1) as f64;
        let (lo, hi) = (pos.floor() as usize, pos.ceil() as usize);
        errors[lo] + (pos - lo as f64) * (errors[hi] - errors[lo])
    };

    assert_eq!(flipper["n_pairs"], paths.len());
    assert_eq!(flipper["n_pairs_compared"], errors.len());
    assert_eq!(flipper["oracle_never_flipped"], 0.0);
    let median = flipper["boundary_error_median"].as_f64().unwrap();
    assert!(
        (median - quantile(0.5)).abs() < 1e-9,
        "{median} vs {errors:?}"
    );
    let iqr = flipper["boundary_error_iqr"].as_f64().unwrap();
    assert!((iqr - (quantile(0.75) - quantile(0.25))).abs() < 1e-9);
    let _ = fs::remove_dir_all(&root);
}

#[test]
fn oracle_multiclass_abstain_on_blackout() {
    let root = temp_dir("oracle-abstain");
    let groups = ["1", "2", "3"];
    let (sim_root, labels) = simulate_multi(&root, "multiclass", &groups);
    let model = fixture_model(
        &root.join("models"),
        "multiclass_tensor",
        "multiclass",
        &["low", "mid", "high"],
    );
    let mut common = vec![
        "--sim-root",
        sim_root.to_str().unwrap(),
        "--labels",
        labels.to_str().unwrap(),
        "--model",
        model.to_str().unwrap(),
        "--task",
        "multiclass",
    ];
    for g in &groups {
        common.extend(["--positive-group-id", g]);
    }
    let cols = [
        "multiclass_tensor__low",
        "multiclass_tensor__mid",
        "multiclass_tensor__high",
    ];
    let preds = predictions_for(&root, "blackout", &common, &cols);
    let argmax = |p: &[f64]| (0..3).max_by(|&a, &b| p[a].total_cmp(&p[b])).unwrap();
    let one_hot = |c: usize| (0..3).map(|j| i64::from(j == c)).collect::<Vec<_>>();
    // The oracle says "mid" (cannot decide) for odd-numbered blackout rows
    // and agrees with the model everywhere else.
    let oracle_class = |v: &str, id: &str, p: &[f64]| {
        if v == "blackout" && id.parse::<u32>().unwrap() % 2 == 1 {
            1
        } else {
            argmax(p)
        }
    };
    let rows: Vec<_> = preds
        .iter()
        .map(|(v, id, p)| (v.clone(), id.clone(), one_hot(oracle_class(v, id, p))))
        .collect();
    let oracle = root.join("oracle.parquet");
    write_oracle(&oracle, &rows, &["group_1", "group_2", "group_3"]);

    let mut expected = [[0usize; 2]; 2]; // [oracle decided][model decided]
    for (v, id, p) in preds.iter().filter(|(v, _, _)| v == "blackout") {
        let oracle_decided = oracle_class(v, id, p) != 1;
        expected[usize::from(oracle_decided)][usize::from(argmax(p) != 1)] += 1;
    }

    let mut args = vec!["evaluate", "blackout"];
    args.extend(&common);
    args.extend(["--oracle-labels", oracle.to_str().unwrap()]);
    args.extend(["--abstain-class", "mid"]);
    let abstain = run_json(&args)["oracle"]["abstain"].clone();
    assert_eq!(abstain["class"], "mid");
    let c = &abstain["confusion"];
    assert_eq!(c["oracle_abstain_model_abstain"], expected[0][0]);
    assert_eq!(c["oracle_abstain_model_decided"], expected[0][1]);
    assert_eq!(c["oracle_decided_model_abstain"], expected[1][0]);
    assert_eq!(c["oracle_decided_model_decided"], expected[1][1]);
    let missing = (expected[0][0] + expected[0][1]) as f64;
    assert!(missing > 0.0);
    let share = abstain["abstain_when_missing"].as_f64().unwrap();
    assert!((share - expected[0][0] as f64 / missing).abs() < 1e-12);
    let decided = abstain["decided_when_missing"].as_f64().unwrap();
    assert!((decided - expected[0][1] as f64 / missing).abs() < 1e-12);

    // The abstain class must be one of the model's classes, and needs an oracle.
    let last = args.len() - 1;
    args[last] = "unsure";
    let out = run_cli(&args);
    assert!(String::from_utf8_lossy(&out.stderr).contains("not one of"));
    let mut no_oracle = vec!["evaluate", "blackout"];
    no_oracle.extend(&common);
    no_oracle.extend(["--abstain-class", "mid"]);
    assert!(!run_cli(&no_oracle).status.success());
    let _ = fs::remove_dir_all(&root);
}

/// `(variant_name, source_row_id, f1)` for every clean and sweep row.
fn sweep_rows(sim_root: &Path) -> Vec<(String, String, f64)> {
    ["clean", "sweep"]
        .iter()
        .flat_map(|variant| {
            let path = sim_root.join(format!("{variant}/{variant}.parquet"));
            let df = ParquetReader::new(fs::File::open(path).unwrap())
                .finish()
                .unwrap();
            let ids = df.column("source_row_id").unwrap().str().unwrap().clone();
            let f1 = df.column("f1").unwrap().f64().unwrap().clone();
            (0..df.height())
                .map(|i| {
                    let (id, v) = (ids.get(i).unwrap(), f1.get(i).unwrap());
                    (variant.to_string(), id.to_string(), v)
                })
                .collect::<Vec<_>>()
        })
        .collect()
}

/// A precomputed predictions file: the row key plus one Float64 column per
/// `(name, value)` that `score` returns.
fn write_scored(
    path: &Path,
    rows: &[(String, String, f64)],
    names: &[&str],
    score: impl Fn(f64) -> Vec<f64>,
) {
    let mut columns: Vec<Column> = vec![
        Series::new(
            "variant_name".into(),
            rows.iter().map(|r| r.0.as_str()).collect::<Vec<_>>(),
        )
        .into(),
        Series::new(
            "source_row_id".into(),
            rows.iter().map(|r| r.1.as_str()).collect::<Vec<_>>(),
        )
        .into(),
    ];
    let scores: Vec<Vec<f64>> = rows.iter().map(|r| score(r.2)).collect();
    for (j, name) in names.iter().enumerate() {
        let values: Vec<f64> = scores.iter().map(|s| s[j]).collect();
        columns.push(Series::new((*name).into(), values).into());
    }
    let mut df = DataFrame::new(rows.len(), columns).unwrap();
    ParquetWriter::new(&mut fs::File::create(path).unwrap())
        .finish(&mut df)
        .unwrap();
}

/// `simulate --sweep`, then precomputed predictions plus an oracle into
/// `compare`: the sweep metrics must locate each model's boundary.
#[test]
fn sweep_scores_precomputed_predictions_against_the_oracle() {
    let root = temp_dir("sweep-oracle");
    let input = root.join("clean.parquet");
    let labels = root.join("groups.parquet");
    let sim_root = root.join("sim");
    write_two_feature_input(&input);
    write_groups(
        &labels,
        &[
            (103, &[7, 8, 9, 10, 11, 12]),
            (1, &[1, 2, 3, 4]),
            (2, &[5, 6, 7, 8]),
            (3, &[9, 10, 11, 12]),
        ],
    );
    // f1 = 1 is the cutoff; 41 steps over 0..2 put a grid point every 0.05.
    #[rustfmt::skip]
    let out = run_cli(&[
        "simulate", "--input", input.to_str().unwrap(), "--output", sim_root.to_str().unwrap(),
        "--feature", "f1", "--jitter-iters", "1",
        "--sweep", "f1=1", "--sweep-range", "0:2", "--sweep-steps", "41",
    ]);
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let rows = sweep_rows(&sim_root);

    let at = |cut: f64| move |v: f64| vec![f64::from(u8::from(v >= cut))];
    write_scored(&root.join("perfect.parquet"), &rows, &["p"], at(1.0));
    write_scored(&root.join("constant.parquet"), &rows, &["p"], |_| vec![0.0]);
    write_scored(&root.join("shifted.parquet"), &rows, &["p"], at(1.1));
    let oracle = root.join("oracle.parquet");
    let truth: Vec<_> = rows
        .iter()
        .map(|(variant, id, v)| (variant.clone(), id.clone(), vec![i64::from(*v >= 1.0)]))
        .collect();
    write_oracle(&oracle, &truth, &["label"]);

    let mut args = vec![
        "compare",
        "sweep",
        "--sim-root",
        sim_root.to_str().unwrap(),
        "--labels",
        labels.to_str().unwrap(),
        "--oracle-labels",
        oracle.to_str().unwrap(),
    ];
    let preds: Vec<String> = ["perfect", "constant", "shifted"]
        .iter()
        .map(|m| root.join(format!("{m}.parquet")).display().to_string())
        .collect();
    for p in &preds {
        args.extend(["--predictions", p.as_str()]);
    }
    let result = run_json(&args);
    let scores: std::collections::HashMap<String, serde_json::Value> = result["evaluations"]
        .as_array()
        .unwrap()
        .iter()
        .map(|e| {
            let eval = &e["evaluation"];
            assert!(eval["oracle"]["accuracy"]["sweep"].is_number(), "{eval}");
            (
                e["model_label"].as_str().unwrap().to_string(),
                eval["sweep"][0].clone(),
            )
        })
        .collect();

    let perfect = &scores["perfect"];
    assert_eq!(perfect["oracle"], true);
    assert_eq!(perfect["n_paths"], 12);
    assert_eq!(perfect["boundary_offset"]["median"], 0.0);
    assert_eq!(perfect["monotonic_rate"], 1.0);
    assert_eq!(perfect["never_flipped"], 0.0);
    assert_eq!(perfect["accuracy"], 1.0);
    assert_eq!(scores["constant"]["never_flipped"], 1.0);
    assert_eq!(scores["constant"]["monotonic_rate"], 0.0);
    let shifted = &scores["shifted"];
    let offset = shifted["boundary_offset"]["median"].as_f64().unwrap();
    assert!((offset - 0.1).abs() <= 0.05, "{shifted}");
    assert_eq!(shifted["monotonic_rate"], 1.0);
    assert!(shifted["accuracy"].as_f64().unwrap() < 1.0);

    // Multiclass with an abstain class: the oracle answers `mid` (cannot
    // decide) near the cutoff, and so does this model.
    let band = |v: f64| match v {
        v if v < 0.9 => 0,
        v if v <= 1.1 => 1,
        _ => 2,
    };
    let one_hot = |c: usize| {
        (0..3)
            .map(|j| f64::from(u8::from(j == c)))
            .collect::<Vec<_>>()
    };
    let multi = root.join("multi.parquet");
    write_scored(&multi, &rows, &["low", "mid", "high"], |v| one_hot(band(v)));
    let truth: Vec<_> = rows
        .iter()
        .map(|(variant, id, v)| {
            let c = band(*v);
            (
                variant.clone(),
                id.clone(),
                (0..3).map(|j| i64::from(j == c)).collect(),
            )
        })
        .collect();
    write_oracle(&oracle, &truth, &["group_1", "group_2", "group_3"]);
    #[rustfmt::skip]
    let result = run_json(&[
        "evaluate", "sweep", "--sim-root", sim_root.to_str().unwrap(),
        "--labels", labels.to_str().unwrap(), "--task", "multiclass",
        "--positive-group-id", "1", "--positive-group-id", "2", "--positive-group-id", "3",
        "--predictions", multi.to_str().unwrap(),
        "--oracle-labels", oracle.to_str().unwrap(), "--abstain-class", "mid",
    ]);
    let sweep = &result["sweep"][0];
    assert_eq!(sweep["accuracy"], 1.0, "{sweep}");
    // Two flips per path (low → mid → high), both where the oracle's are.
    assert_eq!(sweep["extra_flips_rate"], 1.0);
    assert_eq!(result["oracle"]["accuracy"]["sweep"], 1.0);
    let _ = fs::remove_dir_all(&root);
}
