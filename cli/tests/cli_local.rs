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
