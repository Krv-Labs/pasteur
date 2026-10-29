use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use pasteur_core::{is_flipper_meta_column, EvaluationConfig, SimulationVariant, TaskType};
use pasteur_hf::layout;
use polars::prelude::*;

use crate::data::source::LabelSource;
use crate::io::read_parquet;

/// Every variant's `x` holds its features plus the row key
/// (`layout::ROW_KEY_COLS`), so a `Model` can tell which row it is scoring.
/// `feature_order` lists the features only; ONNX models select by it.
pub struct EvaluationInputs {
    pub clean_datasets: HashMap<String, SimulationVariant>,
    pub simulation_variants: HashMap<String, HashMap<String, SimulationVariant>>,
    pub feature_order: Vec<String>,
    pub eval_config: EvaluationConfig,
    pub source_row_id: Series,
    pub dataset_name: String,
}

pub fn build_evaluation_inputs(
    sim_root: &Path,
    sim_type: &str,
    source: &LabelSource,
    task: TaskType,
    dataset_name: &str,
    flip_threshold: f64,
) -> Result<EvaluationInputs> {
    let (clean_df, source_row_id, feature_order) = load_clean_baseline(sim_root)?;
    let labels = source.derive(task, &source_row_id)?;
    let clean = SimulationVariant {
        x: keyed_frame(&clean_df, "clean", &source_row_id)?,
        y: labels.clone(),
    };

    let variant_files = list_variant_files(sim_root, sim_type)?;
    let simulation_variants = load_simulation_variants(
        &variant_files,
        sim_type,
        dataset_name,
        &labels,
        task,
        &source_row_id,
    )?;

    let feature_order = if sim_type == "flipper" {
        simulation_variants
            .values()
            .find_map(|per_dataset| per_dataset.get(dataset_name))
            .map(|variant| flipper_feature_order(&variant.x))
            .unwrap_or(feature_order)
    } else {
        feature_order
    };

    let mut clean_datasets = HashMap::new();
    clean_datasets.insert(dataset_name.to_string(), clean);

    let eval_config = EvaluationConfig {
        n_jitter_iters: variant_files.len(),
        allow_partial: true,
        metrics: vec![match task {
            TaskType::Binary | TaskType::Multiclass | TaskType::Multilabel => "roc_auc",
            TaskType::Regression => "r2",
        }
        .to_string()],
        flip_threshold,
        task,
    };

    Ok(EvaluationInputs {
        clean_datasets,
        simulation_variants,
        feature_order,
        eval_config,
        source_row_id,
        dataset_name: dataset_name.to_string(),
    })
}

fn load_clean_baseline(sim_root: &Path) -> Result<(DataFrame, Series, Vec<String>)> {
    let clean_files = layout::list_parquet_files(sim_root, "clean")
        .map_err(|e| anyhow::anyhow!("{e}"))
        .context("could not find `clean/` parquet files")?;
    let clean_df = read_parquet(&clean_files[0])?;
    let source_row_id = clean_df
        .column(layout::SOURCE_ROW_ID_COL)?
        .as_materialized_series()
        .clone();
    let feature_order: Vec<String> = clean_df
        .get_column_names()
        .iter()
        .map(|s| s.to_string())
        .filter(|name| !is_bookkeeping_column(name))
        .collect();
    Ok((clean_df, source_row_id, feature_order))
}

/// Row key and `row_ordinal`: written by `simulate`, never model features.
fn is_bookkeeping_column(name: &str) -> bool {
    layout::ROW_KEY_COLS.contains(&name) || name == layout::ROW_ORDINAL_COL
}

fn has_column(df: &DataFrame, name: &str) -> bool {
    df.get_column_names().iter().any(|c| c.as_str() == name)
}

/// A variant file as an evaluation `x`: its columns plus the row key
/// (`variant_name`, `source_row_id`), without `row_ordinal`. A file with no
/// `source_row_id` takes the clean ids, row for row.
fn keyed_frame(df: &DataFrame, variant: &str, clean_ids: &Series) -> Result<DataFrame> {
    if has_column(df, layout::VARIANT_NAME_COL) {
        anyhow::bail!(
            "{variant} has a column named {:?}, which is reserved for the row key; rename it",
            layout::VARIANT_NAME_COL
        );
    }
    let mut out = df.clone();
    if has_column(&out, layout::ROW_ORDINAL_COL) {
        out = out.drop(layout::ROW_ORDINAL_COL)?;
    }
    if !has_column(&out, layout::SOURCE_ROW_ID_COL) {
        if out.height() != clean_ids.len() {
            anyhow::bail!(
                "{variant} has {} rows but no {} column and the clean data has {} rows",
                out.height(),
                layout::SOURCE_ROW_ID_COL,
                clean_ids.len()
            );
        }
        out.with_column(clean_ids.clone().into())?;
    }
    let height = out.height();
    out.with_column(
        StringChunked::full(layout::VARIANT_NAME_COL.into(), variant, height)
            .into_series()
            .into(),
    )?;
    Ok(out)
}

fn flipper_feature_order(df: &DataFrame) -> Vec<String> {
    df.get_column_names()
        .iter()
        .map(|s| s.to_string())
        .filter(|name| !is_bookkeeping_column(name) && !is_flipper_meta_column(name))
        .collect()
}

fn list_variant_files(sim_root: &Path, sim_type: &str) -> Result<Vec<PathBuf>> {
    layout::list_parquet_files(sim_root, sim_type)
        .map_err(|e| anyhow::anyhow!("{e}"))
        .with_context(|| format!("could not find `{sim_type}/` parquet files"))
}

fn load_simulation_variants(
    variant_files: &[PathBuf],
    sim_type: &str,
    dataset_name: &str,
    labels: &DataFrame,
    task: TaskType,
    clean_ids: &Series,
) -> Result<HashMap<String, HashMap<String, SimulationVariant>>> {
    let mut simulation_variants = HashMap::new();
    for path in variant_files {
        let name = variant_name(path, sim_type);
        let df = keyed_frame(&read_parquet(path)?, &name, clean_ids)?;
        let (x, y) = if sim_type == "flipper" {
            // Grid rows are synthetic patients with no label of their own:
            // 0 for classifiers, as before, and no target for regression.
            let placeholder: Vec<Column> = labels
                .get_column_names()
                .iter()
                .map(|name| placeholder_label((*name).clone(), task, df.height()).into())
                .collect();
            let height = df.height();
            (df, DataFrame::new(height, placeholder)?)
        } else {
            (df, labels.clone())
        };
        let mut per_dataset = HashMap::new();
        per_dataset.insert(dataset_name.to_string(), SimulationVariant { x, y });
        simulation_variants.insert(name, per_dataset);
    }
    Ok(simulation_variants)
}

fn placeholder_label(name: PlSmallStr, task: TaskType, n: usize) -> Series {
    match task {
        TaskType::Binary | TaskType::Multiclass | TaskType::Multilabel => {
            Int64Chunked::from_vec(name, vec![0i64; n]).into_series()
        }
        TaskType::Regression => Series::full_null(name, n, &DataType::Float64),
    }
}

pub fn variant_source_row_ids(variant: &SimulationVariant, fallback: &Series) -> Result<Series> {
    if variant
        .x
        .get_column_names()
        .iter()
        .any(|n| n.as_str() == layout::SOURCE_ROW_ID_COL)
    {
        Ok(variant
            .x
            .column(layout::SOURCE_ROW_ID_COL)?
            .as_materialized_series()
            .clone())
    } else if variant.x.height() == fallback.len() {
        Ok(fallback.clone())
    } else {
        anyhow::bail!(
            "variant has {} rows but no {} column and fallback has {} rows",
            variant.x.height(),
            layout::SOURCE_ROW_ID_COL,
            fallback.len()
        )
    }
}

pub fn flipper_feature_order_from_inputs(inputs: &EvaluationInputs, sim_type: &str) -> Vec<String> {
    if sim_type != "flipper" {
        return inputs.feature_order.clone();
    }
    for per_dataset in inputs.simulation_variants.values() {
        if let Some(variant) = per_dataset.get(&inputs.dataset_name) {
            return flipper_feature_order(&variant.x);
        }
    }
    inputs.feature_order.clone()
}

fn variant_name(path: &Path, sim_type: &str) -> String {
    path.file_stem()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| sim_type.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write(dir: &Path, name: &str, mut df: DataFrame) -> PathBuf {
        std::fs::create_dir_all(dir).unwrap();
        let path = dir.join(format!("{name}.parquet"));
        let mut file = std::fs::File::create(&path).unwrap();
        ParquetWriter::new(&mut file).finish(&mut df).unwrap();
        path
    }

    fn column_strings(df: &DataFrame, name: &str) -> Vec<String> {
        let ca = df.column(name).unwrap().str().unwrap().clone();
        (0..df.height())
            .map(|i| ca.get(i).unwrap().to_string())
            .collect()
    }

    #[test]
    fn variant_frames_carry_the_row_key() {
        let dir = std::env::temp_dir().join(format!("pasteur-eval-inputs-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let clean_ids = Series::new(layout::SOURCE_ROW_ID_COL.into(), ["a", "b"]);
        let labels = df!["y" => [0i64, 1]].unwrap();
        let files = [
            write(
                &dir,
                "blackout",
                df![
                    "f1" => [1.0, 2.0],
                    layout::SOURCE_ROW_ID_COL => ["a", "b"],
                    layout::ROW_ORDINAL_COL => [0u32, 1],
                ]
                .unwrap(),
            ),
            // No ids of its own: takes the clean ids row for row.
            write(&dir, "jitter_0", df!["f1" => [1.5, 2.5]].unwrap()),
        ];
        let variants = load_simulation_variants(
            &files,
            "jitter",
            "local",
            &labels,
            TaskType::Binary,
            &clean_ids,
        )
        .unwrap();
        for name in ["blackout", "jitter_0"] {
            let x = &variants[name]["local"].x;
            assert_eq!(
                x.get_column_names(),
                ["f1", layout::SOURCE_ROW_ID_COL, layout::VARIANT_NAME_COL]
            );
            assert_eq!(column_strings(x, layout::SOURCE_ROW_ID_COL), ["a", "b"]);
            assert_eq!(column_strings(x, layout::VARIANT_NAME_COL), [name; 2]);
        }

        let grid = write(
            &dir.join("flipper"),
            "flipper",
            df![
                "pair_id" => [0u32, 0],
                "step" => [0u32, 1],
                "f1" => [1.0, 2.0],
                layout::SOURCE_ROW_ID_COL => ["pair_0_step_0", "pair_0_step_1"],
                layout::ROW_ORDINAL_COL => [0u32, 1],
            ]
            .unwrap(),
        );
        let variants = load_simulation_variants(
            &[grid],
            "flipper",
            "local",
            &labels,
            TaskType::Binary,
            &clean_ids,
        )
        .unwrap();
        let x = &variants["flipper"]["local"].x;
        assert_eq!(column_strings(x, layout::VARIANT_NAME_COL), ["flipper"; 2]);
        assert_eq!(
            column_strings(x, layout::SOURCE_ROW_ID_COL),
            ["pair_0_step_0", "pair_0_step_1"]
        );
        assert_eq!(flipper_feature_order(x), ["f1"]);

        let clean = keyed_frame(
            &df!["f1" => [1.0, 2.0], layout::SOURCE_ROW_ID_COL => ["a", "b"]].unwrap(),
            "clean",
            &clean_ids,
        )
        .unwrap();
        assert_eq!(
            column_strings(&clean, layout::VARIANT_NAME_COL),
            ["clean"; 2]
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_feature_named_variant_name_is_refused() {
        let ids = Series::new(layout::SOURCE_ROW_ID_COL.into(), ["a"]);
        let df = df![layout::VARIANT_NAME_COL => ["x"]].unwrap();
        let err = keyed_frame(&df, "clean", &ids).unwrap_err();
        assert!(err.to_string().contains("reserved"), "{err}");
    }
}
