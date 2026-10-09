use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use pasteur_core::{is_flipper_meta_column, EvaluationConfig, SimulationVariant, TaskType};
use pasteur_hf::layout;
use polars::prelude::*;

use crate::data::source::LabelSource;
use crate::io::read_parquet;

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
        x: feature_columns(&clean_df)?,
        y: labels.clone(),
    };

    let variant_files = list_variant_files(sim_root, sim_type)?;
    let simulation_variants =
        load_simulation_variants(&variant_files, sim_type, dataset_name, &labels, task)?;

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
    let clean_x = feature_columns(&clean_df)?;
    let feature_order: Vec<String> = clean_x
        .get_column_names()
        .iter()
        .map(|s| s.to_string())
        .collect();
    Ok((clean_df, source_row_id, feature_order))
}

fn feature_columns(df: &DataFrame) -> Result<DataFrame> {
    let mut out = df.clone();
    for col in [layout::SOURCE_ROW_ID_COL, layout::ROW_ORDINAL_COL] {
        if out
            .get_column_names()
            .iter()
            .any(|name| name.as_str() == col)
        {
            out = out.drop(col)?;
        }
    }
    Ok(out)
}

fn flipper_feature_order(df: &DataFrame) -> Vec<String> {
    df.get_column_names()
        .iter()
        .map(|s| s.to_string())
        .filter(|name| {
            name.as_str() != layout::SOURCE_ROW_ID_COL
                && name.as_str() != layout::ROW_ORDINAL_COL
                && !is_flipper_meta_column(name)
        })
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
) -> Result<HashMap<String, HashMap<String, SimulationVariant>>> {
    let mut simulation_variants = HashMap::new();
    for path in variant_files {
        let name = variant_name(path, sim_type);
        let df = read_parquet(path)?;
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
            (feature_columns(&df)?, labels.clone())
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

pub fn model_feature_frame(
    variant: &SimulationVariant,
    feature_order: &[String],
) -> Result<DataFrame> {
    Ok(variant.x.select(feature_order.to_vec())?)
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
