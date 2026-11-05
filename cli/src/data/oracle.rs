//! `--oracle-labels`: per-row true labels that replace the copied ones.

use std::collections::HashMap;

use anyhow::Result;
use pasteur_core::evaluate::OracleConfig;
use pasteur_core::{SimulationVariant, TaskType};
use pasteur_hf::layout::{ROW_KEY_COLS, SOURCE_ROW_ID_COL, VARIANT_NAME_COL};
use polars::prelude::*;

use crate::args::OracleArgs;
use crate::data::eval_inputs::EvaluationInputs;
use crate::io::read_parquet;

/// Replaces every variant's `y`, clean included, with the oracle's label for
/// the same `(variant_name, source_row_id)`, and marks the run as
/// oracle-labelled. The oracle file holds the row key plus the label columns
/// Pasteur derives from `--labels` (`label`, or `group_<id>` per class or
/// label); other columns are ignored. A row the oracle does not cover is an
/// error. Does nothing without `--oracle-labels`.
pub fn apply_oracle_labels(inputs: &mut EvaluationInputs, args: &OracleArgs) -> Result<()> {
    let Some(path) = &args.oracle_labels else {
        return Ok(());
    };
    if inputs.eval_config.task == TaskType::Regression {
        anyhow::bail!("--oracle-labels scores classifier decisions; regression has no classes");
    }
    let oracle = read_parquet(path)?;
    let clean = inputs
        .clean_datasets
        .get(&inputs.dataset_name)
        .ok_or_else(|| anyhow::anyhow!("no clean dataset {:?}", inputs.dataset_name))?;
    let label_cols: Vec<String> = clean
        .y
        .get_column_names()
        .iter()
        .map(|s| s.to_string())
        .collect();
    let missing: Vec<&String> = label_cols
        .iter()
        .filter(|c| oracle.column(c).is_err())
        .collect();
    if !missing.is_empty() {
        anyhow::bail!(
            "{} has no label column(s) {missing:?}; it needs {:?} plus {label_cols:?}",
            path.display(),
            ROW_KEY_COLS
        );
    }
    let labels = oracle.select(label_cols)?;
    let index = key_index(&oracle)?;

    let relabel = |v: &mut SimulationVariant| -> Result<()> {
        v.y = labels_for(&v.x, &labels, &index)?;
        Ok(())
    };
    for v in inputs.clean_datasets.values_mut() {
        relabel(v)?;
    }
    for v in inputs
        .simulation_variants
        .values_mut()
        .flat_map(|d| d.values_mut())
    {
        relabel(v)?;
    }
    inputs.eval_config.oracle = Some(OracleConfig {
        abstain_class: args.abstain_class.clone(),
    });
    Ok(())
}

type Key = (String, String);

/// The row key of every row of `df`, as text.
fn keys(df: &DataFrame) -> Result<Vec<Key>> {
    let col = |name: &str| -> Result<StringChunked> {
        let c = df
            .column(name)
            .map_err(|_| anyhow::anyhow!("oracle labels need a {name:?} column"))?;
        Ok(c.cast(&DataType::String)?.str()?.clone())
    };
    let (variant, row) = (col(VARIANT_NAME_COL)?, col(SOURCE_ROW_ID_COL)?);
    Ok(variant
        .iter()
        .zip(row.iter())
        .map(|(v, r)| (v.unwrap_or("").to_string(), r.unwrap_or("").to_string()))
        .collect())
}

fn key_index(oracle: &DataFrame) -> Result<HashMap<Key, IdxSize>> {
    let mut index = HashMap::new();
    for (i, key) in keys(oracle)?.into_iter().enumerate() {
        if index.insert(key.clone(), i as IdxSize).is_some() {
            anyhow::bail!("oracle labels list row {key:?} more than once");
        }
    }
    Ok(index)
}

/// The oracle rows for `x`'s keys, in `x`'s row order.
fn labels_for(
    x: &DataFrame,
    labels: &DataFrame,
    index: &HashMap<Key, IdxSize>,
) -> Result<DataFrame> {
    let keys = keys(x)?;
    let missing: Vec<&Key> = keys.iter().filter(|k| !index.contains_key(*k)).collect();
    if !missing.is_empty() {
        let sample: Vec<_> = missing.iter().take(5).collect();
        anyhow::bail!(
            "oracle labels miss {} of {} row(s), e.g. {sample:?} as (variant_name, \
             source_row_id)",
            missing.len(),
            keys.len()
        );
    }
    let idx: Vec<IdxSize> = keys.iter().map(|k| index[k]).collect();
    Ok(labels.take(&IdxCa::from_vec("idx".into(), idx))?)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn x(variant: &str, ids: &[&str]) -> DataFrame {
        df![SOURCE_ROW_ID_COL => ids, VARIANT_NAME_COL => vec![variant; ids.len()]].unwrap()
    }

    #[test]
    fn labels_follow_the_row_key_and_missing_rows_fail() {
        let oracle = df![
            VARIANT_NAME_COL => ["jitter_0", "jitter_0", "clean"],
            SOURCE_ROW_ID_COL => [2i64, 1, 1],
            "label" => [1i64, 0, 1],
        ]
        .unwrap();
        let index = key_index(&oracle).unwrap();
        let labels = oracle.select(["label"]).unwrap();
        let y = labels_for(&x("jitter_0", &["1", "2"]), &labels, &index).unwrap();
        let got: Vec<_> = y.column("label").unwrap().i64().unwrap().iter().collect();
        assert_eq!(got, [Some(0), Some(1)]);

        let err = labels_for(&x("jitter_1", &["1"]), &labels, &index).unwrap_err();
        assert!(err.to_string().contains("miss 1 of 1"), "{err}");

        let dup = df![
            VARIANT_NAME_COL => ["clean", "clean"],
            SOURCE_ROW_ID_COL => ["1", "1"],
        ]
        .unwrap();
        assert!(key_index(&dup).is_err());
    }
}
