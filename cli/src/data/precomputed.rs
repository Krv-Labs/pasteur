//! `--predictions`: a model scored outside Pasteur, read back from parquet
//! and served by row key.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use pasteur_core::{CoreError, Model, TaskType};
use pasteur_hf::layout::{SOURCE_ROW_ID_COL, VARIANT_NAME_COL};
use pasteur_model::{BINARY_OUTPUT_COLUMN, REGRESSION_OUTPUT_COLUMN};
use polars::prelude::*;

use crate::io::read_parquet;

/// Predictions keyed by `(variant_name, source_row_id)`, one value per
/// output. Every row the evaluation asks for must be present.
pub struct PrecomputedPredictions {
    path: PathBuf,
    columns: Vec<String>,
    rows: HashMap<(String, String), Vec<f64>>,
}

impl PrecomputedPredictions {
    /// Reads the row key plus one numeric column per output, in class order.
    /// The label, target and `all_agree` columns `--predictions-out` writes
    /// are skipped, so a single-model predictions file can be fed straight
    /// back in.
    pub fn from_parquet(path: &Path, task: TaskType, n_outputs: usize) -> Result<Self> {
        let df = read_parquet(path)?;
        let names: Vec<String> = df
            .get_column_names()
            .iter()
            .map(|s| s.to_string())
            .filter(|n| !is_bookkeeping_column(n))
            .collect();
        if names.len() != n_outputs {
            anyhow::bail!(
                "{} has prediction columns {names:?} but the labels have {n_outputs} output(s); \
                 give one Float64 column per output, in class order (a --predictions-out file \
                 from several models must be cut down to one model's columns)",
                path.display()
            );
        }
        let key = |name: &str| -> Result<StringChunked> {
            let col = df.column(name).with_context(|| {
                format!("{} has no {name} column (the row key)", path.display())
            })?;
            Ok(col.cast(&DataType::String)?.str()?.clone())
        };
        let variants = key(VARIANT_NAME_COL)?;
        let ids = key(SOURCE_ROW_ID_COL)?;
        let values = names
            .iter()
            .map(|name| {
                let col = df.column(name)?;
                if !col.dtype().is_primitive_numeric() {
                    anyhow::bail!(
                        "{} column {name:?} is {}, not a number",
                        path.display(),
                        col.dtype()
                    );
                }
                Ok(col.cast(&DataType::Float64)?.f64()?.clone())
            })
            .collect::<Result<Vec<_>>>()?;

        let mut rows = HashMap::with_capacity(df.height());
        for i in 0..df.height() {
            let (Some(variant), Some(id)) = (variants.get(i), ids.get(i)) else {
                anyhow::bail!("{} row {i} has a null row key", path.display());
            };
            let preds = values
                .iter()
                .zip(&names)
                .map(|(col, name)| {
                    col.get(i).with_context(|| {
                        format!(
                            "{} has a null {name:?} for variant {variant:?} row {id:?}",
                            path.display()
                        )
                    })
                })
                .collect::<Result<Vec<f64>>>()?;
            if rows
                .insert((variant.to_string(), id.to_string()), preds)
                .is_some()
            {
                anyhow::bail!(
                    "{} has more than one row for variant {variant:?} row {id:?}",
                    path.display()
                );
            }
        }
        Ok(Self {
            path: path.to_path_buf(),
            columns: output_names(names, task),
            rows,
        })
    }

    /// Output column names, in the order `predict_proba` returns them.
    pub fn output_columns(&self) -> &[String] {
        &self.columns
    }
}

impl Model for PrecomputedPredictions {
    fn predict_proba(&self, df: &DataFrame) -> Result<DataFrame, CoreError> {
        let key = |name: &str| -> Result<StringChunked, CoreError> {
            let col = df.column(name).map_err(|_| {
                CoreError::InvalidConfig(format!(
                    "precomputed predictions are looked up by {VARIANT_NAME_COL} and \
                     {SOURCE_ROW_ID_COL}, but the evaluation rows have no {name} column"
                ))
            })?;
            Ok(col.cast(&DataType::String)?.str()?.clone())
        };
        let variants = key(VARIANT_NAME_COL)?;
        let ids = key(SOURCE_ROW_ID_COL)?;
        let n = df.height();
        let mut out = vec![Vec::with_capacity(n); self.columns.len()];
        for i in 0..n {
            let variant = variants.get(i).unwrap_or_default();
            let id = ids.get(i).unwrap_or_default();
            let row = self
                .rows
                .get(&(variant.to_string(), id.to_string()))
                .ok_or_else(|| {
                    CoreError::InvalidConfig(format!(
                        "{} has no prediction for variant {variant:?} row {id:?}",
                        self.path.display()
                    ))
                })?;
            for (col, value) in out.iter_mut().zip(row) {
                col.push(*value);
            }
        }
        let columns = self
            .columns
            .iter()
            .zip(out)
            .map(|(name, values)| Series::new(name.as_str().into(), values).into())
            .collect();
        Ok(DataFrame::new(n, columns)?)
    }
}

/// What `--predictions-out` writes beside the prediction columns.
fn is_bookkeeping_column(name: &str) -> bool {
    matches!(name, "label" | "target" | "all_agree")
        || name.starts_with("label__")
        || name == SOURCE_ROW_ID_COL
        || name == VARIANT_NAME_COL
}

/// Names the outputs as the ONNX path does: binary and regression get the
/// fixed single-column name, multi-output keeps the class names, dropping a
/// `<model>__` prefix that every column shares (as `--predictions-out`
/// writes them).
fn output_names(names: Vec<String>, task: TaskType) -> Vec<String> {
    match task {
        TaskType::Binary => vec![BINARY_OUTPUT_COLUMN.to_string()],
        TaskType::Regression => vec![REGRESSION_OUTPUT_COLUMN.to_string()],
        TaskType::Multiclass | TaskType::Multilabel => {
            let prefix = names[0].split_once("__").map(|(p, _)| format!("{p}__"));
            match prefix {
                Some(p) if names.iter().all(|n| n.starts_with(&p)) => {
                    names.iter().map(|n| n[p.len()..].to_string()).collect()
                }
                _ => names,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write(dir: &Path, mut df: DataFrame) -> PathBuf {
        std::fs::create_dir_all(dir).unwrap();
        let path = dir.join("preds.parquet");
        let mut file = std::fs::File::create(&path).unwrap();
        ParquetWriter::new(&mut file).finish(&mut df).unwrap();
        path
    }

    fn temp_dir(name: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("pasteur_precomputed_{name}_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        dir
    }

    fn rows(variants: &[&str], ids: &[&str]) -> DataFrame {
        df![
            VARIANT_NAME_COL => variants.to_vec(),
            SOURCE_ROW_ID_COL => ids.to_vec(),
            "f1" => vec![0.0; ids.len()],
        ]
        .unwrap()
    }

    #[test]
    fn looks_rows_up_by_key_and_strips_the_model_prefix() {
        let dir = temp_dir("lookup");
        let path = write(
            &dir,
            df![
                SOURCE_ROW_ID_COL => ["1", "2", "1"],
                VARIANT_NAME_COL => ["clean", "clean", "blackout"],
                "label__a" => [1i64, 0, 1],
                "m__a" => [0.9, 0.2, 0.6],
                "m__b" => [0.1, 0.8, 0.4],
                "all_agree" => [true, true, true],
            ]
            .unwrap(),
        );
        let model = PrecomputedPredictions::from_parquet(&path, TaskType::Multiclass, 2).unwrap();
        assert_eq!(model.output_columns(), ["a", "b"]);
        let probs = model
            .predict_proba(&rows(&["blackout", "clean"], &["1", "2"]))
            .unwrap();
        let a: Vec<Option<f64>> = probs.column("a").unwrap().f64().unwrap().to_vec();
        assert_eq!(a, [Some(0.6), Some(0.2)]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn missing_duplicate_and_wrong_width_rows_are_errors() {
        let dir = temp_dir("errors");
        let path = write(
            &dir,
            df![
                SOURCE_ROW_ID_COL => ["1", "2"],
                VARIANT_NAME_COL => ["clean", "clean"],
                "proba" => [0.9, 0.2],
            ]
            .unwrap(),
        );
        let model = PrecomputedPredictions::from_parquet(&path, TaskType::Binary, 1).unwrap();
        let err = model
            .predict_proba(&rows(&["clean"], &["3"]))
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("no prediction for variant \"clean\" row \"3\""),
            "{err}"
        );

        let err = PrecomputedPredictions::from_parquet(&path, TaskType::Multiclass, 3)
            .err()
            .unwrap()
            .to_string();
        assert!(err.contains("labels have 3 output(s)"), "{err}");

        let path = write(
            &dir,
            df![
                SOURCE_ROW_ID_COL => ["1", "1"],
                VARIANT_NAME_COL => ["clean", "clean"],
                "proba" => [0.9, 0.2],
            ]
            .unwrap(),
        );
        let err = PrecomputedPredictions::from_parquet(&path, TaskType::Binary, 1)
            .err()
            .unwrap()
            .to_string();
        assert!(err.contains("more than one row"), "{err}");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
