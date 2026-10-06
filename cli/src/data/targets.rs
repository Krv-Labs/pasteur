use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result};
use polars::prelude::*;

use crate::io::read_parquet;

/// The regression target for `source_row_ids`: one `Float64` column named
/// `target_col`, joined from `targets_path` by ID. IDs are compared as text,
/// so `A00123` and `007` match exactly rather than being integer-coerced.
///
/// Every cohort row needs exactly one finite target. A missing, duplicated or
/// non-finite target stops the run: scoring a model on whichever rows happen
/// to have a value would quietly change the cohort being evaluated.
pub fn derive_target_column(
    targets_path: &Path,
    id_col: &str,
    target_col: &str,
    source_row_ids: &Series,
) -> Result<DataFrame> {
    let table = read_parquet(targets_path)?;
    let column = |name: &str, role: &str| {
        table.column(name).with_context(|| {
            format!(
                "{} has no {role} column {name:?}; its columns are {:?}",
                targets_path.display(),
                table.get_column_names()
            )
        })
    };
    let ids = column(id_col, "ID (--targets-id-col)")?.cast(&DataType::String)?;
    let values = column(target_col, "target (--target-col)")?
        .cast(&DataType::Float64)
        .with_context(|| format!("target column {target_col:?} is not numeric"))?;
    let by_id = index_targets(ids.str()?, values.f64()?)?;

    let row_ids = source_row_ids.str()?;
    let mut target = Vec::with_capacity(row_ids.len());
    let (mut unmatched, mut not_finite) = (Vec::new(), Vec::new());
    for id in row_ids.iter() {
        let id = id.unwrap_or("");
        match by_id.get(id) {
            None => unmatched.push(id),
            Some(Some(v)) if v.is_finite() => target.push(*v),
            Some(_) => not_finite.push(id),
        }
    }

    let problems: Vec<String> = [
        (unmatched, "have no row in --targets"),
        (not_finite, "have a missing or non-finite target"),
    ]
    .into_iter()
    .filter(|(ids, _)| !ids.is_empty())
    .map(|(ids, what)| format!("{} row(s) {what} (e.g. {})", ids.len(), sample(&ids)))
    .collect();
    if !problems.is_empty() {
        anyhow::bail!(
            "targets from {} column {target_col:?} do not cover the cohort: {}",
            targets_path.display(),
            problems.join("; ")
        );
    }
    Ok(DataFrame::new(
        target.len(),
        vec![Series::new(target_col.into(), target).into()],
    )?)
}

/// Target value by ID. An ID listed twice is an error rather than first-wins,
/// since the two rows may disagree.
fn index_targets<'a>(
    ids: &'a StringChunked,
    values: &Float64Chunked,
) -> Result<HashMap<&'a str, Option<f64>>> {
    let mut by_id = HashMap::with_capacity(ids.len());
    let mut duplicates = Vec::new();
    for (id, value) in ids.iter().zip(values.iter()) {
        let Some(id) = id else { continue };
        if by_id.insert(id, value).is_some() {
            duplicates.push(id);
        }
    }
    if !duplicates.is_empty() {
        duplicates.sort_unstable();
        duplicates.dedup();
        anyhow::bail!(
            "--targets lists {} ID(s) more than once (e.g. {}); give each patient one target",
            duplicates.len(),
            sample(&duplicates)
        );
    }
    Ok(by_id)
}

fn sample(ids: &[&str]) -> String {
    ids[..ids.len().min(5)].join(", ")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_targets(dir: &Path, ids: &[&str], values: &[Option<f64>]) -> std::path::PathBuf {
        std::fs::create_dir_all(dir).unwrap();
        let path = dir.join("targets.parquet");
        let mut df = df!["node_id" => ids, "hba1c" => values].unwrap();
        let mut file = std::fs::File::create(&path).unwrap();
        ParquetWriter::new(&mut file).finish(&mut df).unwrap();
        path
    }

    fn dir(name: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!("pasteur-targets-{name}-{}", std::process::id()))
    }

    fn rows(ids: &[&str]) -> Series {
        Series::new("source_row_id".into(), ids)
    }

    #[test]
    fn targets_are_joined_by_id_in_cohort_order() {
        let d = dir("join");
        let path = write_targets(&d, &["b", "A007", "c"], &[Some(7.0), Some(5.5), Some(9.0)]);
        let y = derive_target_column(&path, "node_id", "hba1c", &rows(&["A007", "b"])).unwrap();
        assert_eq!(y.get_column_names(), ["hba1c"]);
        let v: Vec<f64> = y
            .column("hba1c")
            .unwrap()
            .f64()
            .unwrap()
            .into_no_null_iter()
            .collect();
        assert_eq!(v, vec![5.5, 7.0]);
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn missing_and_non_finite_targets_are_reported_together() {
        let d = dir("missing");
        let path = write_targets(&d, &["1", "2", "3"], &[Some(1.0), None, Some(f64::NAN)]);
        let err = derive_target_column(&path, "node_id", "hba1c", &rows(&["1", "2", "3", "4"]))
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("1 row(s) have no row in --targets (e.g. 4)"),
            "{err}"
        );
        assert!(
            err.contains("2 row(s) have a missing or non-finite target (e.g. 2, 3)"),
            "{err}"
        );
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn duplicate_ids_are_rejected() {
        let d = dir("dup");
        let path = write_targets(&d, &["1", "1"], &[Some(1.0), Some(2.0)]);
        let err = derive_target_column(&path, "node_id", "hba1c", &rows(&["1"]))
            .unwrap_err()
            .to_string();
        assert!(err.contains("1 ID(s) more than once (e.g. 1)"), "{err}");
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn a_wrong_column_name_lists_the_real_ones() {
        let d = dir("col");
        let path = write_targets(&d, &["1"], &[Some(1.0)]);
        let err = derive_target_column(&path, "node_id", "a1c", &rows(&["1"]))
            .unwrap_err()
            .to_string();
        assert!(err.contains("\"a1c\"") && err.contains("hba1c"), "{err}");
        let _ = std::fs::remove_dir_all(&d);
    }
}
