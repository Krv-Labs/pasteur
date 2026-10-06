use std::collections::HashSet;
use std::path::Path;

use anyhow::Result;
use pasteur_core::TaskType;
use polars::prelude::*;

use crate::io::read_parquet;

/// The label matrix for `source_row_ids`: one 0/1 column per group in
/// `group_ids` order. Binary takes exactly one group and names its column
/// `label`; multi-output columns are named `group_<id>`. Multiclass rows must
/// fall in exactly one group — a row in none or several has no class, and
/// guessing one would mislabel a patient.
pub fn derive_label_matrix(
    labels_path: &Path,
    group_ids: &[u32],
    task: TaskType,
    source_row_ids: &Series,
) -> Result<DataFrame> {
    check_group_ids(task, group_ids)?;

    let groups = read_parquet(labels_path)?;
    let ids = groups.column("group_id")?.u32()?;
    let columns: Vec<Column> = group_ids
        .iter()
        .map(|&gid| {
            let row_idx = find_positive_group_row(ids, gid, labels_path)?;
            let member_ids = load_member_ids(&groups, row_idx)?;
            let name = match task {
                TaskType::Binary | TaskType::Regression => "label".to_string(),
                TaskType::Multiclass | TaskType::Multilabel => format!("group_{gid}"),
            };
            Ok(labels_for_rows(source_row_ids, &member_ids)
                .with_name(name.as_str().into())
                .into())
        })
        .collect::<Result<_>>()?;
    let y = DataFrame::new(source_row_ids.len(), columns)?;
    if task == TaskType::Multiclass {
        check_one_class_per_row(&y, source_row_ids, group_ids)?;
    }
    Ok(y)
}

fn check_group_ids(task: TaskType, group_ids: &[u32]) -> Result<()> {
    if group_ids.iter().collect::<HashSet<_>>().len() != group_ids.len() {
        anyhow::bail!("--positive-group-id lists a group more than once: {group_ids:?}");
    }
    match (task, group_ids.len()) {
        (TaskType::Binary, 1) | (TaskType::Multilabel, _) => Ok(()),
        (TaskType::Binary, n) => anyhow::bail!(
            "binary evaluation takes one --positive-group-id, got {n}; pass --task multiclass \
             or --task multilabel for more"
        ),
        (TaskType::Multiclass, n) if n < 2 => anyhow::bail!(
            "multiclass needs one --positive-group-id per class (at least two), got {n}"
        ),
        (TaskType::Multiclass, _) => Ok(()),
        (TaskType::Regression, _) => {
            anyhow::bail!("regression is scored against --targets, not --positive-group-id cohorts")
        }
    }
}

/// Class index per row of a one-hot multiclass label matrix.
pub fn class_indices(y: &DataFrame) -> Result<Series> {
    let cols: Vec<&Int64Chunked> = y
        .columns()
        .iter()
        .map(|c| c.i64())
        .collect::<PolarsResult<_>>()?;
    let idx: Vec<i64> = (0..y.height())
        .map(|i| {
            cols.iter()
                .position(|c| c.get(i) == Some(1))
                .map_or(-1, |j| j as i64)
        })
        .collect();
    Ok(Series::new("label".into(), idx))
}

fn check_one_class_per_row(
    y: &DataFrame,
    source_row_ids: &Series,
    group_ids: &[u32],
) -> Result<()> {
    let cols: Vec<&Int64Chunked> = y
        .columns()
        .iter()
        .map(|c| c.i64())
        .collect::<PolarsResult<_>>()?;
    let ids = source_row_ids.str()?;
    let groups_of_row = |i: usize| -> i64 { cols.iter().map(|c| c.get(i).unwrap_or(0)).sum() };
    let ids_where = |keep: fn(i64) -> bool| -> Vec<&str> {
        (0..y.height())
            .filter(|&i| keep(groups_of_row(i)))
            .map(|i| ids.get(i).unwrap_or(""))
            .collect()
    };
    let sample = |rows: &[&str]| rows[..rows.len().min(5)].join(", ");

    let mut problems = Vec::new();
    let in_none = ids_where(|n| n == 0);
    if !in_none.is_empty() {
        problems.push(format!(
            "{} row(s) are in no group (e.g. {}) — add a group for the remaining class, and \
             check IDs are integers",
            in_none.len(),
            sample(&in_none)
        ));
    }
    let in_several = ids_where(|n| n > 1);
    if !in_several.is_empty() {
        problems.push(format!(
            "{} row(s) are in several groups (e.g. {}) — overlapping cohorts are \
             multilabel (--task multilabel)",
            in_several.len(),
            sample(&in_several)
        ));
    }
    if problems.is_empty() {
        return Ok(());
    }
    anyhow::bail!(
        "multiclass labels from groups {group_ids:?} are not one class per row: {}",
        problems.join("; ")
    )
}

fn find_positive_group_row(
    group_ids: &ChunkedArray<UInt32Type>,
    positive_group_id: u32,
    labels_path: &Path,
) -> Result<usize> {
    group_ids
        .iter()
        .position(|v| v == Some(positive_group_id))
        .ok_or_else(|| {
            anyhow::anyhow!(
                "group_id {positive_group_id} not found in {}",
                labels_path.display()
            )
        })
}

fn load_member_ids(groups: &DataFrame, row_idx: usize) -> Result<HashSet<i64>> {
    Ok(groups
        .column("member_ids")?
        .list()?
        .get_as_series(row_idx)
        .ok_or_else(|| anyhow::anyhow!("empty member_ids at row {row_idx}"))?
        .cast(&DataType::Int64)?
        .i64()?
        .into_no_null_iter()
        .collect())
}

fn labels_for_rows(source_row_ids: &Series, member_ids: &HashSet<i64>) -> Series {
    let labels: Vec<i64> = source_row_ids
        .str()
        .expect("source_row_id must be string")
        .iter()
        .map(|s| row_label(s, member_ids))
        .collect();
    Int64Chunked::from_vec("label".into(), labels).into_series()
}

fn row_label(source_id: Option<&str>, member_ids: &HashSet<i64>) -> i64 {
    let in_group = source_id
        .and_then(|s| s.parse::<i64>().ok())
        .map(|id| member_ids.contains(&id))
        .unwrap_or(false);
    i64::from(in_group)
}
