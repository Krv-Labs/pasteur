use std::collections::HashSet;
use std::path::Path;

use anyhow::Result;
use polars::prelude::*;

use crate::io::read_parquet;

pub fn derive_labels_from_file(
    labels_path: &Path,
    positive_group_id: u32,
    source_row_ids: &Series,
) -> Result<Series> {
    let groups = read_parquet(labels_path)?;
    let group_ids = groups.column("group_id")?.u32()?;
    let row_idx = find_positive_group_row(group_ids, positive_group_id, labels_path)?;
    let member_ids = load_member_ids(&groups, row_idx)?;
    Ok(labels_for_rows(source_row_ids, &member_ids))
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
