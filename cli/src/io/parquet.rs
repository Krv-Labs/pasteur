use std::fs;
use std::path::Path;

use anyhow::{Context, Result};
use polars::prelude::*;

pub fn with_ids(
    features: &DataFrame,
    source_row_id: &Column,
    row_ordinal: &Column,
) -> Result<DataFrame> {
    let mut out = features.clone();
    out.insert_column(0, row_ordinal.clone())?;
    out.insert_column(0, source_row_id.clone())?;
    Ok(out)
}

pub fn write_variant(
    output_root: &Path,
    sim_type: &str,
    variant_name: &str,
    df: &DataFrame,
) -> Result<()> {
    let dir = output_root.join(sim_type);
    fs::create_dir_all(&dir)
        .with_context(|| format!("failed to create output dir {}", dir.display()))?;
    let path = dir.join(format!("{variant_name}.parquet"));
    let mut file =
        fs::File::create(&path).with_context(|| format!("failed to create {}", path.display()))?;
    ParquetWriter::new(&mut file)
        .finish(&mut df.clone())
        .with_context(|| format!("failed to write parquet to {}", path.display()))?;
    println!("  wrote {} ({} rows)", path.display(), df.height());
    Ok(())
}

pub fn write_predictions_parquet(mut predictions: DataFrame, out_path: &Path) -> Result<()> {
    if let Some(parent) = out_path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut file = fs::File::create(out_path)
        .with_context(|| format!("failed to create {}", out_path.display()))?;
    ParquetWriter::new(&mut file).finish(&mut predictions)?;
    println!(
        "wrote per-patient predictions ({} rows) to {}",
        predictions.height(),
        out_path.display()
    );
    Ok(())
}

pub fn read_parquet(path: &Path) -> Result<DataFrame> {
    ParquetReader::new(
        fs::File::open(path).with_context(|| format!("failed to open {}", path.display()))?,
    )
    .finish()
    .with_context(|| format!("failed to read parquet {}", path.display()))
}
