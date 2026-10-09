use anyhow::{Context, Result};
use pasteur_core::{
    simulate::flipper::{PAIR_ID_COL, STEP_COL},
    BlackoutConfig, BlackoutSimulator, FlipperConfig, FlipperSimulator, JitterConfig,
    JitterSimulator, Simulator, TaskType,
};
use pasteur_hf::layout;
use polars::prelude::*;

use crate::args::SimulateArgs;
use crate::data::{class_indices, derive_label_matrix};
use crate::io::{read_parquet, with_ids, write_variant};

pub fn run(args: SimulateArgs) -> Result<()> {
    let (features, source_row_id, row_ordinal, provenance) = load_input(&args)?;
    println!(
        "Loaded {} ({} rows x {} cols) [{}]",
        args.input.display(),
        features.height(),
        features.width(),
        provenance
    );

    write_variant(
        &args.output,
        "clean",
        "clean",
        &with_ids(&features, &source_row_id, &row_ordinal)?,
    )?;

    run_blackout(&args, &features, &source_row_id, &row_ordinal)?;
    run_jitter_iters(&args, &features, &source_row_id, &row_ordinal)?;
    if let Some(labels_path) = &args.labels {
        run_flipper(
            &args,
            &features,
            labels_path,
            source_row_id.as_materialized_series(),
        )?;
    } else {
        println!("  skipped flipper/ (pass --labels to generate interpolation grid)");
    }

    layout::validate_sim_bundle(&args.output)
        .context("simulation output failed layout validation")?;
    println!(
        "Done. Simulation variants written under {}",
        args.output.display()
    );
    Ok(())
}

fn load_input(args: &SimulateArgs) -> Result<(DataFrame, Column, Column, String)> {
    let df = read_parquet(&args.input)?;
    let n_rows = df.height();
    let provenance = args.dataset_name.clone().unwrap_or_else(|| {
        args.input
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("local")
            .to_string()
    });

    let mut source_row_id = df
        .column(&args.id_col)
        .with_context(|| format!("id column {:?} not found", args.id_col))?
        .cast(&DataType::String)
        .context("failed to cast id column to string")?;
    source_row_id.rename(layout::SOURCE_ROW_ID_COL.into());

    let row_ordinal: Column =
        UInt32Chunked::from_vec(layout::ROW_ORDINAL_COL.into(), (0..n_rows as u32).collect())
            .into_series()
            .into();

    let features = df
        .drop(&args.id_col)
        .with_context(|| format!("failed to drop id column {:?}", args.id_col))?;

    Ok((features, source_row_id, row_ordinal, provenance))
}

fn run_blackout(
    args: &SimulateArgs,
    features: &DataFrame,
    source_row_id: &Column,
    row_ordinal: &Column,
) -> Result<()> {
    let cfg = BlackoutConfig {
        feature: args.feature.clone(),
        companions: args.blackout_companions.clone(),
        rate: args.blackout_rate,
        window_frac: 0.0,
        patient_id_col: None,
        time_col: None,
        random_state: Some(args.random_state),
    };
    let mut sim = BlackoutSimulator::new(cfg);
    sim.fit(features)?;
    let out = sim.transform(features)?;
    write_variant(
        &args.output,
        "blackout",
        "blackout",
        &with_ids(&out, source_row_id, row_ordinal)?,
    )
}

fn run_jitter_iters(
    args: &SimulateArgs,
    features: &DataFrame,
    source_row_id: &Column,
    row_ordinal: &Column,
) -> Result<()> {
    for i in 0..args.jitter_iters {
        let cfg = JitterConfig {
            feature: args.feature.clone(),
            scale: args.jitter_scale,
            random_state: Some(args.random_state + i as u64),
        };
        let mut sim = JitterSimulator::new(cfg);
        sim.fit(features)?;
        let out = sim.transform(features)?;
        write_variant(
            &args.output,
            "jitter",
            &format!("jitter_{i}"),
            &with_ids(&out, source_row_id, row_ordinal)?,
        )?;
    }
    Ok(())
}

fn run_flipper(
    args: &SimulateArgs,
    features: &DataFrame,
    labels_path: &std::path::Path,
    source_row_id: &Series,
) -> Result<()> {
    let task = TaskType::from(args.task);
    let labels = derive_label_matrix(labels_path, &args.positive_group_ids, task, source_row_id)?;
    let config = FlipperConfig {
        n_pairs: args.flipper_pairs,
        n_steps: args.flipper_steps,
        flip_threshold: 0.5,
        positive_class_label: None,
        random_state: Some(args.random_state),
    };
    let simulator = FlipperSimulator::new(config);
    // Binary pairs across the positive cohort; multiclass across two classes
    // (label_a/label_b hold class indices in --positive-group-id order);
    // multilabel across one label per pair (recorded in `pair_label`).
    let grid = match task {
        TaskType::Binary => simulator.generate(features, labels[0].as_materialized_series())?,
        TaskType::Multiclass => simulator.generate(features, &class_indices(&labels)?)?,
        TaskType::Multilabel => simulator.generate_multilabel(features, &labels)?,
    };
    let out = with_flipper_ids(&grid)?;
    write_variant(&args.output, "flipper", "flipper", &out)
}

fn with_flipper_ids(grid: &DataFrame) -> Result<DataFrame> {
    let n_rows = grid.height();
    let pair_id = grid.column(PAIR_ID_COL)?.u32()?;
    let step = grid.column(STEP_COL)?.u32()?;
    let source_row_id: Vec<String> = (0..n_rows)
        .map(|i| {
            format!(
                "pair_{}_step_{}",
                pair_id.get(i).unwrap_or(0),
                step.get(i).unwrap_or(0)
            )
        })
        .collect();
    let row_ordinal: Vec<u32> = (0..n_rows as u32).collect();

    let mut out = grid.clone();
    out.with_column(Series::new(layout::SOURCE_ROW_ID_COL.into(), source_row_id).into())?;
    out.with_column(Series::new(layout::ROW_ORDINAL_COL.into(), row_ordinal).into())?;
    Ok(out)
}
