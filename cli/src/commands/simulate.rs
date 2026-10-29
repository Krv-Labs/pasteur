use anyhow::{Context, Result};
use pasteur_core::{
    simulate::flipper::{PAIR_ID_COL, STEP_COL},
    simulate::sweep::{SWEEP_CASE_ROW_COL, SWEEP_FEATURE_COL, SWEEP_STEP_COL},
    BlackoutConfig, BlackoutSimulator, FlipperConfig, FlipperSimulator, JitterConfig,
    JitterSimulator, Simulator, SweepConfig, SweepFeature, SweepSimulator, TaskType,
};
use pasteur_hf::layout;
use polars::prelude::*;

use crate::args::SimulateArgs;
use crate::data::{class_indices, LabelSource};
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
    run_flipper(&args, &features, source_row_id.as_materialized_series())?;
    run_sweep(&args, &features, source_row_id.as_materialized_series())?;

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

/// Writes the flipper grid when the flags give it labels to pair across:
/// cohorts for a classifier, targets plus a cutoff for regression.
fn run_flipper(args: &SimulateArgs, features: &DataFrame, source_row_id: &Series) -> Result<()> {
    let task = TaskType::from(args.task);
    let source = LabelSource::from_args(
        task,
        args.labels.as_deref(),
        &args.positive_group_ids,
        &args.targets,
    )?;
    let cutoff = match (task, args.flip_threshold) {
        (TaskType::Binary | TaskType::Multiclass | TaskType::Multilabel, Some(_)) => {
            anyhow::bail!(
                "simulate reads --flip-threshold only for --task regression; a {task} grid \
                 pairs patients across cohorts, and the threshold is set at evaluate time"
            )
        }
        (TaskType::Binary | TaskType::Multiclass | TaskType::Multilabel, None) => None,
        (TaskType::Regression, cutoff) => cutoff,
    };
    let Some(source) = source else {
        let needs = match task {
            TaskType::Binary | TaskType::Multiclass | TaskType::Multilabel => "--labels",
            TaskType::Regression => "--targets, --target-col and --flip-threshold",
        };
        println!("  skipped flipper/ (pass {needs} to generate interpolation grid)");
        return Ok(());
    };
    if task == TaskType::Regression && cutoff.is_none() {
        println!(
            "  skipped flipper/ (pass --flip-threshold, the clinical cutoff in target units, \
             to generate interpolation grid)"
        );
        return Ok(());
    }
    let labels = source.derive(task, source_row_id)?;
    // Set by now for regression; classifier grids do not read it.
    let flip_threshold = cutoff.unwrap_or(0.5);
    let config = FlipperConfig {
        n_pairs: args.flipper_pairs,
        n_steps: args.flipper_steps,
        flip_threshold,
        positive_class_label: None,
        random_state: Some(args.random_state),
    };
    let simulator = FlipperSimulator::new(config);
    // Binary pairs across the positive cohort; multiclass across two classes
    // (label_a/label_b hold class indices in --positive-group-id order);
    // multilabel across one label per pair (recorded in `pair_label`);
    // regression across the cutoff (label_a/label_b hold true targets).
    let grid = match task {
        TaskType::Binary => simulator.generate(features, labels[0].as_materialized_series())?,
        TaskType::Multiclass => simulator.generate(features, &class_indices(&labels)?)?,
        TaskType::Multilabel => simulator.generate_multilabel(features, &labels)?,
        TaskType::Regression => simulator.generate_regression(
            features,
            labels[0].as_materialized_series(),
            flip_threshold,
        )?,
    };
    let out = with_flipper_ids(&grid)?;
    write_variant(&args.output, "flipper", "flipper", &out)
}

/// Writes the single-feature sweep grid when `--sweep` names a feature.
fn run_sweep(args: &SimulateArgs, features: &DataFrame, source_row_id: &Series) -> Result<()> {
    let sweep = &args.sweep;
    if sweep.features.is_empty() {
        if sweep.sweep_range.is_some()
            || sweep.sweep_relative_to.is_some()
            || sweep.sweep_near.is_some()
        {
            anyhow::bail!("--sweep-range, --sweep-relative-to and --sweep-near need --sweep");
        }
        return Ok(());
    }
    let range = sweep
        .sweep_range
        .as_deref()
        .context("--sweep needs --sweep-range lo:hi")?;
    let (lo, hi) = range
        .split_once(':')
        .and_then(|(lo, hi)| Some((lo.trim().parse().ok()?, hi.trim().parse().ok()?)))
        .with_context(|| format!("--sweep-range must be lo:hi, got {range:?}"))?;
    let features_to_sweep = sweep
        .features
        .iter()
        .map(|spec| {
            let (name, cutoff) = match spec.split_once('=') {
                Some((name, cutoff)) => {
                    let cutoff = cutoff
                        .trim()
                        .parse()
                        .with_context(|| format!("--sweep {spec:?}: cutoff is not a number"))?;
                    (name, Some(cutoff))
                }
                None => (spec.as_str(), None),
            };
            Ok(SweepFeature {
                name: name.trim().to_string(),
                cutoff,
            })
        })
        .collect::<Result<_>>()?;
    let config = SweepConfig {
        features: features_to_sweep,
        range: (lo, hi),
        relative_to: sweep.sweep_relative_to,
        n_steps: sweep.sweep_steps,
        near: sweep.sweep_near,
    };
    let grid = SweepSimulator::new(config).generate(features)?;
    write_variant(
        &args.output,
        "sweep",
        "sweep",
        &with_sweep_ids(&grid, source_row_id)?,
    )
}

/// `source_row_id` is `sweep_<case id>_<feature>_step_<n>`.
fn with_sweep_ids(grid: &DataFrame, source_row_id: &Series) -> Result<DataFrame> {
    let n_rows = grid.height();
    let case = grid.column(SWEEP_CASE_ROW_COL)?.u32()?;
    let feature = grid.column(SWEEP_FEATURE_COL)?.str()?;
    let step = grid.column(SWEEP_STEP_COL)?.u32()?;
    let case_ids = source_row_id.str()?;
    let ids: Vec<String> = (0..n_rows)
        .map(|i| {
            let case_id = case
                .get(i)
                .and_then(|r| case_ids.get(r as usize))
                .unwrap_or("");
            format!(
                "sweep_{case_id}_{}_step_{}",
                feature.get(i).unwrap_or(""),
                step.get(i).unwrap_or(0)
            )
        })
        .collect();
    let row_ordinal: Vec<u32> = (0..n_rows as u32).collect();

    let mut out = grid.clone();
    out.with_column(Series::new(layout::SOURCE_ROW_ID_COL.into(), ids).into())?;
    out.with_column(Series::new(layout::ROW_ORDINAL_COL.into(), row_ordinal).into())?;
    Ok(out)
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
