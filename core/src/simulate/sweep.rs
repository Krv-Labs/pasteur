use crate::error::CoreError;
use crate::schema::SweepConfig;
use polars::prelude::*;

/// 0-based positional row index of the case the path was swept from.
pub const SWEEP_CASE_ROW_COL: &str = "sweep_case_row";
/// Name of the one feature this path moves.
pub const SWEEP_FEATURE_COL: &str = "sweep_feature";
/// Position on the path; steps are in ascending `sweep_value` order.
pub const SWEEP_STEP_COL: &str = "sweep_step";
/// The value the swept feature takes at this step.
pub const SWEEP_VALUE_COL: &str = "sweep_value";
/// The feature's declared cutoff, or null when none was given.
pub const SWEEP_CUTOFF_COL: &str = "sweep_cutoff";

/// Meta columns written by `SweepSimulator::generate`. Prefixed so they
/// cannot collide with data columns.
pub const SWEEP_META_COLS: [&str; 5] = [
    SWEEP_CASE_ROW_COL,
    SWEEP_FEATURE_COL,
    SWEEP_STEP_COL,
    SWEEP_VALUE_COL,
    SWEEP_CUTOFF_COL,
];

pub fn is_sweep_meta_column(name: &str) -> bool {
    SWEEP_META_COLS.contains(&name)
}

/// Flipper's single-feature sibling: for each source row and each declared
/// feature, sets **only that feature** to each value of a grid and copies
/// every other column unchanged. Where Flipper moves every feature at once
/// between two patients, a sweep asks whether the model reads one feature,
/// and where along it the decision flips.
///
/// Like Flipper, the output is a grid of synthetic rows rather than a
/// row-preserving perturbation, so this doesn't implement `Simulator`.
pub struct SweepSimulator {
    config: SweepConfig,
}

impl SweepSimulator {
    pub fn new(config: SweepConfig) -> Self {
        Self { config }
    }

    /// Output: the `SWEEP_META_COLS`, then every column of `x` in order,
    /// with every swept feature cast to `Float64`. Rows are grouped by feature,
    /// then source row, then step.
    pub fn generate(&self, x: &DataFrame) -> Result<DataFrame, CoreError> {
        let cfg = &self.config;
        if cfg.features.is_empty() {
            return Err(CoreError::InvalidConfig(
                "sweep needs at least one feature".to_string(),
            ));
        }
        if u32::try_from(x.height()).is_err() {
            return Err(CoreError::InvalidConfig(format!(
                "sweep grid indices must fit in u32: {} rows",
                x.height()
            )));
        }
        // Every swept column becomes Float64 in every block, so the blocks
        // share one schema.
        let mut x = x.clone();
        for feature in &cfg.features {
            let col = x.column(&feature.name).map_err(|_| {
                CoreError::InvalidConfig(format!("sweep feature {:?} not found", feature.name))
            })?;
            if !col.dtype().is_numeric() {
                return Err(CoreError::InvalidConfig(format!(
                    "sweep feature {:?} is {}, not numeric",
                    feature.name,
                    col.dtype()
                )));
            }
            let col = col.cast(&DataType::Float64)?;
            x.with_column(col)?;
        }
        let mut out: Option<DataFrame> = None;
        for feature in &cfg.features {
            let cutoff = feature.cutoff.or(cfg.relative_to);
            let grid = self.grid(&feature.name, cutoff)?;
            let block = sweep_block(&x, &feature.name, cutoff, &grid)?;
            match out.as_mut() {
                Some(df) => {
                    df.vstack_mut(&block)?;
                }
                None => out = Some(block),
            }
        }
        let mut out = out.expect("features is non-empty");
        out.rechunk_mut();
        Ok(out)
    }

    /// The ascending grid for one feature: `n_steps` points over the range
    /// (times `cutoff` in relative mode), plus `n_steps` more within
    /// ±`near` of the cutoff when `near` is set.
    fn grid(&self, feature: &str, cutoff: Option<f64>) -> Result<Vec<f64>, CoreError> {
        let cfg = &self.config;
        let (lo, hi) = cfg.range;
        if !(lo.is_finite() && hi.is_finite() && lo < hi) {
            return Err(CoreError::InvalidConfig(format!(
                "sweep range must be finite with lo < hi, got {lo}:{hi}"
            )));
        }
        if cfg.n_steps < 2 {
            return Err(CoreError::InvalidConfig(format!(
                "sweep needs at least 2 steps, got {}",
                cfg.n_steps
            )));
        }
        let need_cutoff = |why: &str| match cutoff {
            Some(c) if c.is_finite() && c != 0.0 => Ok(c),
            _ => Err(CoreError::InvalidConfig(format!(
                "sweep feature {feature:?} needs a finite, non-zero cutoff {why} \
                 (`--sweep {feature}=<cutoff>` or `--sweep-relative-to`)"
            ))),
        };
        let mut values = linspace(lo, hi, cfg.n_steps);
        if cfg.relative_to.is_some() {
            let c = need_cutoff("for a relative range")?;
            values.iter_mut().for_each(|v| *v *= c);
        }
        if let Some(frac) = cfg.near {
            if !(frac.is_finite() && frac > 0.0) {
                return Err(CoreError::InvalidConfig(format!(
                    "sweep near fraction must be positive, got {frac}"
                )));
            }
            let c = need_cutoff("for --sweep-near")?;
            values.extend(linspace(c * (1.0 - frac), c * (1.0 + frac), cfg.n_steps));
        }
        // A negative cutoff reverses a relative range; `near` interleaves.
        values.sort_by(f64::total_cmp);
        values.dedup();
        if u32::try_from(values.len()).is_err() {
            return Err(CoreError::InvalidConfig(format!(
                "sweep grid has {} steps, more than fit in u32",
                values.len()
            )));
        }
        Ok(values)
    }
}

/// Every row of `x` repeated once per grid value, with `feature` set to it.
fn sweep_block(
    x: &DataFrame,
    feature: &str,
    cutoff: Option<f64>,
    grid: &[f64],
) -> Result<DataFrame, CoreError> {
    let n_steps = grid.len();
    let n = x.height() * n_steps;
    // Row indices double as the `sweep_case_row` column (polars' IdxSize is
    // u32 without the `bigidx` feature; `generate` checked they fit).
    let case_rows: Vec<u32> = (0..x.height() as u32)
        .flat_map(|r| std::iter::repeat_n(r, n_steps))
        .collect();
    let mut data = x.take(&IdxCa::from_vec("idx".into(), case_rows.clone()))?;
    let values: Vec<f64> = grid.iter().copied().cycle().take(n).collect();
    data.with_column(Series::new(feature.into(), values.clone()).into())?;

    let steps: Vec<u32> = (0..n_steps as u32).cycle().take(n).collect();
    let mut columns: Vec<Column> = vec![
        Series::new(SWEEP_CASE_ROW_COL.into(), case_rows).into(),
        StringChunked::full(SWEEP_FEATURE_COL.into(), feature, n)
            .into_series()
            .into(),
        Series::new(SWEEP_STEP_COL.into(), steps).into(),
        Series::new(SWEEP_VALUE_COL.into(), values).into(),
        Series::new(SWEEP_CUTOFF_COL.into(), vec![cutoff; n]).into(),
    ];
    columns.extend(data.into_columns());
    Ok(DataFrame::new(n, columns)?)
}

/// `n` evenly spaced points from `lo` to `hi`, both included (`n >= 2`).
fn linspace(lo: f64, hi: f64, n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| lo + (hi - lo) * i as f64 / (n - 1) as f64)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::SweepFeature;

    fn config(features: &[(&str, Option<f64>)], range: (f64, f64), n_steps: usize) -> SweepConfig {
        SweepConfig {
            features: features
                .iter()
                .map(|(name, cutoff)| SweepFeature {
                    name: name.to_string(),
                    cutoff: *cutoff,
                })
                .collect(),
            range,
            relative_to: None,
            n_steps,
            near: None,
        }
    }

    fn sample() -> DataFrame {
        df![
            "a" => [1.0, 2.0],
            "b" => [10i64, 20],
            "note" => ["x", "y"],
        ]
        .unwrap()
    }

    #[test]
    fn moves_only_the_swept_feature() -> Result<(), CoreError> {
        let grid = SweepSimulator::new(config(&[("a", None), ("b", Some(15.0))], (0.0, 4.0), 5))
            .generate(&sample())?;
        assert_eq!(grid.height(), 2 * 2 * 5);
        let names: Vec<&str> = grid.get_column_names().iter().map(|s| s.as_str()).collect();
        assert_eq!(&names[..5], SWEEP_META_COLS);
        assert_eq!(&names[5..], ["a", "b", "note"]);

        let feature = grid.column(SWEEP_FEATURE_COL)?.str()?;
        let case = grid.column(SWEEP_CASE_ROW_COL)?.u32()?;
        let step = grid.column(SWEEP_STEP_COL)?.u32()?;
        let value = grid.column(SWEEP_VALUE_COL)?.f64()?;
        let cutoff = grid.column(SWEEP_CUTOFF_COL)?.f64()?;
        let a = grid.column("a")?.cast(&DataType::Float64)?;
        let b = grid.column("b")?.cast(&DataType::Float64)?;
        let note = grid.column("note")?.str()?;
        for i in 0..grid.height() {
            let r = case.get(i).unwrap() as usize;
            let v = value.get(i).unwrap();
            assert_eq!(v, step.get(i).unwrap() as f64);
            assert_eq!(note.get(i), Some(["x", "y"][r]));
            if feature.get(i) == Some("a") {
                assert_eq!(a.f64()?.get(i), Some(v));
                assert_eq!(b.f64()?.get(i), Some([10.0, 20.0][r]));
                assert_eq!(cutoff.get(i), None);
            } else {
                assert_eq!(b.f64()?.get(i), Some(v));
                assert_eq!(a.f64()?.get(i), Some([1.0, 2.0][r]));
                assert_eq!(cutoff.get(i), Some(15.0));
            }
        }
        Ok(())
    }

    #[test]
    fn relative_range_scales_by_the_cutoff_and_near_adds_points() -> Result<(), CoreError> {
        let mut cfg = config(&[("a", Some(2.0)), ("b", None)], (0.5, 2.0), 4);
        cfg.relative_to = Some(10.0);
        let grid = SweepSimulator::new(cfg.clone()).generate(&sample())?;
        let values = |df: &DataFrame, f: &str| -> Vec<f64> {
            let mask = df
                .column(SWEEP_FEATURE_COL)
                .unwrap()
                .str()
                .unwrap()
                .equal(f);
            let rows = df.filter(&mask).unwrap();
            let case = rows
                .column(SWEEP_CASE_ROW_COL)
                .unwrap()
                .u32()
                .unwrap()
                .equal(0);
            let rows = rows.filter(&case).unwrap();
            let v = rows.column(SWEEP_VALUE_COL).unwrap().f64().unwrap();
            v.into_no_null_iter().collect()
        };
        // `a` uses its own cutoff, `b` falls back to --sweep-relative-to.
        assert_eq!(values(&grid, "a"), [1.0, 2.0, 3.0, 4.0]);
        assert_eq!(values(&grid, "b"), [5.0, 10.0, 15.0, 20.0]);

        cfg.near = Some(0.1);
        let grid = SweepSimulator::new(cfg).generate(&sample())?;
        let a = values(&grid, "a");
        assert_eq!(a.len(), 8);
        assert!(a.windows(2).all(|w| w[0] < w[1]), "{a:?}");
        assert_eq!(a.iter().filter(|v| (1.8..=2.2).contains(*v)).count(), 5);
        Ok(())
    }

    #[test]
    fn rejects_bad_configs() {
        let x = sample();
        let gen = |cfg: SweepConfig| SweepSimulator::new(cfg).generate(&x).unwrap_err();
        assert!(gen(config(&[("zz", None)], (0.0, 1.0), 3))
            .to_string()
            .contains("not found"));
        assert!(gen(config(&[("note", None)], (0.0, 1.0), 3))
            .to_string()
            .contains("not numeric"));
        assert!(gen(config(&[("a", None)], (1.0, 0.0), 3))
            .to_string()
            .contains("lo < hi"));
        assert!(gen(config(&[("a", None)], (0.0, 1.0), 1))
            .to_string()
            .contains("2 steps"));
        let mut near = config(&[("a", None)], (0.0, 1.0), 3);
        near.near = Some(0.1);
        assert!(gen(near).to_string().contains("cutoff"));
    }
}
