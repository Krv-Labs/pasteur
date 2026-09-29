use crate::error::CoreError;
use crate::schema::FlipperConfig;
use polars::prelude::*;
use rand::prelude::*;
use std::collections::HashMap;

pub const PAIR_ID_COL: &str = "pair_id";
pub const STEP_COL: &str = "step";
pub const T_COL: &str = "t";
pub const SOURCE_A_ROW_COL: &str = "source_a_row";
pub const SOURCE_B_ROW_COL: &str = "source_b_row";
pub const LABEL_A_COL: &str = "label_a";
pub const LABEL_B_COL: &str = "label_b";

/// Meta columns written by `generate` and required by flipper evaluation.
pub const FLIPPER_META_COLS: [&str; 7] = [
    PAIR_ID_COL,
    STEP_COL,
    T_COL,
    SOURCE_A_ROW_COL,
    SOURCE_B_ROW_COL,
    LABEL_A_COL,
    LABEL_B_COL,
];

pub fn is_flipper_meta_column(name: &str) -> bool {
    FLIPPER_META_COLS.contains(&name)
}

/// Generates synthetic "patients" by sampling pairs of real rows with
/// different label values and linearly interpolating their raw feature
/// vectors between them — an alternative to PCA-axis perturbation that
/// needs no inverse transform back into model-input space, since the
/// interpolation already lives in that space.
///
/// Unlike `BlackoutSimulator`/`JitterSimulator`, this isn't a row-preserving
/// perturbation of the input `DataFrame` — every output row is a *synthetic*
/// point between two real ones, so it doesn't implement the `Simulator`
/// trait's `fit`/`transform(df) -> DataFrame` shape (nothing else in this
/// workspace calls Flipper polymorphically through that trait). It also
/// needs labels to pick cross-class pairs, which `Simulator::fit` doesn't
/// have access to — so `generate` takes `y` directly.
pub struct FlipperSimulator {
    config: FlipperConfig,
}

impl FlipperSimulator {
    pub fn new(config: FlipperConfig) -> Self {
        Self { config }
    }

    /// Output columns: `pair_id`, `step`, `t`, `source_a_row`,
    /// `source_b_row` (0-based positional row indices into `x`/`y`),
    /// `label_a`, `label_b`, plus every numeric column of `x` (interpolated;
    /// non-numeric columns are dropped, matching the old PCA path's feature
    /// selection). `t` is `step / (n_steps - 1)`, inclusive of both
    /// endpoints — `t=0` reproduces row `source_a_row` exactly, `t=1`
    /// reproduces `source_b_row` exactly.
    pub fn generate(&self, x: &DataFrame, y: &Series) -> Result<DataFrame, CoreError> {
        if x.height() != y.len() {
            return Err(CoreError::InvalidConfig(format!(
                "x has {} rows but y has {} — must match",
                x.height(),
                y.len()
            )));
        }

        let feature_cols: Vec<String> = x
            .columns()
            .iter()
            .filter(|s| s.dtype().is_numeric())
            .map(|s| s.name().to_string())
            .collect();
        if feature_cols.is_empty() {
            return Err(CoreError::InvalidConfig(
                "FlipperSimulator requires at least one numeric feature column".to_string(),
            ));
        }
        let feature_series: Vec<Float64Chunked> = feature_cols
            .iter()
            .map(|c| {
                x.column(c)?
                    .cast(&DataType::Float64)
                    .map(|s| s.f64().unwrap().clone())
            })
            .collect::<Result<_, PolarsError>>()?;

        let label_groups = group_rows_by_label(y)?;
        // Sorted for determinism: HashMap iteration order isn't guaranteed
        // stable across separately-constructed maps with the same content,
        // which would otherwise make key_a/key_b assignment (and hence
        // source_a/source_b) non-reproducible across runs with the same seed.
        let mut label_keys: Vec<String> = label_groups.keys().cloned().collect();
        label_keys.sort();
        if label_keys.len() < 2 {
            return Err(CoreError::InvalidConfig(
                "FlipperSimulator needs at least two distinct label values to pair across"
                    .to_string(),
            ));
        }
        let y_f64 = y.cast(&DataType::Float64)?;
        let y_f64 = y_f64.f64()?;

        let mut rng = match self.config.random_state {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => StdRng::from_entropy(),
        };
        let n_steps = self.config.n_steps.max(2);
        let n_pairs = self.config.n_pairs;

        let mut pair_id_col = Vec::with_capacity(n_pairs * n_steps);
        let mut step_col = Vec::with_capacity(n_pairs * n_steps);
        let mut t_col = Vec::with_capacity(n_pairs * n_steps);
        let mut source_a_col = Vec::with_capacity(n_pairs * n_steps);
        let mut source_b_col = Vec::with_capacity(n_pairs * n_steps);
        let mut label_a_col = Vec::with_capacity(n_pairs * n_steps);
        let mut label_b_col = Vec::with_capacity(n_pairs * n_steps);
        let mut feature_data: Vec<Vec<f64>> = feature_cols
            .iter()
            .map(|_| Vec::with_capacity(n_pairs * n_steps))
            .collect();

        for pair_id in 0..n_pairs {
            let (key_a, key_b) = pick_two_distinct_labels(&label_keys, &mut rng);
            let row_a = *label_groups[&key_a].choose(&mut rng).unwrap();
            let row_b = *label_groups[&key_b].choose(&mut rng).unwrap();
            let label_a = y_f64.get(row_a).unwrap_or(f64::NAN);
            let label_b = y_f64.get(row_b).unwrap_or(f64::NAN);

            for step in 0..n_steps {
                let t = step as f64 / (n_steps - 1) as f64;
                pair_id_col.push(pair_id as u32);
                step_col.push(step as u32);
                t_col.push(t);
                source_a_col.push(row_a as u32);
                source_b_col.push(row_b as u32);
                label_a_col.push(label_a);
                label_b_col.push(label_b);
                for (j, ca) in feature_series.iter().enumerate() {
                    let a_val = ca.get(row_a).unwrap_or(f64::NAN);
                    let b_val = ca.get(row_b).unwrap_or(f64::NAN);
                    feature_data[j].push(a_val + t * (b_val - a_val));
                }
            }
        }

        let n_rows = pair_id_col.len();
        let mut columns: Vec<Column> = vec![
            Series::new(PAIR_ID_COL.into(), pair_id_col).into(),
            Series::new(STEP_COL.into(), step_col).into(),
            Series::new(T_COL.into(), t_col).into(),
            Series::new(SOURCE_A_ROW_COL.into(), source_a_col).into(),
            Series::new(SOURCE_B_ROW_COL.into(), source_b_col).into(),
            Series::new(LABEL_A_COL.into(), label_a_col).into(),
            Series::new(LABEL_B_COL.into(), label_b_col).into(),
        ];
        for (name, data) in feature_cols.into_iter().zip(feature_data) {
            columns.push(Series::new(name.into(), data).into());
        }

        Ok(DataFrame::new(n_rows, columns)?)
    }
}

fn group_rows_by_label(y: &Series) -> Result<HashMap<String, Vec<usize>>, CoreError> {
    let mut groups: HashMap<String, Vec<usize>> = HashMap::new();
    for i in 0..y.len() {
        let key = format!("{}", y.get(i)?);
        groups.entry(key).or_default().push(i);
    }
    Ok(groups)
}

fn pick_two_distinct_labels(keys: &[String], rng: &mut StdRng) -> (String, String) {
    if keys.len() == 2 {
        return (keys[0].clone(), keys[1].clone());
    }
    let mut idxs: Vec<usize> = (0..keys.len()).collect();
    idxs.shuffle(rng);
    (keys[idxs[0]].clone(), keys[idxs[1]].clone())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_df() -> (DataFrame, Series) {
        let x = df![
            "f1" => [1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0],
            "f2" => [0.0, 0.0, 0.0, 0.0, 100.0, 100.0, 100.0, 100.0],
        ]
        .unwrap();
        let y = Series::new("label".into(), &[0i64, 0, 0, 0, 1, 1, 1, 1]);
        (x, y)
    }

    #[test]
    fn generate_pairs_rows_from_different_labels_only() {
        let (x, y) = sample_df();
        let config = FlipperConfig {
            n_pairs: 20,
            n_steps: 5,
            flip_threshold: 0.5,
            positive_class_label: None,
            random_state: Some(7),
        };
        let grid = FlipperSimulator::new(config).generate(&x, &y).unwrap();

        assert_eq!(grid.height(), 20 * 5);
        let label_a = grid.column(LABEL_A_COL).unwrap().f64().unwrap();
        let label_b = grid.column(LABEL_B_COL).unwrap().f64().unwrap();
        let source_a = grid.column(SOURCE_A_ROW_COL).unwrap().u32().unwrap();
        let source_b = grid.column(SOURCE_B_ROW_COL).unwrap().u32().unwrap();
        for i in 0..grid.height() {
            assert_ne!(label_a.get(i), label_b.get(i));
            let a = source_a.get(i).unwrap();
            let b = source_b.get(i).unwrap();
            assert_ne!(
                a < 4,
                b < 4,
                "a={a} b={b} must be in different label groups"
            );
        }
    }

    #[test]
    fn generate_interpolates_endpoints_exactly() -> Result<(), CoreError> {
        let (x, y) = sample_df();
        let config = FlipperConfig {
            n_pairs: 1,
            n_steps: 5,
            flip_threshold: 0.5,
            positive_class_label: None,
            random_state: Some(42),
        };
        let grid = FlipperSimulator::new(config).generate(&x, &y)?;

        let source_a = grid.column(SOURCE_A_ROW_COL)?.u32()?.get(0).unwrap();
        let source_b = grid.column(SOURCE_B_ROW_COL)?.u32()?.get(0).unwrap();
        let f1 = grid.column("f1")?.f64()?;
        let t = grid.column(T_COL)?.f64()?;

        let a_val = x.column("f1")?.f64()?.get(source_a as usize).unwrap();
        let b_val = x.column("f1")?.f64()?.get(source_b as usize).unwrap();

        assert_eq!(t.get(0), Some(0.0));
        assert_eq!(f1.get(0), Some(a_val));
        let last = grid.height() - 1;
        assert_eq!(t.get(last), Some(1.0));
        assert_eq!(f1.get(last), Some(b_val));
        let mid = grid.height() / 2;
        assert_eq!(t.get(mid), Some(0.5));
        assert!((f1.get(mid).unwrap() - (a_val + b_val) / 2.0).abs() < 1e-9);
        Ok(())
    }

    #[test]
    fn generate_is_deterministic_given_same_random_state() -> Result<(), CoreError> {
        let (x, y) = sample_df();
        let config = FlipperConfig {
            n_pairs: 10,
            n_steps: 4,
            flip_threshold: 0.5,
            positive_class_label: None,
            random_state: Some(123),
        };
        let a = FlipperSimulator::new(config.clone()).generate(&x, &y)?;
        let b = FlipperSimulator::new(config).generate(&x, &y)?;

        let a_f1 = a.column("f1")?.f64()?;
        let b_f1 = b.column("f1")?.f64()?;
        for i in 0..a.height() {
            assert_eq!(a_f1.get(i), b_f1.get(i));
        }
        Ok(())
    }

    #[test]
    fn generate_errors_with_fewer_than_two_label_groups() {
        let x = df!["f1" => [1.0, 2.0, 3.0]].unwrap();
        let y = Series::new("label".into(), &[0i64, 0, 0]);
        let config = FlipperConfig {
            n_pairs: 1,
            n_steps: 3,
            flip_threshold: 0.5,
            positive_class_label: None,
            random_state: Some(1),
        };
        assert!(FlipperSimulator::new(config).generate(&x, &y).is_err());
    }
}
