//! A site: several of the existing perturbations applied to one cohort, the
//! way one hospital's export is wrong in several columns at once (FT4 in
//! pmol/L, TSH censored, an assay never ordered). No perturbation logic lives
//! here; each step builds one of the single-feature simulators.

use crate::error::CoreError;
use crate::schema::{BlackoutConfig, ScrambleConfig, ScrambleEncoding, UnitsConfig};
use polars::prelude::*;
use serde::{Deserialize, Serialize};

use super::rng::stable_unit_floats;
use super::{
    unit_preset, BlackoutSimulator, ScrambleSimulator, Simulator, UnitsSimulator, UNIT_PRESETS,
};

fn always() -> f64 {
    1.0
}

/// One perturbation of one feature. `chance` is the probability a random
/// site draw keeps the step; the site as declared (no draw) applies every
/// step whatever its chance.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub enum SiteStep {
    Blackout {
        feature: String,
        #[serde(default)]
        companions: Vec<String>,
        rate: f64,
        #[serde(default = "always")]
        chance: f64,
    },
    Scramble {
        feature: String,
        rate: f64,
        /// Empty means all encodings.
        #[serde(default)]
        encodings: Vec<ScrambleEncoding>,
        #[serde(default = "always")]
        chance: f64,
    },
    /// Exactly one of `preset` or `factor` (with optional `offset`).
    Units {
        feature: String,
        rate: f64,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        preset: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        factor: Option<f64>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        offset: Option<f64>,
        #[serde(default = "always")]
        chance: f64,
    },
}

impl SiteStep {
    fn feature(&self) -> &str {
        match self {
            SiteStep::Blackout { feature, .. }
            | SiteStep::Scramble { feature, .. }
            | SiteStep::Units { feature, .. } => feature,
        }
    }

    fn chance(&self) -> f64 {
        match self {
            SiteStep::Blackout { chance, .. }
            | SiteStep::Scramble { chance, .. }
            | SiteStep::Units { chance, .. } => *chance,
        }
    }

    /// Every column the step writes: its feature plus blackout companions.
    fn columns(&self) -> Vec<&str> {
        let companions: &[String] = match self {
            SiteStep::Blackout { companions, .. } => companions,
            _ => &[],
        };
        std::iter::once(self.feature())
            .chain(companions.iter().map(String::as_str))
            .collect()
    }

    /// The validated single-feature simulator for this step.
    fn build(&self, seed: Option<u64>) -> Result<Box<dyn Simulator>, CoreError> {
        match self {
            SiteStep::Blackout {
                feature,
                companions,
                rate,
                ..
            } => {
                let sim = BlackoutSimulator::new(BlackoutConfig {
                    feature: feature.clone(),
                    companions: companions.clone(),
                    rate: *rate,
                    window_frac: 0.0,
                    patient_id_col: None,
                    time_col: None,
                    random_state: seed,
                });
                sim.validate()?;
                Ok(Box::new(sim))
            }
            SiteStep::Scramble {
                feature,
                rate,
                encodings,
                ..
            } => {
                let encodings = if encodings.is_empty() {
                    ScrambleEncoding::ALL.to_vec()
                } else {
                    encodings.clone()
                };
                let sim = ScrambleSimulator::new(ScrambleConfig {
                    feature: feature.clone(),
                    rate: *rate,
                    encodings,
                    random_state: seed,
                });
                sim.validate()?;
                Ok(Box::new(sim))
            }
            SiteStep::Units {
                feature,
                rate,
                preset,
                factor,
                offset,
                ..
            } => {
                let (factor, offset) = match (preset, factor, offset) {
                    (Some(name), None, None) => unit_preset(name).ok_or_else(|| {
                        let names: Vec<&str> = UNIT_PRESETS.iter().map(|(n, _, _)| *n).collect();
                        CoreError::InvalidConfig(format!(
                            "unknown unit preset {name:?}; valid: {}",
                            names.join(", ")
                        ))
                    })?,
                    (None, Some(factor), offset) => (*factor, offset.unwrap_or(0.0)),
                    _ => {
                        return Err(CoreError::InvalidConfig(
                            "a units step takes exactly one of `preset` or `factor` (with \
                             optional `offset`)"
                                .to_string(),
                        ))
                    }
                };
                let sim = UnitsSimulator::new(UnitsConfig {
                    feature: feature.clone(),
                    factor,
                    offset,
                    rate: *rate,
                    random_state: seed,
                });
                sim.validate()?;
                Ok(Box::new(sim))
            }
        }
    }
}

/// The site file: `{"steps": [{"units": {...}}, {"scramble": {...}}, ...]}`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SiteConfig {
    pub steps: Vec<SiteStep>,
    /// Set by the caller, never read from the site file.
    #[serde(skip)]
    pub random_state: Option<u64>,
}

pub struct SiteSimulator {
    config: SiteConfig,
}

fn in_step(i: usize, msg: String) -> CoreError {
    CoreError::InvalidConfig(format!("site step {i}: {msg}"))
}

impl SiteSimulator {
    pub fn new(config: SiteConfig) -> Self {
        Self { config }
    }

    /// Config checks that need no data, so callers can run them before
    /// writing anything.
    pub fn validate(&self) -> Result<(), CoreError> {
        let steps = &self.config.steps;
        if steps.is_empty() {
            return Err(CoreError::InvalidConfig(
                "a site needs at least one step".to_string(),
            ));
        }
        let mut scrambled: Vec<&str> = Vec::new();
        for (i, step) in steps.iter().enumerate() {
            if !(0.0..=1.0).contains(&step.chance()) {
                return Err(in_step(
                    i,
                    format!("chance must be in [0, 1], got {}", step.chance()),
                ));
            }
            step.build(None).map_err(|e| match e {
                CoreError::InvalidConfig(msg) => in_step(i, msg),
                other => in_step(i, other.to_string()),
            })?;
            // Scramble leaves the column as text, which a later numeric step
            // would silently null.
            if let Some(name) = step.columns().into_iter().find(|n| scrambled.contains(n)) {
                return Err(in_step(
                    i,
                    format!(
                        "{name:?} was scrambled to text by an earlier step; make scramble the \
                         last step for that feature"
                    ),
                ));
            }
            if matches!(step, SiteStep::Scramble { .. }) {
                scrambled.push(step.feature());
            }
        }
        Ok(())
    }

    /// Fails when a step names a column `df` does not have, or when a units
    /// or scramble step names one that is not numeric: both would cast it
    /// leniently and null every value without an error.
    pub fn check_columns(&self, df: &DataFrame) -> Result<(), CoreError> {
        for (i, step) in self.config.steps.iter().enumerate() {
            for name in step.columns() {
                if df.column(name).is_err() {
                    return Err(CoreError::FeatureNotFound(name.to_string()));
                }
            }
            let dtype = df.column(step.feature())?.dtype();
            if !matches!(step, SiteStep::Blackout { .. }) && !dtype.is_numeric() {
                return Err(in_step(
                    i,
                    format!(
                        "feature {:?} is {dtype}, but units and scramble steps need a numeric column",
                        step.feature()
                    ),
                ));
            }
        }
        Ok(())
    }

    /// Indices of the steps a draw applies. `None` is the site as declared:
    /// every step. `Some(d)` keeps each step with its `chance`, decided by
    /// `(random_state, d, step)` alone so a draw never depends on the others.
    pub fn steps_in(&self, draw: Option<usize>) -> Vec<usize> {
        (0..self.config.steps.len())
            .filter(|&j| match draw {
                None => true,
                Some(d) => {
                    let (u, _) =
                        stable_unit_floats(self.config.random_state, &format!("site:{d}:step:{j}"));
                    u < self.config.steps[j].chance()
                }
            })
            .collect()
    }

    /// Each step and each draw gets its own seed. The single-feature
    /// simulators all pick rows from `random_state` alone, so one shared seed
    /// would hit the same rows in every step at equal rates, and in every draw.
    fn step_seed(&self, draw: Option<usize>, step: usize) -> Option<u64> {
        let draw_offset = draw.map_or(0, |d| (d as u64 + 1).wrapping_mul(1_000_003));
        self.config
            .random_state
            .map(|s| s.wrapping_add(step as u64).wrapping_add(draw_offset))
    }

    pub fn transform_draw(
        &self,
        df: &DataFrame,
        draw: Option<usize>,
    ) -> Result<DataFrame, CoreError> {
        self.validate()?;
        self.steps_in(draw)
            .into_iter()
            .try_fold(df.clone(), |df, j| {
                self.config.steps[j]
                    .build(self.step_seed(draw, j))?
                    .transform(&df)
            })
    }
}

impl Simulator for SiteSimulator {
    fn fit(&mut self, _df: &DataFrame) -> Result<(), CoreError> {
        Ok(())
    }

    fn transform(&self, df: &DataFrame) -> Result<DataFrame, CoreError> {
        self.transform_draw(df, None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn site(json: &str) -> SiteSimulator {
        let mut config: SiteConfig = serde_json::from_str(json).unwrap();
        config.random_state = Some(7);
        SiteSimulator::new(config)
    }

    fn frame() -> DataFrame {
        let n = 100;
        df![
            "ft4" => (0..n).map(|i| 1.0 + i as f64 / 100.0).collect::<Vec<_>>(),
            "tsh" => (0..n).map(|i| 0.5 + i as f64).collect::<Vec<_>>(),
            "t3" => (0..n).map(|i| i as f64).collect::<Vec<_>>(),
            "t3_measured" => vec![1i64; n],
        ]
        .unwrap()
    }

    fn nulls(df: &DataFrame, name: &str) -> Vec<bool> {
        df.column(name)
            .unwrap()
            .is_null()
            .iter()
            .map(|v| v == Some(true))
            .collect()
    }

    #[test]
    fn steps_compose_across_features() {
        let sim = site(
            r#"{"steps": [
                {"units": {"feature": "ft4", "preset": "ft4_ngdl_to_pmoll", "rate": 1.0}},
                {"scramble": {"feature": "tsh", "rate": 1.0, "encodings": ["censored"]}},
                {"blackout": {"feature": "t3", "companions": ["t3_measured"], "rate": 1.0}}
            ]}"#,
        );
        let df = frame();
        let out = sim.transform(&df).unwrap();
        assert_eq!(out.get_column_names(), df.get_column_names());
        let ft4 = out.column("ft4").unwrap().f64().unwrap().get(0).unwrap();
        assert!((ft4 - 12.87).abs() < 1e-12);
        assert_eq!(out.column("tsh").unwrap().dtype(), &DataType::String);
        assert_eq!(out.column("t3").unwrap().null_count(), 100);
        assert_eq!(out.column("t3_measured").unwrap().null_count(), 100);
    }

    #[test]
    fn equal_rates_hit_different_rows_per_step() {
        let sim = site(
            r#"{"steps": [
                {"blackout": {"feature": "ft4", "rate": 0.5}},
                {"blackout": {"feature": "tsh", "rate": 0.5}}
            ]}"#,
        );
        let out = sim.transform(&frame()).unwrap();
        assert_eq!(out.column("ft4").unwrap().null_count(), 50);
        assert_ne!(nulls(&out, "ft4"), nulls(&out, "tsh"));
    }

    #[test]
    fn zero_rates_leave_the_cohort_unchanged() {
        let sim = site(
            r#"{"steps": [
                {"units": {"feature": "ft4", "factor": 10, "rate": 0.0}},
                {"blackout": {"feature": "t3", "rate": 0.0}}
            ]}"#,
        );
        let df = frame();
        assert!(sim.transform(&df).unwrap().equals_missing(&df));
    }

    #[test]
    fn draws_keep_steps_by_chance_and_are_reproducible() {
        let sim = site(
            r#"{"steps": [
                {"units": {"feature": "ft4", "factor": 10, "rate": 1.0}},
                {"blackout": {"feature": "tsh", "rate": 1.0, "chance": 0.5}},
                {"blackout": {"feature": "t3", "rate": 1.0, "chance": 0.0}}
            ]}"#,
        );
        assert_eq!(sim.steps_in(None), vec![0, 1, 2]);
        let draws: Vec<Vec<usize>> = (0..40).map(|d| sim.steps_in(Some(d))).collect();
        assert!(draws.iter().all(|s| s.contains(&0) && !s.contains(&2)));
        let kept = draws.iter().filter(|s| s.contains(&1)).count();
        assert!((5..35).contains(&kept), "chance 0.5 kept {kept} of 40");

        let df = frame();
        let a = sim.transform_draw(&df, Some(3)).unwrap();
        assert!(a.equals_missing(&sim.transform_draw(&df, Some(3)).unwrap()));
    }

    #[test]
    fn draws_pick_different_rows() {
        let sim = site(r#"{"steps": [{"blackout": {"feature": "ft4", "rate": 0.5}}]}"#);
        let df = frame();
        let a = sim.transform_draw(&df, Some(0)).unwrap();
        let b = sim.transform_draw(&df, Some(1)).unwrap();
        assert_ne!(nulls(&a, "ft4"), nulls(&b, "ft4"));
    }

    fn validate_err(json: &str) -> String {
        site(json).validate().unwrap_err().to_string()
    }

    #[test]
    fn bad_sites_are_refused_before_any_data_is_read() {
        assert!(validate_err(r#"{"steps": []}"#).contains("at least one step"));
        assert!(
            validate_err(r#"{"steps": [{"blackout": {"feature": "a", "rate": 1.5}}]}"#)
                .contains("site step 0")
        );
        assert!(validate_err(
            r#"{"steps": [{"blackout": {"feature": "a", "rate": 0.5, "chance": 2}}]}"#
        )
        .contains("chance"));
        assert!(validate_err(
            r#"{"steps": [{"units": {"feature": "a", "preset": "nope", "rate": 1.0}}]}"#
        )
        .contains("unknown unit preset"));
        assert!(validate_err(
            r#"{"steps": [{"units": {"feature": "a", "preset": "temp_f_to_c", "factor": 2, "rate": 1.0}}]}"#
        )
        .contains("exactly one of"));
        assert!(validate_err(
            r#"{"steps": [
                {"scramble": {"feature": "a", "rate": 0.5}},
                {"units": {"feature": "a", "factor": 10, "rate": 0.5}}
            ]}"#
        )
        .contains("site step 1"));
    }

    #[test]
    fn unknown_fields_and_missing_columns_are_errors() {
        let typo = r#"{"steps": [{"blackout": {"feature": "a", "rate": 0.5, "chanse": 0.5}}]}"#;
        assert!(serde_json::from_str::<SiteConfig>(typo).is_err());
        let seeded = r#"{"steps": [], "random_state": 1}"#;
        assert!(serde_json::from_str::<SiteConfig>(seeded).is_err());

        let sim = site(r#"{"steps": [{"blackout": {"feature": "nope", "rate": 0.5}}]}"#);
        let err = sim.check_columns(&frame()).unwrap_err();
        assert!(matches!(err, CoreError::FeatureNotFound(ref n) if n == "nope"));
    }
}
