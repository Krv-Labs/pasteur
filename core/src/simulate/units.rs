use crate::error::CoreError;
use crate::schema::UnitsConfig;
use polars::prelude::*;

use super::rng::{row_mask, seeded_rng};
use super::Simulator;

/// Named conversions, conventional -> SI: `(name, factor, offset)`.
/// Molar-mass-dependent factors (mg/dL -> mmol/L etc.) are per analyte.
pub const UNIT_PRESETS: &[(&str, f64, f64)] = &[
    ("glucose_mgdl_to_mmoll", 1.0 / 18.016, 0.0),
    ("creatinine_mgdl_to_umoll", 88.42, 0.0),
    ("cholesterol_mgdl_to_mmoll", 1.0 / 38.67, 0.0),
    ("triglycerides_mgdl_to_mmoll", 0.01129, 0.0),
    ("hemoglobin_gdl_to_gl", 10.0, 0.0),
    ("ft4_ngdl_to_pmoll", 12.87, 0.0),
    ("hba1c_ngsp_to_ifcc", 10.929, -23.50),
    ("temp_f_to_c", 5.0 / 9.0, -160.0 / 9.0),
    ("weight_lb_to_kg", 0.45359237, 0.0),
];

/// `(factor, offset)` for a preset name.
pub fn unit_preset(name: &str) -> Option<(f64, f64)> {
    UNIT_PRESETS
        .iter()
        .find(|(n, _, _)| *n == name)
        .map(|&(_, f, o)| (f, o))
}

pub struct UnitsSimulator {
    config: UnitsConfig,
}

impl UnitsSimulator {
    pub fn new(config: UnitsConfig) -> Self {
        Self { config }
    }

    fn validate(&self) -> Result<(), CoreError> {
        let c = &self.config;
        if !c.factor.is_finite() || c.factor == 0.0 {
            return Err(CoreError::InvalidConfig(format!(
                "units factor must be finite and non-zero, got {}",
                c.factor
            )));
        }
        if !c.offset.is_finite() {
            return Err(CoreError::InvalidConfig(format!(
                "units offset must be finite, got {}",
                c.offset
            )));
        }
        if !(0.0..=1.0).contains(&c.rate) {
            return Err(CoreError::InvalidConfig(format!(
                "units rate must be in [0, 1], got {}",
                c.rate
            )));
        }
        Ok(())
    }
}

impl Simulator for UnitsSimulator {
    fn fit(&mut self, _df: &DataFrame) -> Result<(), CoreError> {
        Ok(())
    }

    /// Rewrites a seeded fraction of rows as `value * factor + offset`: one
    /// site exporting the analyte in a different unit. Output is Float64.
    fn transform(&self, df: &DataFrame) -> Result<DataFrame, CoreError> {
        self.validate()?;
        let feature = self.config.feature.as_str();
        if !df.get_column_names().iter().any(|c| c.as_str() == feature) {
            return Err(CoreError::FeatureNotFound(feature.to_string()));
        }

        let mask = row_mask(
            df.height(),
            self.config.rate,
            &mut seeded_rng(self.config.random_state),
        );
        let mut df_with_mask = df.clone();
        df_with_mask.with_column(
            BooleanChunked::from_slice("mask".into(), &mask)
                .into_series()
                .into(),
        )?;

        let value = col(feature).cast(DataType::Float64);
        let converted = when(col("mask"))
            .then(value.clone() * lit(self.config.factor) + lit(self.config.offset))
            .otherwise(value)
            .alias(feature);

        Ok(df_with_mask
            .lazy()
            .with_column(converted)
            .drop(cols(["mask"]))
            .collect()?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frame() -> DataFrame {
        df![
            "val" => [90.0, 100.0, 110.0, 120.0, 130.0, 140.0, 150.0, 160.0, 170.0, 180.0],
            "other" => [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        ]
        .unwrap()
    }

    fn config(factor: f64, offset: f64, rate: f64) -> UnitsConfig {
        UnitsConfig {
            feature: "val".to_string(),
            factor,
            offset,
            rate,
            random_state: Some(3),
        }
    }

    fn values(df: &DataFrame, name: &str) -> Vec<Option<f64>> {
        df.column(name).unwrap().f64().unwrap().iter().collect()
    }

    #[test]
    fn same_random_state_is_deterministic() {
        let sim = UnitsSimulator::new(config(2.0, 1.0, 0.5));
        let df = frame();
        assert_eq!(
            values(&sim.transform(&df).unwrap(), "val"),
            values(&sim.transform(&df).unwrap(), "val")
        );
    }

    #[test]
    fn glucose_preset_at_full_rate_converts_every_row() {
        let (factor, offset) = unit_preset("glucose_mgdl_to_mmoll").unwrap();
        let df = frame();
        let out = UnitsSimulator::new(config(factor, offset, 1.0))
            .transform(&df)
            .unwrap();
        for (got, orig) in values(&out, "val").iter().zip(values(&df, "val")) {
            assert!((got.unwrap() - orig.unwrap() / 18.016).abs() < 1e-12);
        }
    }

    #[test]
    fn temp_preset_applies_affine_offset() {
        let (factor, offset) = unit_preset("temp_f_to_c").unwrap();
        let df = df!["val" => [212i64, 32]].unwrap();
        let out = UnitsSimulator::new(config(factor, offset, 1.0))
            .transform(&df)
            .unwrap();
        let v = values(&out, "val");
        assert!((v[0].unwrap() - 100.0).abs() < 1e-12);
        assert!(v[1].unwrap().abs() < 1e-12);
    }

    #[test]
    fn half_rate_changes_exactly_half_and_leaves_rest_untouched() {
        let df = frame();
        let out = UnitsSimulator::new(config(10.0, 0.0, 0.5))
            .transform(&df)
            .unwrap();
        assert_eq!(out.get_column_names(), df.get_column_names());
        let (before, after) = (values(&df, "val"), values(&out, "val"));
        let changed = before
            .iter()
            .zip(&after)
            .filter(|(b, a)| b.unwrap().to_bits() != a.unwrap().to_bits())
            .count();
        assert_eq!(changed, 5);
        assert_eq!(values(&out, "other"), values(&df, "other"));
    }

    #[test]
    fn rejects_bad_config_and_missing_feature() {
        let df = frame();
        for (factor, offset) in [(0.0, 0.0), (f64::NAN, 0.0), (1.0, f64::INFINITY)] {
            let err = UnitsSimulator::new(config(factor, offset, 0.5))
                .transform(&df)
                .unwrap_err();
            assert!(matches!(err, CoreError::InvalidConfig(_)), "got {err:?}");
        }
        let mut missing = config(2.0, 0.0, 0.5);
        missing.feature = "nope".to_string();
        let err = UnitsSimulator::new(missing).transform(&df).unwrap_err();
        assert!(matches!(err, CoreError::FeatureNotFound(ref n) if n == "nope"));
        assert!(unit_preset("nope").is_none());
    }
}
