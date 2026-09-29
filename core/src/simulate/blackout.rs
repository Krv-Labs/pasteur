use crate::error::CoreError;
use crate::schema::BlackoutConfig;
use polars::prelude::*;
use rand::prelude::*;

use super::patient::{
    apply_patient_window_mask, group_rows_by_patient, selected_patient_keys, time_sort_keys,
};
use super::Simulator;

pub struct BlackoutSimulator {
    config: BlackoutConfig,
}

impl BlackoutSimulator {
    pub fn new(config: BlackoutConfig) -> Self {
        Self { config }
    }
}

impl Simulator for BlackoutSimulator {
    fn fit(&mut self, _df: &DataFrame) -> Result<(), CoreError> {
        Ok(())
    }

    fn transform(&self, df: &DataFrame) -> Result<DataFrame, CoreError> {
        let mut rng = if let Some(seed) = self.config.random_state {
            StdRng::seed_from_u64(seed)
        } else {
            StdRng::from_entropy()
        };

        if let Some(ref patient_id_col) = self.config.patient_id_col {
            self.transform_patient_level(df, &mut rng, patient_id_col)
        } else {
            self.transform_row_level(df, &mut rng)
        }
    }
}

impl BlackoutSimulator {
    fn transform_row_level(
        &self,
        df: &DataFrame,
        rng: &mut StdRng,
    ) -> Result<DataFrame, CoreError> {
        let n_rows = df.height();
        // Matches blackout.py's `_fit_row_mode`: truncating int(), not ceil().
        let n_blackout = (n_rows as f64 * self.config.rate) as usize;

        let mut indices: Vec<usize> = (0..n_rows).collect();
        indices.shuffle(rng);
        let blackout_indices = &indices[0..n_blackout.min(n_rows)];

        let mut blackout_mask = vec![false; n_rows];
        for &idx in blackout_indices {
            blackout_mask[idx] = true;
        }

        self.apply_mask(df, &blackout_mask)
    }

    /// Patient-aware blackout: select a fraction of patients, then black out
    /// a contiguous window of each selected patient's rows (optionally
    /// sorted by `time_col`). Mirrors `blackout.py`'s `_fit_patient_mode`:
    /// * `rate` picks *which patients* (global RNG, order-shuffled selection
    ///   in place of numpy's `rng.choice(..., replace=False)`).
    /// * `window_frac` picks *how much of each patient's timeline*, with the
    ///   window's start position derived from a SHA256 seed mixed from
    ///   `(random_state, patient_id)` — so the window choice is stable
    ///   regardless of patient iteration/selection order.
    fn transform_patient_level(
        &self,
        df: &DataFrame,
        rng: &mut StdRng,
        patient_id_col: &str,
    ) -> Result<DataFrame, CoreError> {
        let n_rows = df.height();
        let (patient_order, patient_rows) = group_rows_by_patient(df, patient_id_col)?;
        let target_patient_keys = selected_patient_keys(&patient_order, self.config.rate, rng);
        let sort_keys = time_sort_keys(df, n_rows, self.config.time_col.as_deref());
        let mut blackout_mask = vec![false; n_rows];

        for pid in &target_patient_keys {
            apply_patient_window_mask(
                &mut blackout_mask,
                &patient_rows,
                pid,
                sort_keys.as_deref(),
                self.config.window_frac,
                self.config.random_state,
            );
        }

        self.apply_mask(df, &blackout_mask)
    }

    /// Every column blacked out together: the primary `feature` plus any
    /// declared `companions`. Masking them in one pass is the whole point —
    /// nulling the assay while its `_measured` flag still reads 1 produces a
    /// row no EHR emits.
    fn masked_columns(&self) -> Vec<&str> {
        std::iter::once(self.config.feature.as_str())
            .chain(self.config.companions.iter().map(|s| s.as_str()))
            .collect()
    }

    fn apply_mask(&self, df: &DataFrame, blackout_mask: &[bool]) -> Result<DataFrame, CoreError> {
        let targets = self.masked_columns();
        // Fail loudly on a name that isn't there: a silently-skipped companion
        // is exactly the incoherence this masking exists to prevent.
        for name in &targets {
            if !df.get_column_names().iter().any(|c| c.as_str() == *name) {
                return Err(CoreError::FeatureNotFound((*name).to_string()));
            }
        }

        let mask_ca = BooleanChunked::from_slice("mask".into(), blackout_mask);

        let mut df_with_mask = df.clone();
        df_with_mask.with_column(mask_ca.into_series().into())?;

        let nulled: Vec<Expr> = targets
            .iter()
            .map(|name| {
                when(col("mask"))
                    .then(lit(NULL))
                    .otherwise(col(*name))
                    .alias(*name)
            })
            .collect();

        let result = df_with_mask
            .lazy()
            .with_columns(nulled)
            .drop(cols(["mask"]))
            .collect()?;

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blackout_patient_level_is_deterministic_and_masks_only_target_windows() {
        // 3 patients x 4 rows each, sequential values so we can see exactly
        // what got blacked out.
        let df = df![
            "patient" => ["a", "a", "a", "a", "b", "b", "b", "b", "c", "c", "c", "c"],
            "val" => [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
        ]
        .unwrap();

        let config = BlackoutConfig {
            feature: "val".to_string(),
            companions: vec![],
            rate: 0.5,        // half the patients
            window_frac: 0.5, // half each selected patient's rows
            patient_id_col: Some("patient".to_string()),
            time_col: None,
            random_state: Some(7),
        };
        let sim = BlackoutSimulator::new(config);

        let out1 = sim.transform(&df).unwrap();
        let out2 = sim.transform(&df).unwrap();

        // Determinism: same random_state -> identical masking every call.
        let v1 = out1.column("val").unwrap().f64().unwrap();
        let v2 = out2.column("val").unwrap().f64().unwrap();
        for i in 0..12 {
            assert_eq!(v1.get(i), v2.get(i));
        }

        // Exactly one patient (n_blackout = floor(3*0.5) with min-1 fallback = 1)
        // should have exactly 2 of its 4 rows (window_frac=0.5) nulled out;
        // the rest of the data must be untouched.
        let null_count = v1.iter().filter(|v| v.is_none()).count();
        assert_eq!(
            null_count, 2,
            "expected exactly one patient's half-window blacked out"
        );
    }

    fn coherence_frame() -> DataFrame {
        df![
            "val" => [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            "val_measured" => [1i64, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            "other" => [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        ]
        .unwrap()
    }

    fn coherence_config(companions: Vec<String>) -> BlackoutConfig {
        BlackoutConfig {
            feature: "val".to_string(),
            companions,
            rate: 0.5,
            window_frac: 0.0,
            patient_id_col: None,
            time_col: None,
            random_state: Some(11),
        }
    }

    #[test]
    fn companions_are_nulled_on_exactly_the_same_rows_as_the_feature() {
        let df = coherence_frame();
        let sim = BlackoutSimulator::new(coherence_config(vec!["val_measured".to_string()]));
        let out = sim.transform(&df).unwrap();

        let val = out.column("val").unwrap();
        let measured = out.column("val_measured").unwrap();
        let val_nulls: Vec<bool> = (0..out.height())
            .map(|i| val.get(i).unwrap().is_null())
            .collect();
        let measured_nulls: Vec<bool> = (0..out.height())
            .map(|i| measured.get(i).unwrap().is_null())
            .collect();

        assert_eq!(
            val_nulls, measured_nulls,
            "a blacked-out assay must not keep claiming it was measured"
        );
        assert_eq!(
            val_nulls.iter().filter(|n| **n).count(),
            5,
            "rate 0.5 over 10 rows should null 5"
        );

        // Undeclared columns stay untouched — masking is opt-in per column.
        let other = out.column("other").unwrap();
        assert_eq!(
            (0..out.height())
                .filter(|i| other.get(*i).unwrap().is_null())
                .count(),
            0
        );
    }

    #[test]
    fn unknown_companion_is_an_error_not_a_silent_skip() {
        let df = coherence_frame();
        let sim = BlackoutSimulator::new(coherence_config(vec!["val_mesured".to_string()]));
        let err = sim.transform(&df).expect_err("typo'd companion must fail");
        assert!(
            matches!(err, CoreError::FeatureNotFound(ref n) if n == "val_mesured"),
            "got {err:?}"
        );
    }
}
