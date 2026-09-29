use crate::error::CoreError;
use crate::schema::JitterConfig;
use polars::prelude::*;

use super::rng::stable_normal_sample;
use super::Simulator;

pub struct JitterSimulator {
    config: JitterConfig,
    feature_std: f64,
}

impl JitterSimulator {
    pub fn new(config: JitterConfig) -> Self {
        Self {
            config,
            feature_std: 0.0,
        }
    }
}

impl Simulator for JitterSimulator {
    fn fit(&mut self, df: &DataFrame) -> Result<(), CoreError> {
        let series = df.column(&self.config.feature)?;
        self.feature_std = series.as_materialized_series().std(1).unwrap_or(0.0);
        Ok(())
    }

    fn transform(&self, df: &DataFrame) -> Result<DataFrame, CoreError> {
        let n_rows = df.height();
        let scale = self.config.scale * self.feature_std;

        // Deterministic per-row SHA256-seeded draw (see `stable_normal_sample`
        // in rng.rs) instead of a single shared RNG stream — matches jitter.py's
        // batch-independence guarantee: splitting rows into any batches and
        // concatenating the results reproduces `transform(df)` exactly.
        let noise: Vec<f64> = (0..n_rows)
            .map(|i| stable_normal_sample(self.config.random_state, i) * scale)
            .collect();
        let noise_series = Float64Chunked::from_vec("noise".into(), noise).into_series();

        let result = df
            .clone()
            .lazy()
            .with_column(lit(noise_series).alias("noise"))
            .with_column((col(&self.config.feature) + col("noise")).alias(&self.config.feature))
            .drop(cols(["noise"]))
            .collect()?;

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn jitter_transform_is_deterministic_given_same_random_state() {
        let df = df!["val" => [1.0, 2.0, 3.0, 4.0, 5.0]].unwrap();
        let config = JitterConfig {
            feature: "val".to_string(),
            scale: 0.5,
            random_state: Some(42),
        };
        let mut sim_a = JitterSimulator::new(config.clone());
        sim_a.fit(&df).unwrap();
        let out_a = sim_a.transform(&df).unwrap();

        let mut sim_b = JitterSimulator::new(config);
        sim_b.fit(&df).unwrap();
        let out_b = sim_a.transform(&df).unwrap();
        let out_b2 = sim_b.transform(&df).unwrap();

        let a = out_a.column("val").unwrap().f64().unwrap();
        let b = out_b.column("val").unwrap().f64().unwrap();
        let b2 = out_b2.column("val").unwrap().f64().unwrap();
        let orig = df.column("val").unwrap().f64().unwrap();

        for i in 0..5 {
            // Same simulator instance, same input -> exact same noise every call.
            assert_eq!(a.get(i), b.get(i));
            // Independently-constructed simulator with the same random_state
            // and the same rows -> exact same noise too (batch-independence).
            assert_eq!(a.get(i), b2.get(i));
            // Noise actually got applied.
            assert_ne!(a.get(i), orig.get(i));
        }
    }
}
