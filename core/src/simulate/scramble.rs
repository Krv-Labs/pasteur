use crate::error::CoreError;
use crate::schema::{ScrambleConfig, ScrambleEncoding};
use polars::prelude::*;

use super::rng::{row_mask, seeded_rng, stable_unit_floats};
use super::Simulator;

pub struct ScrambleSimulator {
    config: ScrambleConfig,
}

impl ScrambleSimulator {
    pub fn new(config: ScrambleConfig) -> Self {
        Self { config }
    }
}

impl Simulator for ScrambleSimulator {
    fn fit(&mut self, _df: &DataFrame) -> Result<(), CoreError> {
        Ok(())
    }

    fn transform(&self, df: &DataFrame) -> Result<DataFrame, CoreError> {
        let cfg = &self.config;
        if cfg.encodings.is_empty() {
            return Err(CoreError::InvalidConfig(
                "scramble needs at least one encoding".to_string(),
            ));
        }
        if !(0.0..=1.0).contains(&cfg.rate) {
            return Err(CoreError::InvalidConfig(format!(
                "scramble rate must be in [0, 1], got {}",
                cfg.rate
            )));
        }
        let column = df
            .column(&cfg.feature)
            .map_err(|_| CoreError::FeatureNotFound(cfg.feature.clone()))?;

        let values = column.cast(&DataType::Float64)?;
        let values = values.f64()?;
        let mask = row_mask(df.height(), cfg.rate, &mut seeded_rng(cfg.random_state));

        // Untouched rows use `{v}`, which round-trips exactly, so only the
        // selected rows differ once the loader casts the column back.
        let text: StringChunked = values
            .iter()
            .zip(&mask)
            .enumerate()
            .map(|(i, (v, &selected))| {
                v.map(|v| match selected {
                    true => self.scramble(i, v),
                    false => format!("{v}"),
                })
            })
            .collect();

        let mut out = df.clone();
        out.with_column(
            text.with_name(cfg.feature.as_str().into())
                .into_series()
                .into(),
        )?;
        Ok(out)
    }
}

impl ScrambleSimulator {
    /// Keyed by row position so a row's encoding doesn't depend on which
    /// other rows were selected.
    fn scramble(&self, row: usize, v: f64) -> String {
        let (u1, u2) = stable_unit_floats(self.config.random_state, &format!("scramble:{row}"));
        let pick = |u: f64, n: usize| ((u * n as f64) as usize).min(n - 1);
        let encoding = self.config.encodings[pick(u1, self.config.encodings.len())];
        let variants: Vec<String> = match encoding {
            ScrambleEncoding::NumericText => vec![
                decimal_comma(v),
                thousands(v),
                format!(" {v} "),
                format!("{v:e}"),
            ],
            ScrambleEncoding::Censored => vec![
                format!("<{v}"),
                format!(">{v}"),
                "ND".to_string(),
                "QNS".to_string(),
            ],
            ScrambleEncoding::UnitSuffix => vec![format!("{v} mg/dL")],
            ScrambleEncoding::Sentinel => vec!["-999".to_string(), "9999".to_string()],
        };
        variants[pick(u2, variants.len())].clone()
    }
}

/// `5,2`. Whole numbers get `,0` (`5` -> `5,0`): `{v}` prints them without a
/// decimal point, so a plain swap would leave text that still parses.
fn decimal_comma(v: f64) -> String {
    let s = format!("{v}");
    if s.contains('.') {
        s.replace('.', ",")
    } else {
        format!("{s},0")
    }
}

/// `1,200.5` for |v| >= 1000; smaller values have no thousands to group, so
/// they get the decimal-comma form instead.
fn thousands(v: f64) -> String {
    if v.abs() < 1000.0 {
        return decimal_comma(v);
    }
    let s = format!("{}", v.abs());
    let (int, frac) = s.split_once('.').map_or((s.as_str(), ""), |(i, f)| (i, f));
    let mut grouped = String::new();
    for (i, c) in int.chars().enumerate() {
        if i > 0 && (int.len() - i) % 3 == 0 {
            grouped.push(',');
        }
        grouped.push(c);
    }
    let sign = if v < 0.0 { "-" } else { "" };
    let frac = if frac.is_empty() {
        String::new()
    } else {
        format!(".{frac}")
    };
    format!("{sign}{grouped}{frac}")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config(rate: f64, encodings: Vec<ScrambleEncoding>) -> ScrambleConfig {
        ScrambleConfig {
            feature: "val".to_string(),
            rate,
            encodings,
            random_state: Some(3),
        }
    }

    fn frame() -> DataFrame {
        df![
            "val" => [1.5, 2.0, 3.25, 1200.5, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            "other" => [1i64, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        ]
        .unwrap()
    }

    /// What the model loader sees: a non-strict cast back to f64.
    fn parsed(df: &DataFrame) -> Vec<Option<f64>> {
        let s = df.column("val").unwrap().cast(&DataType::Float64).unwrap();
        s.f64().unwrap().iter().collect()
    }

    #[test]
    fn deterministic_and_string_typed() {
        let sim = ScrambleSimulator::new(config(0.5, ScrambleEncoding::ALL.to_vec()));
        let out1 = sim.transform(&frame()).unwrap();
        let out2 = sim.transform(&frame()).unwrap();
        assert!(out1.equals_missing(&out2));
        assert_eq!(out1.column("val").unwrap().dtype(), &DataType::String);
        assert_eq!(out1.get_column_names(), frame().get_column_names());
    }

    #[test]
    fn rate_zero_round_trips_exactly() {
        let df = df!["val" => [Some(0.1), None, Some(1e-7), Some(-1234.5678)]].unwrap();
        let sim = ScrambleSimulator::new(config(0.0, ScrambleEncoding::ALL.to_vec()));
        let out = sim.transform(&df).unwrap();
        let input: Vec<Option<f64>> = df.column("val").unwrap().f64().unwrap().iter().collect();
        assert_eq!(parsed(&out), input);
    }

    #[test]
    fn censored_rows_silently_become_null() {
        let df = frame();
        let sim = ScrambleSimulator::new(config(0.5, vec![ScrambleEncoding::Censored]));
        let out = sim.transform(&df).unwrap();
        let input: Vec<Option<f64>> = df.column("val").unwrap().f64().unwrap().iter().collect();
        let got = parsed(&out);
        assert_eq!(got.iter().filter(|v| v.is_none()).count(), 5);
        for (g, i) in got.iter().zip(&input) {
            assert!(g.is_none() || g == i, "{g:?} vs {i:?}");
        }
        assert!(out
            .column("other")
            .unwrap()
            .equals(df.column("other").unwrap()));
    }

    #[test]
    fn sentinels_parse_as_numbers() {
        let sim = ScrambleSimulator::new(config(1.0, vec![ScrambleEncoding::Sentinel]));
        let out = sim.transform(&frame()).unwrap();
        for v in parsed(&out) {
            assert!(matches!(v, Some(-999.0) | Some(9999.0)), "{v:?}");
        }
    }

    #[test]
    fn numeric_text_forms_and_what_the_loader_makes_of_them() {
        assert_eq!(decimal_comma(5.0), "5,0");
        assert_eq!(decimal_comma(5.2), "5,2");
        assert_eq!(thousands(1200.0), "1,200");
        assert_eq!(thousands(-1234567.5), "-1,234,567.5");
        assert_eq!(thousands(12.0), "12,0");

        // Only scientific notation survives the loader's lenient cast.
        let text = df!["val" => ["5,0", "1,200", " 5 ", "5e0", "5 mg/dL"]].unwrap();
        assert_eq!(parsed(&text), vec![None, None, None, Some(5.0), None]);
    }

    #[test]
    fn bad_config_is_an_error() {
        let mut missing = config(0.5, ScrambleEncoding::ALL.to_vec());
        missing.feature = "nope".to_string();
        let err = ScrambleSimulator::new(missing)
            .transform(&frame())
            .unwrap_err();
        assert!(matches!(err, CoreError::FeatureNotFound(ref n) if n == "nope"));

        let err = ScrambleSimulator::new(config(0.5, vec![]))
            .transform(&frame())
            .unwrap_err();
        assert!(matches!(err, CoreError::InvalidConfig(_)));
    }
}
