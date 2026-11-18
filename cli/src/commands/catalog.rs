use anyhow::Result;
use pasteur_core::{ScrambleEncoding, SiteConfig, SiteStep, UNIT_PRESETS};
use serde_json::{json, Value};

/// A site file that uses every step kind. Built from the real types, so the
/// example in the catalog cannot drift from what `simulate --site` accepts.
fn example_site() -> SiteConfig {
    SiteConfig {
        steps: vec![
            SiteStep::Units {
                feature: "FT4".to_string(),
                rate: 1.0,
                preset: Some("ft4_ngdl_to_pmoll".to_string()),
                factor: None,
                offset: None,
                chance: 1.0,
            },
            SiteStep::Blackout {
                feature: "T3".to_string(),
                companions: vec!["T3_measured".to_string()],
                rate: 1.0,
                chance: 0.5,
            },
            SiteStep::Scramble {
                feature: "TSH".to_string(),
                rate: 0.3,
                encodings: vec![ScrambleEncoding::Censored, ScrambleEncoding::NumericText],
                chance: 1.0,
            },
        ],
        random_state: None,
    }
}

fn catalog() -> Result<Value> {
    let presets: Vec<Value> = UNIT_PRESETS
        .iter()
        .map(|(name, factor, offset)| json!({"name": name, "factor": factor, "offset": offset}))
        .collect();
    Ok(json!({
        "version": env!("CARGO_PKG_VERSION"),
        "simulators": {
            "blackout": "nulls a seeded share of one feature's rows (and companions)",
            "jitter": "adds seeded measurement noise to one numeric feature",
            "flipper": "interpolates between pairs of opposite-label rows",
            "sweep": "moves one feature over a grid, the rest of the row fixed",
            "scramble": "re-encodes a seeded share of one feature's rows as messy export text",
            "units": "rewrites a seeded share of one feature's rows as value * factor + offset",
            "site": "applies several blackout/scramble/units steps together as one site",
        },
        "stress_sim_types": ["blackout", "scramble", "units", "site"],
        "scramble_encodings": ScrambleEncoding::ALL,
        "unit_presets": presets,
        "site_file": {
            "flags": {
                "--site <path>": "write site/site.parquet with every step applied",
                "--site-draws <n>": "write site/site_<i>.parquet instead; each draw keeps each step with its chance",
            },
            "steps": {
                "blackout": {"feature": "column", "rate": "0..1", "companions": "[column], optional", "chance": "0..1, default 1"},
                "scramble": {"feature": "column", "rate": "0..1", "encodings": "[encoding], default all", "chance": "0..1, default 1"},
                "units": {"feature": "column", "rate": "0..1", "preset": "preset name, or", "factor": "number, with optional", "offset": "number", "chance": "0..1, default 1"},
            },
            "rules": [
                "steps run in order; each step and each draw picks its own rows",
                "nothing may touch a feature after a scramble step: scramble leaves text",
                "encoding names here use underscores (numeric_text); the --scramble-encoding flag uses hyphens",
            ],
            "example": example_site(),
        },
    }))
}

pub fn run() -> Result<()> {
    println!("{}", serde_json::to_string_pretty(&catalog()?)?);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use pasteur_core::SiteSimulator;

    #[test]
    fn catalog_example_is_a_valid_site_file() {
        let catalog = catalog().unwrap();
        assert_eq!(catalog["unit_presets"].as_array().unwrap().len(), 9);
        assert_eq!(
            catalog["scramble_encodings"],
            json!(["numeric_text", "censored", "unit_suffix", "sentinel"])
        );
        let example = serde_json::to_string(&catalog["site_file"]["example"]).unwrap();
        let config: SiteConfig = serde_json::from_str(&example).unwrap();
        SiteSimulator::new(config).validate().unwrap();
    }
}
