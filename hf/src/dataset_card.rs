use serde::Deserialize;

/// Source dataset metadata for building a simulation-output dataset card.
/// Populate from agent-provided JSON (`hf datasets info --format json`) or CLI flags.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct SourceDatasetInfo {
    #[serde(default)]
    pub tags: Vec<String>,
    #[serde(default)]
    pub description: String,
}

/// Renders a minimal HF dataset card (YAML front matter + body) for a
/// simulation-output repo, carrying over tags/description from the source
/// dataset it was derived from.
pub fn render_dataset_card(
    repo_id: &str,
    source_dataset_id: &str,
    source_tags: &[String],
    source_description: &str,
    simulation_types: &[String],
) -> String {
    let mut tags = source_tags.to_vec();
    tags.push("pasteur-simulation".to_string());
    let tags_yaml = tags
        .iter()
        .map(|t| format!("  - {t}"))
        .collect::<Vec<_>>()
        .join("\n");
    let layout = simulation_types
        .iter()
        .map(|t| format!("- `{t}/` — variant parquet files"))
        .collect::<Vec<_>>()
        .join("\n");

    format!(
        "---\ntags:\n{tags_yaml}\n---\n\n\
# {repo_id}\n\n\
Pasteur simulation outputs derived from [{source_dataset_id}](https://huggingface.co/datasets/{source_dataset_id}).\n\n\
{source_description}\n\n\
## Source baseline (not in this repo)\n\n\
This bundle contains **perturbed variants only**. The unperturbed baseline lives on the source dataset \
(`data/uci-thyroid/processed/clean-default/clean.parquet` on `{source_dataset_id}`), not duplicated here. \
Agents running `pasteur-cli evaluate` / `compare` should download that file locally as `clean/clean.parquet` under `--sim-root`.\n\n\
## Layout\n\n\
{layout}\n\n\
Optional: `predictions/<sim_type>.parquet` — per-patient model probabilities from `pasteur-cli compare`.\n\n\
Publish this bundle as a private repository with the Hugging Face CLI (`hf repos create --private`, `hf upload`). See `AGENTS.md` in pasteur-core.\n"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn card_includes_source_tags_and_simulation_folders() {
        let card = render_dataset_card(
            "krv-labs/uci-thyroid-simulations",
            "krv-labs/uci-thyroid",
            &["clinical".to_string(), "thyroid".to_string()],
            "UCI Thyroid Clinical & Graph Dataset",
            &["blackout".to_string(), "jitter".to_string()],
        );
        assert!(card.contains("  - clinical"));
        assert!(card.contains("  - pasteur-simulation"));
        assert!(card.contains("krv-labs/uci-thyroid"));
        assert!(card.contains("- `blackout/`"));
        assert!(card.contains("- `jitter/`"));
        assert!(card.contains("hf upload"));
    }
}
