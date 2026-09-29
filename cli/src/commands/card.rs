use std::fs;

use anyhow::{Context, Result};
use pasteur_hf::dataset_card::{render_dataset_card, SourceDatasetInfo};
use pasteur_hf::layout;

use crate::args::CardArgs;

pub fn run(args: CardArgs) -> Result<()> {
    let info = load_source_info(&args)?;
    let sim_types: Vec<String> = layout::list_sim_types(&args.input)
        .map_err(|e| anyhow::anyhow!("{e}"))
        .context("failed to list simulation types")?
        .into_iter()
        .filter(|t| t != "clean")
        .collect();

    let card = render_dataset_card(
        &args.dest_repo,
        &args.source_dataset,
        &info.tags,
        &info.description,
        &sim_types,
    );

    let readme_path = args
        .output_readme
        .unwrap_or_else(|| args.input.join("README.md"));
    if let Some(parent) = readme_path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(&readme_path, &card)?;
    println!("wrote dataset card to {}", readme_path.display());
    Ok(())
}

fn load_source_info(args: &CardArgs) -> Result<SourceDatasetInfo> {
    if let Some(path) = &args.source_info {
        let raw = fs::read_to_string(path)
            .with_context(|| format!("failed to read source info {}", path.display()))?;
        return serde_json::from_str(&raw)
            .with_context(|| format!("failed to parse source info JSON from {}", path.display()));
    }
    Ok(SourceDatasetInfo {
        tags: args.tags.clone(),
        description: args.description.clone().unwrap_or_default(),
    })
}
