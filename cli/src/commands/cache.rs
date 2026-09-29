use std::path::{Path, PathBuf};

use anyhow::{bail, Result};
use pasteur_hf::cache::{self, delete_entry, list_dir_entries, models_dir, simulations_dir};

use crate::args::{CacheClearArgs, CacheCommand, CacheListArgs, CacheRmArgs};

pub fn run(args: crate::args::CacheArgs) -> Result<()> {
    match args.command {
        CacheCommand::List(args) => run_list(args),
        CacheCommand::Rm(args) => run_rm(args),
        CacheCommand::Clear(args) => run_clear(args),
    }
}

fn run_list(args: CacheListArgs) -> Result<()> {
    let root = cache_root(&args.common.cache_dir);
    println!("cache root: {}", root.display());
    print_bucket("models", &models_dir(&root))?;
    print_bucket("simulations", &simulations_dir(&root))?;
    Ok(())
}

fn run_rm(args: CacheRmArgs) -> Result<()> {
    let root = cache_root(&args.common.cache_dir);
    let kind = args.kind.as_str();
    let dir = resolve_bucket_dir(&root, kind, &args.name)?;
    delete_entry(&dir, &args.name).map_err(|e| anyhow::anyhow!("{e}"))?;
    println!("removed {} from {}", args.name, dir.display());
    Ok(())
}

fn run_clear(args: CacheClearArgs) -> Result<()> {
    let root = cache_root(&args.common.cache_dir);
    let removed = match args.kind.as_deref() {
        Some(kind) => clear_bucket(&root, kind)?,
        None => clear_bucket(&root, "models")? + clear_bucket(&root, "simulations")?,
    };
    println!(
        "cleared {removed} cache entr{}",
        if removed == 1 { "y" } else { "ies" }
    );
    Ok(())
}

fn cache_root(override_dir: &Option<PathBuf>) -> PathBuf {
    cache::cache_root(override_dir.as_deref())
}

fn print_bucket(label: &str, dir: &Path) -> Result<()> {
    let entries = list_dir_entries(dir).map_err(|e| anyhow::anyhow!("{e}"))?;
    println!();
    println!("{label}/");
    if entries.is_empty() {
        println!("  (empty)");
        return Ok(());
    }
    for entry in entries {
        println!("  {}  {}", format_size(entry.size_bytes), entry.name);
    }
    Ok(())
}

fn clear_bucket(root: &Path, kind: &str) -> Result<usize> {
    let dir = bucket_dir(root, kind)?;
    let entries = list_dir_entries(&dir).map_err(|e| anyhow::anyhow!("{e}"))?;
    for entry in &entries {
        delete_entry(&dir, &entry.name).map_err(|e| anyhow::anyhow!("{e}"))?;
    }
    Ok(entries.len())
}

fn resolve_bucket_dir(root: &Path, kind: &str, name: &str) -> Result<PathBuf> {
    match kind {
        "models" => Ok(models_dir(root)),
        "simulations" => Ok(simulations_dir(root)),
        "auto" => find_entry_dir(root, name),
        other => bail!("unknown cache kind {other:?} (expected models, simulations, or auto)"),
    }
}

fn find_entry_dir(root: &Path, name: &str) -> Result<PathBuf> {
    for dir in [models_dir(root), simulations_dir(root)] {
        if dir.join(name).exists() {
            return Ok(dir);
        }
    }
    bail!("no cache entry named {name:?} under {}", root.display())
}

fn bucket_dir(root: &Path, kind: &str) -> Result<PathBuf> {
    match kind {
        "models" => Ok(models_dir(root)),
        "simulations" => Ok(simulations_dir(root)),
        other => bail!("unknown cache kind {other:?} (expected models or simulations)"),
    }
}

fn format_size(bytes: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = KB * 1024;
    const GB: u64 = MB * 1024;
    if bytes >= GB {
        format!("{:>7.1} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:>7.1} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:>7.1} KB", bytes as f64 / KB as f64)
    } else {
        format!("{:>7} B", bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn temp_cache_root(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!("pasteur-cli-cache-{name}-{}", std::process::id()))
    }

    #[test]
    fn clear_bucket_removes_all_entries() {
        let root = temp_cache_root("clear");
        let _ = fs::remove_dir_all(&root);
        let models = models_dir(&root);
        fs::create_dir_all(&models).unwrap();
        fs::write(models.join("a.onnx"), b"x").unwrap();
        fs::write(models.join("b.onnx"), b"yy").unwrap();

        assert_eq!(clear_bucket(&root, "models").unwrap(), 2);
        assert!(list_dir_entries(&models).unwrap().is_empty());

        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn find_entry_dir_searches_both_buckets() {
        let root = temp_cache_root("find");
        let _ = fs::remove_dir_all(&root);
        let sims = simulations_dir(&root);
        fs::create_dir_all(sims.join("run-1")).unwrap();

        let dir = find_entry_dir(&root, "run-1").unwrap();
        assert_eq!(dir, sims);

        let _ = fs::remove_dir_all(&root);
    }
}
