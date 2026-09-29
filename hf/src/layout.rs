use pasteur_core::CoreError;
use std::path::{Path, PathBuf};

pub const SOURCE_ROW_ID_COL: &str = "source_row_id";
pub const ROW_ORDINAL_COL: &str = "row_ordinal";

/// Lists immediate child directories under `root` (simulation-type folders).
pub fn list_sim_types(root: &Path) -> Result<Vec<String>, CoreError> {
    if !root.exists() {
        return Err(CoreError::InvalidConfig(format!(
            "simulation bundle root {root:?} does not exist"
        )));
    }
    let mut types = Vec::new();
    for entry in std::fs::read_dir(root)? {
        let entry = entry?;
        if entry.path().is_dir() {
            types.push(entry.file_name().to_string_lossy().to_string());
        }
    }
    types.sort();
    if types.is_empty() {
        return Err(CoreError::InvalidConfig(format!(
            "no simulation-type folders found under {root:?}"
        )));
    }
    Ok(types)
}

/// Lists `.parquet` files directly under `<sim_root>/<sim_type>/`.
pub fn list_parquet_files(sim_root: &Path, sim_type: &str) -> Result<Vec<PathBuf>, CoreError> {
    let dir = sim_root.join(sim_type);
    if !dir.is_dir() {
        return Err(CoreError::InvalidConfig(format!(
            "simulation type folder {dir:?} does not exist"
        )));
    }
    let mut paths = Vec::new();
    for entry in std::fs::read_dir(&dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_file() && path.extension().is_some_and(|e| e == "parquet") {
            paths.push(path);
        }
    }
    paths.sort();
    if paths.is_empty() {
        return Err(CoreError::InvalidConfig(format!(
            "no parquet files found under {dir:?}"
        )));
    }
    Ok(paths)
}

/// Validates that `root` has at least one sim-type folder and each contains parquet files.
pub fn validate_sim_bundle(root: &Path) -> Result<(), CoreError> {
    for sim_type in list_sim_types(root)? {
        list_parquet_files(root, &sim_type)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn temp_root(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!("pasteur-hf-layout-{name}-{}", std::process::id()))
    }

    #[test]
    fn list_sim_types_and_parquet_files() {
        let root = temp_root("list");
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(root.join("clean")).unwrap();
        fs::create_dir_all(root.join("blackout")).unwrap();
        fs::write(root.join("clean/clean.parquet"), b"fake").unwrap();
        fs::write(root.join("blackout/blackout.parquet"), b"fake").unwrap();

        let types = list_sim_types(&root).unwrap();
        assert_eq!(types, vec!["blackout", "clean"]);
        assert_eq!(list_parquet_files(&root, "clean").unwrap().len(), 1);

        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn validate_sim_bundle_requires_parquet_files() {
        let root = temp_root("validate");
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(root.join("empty")).unwrap();
        assert!(validate_sim_bundle(&root).is_err());
        let _ = fs::remove_dir_all(&root);
    }
}
