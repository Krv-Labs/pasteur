use pasteur_core::CoreError;
use std::path::{Path, PathBuf};

pub const CACHE_DIR_ENV: &str = "PASTEUR_RS_CACHE_DIR";

/// Precedence: explicit override, then `$PASTEUR_RS_CACHE_DIR`, then the OS
/// cache dir, then `./.pasteur-rs-cache`. Independent of the Python CLI
/// `.pasteur/` project layout.
pub fn cache_root(override_dir: Option<&Path>) -> PathBuf {
    if let Some(p) = override_dir {
        return p.to_path_buf();
    }
    if let Ok(p) = std::env::var(CACHE_DIR_ENV) {
        return PathBuf::from(p);
    }
    match dirs::cache_dir() {
        Some(base) => base.join("pasteur-rs"),
        None => PathBuf::from(".pasteur-rs-cache"),
    }
}

pub fn models_dir(root: &Path) -> PathBuf {
    root.join("models")
}

pub fn simulations_dir(root: &Path) -> PathBuf {
    root.join("simulations")
}

#[derive(Debug, Clone)]
pub struct CacheEntry {
    pub name: String,
    pub path: PathBuf,
    pub size_bytes: u64,
}

/// Lists the immediate children of `dir` (one entry per cached model file or
/// simulation run directory), each with its total on-disk size.
pub fn list_dir_entries(dir: &Path) -> Result<Vec<CacheEntry>, CoreError> {
    if !dir.exists() {
        return Ok(Vec::new());
    }
    let mut out = Vec::new();
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        let size_bytes = dir_size_bytes(&path)?;
        out.push(CacheEntry {
            name: entry.file_name().to_string_lossy().to_string(),
            path,
            size_bytes,
        });
    }
    out.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(out)
}

fn dir_size_bytes(path: &Path) -> Result<u64, CoreError> {
    let meta = std::fs::metadata(path)?;
    if meta.is_file() {
        return Ok(meta.len());
    }
    let mut total = 0u64;
    for entry in std::fs::read_dir(path)? {
        total += dir_size_bytes(&entry?.path())?;
    }
    Ok(total)
}

/// Deletes one named cache entry (a cached model file or a whole simulation
/// run directory) under `dir`.
pub fn delete_entry(dir: &Path, name: &str) -> Result<(), CoreError> {
    let path = dir.join(name);
    if !path.exists() {
        return Err(CoreError::InvalidConfig(format!(
            "no cache entry named {name:?} in {dir:?}"
        )));
    }
    if path.is_dir() {
        std::fs::remove_dir_all(path)?;
    } else {
        std::fs::remove_file(path)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cache_root_precedence_prefers_explicit_override() {
        let explicit = Path::new("/tmp/explicit-dir");
        assert_eq!(cache_root(Some(explicit)), explicit);
    }

    #[test]
    fn list_and_delete_round_trip_on_a_temp_tree() {
        let dir =
            std::env::temp_dir().join(format!("pasteur-hf-cache-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("run-a")).unwrap();
        std::fs::write(dir.join("run-a").join("manifest.json"), b"{}").unwrap();
        std::fs::write(dir.join("model.onnx"), b"fake-onnx-bytes").unwrap();

        let entries = list_dir_entries(&dir).unwrap();
        assert_eq!(entries.len(), 2);
        let run_a = entries.iter().find(|e| e.name == "run-a").unwrap();
        assert!(run_a.size_bytes > 0);

        delete_entry(&dir, "run-a").unwrap();
        assert_eq!(list_dir_entries(&dir).unwrap().len(), 1);
        assert!(delete_entry(&dir, "does-not-exist").is_err());

        std::fs::remove_dir_all(&dir).unwrap();
    }
}
