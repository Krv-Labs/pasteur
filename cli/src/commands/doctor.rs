use anyhow::Result;
use pasteur_model::runtime;

/// Load ONNX Runtime the way `evaluate` and `compare` will, and say where it
/// came from. Exits non-zero, with the search list and the fixes, when it
/// cannot be found: a check worth running before scoring a cohort.
pub fn run() -> Result<()> {
    println!("pasteur-cli {}", env!("CARGO_PKG_VERSION"));
    let info = runtime::ensure()?;
    println!("onnx runtime: {}", info.path.display());
    println!("{}", info.build);
    Ok(())
}
