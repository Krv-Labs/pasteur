use pyo3::prelude::*;

pub mod simulators;

use simulators::blackout::PyBlackoutSimulator;
use simulators::jitter::PyJitterSimulator;

/// Register the common Pasteur components to a Python module.
pub fn register_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyBlackoutSimulator>()?;
    m.add_class::<PyJitterSimulator>()?;
    Ok(())
}

#[pymodule]
fn pypasteur(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    register_module(m)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    // maturin's generated `__init__.py` re-exports with `import *`, which
    // skips dunder names unless they are listed in `__all__`.
    m.add(
        "__all__",
        ["BlackoutSimulator", "JitterSimulator", "__version__"],
    )?;
    Ok(())
}
