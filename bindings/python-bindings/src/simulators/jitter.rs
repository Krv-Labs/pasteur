use pasteur_core::{JitterConfig, JitterSimulator as CoreJitterSimulator, Simulator};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3_polars::PyDataFrame;

#[pyclass(name = "JitterSimulator")]
pub struct PyJitterSimulator {
    inner: CoreJitterSimulator,
}

#[pymethods]
impl PyJitterSimulator {
    #[new]
    #[pyo3(signature = (feature, scale=1.0, random_state=None))]
    fn new(feature: String, scale: f64, random_state: Option<u64>) -> PyResult<Self> {
        let config = JitterConfig {
            feature,
            scale,
            random_state,
        };
        Ok(Self {
            inner: CoreJitterSimulator::new(config),
        })
    }

    fn fit(&mut self, df: PyDataFrame) -> PyResult<()> {
        self.inner
            .fit(&df.into())
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(())
    }

    fn transform(&self, df: PyDataFrame) -> PyResult<PyDataFrame> {
        let result = self
            .inner
            .transform(&df.into())
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(PyDataFrame(result))
    }
}
