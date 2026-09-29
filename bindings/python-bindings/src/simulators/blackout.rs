use pasteur_core::{BlackoutConfig, BlackoutSimulator as CoreBlackoutSimulator, Simulator};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3_polars::PyDataFrame;

#[pyclass(name = "BlackoutSimulator")]
pub struct PyBlackoutSimulator {
    inner: CoreBlackoutSimulator,
}

#[pymethods]
impl PyBlackoutSimulator {
    #[new]
    // `companions` goes last so existing positional callers keep working.
    #[pyo3(signature = (feature, rate=0.1, window_frac=0.1, patient_id_col=None, time_col=None, random_state=None, companions=None))]
    fn new(
        feature: String,
        rate: f64,
        window_frac: f64,
        patient_id_col: Option<String>,
        time_col: Option<String>,
        random_state: Option<u64>,
        companions: Option<Vec<String>>,
    ) -> PyResult<Self> {
        let config = BlackoutConfig {
            feature,
            companions: companions.unwrap_or_default(),
            rate,
            window_frac,
            patient_id_col,
            time_col,
            random_state,
        };
        Ok(Self {
            inner: CoreBlackoutSimulator::new(config),
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
