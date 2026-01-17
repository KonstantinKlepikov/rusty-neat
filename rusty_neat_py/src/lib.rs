use pyo3::prelude::*;
use pyo3::exceptions::PyRuntimeError;
use std::sync::{Arc, RwLock};

/// Thin PyO3 wrapper for `rusty_neat::Genome`
#[pyclass]
pub struct PyGenome {
    inner: Arc<RwLock<rusty_neat::Genome>>,
}

#[pymethods]
impl PyGenome {
    /// Construct a new Genome with basic dimensions. Other parameters use sensible defaults.
    #[new]
    fn new(id: u64, num_inputs: usize, num_hidden: usize, num_outputs: usize) -> Self {
        let params = rusty_neat::GenomeInitStruct {
            num_inputs,
            num_hidden,
            num_outputs,
            fs_neat: false,
            output_act_type: rusty_neat::genes::ActivationFunction::SignedSigmoid,
            hidden_act_type: rusty_neat::genes::ActivationFunction::SignedSigmoid,
            seed_type: rusty_neat::genome::GenomeSeedType::Perceptron,
            num_layers: 1,
            fs_neat_links: 0,
        };

        let g = rusty_neat::Genome::new(id, &params);
        PyGenome {
            inner: Arc::new(RwLock::new(g)),
        }
    }

    /// Load genome from a file (delegates to rusty_neat::Genome::from_file)
    #[staticmethod]
    fn from_file(path: &str) -> PyResult<Self> {
        match rusty_neat::Genome::from_file(path) {
            Ok(g) => Ok(PyGenome {
                inner: Arc::new(RwLock::new(g)),
            }),
            Err(e) => Err(PyRuntimeError::new_err(format!("IO error: {}", e))),
        }
    }

    fn get_id(&self) -> PyResult<u64> {
        let g = self.inner.read().map_err(|_| PyRuntimeError::new_err("lock poisoned"))?;
        Ok(g.id)
    }

    fn set_id(&mut self, id: u64) -> PyResult<()> {
        let mut g = self.inner.write().map_err(|_| PyRuntimeError::new_err("lock poisoned"))?;
        g.id = id;
        Ok(())
    }
}

/// Thin PyO3 wrapper for `rusty_neat::NeuralNetwork`
#[pyclass]
pub struct PyNeuralNetwork {
    inner: Arc<RwLock<rusty_neat::NeuralNetwork>>,
}

#[pymethods]
impl PyNeuralNetwork {
    #[new]
    fn new() -> Self {
        PyNeuralNetwork {
            inner: Arc::new(RwLock::new(rusty_neat::NeuralNetwork::new())),
        }
    }

    /// Provide inputs (accepts any Python sequence convertible to Vec<f64>)
    fn input(&self, inputs: &PyAny) -> PyResult<()> {
        let seq = inputs.extract::<Vec<f64>>().map_err(|e| PyRuntimeError::new_err(format!("expected sequence of floats: {}", e)))?;
        let mut nn = self.inner.write().map_err(|_| PyRuntimeError::new_err("lock poisoned"))?;
        nn.input(seq);
        Ok(())
    }

    /// Run activation step
    fn activate(&self) -> PyResult<()> {
        let mut nn = self.inner.write().map_err(|_| PyRuntimeError::new_err("lock poisoned"))?;
        nn.activate();
        Ok(())
    }

    /// Get outputs as a Python list of floats
    fn output(&self, py: Python) -> PyResult<PyObject> {
        let nn = self.inner.read().map_err(|_| PyRuntimeError::new_err("lock poisoned"))?;
        let out = nn.output();
        Ok(out.to_object(py))
    }
}

#[pymodule]
fn rusty_neat_py(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyGenome>()?;
    m.add_class::<PyNeuralNetwork>()?;
    Ok(())
}
