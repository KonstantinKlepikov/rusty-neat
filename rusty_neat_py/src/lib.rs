use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
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
        let g = self
            .inner
            .read()
            .map_err(|_| PyRuntimeError::new_err("lock poisoned"))?;
        Ok(g.id)
    }

    fn set_id(&mut self, id: u64) -> PyResult<()> {
        let mut g = self
            .inner
            .write()
            .map_err(|_| PyRuntimeError::new_err("lock poisoned"))?;
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
        let seq = inputs
            .extract::<Vec<f64>>()
            .map_err(|e| PyRuntimeError::new_err(format!("expected sequence of floats: {}", e)))?;
        let mut nn = self
            .inner
            .write()
            .map_err(|_| PyRuntimeError::new_err("lock poisoned"))?;
        nn.input(seq);
        Ok(())
    }

    /// Run activation step
    fn activate(&self) -> PyResult<()> {
        let mut nn = self
            .inner
            .write()
            .map_err(|_| PyRuntimeError::new_err("lock poisoned"))?;
        nn.activate();
        Ok(())
    }

    /// Get outputs as a Python list of floats
    fn output(&self, py: Python) -> PyResult<PyObject> {
        let nn = self
            .inner
            .read()
            .map_err(|_| PyRuntimeError::new_err("lock poisoned"))?;
        let out = nn.output();
        Ok(out.to_object(py))
    }
}

/// Thin PyO3 wrapper for `rusty_neat::Parameters`
#[pyclass]
pub struct PyParameters {
    inner: Arc<RwLock<rusty_neat::Parameters>>,
}

#[pymethods]
impl PyParameters {
    #[new]
    fn new() -> Self {
        PyParameters {
            inner: Arc::new(RwLock::new(rusty_neat::Parameters::default())),
        }
    }

    fn get_min_weight(&self) -> PyResult<f64> {
        let p = self
            .inner
            .read()
            .map_err(|_| PyRuntimeError::new_err("lock poisoned"))?;
        Ok(p.min_weight)
    }

    fn set_min_weight(&mut self, v: f64) -> PyResult<()> {
        let mut p = self
            .inner
            .write()
            .map_err(|_| PyRuntimeError::new_err("lock poisoned"))?;
        p.min_weight = v;
        Ok(())
    }

    fn reset(&mut self) -> PyResult<()> {
        let mut p = self
            .inner
            .write()
            .map_err(|_| PyRuntimeError::new_err("lock poisoned"))?;
        *p = rusty_neat::Parameters::default();
        Ok(())
    }
}

/// Thin PyO3 wrapper for `rusty_neat::Substrate`
#[pyclass]
pub struct PySubstrate {
    inner: Arc<RwLock<rusty_neat::Substrate>>,
}

#[pymethods]
impl PySubstrate {
    #[new]
    fn new() -> Self {
        PySubstrate {
            inner: Arc::new(RwLock::new(rusty_neat::Substrate::new())),
        }
    }

    /// Construct with explicit coords: accepts sequences of sequences of floats
    #[staticmethod]
    fn with_coords(inputs: &PyAny, hidden: &PyAny, outputs: &PyAny) -> PyResult<Self> {
        let in_coords = inputs.extract::<Vec<Vec<f64>>>().map_err(|e| {
            PyRuntimeError::new_err(format!(
                "expected nested sequence of floats for inputs: {}",
                e
            ))
        })?;
        let hidden_coords = hidden.extract::<Vec<Vec<f64>>>().map_err(|e| {
            PyRuntimeError::new_err(format!(
                "expected nested sequence of floats for hidden: {}",
                e
            ))
        })?;
        let out_coords = outputs.extract::<Vec<Vec<f64>>>().map_err(|e| {
            PyRuntimeError::new_err(format!(
                "expected nested sequence of floats for outputs: {}",
                e
            ))
        })?;

        Ok(PySubstrate {
            inner: Arc::new(RwLock::new(rusty_neat::Substrate::with_coords(
                in_coords,
                hidden_coords,
                out_coords,
            ))),
        })
    }

    fn get_min_cppn_inputs(&self) -> PyResult<usize> {
        let s = self
            .inner
            .read()
            .map_err(|_| PyRuntimeError::new_err("lock poisoned"))?;
        Ok(s.get_min_cppn_inputs())
    }

    fn get_min_cppn_outputs(&self) -> PyResult<usize> {
        let s = self
            .inner
            .read()
            .map_err(|_| PyRuntimeError::new_err("lock poisoned"))?;
        Ok(s.get_min_cppn_outputs())
    }

    fn print_info(&self) -> PyResult<()> {
        let s = self
            .inner
            .read()
            .map_err(|_| PyRuntimeError::new_err("lock poisoned"))?;
        s.print_info();
        Ok(())
    }

    fn set_neurons(
        &mut self,
        inputs: Vec<Vec<f64>>,
        hidden: Vec<Vec<f64>>,
        outputs: Vec<Vec<f64>>,
    ) -> PyResult<()> {
        let mut s = self
            .inner
            .write()
            .map_err(|_| PyRuntimeError::new_err("lock poisoned"))?;
        s.set_neurons(inputs, hidden, outputs);
        Ok(())
    }

    fn clear_custom_connectivity(&mut self) -> PyResult<()> {
        let mut s = self
            .inner
            .write()
            .map_err(|_| PyRuntimeError::new_err("lock poisoned"))?;
        s.clear_custom_connectivity();
        Ok(())
    }

    fn set_custom_connectivity(&mut self, conns: Vec<Vec<i32>>) -> PyResult<()> {
        let mut s = self
            .inner
            .write()
            .map_err(|_| PyRuntimeError::new_err("lock poisoned"))?;
        s.set_custom_connectivity(conns);
        Ok(())
    }
}

/// Thin PyO3 wrapper for random utilities (rusty_neat::random::Random)
#[pyclass]
pub struct PyRNG;

#[pymethods]
impl PyRNG {
    #[new]
    fn new() -> Self {
        PyRNG {}
    }

    fn rand_float(&self) -> PyResult<f64> {
        Ok(rusty_neat::random::Random::rand_float())
    }

    fn rand_int(&self, max: u64) -> PyResult<u64> {
        Ok(rusty_neat::random::Random::rand_int(max))
    }
}

#[pymodule]
fn rusty_neat_py(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyGenome>()?;
    m.add_class::<PyNeuralNetwork>()?;
    m.add_class::<PyParameters>()?;
    m.add_class::<PySubstrate>()?;
    m.add_class::<PyRNG>()?;
    Ok(())
}
